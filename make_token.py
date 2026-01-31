# build_text_llm_structured_tokens_no_locf_optionB_sharded.py
# ============================================================
# (Option B + SHARDED MULTI-PROC) LLM -> JSON 구조화 특징 + 토큰리스트(EmbeddingBag) 빌더
#   - ICUSTAYS 기반 / NO-LOCF
#
# ✅ Option B:
#   - exam(=exam_name)을 LLM 프롬프트 컨텍스트 + 캐시 키에만 사용
#   - class_label은 사용하지 않음 (modality/region으로 충분하므로 제거)
#
# ✅ Sharding (multi-proc / multi-gpu) 지원:
#   - note_id를 (재현 가능한) 랜덤 셔플 후 shard_id로 분할
#   - cache_only 모드로 "캐시만" 병렬로 채운 뒤, 마지막에 1회 전체 빌드(집계/npz/csv) 권장
#
# 권장 워크플로우 (예: GPU 3개):
#   1) (각 GPU별) LLM 서버 3개 띄우기 (ollama 3개 인스턴스 or vLLM 3개 서버)
#   2) cache_only shard 3개를 동시에 실행하여 캐시 채우기
#   3) cache가 채워진 뒤, 단일 프로세스로 전체 빌드 (LLM 호출 거의/전혀 없음)
#
# 출력:
#   out_dir/text_llm_struct_tokens.csv
#   out_dir/meta_text_llm_struct_tokens.json
#   out_dir/meta_bags.npz
# ============================================================

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from urllib import request
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
import pandas as pd


# -------------------------
# Cache helpers (sharded cache to avoid huge single directories)
# -------------------------
def _cache_paths(cfg: "CFG", h: str) -> Tuple[str, str]:
    base = os.path.join(cfg.cache_dir, "llm_json")
    h = str(h)
    if len(h) >= 4:
        primary = os.path.join(base, h[:2], h[2:4], f"{h}.json")
    else:
        primary = os.path.join(base, "_", "_", f"{h}.json")
    fallback = os.path.join(base, f"{h}.json")
    return primary, fallback


def _load_cache_json(cfg: "CFG", h: str) -> Optional[Dict[str, Any]]:
    if not cfg.use_cache:
        return None
    primary, fallback = _cache_paths(cfg, h)

    if os.path.exists(primary):
        try:
            with open(primary, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # legacy fallback
    if os.path.exists(fallback):
        try:
            with open(fallback, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _save_cache_json(cfg: "CFG", h: str, obj: Dict[str, Any]) -> None:
    if not cfg.use_cache:
        return
    primary, _ = _cache_paths(cfg, h)
    os.makedirs(os.path.dirname(primary), exist_ok=True)
    tmp = primary + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, primary)


# -------------------------
# Utils
# -------------------------
def _safe_makedirs(path: str):
    os.makedirs(path, exist_ok=True)


def _dbg(cfg: "CFG", msg: str, **kwargs):
    if cfg.debug:
        print(f"[DBG] {msg}")
        for k, v in kwargs.items():
            print(f"  - {k}: {v}")


def _sanitize_token(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^0-9a-zA-Z가-힣_+\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_simple(s: str, max_tokens: int = 256) -> List[str]:
    s = _sanitize_token(s)
    if not s:
        return []
    toks = s.split(" ")
    toks = [t for t in toks if t and len(t) <= 64]
    if max_tokens > 0:
        toks = toks[:max_tokens]
    return toks


def _sanitize_cat(s: str) -> str:
    s = str(s)
    s = s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    for ch in ["(", ")", "[", "]", "{", "}", ":", ";", ",", "."]:
        s = s.replace(ch, "_")
    s = s.strip()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s if s else "Unknown")[:64]


def _infer_existing_path(candidates: List[str]) -> str:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def _fix_note_id_series(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def _to_key_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float64")


def _safe_dt(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _delta_from_mask_and_grid_per_stay(
    df_sorted: pd.DataFrame,
    mask_col: str,
    grid_col: str,
    stay_col: str,
    grid_hours: int,
) -> np.ndarray:
    out = np.zeros(len(df_sorted), dtype=np.int32)
    for _sid, idx in df_sorted.groupby(stay_col).indices.items():
        idx = np.asarray(idx, dtype=np.int64)
        mask = df_sorted.iloc[idx][mask_col].to_numpy(dtype=np.int8, copy=False)
        grid = df_sorted.iloc[idx][grid_col].to_numpy(dtype=np.int32, copy=False)
        last_g = None
        for j in range(len(idx)):
            g = int(grid[j])
            if mask[j] == 1:
                out[idx[j]] = 0
                last_g = g
            else:
                if last_g is None:
                    out[idx[j]] = int((g + 1) * grid_hours)
                else:
                    out[idx[j]] = int((g - last_g) * grid_hours)
    return out


def build_embeddingbag_inputs(
    row_ids: np.ndarray,
    item_ids: np.ndarray,
    n_rows: int,
    unknown_id: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    row_ids = row_ids.astype(np.int64, copy=False)
    item_ids = item_ids.astype(np.int32, copy=False)

    if n_rows <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((1,), dtype=np.int64)

    if row_ids.size == 0:
        ids = np.full(n_rows, unknown_id, dtype=np.int32)
        offsets = np.arange(0, n_rows + 1, dtype=np.int64)
        return ids, offsets

    order = np.argsort(row_ids, kind="mergesort")
    r = row_ids[order]
    v = item_ids[order]

    counts = np.bincount(r, minlength=n_rows).astype(np.int64)
    counts2 = counts.copy()
    empty = counts2 == 0
    counts2[empty] = 1

    offsets1 = np.zeros(n_rows + 1, dtype=np.int64)
    offsets1[1:] = np.cumsum(counts)

    offsets2 = np.zeros(n_rows + 1, dtype=np.int64)
    offsets2[1:] = np.cumsum(counts2)

    ids2 = np.full(offsets2[-1], unknown_id, dtype=np.int32)

    within = np.arange(len(v), dtype=np.int64) - offsets1[r]
    new_pos = offsets2[r] + within
    ids2[new_pos] = v

    return ids2, offsets2


# -------------------------
# Local LLM caller
# -------------------------
def _http_json(url: str, payload: Dict[str, Any], timeout: int = 120) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        b = resp.read()
    return json.loads(b.decode("utf-8", errors="ignore"))


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        cand = m.group(1).strip()
        try:
            return json.loads(cand)
        except Exception:
            pass

    i = text.find("{")
    j = text.rfind("}")
    if i >= 0 and j > i:
        cand = text[i : j + 1].strip()
        try:
            return json.loads(cand)
        except Exception:
            cand2 = re.sub(r",\s*}", "}", cand)
            cand2 = re.sub(r",\s*]", "]", cand2)
            try:
                return json.loads(cand2)
            except Exception:
                return None
    return None


def llm_extract_json(
    *,
    backend: str,
    text: str,
    modality: str,
    region: str,
    exam: str,
    cfg: "CFG",
) -> Dict[str, Any]:
    text = str(text or "")
    text_in = text[: int(cfg.text_max_chars)]

    key = (
        cfg.schema_version
        + "|" + str(modality)
        + "|" + str(region)
        + "|" + str(exam)
        + "|" + text_in
    )
    h = hashlib.md5(key.encode("utf-8", errors="ignore")).hexdigest()

    cached = _load_cache_json(cfg, h)
    if cached is not None:
        return cached

    system = (
        "You are a clinical text extraction assistant. "
        "Return ONLY a single JSON object (no extra text). "
        "If unknown, use null/empty list. Do not hallucinate patient identifiers."
    )
    schema = {
        "schema_version": cfg.schema_version,
        "summary": "string (<=40 words)",
        "critical": "boolean",
        "abnormal": "boolean",
        "severity": "integer 0-3",
        "uncertainty": "integer 0-3",
        "keywords": ["short strings"],
        "diagnoses": ["short strings"],
        "anatomy": ["short strings"],
        "procedures": ["short strings"],
        "negated": ["short strings"],
    }
    user = (
        f"MODALITY={modality}\nREGION={region}\nEXAM={exam}\n\n"
        f"TEXT:\n{text_in}\n\n"
        f"Extract into this JSON schema:\n{json.dumps(schema)}"
    )

    out_text = ""
    try:
        if backend == "ollama":
            url = cfg.ollama_url.rstrip("/") + "/api/chat"
            payload = {
                "model": cfg.ollama_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "options": {"temperature": cfg.llm_temperature},
                "keep_alive": -1,
                "stream": False,
            }
            res = _http_json(url, payload, timeout=cfg.llm_timeout)
            out_text = (res.get("message") or {}).get("content", "") or ""
        elif backend == "openai_compat":
            url = cfg.openai_base_url.rstrip("/") + "/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.openai_api_key:
                headers["Authorization"] = f"Bearer {cfg.openai_api_key}"
            payload = {
                "model": cfg.openai_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": cfg.llm_temperature,
            }
            data = json.dumps(payload).encode("utf-8")
            req = request.Request(url, data=data, headers=headers)
            with request.urlopen(req, timeout=cfg.llm_timeout) as resp:
                res = json.loads(resp.read().decode("utf-8", errors="ignore"))
            choices = res.get("choices") or []
            if choices:
                out_text = (((choices[0] or {}).get("message") or {}).get("content")) or ""
        else:
            raise ValueError(f"Unknown llm_backend='{backend}'")
    except (HTTPError, URLError, TimeoutError, ValueError) as e:
        if cfg.debug:
            print(f"[WARN] LLM call failed: {type(e).__name__}: {e}")
        out_text = ""

    obj = _extract_json_object(out_text) or {}
    _save_cache_json(cfg, h, obj)

    if cfg.llm_sleep_ms > 0:
        time.sleep(cfg.llm_sleep_ms / 1000.0)

    return obj


# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    grid_hours: int = 1

    mimic_iv_dir: str = "mimiciv/3.1"
    icustays_path: Optional[str] = None
    max_stay_hours: Optional[int] = None

    radiology_with_class_path: str = "radiology_with_exam_and_class.csv"
    radiology_csv_path: str = "mimic-iv-note/2.2/note/radiology.csv"

    llm_backend: str = "ollama"   # ollama | openai_compat
    llm_temperature: float = 0.0
    llm_timeout: int = 180
    llm_sleep_ms: int = 0
    llm_workers: int = 4

    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1:8b"

    openai_base_url: str = "http://localhost:8000"
    openai_model: str = "llama"
    openai_api_key: str = ""

    schema_version: str = "v1"

    token_min_freq: int = 3
    token_max_vocab: int = 20000
    token_max_per_note: int = 256

    text_max_chars: int = 4000

    cache_dir: str = "artifacts/note_build_cache_llm"
    use_cache: bool = True

    out_dir: str = "artifacts/note_llm_struct_tokens"
    out_csv: str = "text_llm_struct_tokens.csv"
    out_meta: str = "meta_text_llm_struct_tokens.json"

    max_notes: int = 0
    debug: bool = True

    # ✅ sharding
    num_shards: int = 1
    shard_id: int = 0
    shard_seed: int = 13
    cache_only: bool = False


# -------------------------
# resolve paths
# -------------------------
def resolve_paths(cfg: CFG) -> Dict[str, str]:
    icustays = cfg.icustays_path or _infer_existing_path([
        os.path.join(cfg.mimic_iv_dir, "icu", "icustays.csv.gz"),
        os.path.join(cfg.mimic_iv_dir, "icu", "icustays.csv"),
    ])
    return {"icustays": icustays}


# -------------------------
# base_grid
# -------------------------
def load_icustays_base_grid(cfg: CFG, icustays_path: str) -> pd.DataFrame:
    icu = pd.read_csv(icustays_path, low_memory=False)

    need = ["stay_id", "subject_id", "hadm_id", "intime", "outtime"]
    for c in need:
        if c not in icu.columns:
            raise ValueError(f"icustays missing column: {c}")

    icu["stay_id"] = pd.to_numeric(icu["stay_id"], errors="coerce")
    icu["subject_id"] = _to_key_float(icu["subject_id"])
    icu["hadm_id"] = _to_key_float(icu["hadm_id"])
    icu["intime"] = _safe_dt(icu["intime"])
    icu["outtime"] = _safe_dt(icu["outtime"])
    icu = icu.dropna(subset=["stay_id", "subject_id", "hadm_id", "intime", "outtime"]).copy()
    icu["stay_id"] = icu["stay_id"].astype(np.int64)

    los_h = (icu["outtime"] - icu["intime"]).dt.total_seconds() / 3600.0
    los_h = los_h.clip(lower=0.0)
    if cfg.max_stay_hours is not None:
        los_h = np.minimum(los_h, float(cfg.max_stay_hours))
    icu["los_hours"] = los_h
    icu["max_grid"] = np.floor(icu["los_hours"].values / float(cfg.grid_hours)).astype(np.int32)

    _dbg(cfg, "icustays loaded", rows=len(icu), stays=int(icu["stay_id"].nunique()))

    parts = []
    for sid, mg in zip(icu["stay_id"].tolist(), icu["max_grid"].tolist()):
        mg = int(mg)
        if mg < 0:
            continue
        parts.append(pd.DataFrame({"stay_id": sid, "grid": np.arange(mg + 1, dtype=np.int32)}))
    base = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame(columns=["stay_id", "grid"])

    base = base.merge(
        icu[["stay_id", "subject_id", "hadm_id", "intime", "outtime", "max_grid"]],
        on="stay_id",
        how="left"
    ).rename(columns={"intime": "baseline_time"})

    base = base.sort_values(["stay_id", "grid"]).reset_index(drop=True)
    base["row_id"] = np.arange(len(base), dtype=np.int64)

    _dbg(cfg, "base_grid built", rows=len(base), stays=int(base["stay_id"].nunique()))
    return base


# -------------------------
# radiology_with_exam_and_class load + enrich
# -------------------------
def load_radiology_with_class(cfg: CFG) -> pd.DataFrame:
    rad = pd.read_csv(cfg.radiology_with_class_path, low_memory=False)
    must = ["note_id", "text", "modality", "region"]
    for c in must:
        if c not in rad.columns:
            raise ValueError(f"radiology_with_exam_and_class missing required column: {c}")

    rad = rad.copy()
    rad["note_id"] = _fix_note_id_series(rad["note_id"])
    rad["text"] = rad["text"].astype(str).fillna("")
    rad["modality"] = rad["modality"].astype(str).fillna("Unknown").map(_sanitize_cat)
    rad["region"] = rad["region"].astype(str).fillna("Unknown").map(_sanitize_cat)

    if "exam" not in rad.columns:
        if "exam_name" in rad.columns:
            rad["exam"] = rad["exam_name"].astype(str).fillna("")
        else:
            rad["exam"] = ""
    rad["exam"] = rad["exam"].astype(str).fillna("").map(_sanitize_cat)

    if "subject_id" in rad.columns:
        rad["subject_id"] = _to_key_float(rad["subject_id"])
    if "hadm_id" in rad.columns:
        rad["hadm_id"] = _to_key_float(rad["hadm_id"])
    if "note_time" in rad.columns:
        rad["note_time"] = _safe_dt(rad["note_time"])

    return rad


def enrich_rad_with_radiology_csv_if_needed(cfg: CFG, rad: pd.DataFrame) -> pd.DataFrame:
    has_hadm = "hadm_id" in rad.columns and rad["hadm_id"].notna().any()
    has_time = "note_time" in rad.columns and rad["note_time"].notna().any()
    if has_hadm and has_time:
        out = rad.copy()
        out["hadm_id"] = _to_key_float(out["hadm_id"])
        out["note_time"] = _safe_dt(out["note_time"])
        return out

    need_ids = set(rad["note_id"].astype(str).unique().tolist())
    rx_parts = []
    for chunk in pd.read_csv(cfg.radiology_csv_path, low_memory=False, chunksize=500_000):
        if "note_id" not in chunk.columns:
            continue
        chunk = chunk.copy()
        chunk["note_id"] = _fix_note_id_series(chunk["note_id"])
        chunk = chunk[chunk["note_id"].isin(need_ids)]
        if len(chunk):
            rx_parts.append(chunk)

    if not rx_parts:
        raise RuntimeError("Could not find matching note_id in radiology.csv. Check paths/version.")

    rx = pd.concat(rx_parts, axis=0, ignore_index=True)

    need = ["note_id", "subject_id", "hadm_id"]
    for c in need:
        if c not in rx.columns:
            raise ValueError(f"radiology.csv missing required column: {c}")

    rx["subject_id"] = _to_key_float(rx["subject_id"])
    rx["hadm_id"] = _to_key_float(rx["hadm_id"])
    if "charttime" in rx.columns:
        rx["note_time"] = _safe_dt(rx["charttime"])
    elif "storetime" in rx.columns:
        rx["note_time"] = _safe_dt(rx["storetime"])
    else:
        rx["note_time"] = pd.NaT

    rx = rx[["note_id", "subject_id", "hadm_id", "note_time"]].drop_duplicates("note_id")

    out = rad.merge(rx, on="note_id", how="left", suffixes=("", "_rx"))
    if "subject_id" not in out.columns:
        out["subject_id"] = out["subject_id_rx"]
    else:
        out["subject_id"] = out["subject_id"].fillna(out["subject_id_rx"])

    out = out.drop(columns=[c for c in out.columns if c.endswith("_rx")], errors="ignore")
    out["subject_id"] = _to_key_float(out["subject_id"])
    out["hadm_id"] = _to_key_float(out["hadm_id"])
    out["note_time"] = _safe_dt(out["note_time"])
    return out


def map_notes_to_stay_grid(cfg: CFG, base: pd.DataFrame, rad: pd.DataFrame) -> pd.DataFrame:
    stay_tbl = base[["stay_id", "subject_id", "hadm_id", "baseline_time", "outtime", "max_grid"]].drop_duplicates("stay_id").copy()
    stay_tbl["subject_id"] = _to_key_float(stay_tbl["subject_id"])
    stay_tbl["hadm_id"] = _to_key_float(stay_tbl["hadm_id"])
    stay_tbl["baseline_time"] = _safe_dt(stay_tbl["baseline_time"])
    stay_tbl["outtime"] = _safe_dt(stay_tbl["outtime"])

    rad2 = rad.copy()
    if "subject_id" not in rad2.columns:
        rad2["subject_id"] = np.nan
    rad2["subject_id"] = _to_key_float(rad2["subject_id"])
    rad2["hadm_id"] = _to_key_float(rad2.get("hadm_id", pd.Series([np.nan] * len(rad2))))
    rad2["note_time"] = _safe_dt(rad2.get("note_time"))
    rad2 = rad2.dropna(subset=["subject_id", "hadm_id", "note_time"]).copy()

    cand = rad2.merge(stay_tbl, on=["subject_id", "hadm_id"], how="inner")
    cand = cand[(cand["note_time"] >= cand["baseline_time"]) & (cand["note_time"] <= cand["outtime"])].copy()
    if len(cand) == 0:
        _dbg(cfg, "note->stay empty after time filter")
        return pd.DataFrame(columns=["row_id", "stay_id", "grid", "note_id", "text", "modality", "region", "exam"])

    cand["_dt_from_start"] = (cand["note_time"] - cand["baseline_time"]).abs()
    cand = cand.sort_values(["note_id", "_dt_from_start"]).drop_duplicates(subset=["note_id"], keep="first").copy()

    delta_h = (cand["note_time"] - cand["baseline_time"]).dt.total_seconds() / 3600.0
    cand["grid"] = np.floor(delta_h / float(cfg.grid_hours)).astype(np.int32)
    cand["grid"] = np.clip(cand["grid"].values, 0, cand["max_grid"].values.astype(np.int32))

    key = base[["stay_id", "grid", "row_id"]].copy()
    cand = cand.merge(key, on=["stay_id", "grid"], how="inner")

    out = cand[["row_id", "stay_id", "grid", "note_id", "text", "modality", "region", "exam"]].copy()
    out["note_id"] = _fix_note_id_series(out["note_id"])
    out["text"] = out["text"].astype(str).fillna("").str.slice(0, int(cfg.text_max_chars))
    out["modality"] = out["modality"].astype(str).fillna("Unknown").map(_sanitize_cat)
    out["region"] = out["region"].astype(str).fillna("Unknown").map(_sanitize_cat)
    out["exam"] = out["exam"].astype(str).fillna("").map(_sanitize_cat)

    _dbg(cfg, "note->grid mapped", rows=len(out), notes=int(out["note_id"].nunique()), stays=int(out["stay_id"].nunique()))
    return out


# -------------------------
# Sharding helper (random but reproducible)
# -------------------------
def shard_note_tbl(note_tbl: pd.DataFrame, num_shards: int, shard_id: int, seed: int) -> pd.DataFrame:
    num_shards = int(num_shards)
    shard_id = int(shard_id)
    if num_shards <= 1:
        return note_tbl.reset_index(drop=True)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be in [0, {num_shards-1}], got {shard_id}")

    # 안정성을 위해 note_id 기준 정렬 후, seed로 permute
    tbl = note_tbl.sort_values("note_id").reset_index(drop=True)
    n = len(tbl)
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(n)
    take = perm[shard_id::num_shards]
    out = tbl.iloc[take].reset_index(drop=True)
    return out


# -------------------------
# Vocab / feature conversion
# -------------------------
def build_vocabs(rad_map: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    mods = sorted(set(rad_map["modality"].astype(str).tolist()))
    regs = sorted(set(rad_map["region"].astype(str).tolist()))
    mod2id = {"__UNK__": 0}
    reg2id = {"__UNK__": 0}
    for m in mods:
        if m not in mod2id:
            mod2id[m] = len(mod2id)
    for r in regs:
        if r not in reg2id:
            reg2id[r] = len(reg2id)
    return {"mod2id": mod2id, "reg2id": reg2id}


def json_to_struct_and_tokens(obj: Dict[str, Any], cfg: CFG) -> Tuple[Dict[str, float], List[str]]:
    critical = 1.0 if bool(obj.get("critical", False)) else 0.0
    abnormal = 1.0 if bool(obj.get("abnormal", False)) else 0.0

    def _clip_int(x, lo, hi, default=0):
        try:
            v = int(x)
        except Exception:
            v = default
        v = max(lo, min(hi, v))
        return float(v)

    severity = _clip_int(obj.get("severity", 0), 0, 3, default=0)
    uncertainty = _clip_int(obj.get("uncertainty", 0), 0, 3, default=0)

    kws = obj.get("keywords") or []
    diags = obj.get("diagnoses") or []
    anat = obj.get("anatomy") or []
    procs = obj.get("procedures") or []
    neg = obj.get("negated") or []

    def _as_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return x
        if isinstance(x, str):
            return [x]
        return []

    kws = _as_list(kws)
    diags = _as_list(diags)
    anat = _as_list(anat)
    procs = _as_list(procs)
    neg = _as_list(neg)

    toks: List[str] = []
    for bucket in (kws, diags, anat, procs):
        for s in bucket:
            toks.extend(_tokenize_simple(s, max_tokens=cfg.token_max_per_note))
    for s in neg:
        for t in _tokenize_simple(s, max_tokens=cfg.token_max_per_note):
            toks.append("neg_" + t)

    seen = set()
    toks2: List[str] = []
    for t in toks:
        if not t:
            continue
        if t not in seen:
            toks2.append(t)
            seen.add(t)
    if cfg.token_max_per_note and cfg.token_max_per_note > 0:
        toks2 = toks2[: int(cfg.token_max_per_note)]

    feats = {
        "txt__has_note": 1.0,
        "txt__critical": critical,
        "txt__abnormal": abnormal,
        "txt__severity": severity,
        "txt__uncertainty": uncertainty,
        "txt__n_kw": float(len(kws)),
        "txt__n_diag": float(len(diags)),
        "txt__n_anat": float(len(anat)),
        "txt__n_proc": float(len(procs)),
        "txt__n_neg": float(len(neg)),
        "txt__n_toks": float(len(toks2)),
    }
    return feats, toks2


def build_token_vocab(all_token_lists: List[List[str]], cfg: CFG) -> Dict[str, int]:
    from collections import Counter
    ctr = Counter()
    for toks in all_token_lists:
        ctr.update(toks)
    items = [(t, c) for t, c in ctr.items() if c >= int(cfg.token_min_freq)]
    items.sort(key=lambda x: (-x[1], x[0]))

    tok2id = {"__UNK__": 0}
    max_keep = max(0, int(cfg.token_max_vocab) - 1)
    for t, _ in items[:max_keep]:
        if t not in tok2id:
            tok2id[t] = len(tok2id)
    return tok2id


# -------------------------
# Main
# -------------------------
def main(cfg: CFG):
    _safe_makedirs(cfg.out_dir)
    _safe_makedirs(cfg.cache_dir)

    paths = resolve_paths(cfg)
    base = load_icustays_base_grid(cfg, paths["icustays"])

    rad = load_radiology_with_class(cfg)
    rad = enrich_rad_with_radiology_csv_if_needed(cfg, rad)
    rad_map = map_notes_to_stay_grid(cfg, base, rad)

    if cfg.max_notes and int(cfg.max_notes) > 0 and len(rad_map):
        note_ids = rad_map["note_id"].astype(str).unique().tolist()[: int(cfg.max_notes)]
        rad_map = rad_map[rad_map["note_id"].astype(str).isin(note_ids)].copy()
        _dbg(cfg, "max_notes applied", max_notes=int(cfg.max_notes), mapped_rows=len(rad_map))

    # note_tbl: note_id별 1회만 호출
    note_tbl_cols = ["note_id", "text", "modality", "region", "exam"]
    note_tbl_all = rad_map[note_tbl_cols].drop_duplicates("note_id").copy()
    note_tbl_all["note_id"] = note_tbl_all["note_id"].astype(str)

    # sharding
    note_tbl = shard_note_tbl(note_tbl_all, cfg.num_shards, cfg.shard_id, cfg.shard_seed)

    _dbg(cfg, "LLM extraction start (sharded)",
         notes_total=len(note_tbl_all), notes_this_shard=len(note_tbl),
         shard_id=cfg.shard_id, num_shards=cfg.num_shards, shard_seed=cfg.shard_seed,
         cache_only=bool(cfg.cache_only),
         backend=cfg.llm_backend, ollama_url=cfg.ollama_url, openai_base_url=cfg.openai_base_url)

    rows = list(note_tbl.itertuples(index=False))
    n_total = len(rows)
    max_workers = max(1, int(getattr(cfg, "llm_workers", 1) or 1))

    def _maybe_log(done_cnt: int):
        if cfg.debug and (done_cnt % 500 == 0 or done_cnt == n_total):
            print(f"[LLM] {done_cnt}/{n_total} done")

    # cache_only면 결과를 메모리에 안 쌓고, 캐시만 채운다.
    if cfg.cache_only:
        def _call_only(r):
            _ = llm_extract_json(
                backend=cfg.llm_backend,
                text=r.text,
                modality=str(r.modality),
                region=str(r.region),
                exam=str(getattr(r, "exam", "")),
                cfg=cfg,
            )
            return 1

        if max_workers == 1:
            for i, r in enumerate(rows, start=1):
                _call_only(r)
                _maybe_log(i)
        else:
            inflight = max_workers * 2
            it = iter(rows)
            futs = {}
            done_cnt = 0
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for _ in range(inflight):
                    r = next(it, None)
                    if r is None:
                        break
                    futs[ex.submit(_call_only, r)] = 1
                while futs:
                    done_set, _ = wait(list(futs.keys()), return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        try:
                            _ = fut.result()
                        except Exception as e:
                            if cfg.debug:
                                print(f"[WARN] worker failed: {type(e).__name__}: {e}")
                        done_cnt += 1
                        futs.pop(fut, None)
                        _maybe_log(done_cnt)
                        r = next(it, None)
                        if r is not None:
                            futs[ex.submit(_call_only, r)] = 1

        print("\n✅ DONE (cache_only shard)")
        print(f" - cache_dir: {cfg.cache_dir}")
        print(f" - shard: {cfg.shard_id}/{cfg.num_shards} (seed={cfg.shard_seed})")
        print(f" - notes processed (this shard): {n_total}")
        return

    # --------- full build below (uses cache; should be fast after cache_only stage) ---------
    out = base[["stay_id", "subject_id", "hadm_id", "baseline_time", "outtime", "max_grid", "grid", "row_id"]].copy()
    out = out.sort_values(["stay_id", "grid"]).reset_index(drop=True)

    if len(rad_map) == 0:
        _dbg(cfg, "no mapped notes; writing empty outputs")
        feat_cols = [
            "txt__has_note", "txt__critical", "txt__abnormal", "txt__severity", "txt__uncertainty",
            "txt__n_kw", "txt__n_diag", "txt__n_anat", "txt__n_proc", "txt__n_neg", "txt__n_toks",
            "txt__n_notes",
        ]
        for c in feat_cols:
            out[c] = 0.0
        out["mask_text"] = 0
        out["delta_text_hours"] = _delta_from_mask_and_grid_per_stay(out, "mask_text", "grid", "stay_id", cfg.grid_hours)

        n_rows = int(out["row_id"].max()) + 1 if len(out) else 0
        mod2id, reg2id, tok2id = {"__UNK__": 0}, {"__UNK__": 0}, {"__UNK__": 0}

        mod_ids, mod_offsets = build_embeddingbag_inputs(np.zeros((0,), np.int64), np.zeros((0,), np.int32), n_rows, 0)
        reg_ids, reg_offsets = build_embeddingbag_inputs(np.zeros((0,), np.int64), np.zeros((0,), np.int32), n_rows, 0)
        tok_ids, tok_offsets = build_embeddingbag_inputs(np.zeros((0,), np.int64), np.zeros((0,), np.int32), n_rows, 0)

        npz_path = os.path.join(cfg.out_dir, "meta_bags.npz")
        np.savez_compressed(
            npz_path,
            row_id=out["row_id"].to_numpy(np.int64, copy=False),
            mod_ids=mod_ids.astype(np.int32, copy=False),
            mod_offsets=mod_offsets.astype(np.int64, copy=False),
            reg_ids=reg_ids.astype(np.int32, copy=False),
            reg_offsets=reg_offsets.astype(np.int64, copy=False),
            tok_ids=tok_ids.astype(np.int32, copy=False),
            tok_offsets=tok_offsets.astype(np.int64, copy=False),
            n_mod=np.int64(len(mod2id)),
            n_reg=np.int64(len(reg2id)),
            n_tok=np.int64(len(tok2id)),
        )

        out_csv_path = os.path.join(cfg.out_dir, cfg.out_csv)
        out.to_csv(out_csv_path, index=False)

        meta = {
            "builder": "build_text_llm_structured_tokens_no_locf_optionB_sharded",
            "cfg": asdict(cfg),
            "vocab": {"mod2id": mod2id, "reg2id": reg2id, "tok2id_size": int(len(tok2id))},
            "paths": {"out_csv": out_csv_path, "meta_bags_npz": npz_path},
            "notes": {"n_note_ids_used": 0, "n_mapped_rows": 0, "n_grid_rows": int(len(out))},
            "bags": {"names": ["mod", "reg", "tok"], "npz_keys": ["mod_ids", "mod_offsets", "reg_ids", "reg_offsets", "tok_ids", "tok_offsets", "row_id"]},
        }
        meta_path = os.path.join(cfg.out_dir, cfg.out_meta)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print("\n✅ DONE (no notes)")
        print(f" - csv : {out_csv_path}")
        print(f" - meta: {meta_path}")
        print(f" - npz : {npz_path}")
        return

    voc = build_vocabs(rad_map)
    mod2id, reg2id = voc["mod2id"], voc["reg2id"]

    # ---------- note-level: cache -> feats/tokens ----------
    note_feats: Dict[str, Dict[str, float]] = {}
    note_tokens: Dict[str, List[str]] = {}

    rows_all = list(note_tbl_all.itertuples(index=False))
    n_total2 = len(rows_all)
    _dbg(cfg, "LLM features build (should hit cache)", notes=n_total2)

    def _proc_one_full(r):
        nid = str(r.note_id)
        obj = llm_extract_json(
            backend=cfg.llm_backend,
            text=r.text,
            modality=str(r.modality),
            region=str(r.region),
            exam=str(getattr(r, "exam", "")),
            cfg=cfg,
        )
        feats, toks = json_to_struct_and_tokens(obj, cfg)
        return nid, feats, toks

    def _maybe_log2(done_cnt: int):
        if cfg.debug and (done_cnt % 500 == 0 or done_cnt == n_total2):
            print(f"[LLM] {done_cnt}/{n_total2} done")

    if max_workers == 1:
        for i, r in enumerate(rows_all, start=1):
            nid, feats, toks = _proc_one_full(r)
            note_feats[nid] = feats
            note_tokens[nid] = toks
            _maybe_log2(i)
    else:
        inflight = max_workers * 2
        it = iter(rows_all)
        futs = {}
        done_cnt = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for _ in range(inflight):
                r = next(it, None)
                if r is None:
                    break
                futs[ex.submit(_proc_one_full, r)] = 1
            while futs:
                done_set, _ = wait(list(futs.keys()), return_when=FIRST_COMPLETED)
                for fut in done_set:
                    try:
                        nid, feats, toks = fut.result()
                        note_feats[nid] = feats
                        note_tokens[nid] = toks
                    except Exception as e:
                        if cfg.debug:
                            print(f"[WARN] worker failed: {type(e).__name__}: {e}")
                    done_cnt += 1
                    futs.pop(fut, None)
                    _maybe_log2(done_cnt)
                    r = next(it, None)
                    if r is not None:
                        futs[ex.submit(_proc_one_full, r)] = 1

    all_toks = [note_tokens.get(str(nid), []) for nid in note_tbl_all["note_id"].tolist()]
    tok2id = build_token_vocab(all_toks, cfg)
    _dbg(cfg, "token vocab built", vocab_size=len(tok2id), min_freq=cfg.token_min_freq, max_vocab=cfg.token_max_vocab)

    # ---------- per-grid aggregate numeric features ----------
    feat_cols = [
        "txt__has_note", "txt__critical", "txt__abnormal", "txt__severity", "txt__uncertainty",
        "txt__n_kw", "txt__n_diag", "txt__n_anat", "txt__n_proc", "txt__n_neg", "txt__n_toks",
        "txt__n_notes",
    ]
    for c in feat_cols:
        out[c] = 0.0

    rm = rad_map[["row_id", "note_id"]].copy()
    rm["note_id"] = rm["note_id"].astype(str)

    nf_rows = []
    for nid, feats in note_feats.items():
        d = {"note_id": nid}
        d.update(feats)
        nf_rows.append(d)
    nf = pd.DataFrame(nf_rows) if nf_rows else pd.DataFrame(columns=["note_id"])

    rm = rm.merge(nf, on="note_id", how="left")

    for c in feat_cols:
        if c == "txt__n_notes":
            continue
        rm[c] = pd.to_numeric(rm.get(c, 0.0), errors="coerce").fillna(0.0).astype(np.float32)

    grp = rm.groupby("row_id", as_index=False)
    agg = grp[[
        "txt__has_note", "txt__critical", "txt__abnormal", "txt__severity", "txt__uncertainty",
        "txt__n_kw", "txt__n_diag", "txt__n_anat", "txt__n_proc", "txt__n_neg", "txt__n_toks",
    ]].mean()
    nn = grp.size()

    if isinstance(nn, pd.Series):
        nn = nn.reset_index(name="txt__n_notes")
    else:
        # groupby(as_index=False).size() -> DataFrame with column "size"
        nn = nn.rename(columns={"size": "txt__n_notes"})
    agg = agg.merge(nn, on="row_id", how="left")

    out = out.merge(agg, on="row_id", how="left", suffixes=("", "_agg"))
    for c in feat_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(np.float32)

    out["mask_text"] = (out["txt__n_notes"].to_numpy() > 0).astype(np.int8)
    out = out.sort_values(["stay_id", "grid"]).reset_index(drop=True)
    out["delta_text_hours"] = _delta_from_mask_and_grid_per_stay(out, "mask_text", "grid", "stay_id", cfg.grid_hours)

    # ---------- meta_bags ----------
    rm2 = rad_map[["row_id", "note_id", "modality", "region"]].copy()
    rm2["mod_id"] = rm2["modality"].map(lambda x: mod2id.get(str(x), 0)).astype(np.int32)
    rm2["reg_id"] = rm2["region"].map(lambda x: reg2id.get(str(x), 0)).astype(np.int32)

    n_rows = int(out["row_id"].max()) + 1 if len(out) else 0

    mod_ids, mod_offsets = build_embeddingbag_inputs(
        row_ids=rm2["row_id"].to_numpy(np.int64, copy=False),
        item_ids=rm2["mod_id"].to_numpy(np.int32, copy=False),
        n_rows=n_rows,
        unknown_id=0,
    )
    reg_ids, reg_offsets = build_embeddingbag_inputs(
        row_ids=rm2["row_id"].to_numpy(np.int64, copy=False),
        item_ids=rm2["reg_id"].to_numpy(np.int32, copy=False),
        n_rows=n_rows,
        unknown_id=0,
    )

    tok_pairs_r: List[int] = []
    tok_pairs_t: List[int] = []
    for rr in rad_map[["row_id", "note_id"]].itertuples(index=False):
        rid = int(rr.row_id)
        nid = str(rr.note_id)
        toks = note_tokens.get(nid, [])
        if not toks:
            continue
        for t in toks:
            tok_pairs_r.append(rid)
            tok_pairs_t.append(tok2id.get(t, 0))

    tok_row = np.asarray(tok_pairs_r, dtype=np.int64) if tok_pairs_r else np.zeros((0,), dtype=np.int64)
    tok_id = np.asarray(tok_pairs_t, dtype=np.int32) if tok_pairs_t else np.zeros((0,), dtype=np.int32)

    tok_ids, tok_offsets = build_embeddingbag_inputs(
        row_ids=tok_row,
        item_ids=tok_id,
        n_rows=n_rows,
        unknown_id=0,
    )

    npz_path = os.path.join(cfg.out_dir, "meta_bags.npz")
    np.savez_compressed(
        npz_path,
        row_id=out["row_id"].to_numpy(np.int64, copy=False),
        mod_ids=mod_ids.astype(np.int32, copy=False),
        mod_offsets=mod_offsets.astype(np.int64, copy=False),
        reg_ids=reg_ids.astype(np.int32, copy=False),
        reg_offsets=reg_offsets.astype(np.int64, copy=False),
        tok_ids=tok_ids.astype(np.int32, copy=False),
        tok_offsets=tok_offsets.astype(np.int64, copy=False),
        n_mod=np.int64(len(mod2id)),
        n_reg=np.int64(len(reg2id)),
        n_tok=np.int64(len(tok2id)),
    )

    out_csv_path = os.path.join(cfg.out_dir, cfg.out_csv)
    out.to_csv(out_csv_path, index=False)

    meta = {
        "builder": "build_text_llm_structured_tokens_no_locf_optionB_sharded",
        "cfg": asdict(cfg),
        "vocab": {
            "mod2id": mod2id,
            "reg2id": reg2id,
            "tok2id_size": int(len(tok2id)),
        },
        "paths": {
            "out_csv": out_csv_path,
            "meta_bags_npz": npz_path,
        },
        "notes": {
            "n_note_ids_used": int(len(note_tbl_all)),
            "n_mapped_rows": int(len(rad_map)),
            "n_grid_rows": int(len(out)),
        },
        "bags": {
            "names": ["mod", "reg", "tok"],
            "npz_keys": ["mod_ids", "mod_offsets", "reg_ids", "reg_offsets", "tok_ids", "tok_offsets", "row_id"],
        },
    }
    meta_path = os.path.join(cfg.out_dir, cfg.out_meta)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n✅ DONE")
    print(f" - csv : {out_csv_path}")
    print(f" - meta: {meta_path}")
    print(f" - npz : {npz_path}")
    print(f" - vocab sizes: mod={len(mod2id)} reg={len(reg2id)} tok={len(tok2id)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--grid_hours", type=int, default=CFG.grid_hours)
    p.add_argument("--mimic_iv_dir", type=str, default=CFG.mimic_iv_dir)
    p.add_argument("--icustays_path", type=str, default=CFG.icustays_path)
    p.add_argument("--max_stay_hours", type=int, default=CFG.max_stay_hours)

    p.add_argument("--radiology_with_class_path", type=str, default=CFG.radiology_with_class_path)
    p.add_argument("--radiology_csv_path", type=str, default=CFG.radiology_csv_path)
    p.add_argument("--text_max_chars", type=int, default=CFG.text_max_chars)

    p.add_argument("--llm_backend", type=str, default=CFG.llm_backend, choices=["ollama", "openai_compat"])
    p.add_argument("--llm_temperature", type=float, default=CFG.llm_temperature)
    p.add_argument("--llm_timeout", type=int, default=CFG.llm_timeout)
    p.add_argument("--llm_sleep_ms", type=int, default=CFG.llm_sleep_ms)
    p.add_argument("--llm_workers", type=int, default=CFG.llm_workers)

    p.add_argument("--ollama_url", type=str, default=CFG.ollama_url)
    p.add_argument("--ollama_model", type=str, default=CFG.ollama_model)

    p.add_argument("--openai_base_url", type=str, default=CFG.openai_base_url)
    p.add_argument("--openai_model", type=str, default=CFG.openai_model)
    p.add_argument("--openai_api_key", type=str, default=CFG.openai_api_key)

    p.add_argument("--schema_version", type=str, default=CFG.schema_version)

    p.add_argument("--token_min_freq", type=int, default=CFG.token_min_freq)
    p.add_argument("--token_max_vocab", type=int, default=CFG.token_max_vocab)
    p.add_argument("--token_max_per_note", type=int, default=CFG.token_max_per_note)

    p.add_argument("--cache_dir", type=str, default=CFG.cache_dir)
    p.add_argument("--use_cache", action="store_true", default=CFG.use_cache)
    p.add_argument("--no_cache", action="store_false", dest="use_cache")

    p.add_argument("--out_dir", type=str, default=CFG.out_dir)
    p.add_argument("--out_csv", type=str, default=CFG.out_csv)
    p.add_argument("--out_meta", type=str, default=CFG.out_meta)

    p.add_argument("--max_notes", type=int, default=CFG.max_notes)
    p.add_argument("--debug", action="store_true", default=CFG.debug)
    p.add_argument("--no_debug", action="store_false", dest="debug")

    # ✅ sharding args
    p.add_argument("--num_shards", type=int, default=CFG.num_shards)
    p.add_argument("--shard_id", type=int, default=CFG.shard_id)
    p.add_argument("--shard_seed", type=int, default=CFG.shard_seed)
    p.add_argument("--cache_only", action="store_true", default=CFG.cache_only)

    args = p.parse_args()
    cfg = CFG(**vars(args))
    main(cfg)
