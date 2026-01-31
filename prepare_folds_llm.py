# prepare_folds_llm_firststay.py
# ============================================================
# Prepared-folds builder (model_data + LLM token meta-bags)
#
# ✅ Filters BEFORE split:
#   (0) keep ONLY the first ICU stay per subject_id (drop subsequent stays/admissions)
#   (1) drop stays with ICU LOS <= --min_icu_los_hours
#   (2) drop stays with "NO REAL TOKENS" in text
#
# IMPORTANT:
# - EmbeddingBag inputs are NEVER left empty:
#   empty rows are filled with __UNK__ token id=0 (and offsets advance by 1).
#   Therefore, "tok_offsets[i+1]-tok_offsets[i] > 0" is NOT a valid "has note" criterion.
#
# - Here we define "real text exists" as:
#       any token id != 0 within that row (non-UNK token)
#   and "stay has note" if ANY row in that stay has non-UNK token.
#
# Requires:
#   - text CSV has columns: stay_id, row_id (+ usual keys)
#   - meta_bags.npz has arrays: tok_ids, tok_offsets, row_id
# ============================================================

from __future__ import annotations

import os, json, shutil
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Set, List

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except Exception as e:
    raise RuntimeError("scikit-learn required: pip install scikit-learn") from e


@dataclass
class CFG:
    model_data_csv: str = "model_data_mimic4.csv"
    text_data_csv: str = "artifacts/note_llm_struct_tokens_1h/text_llm_struct_tokens.csv"
    text_meta_json: str = "artifacts/note_llm_struct_tokens_1h/meta_text_llm_struct_tokens.json"
    text_bags_npz: str = "artifacts/note_llm_struct_tokens_1h/meta_bags.npz"

    out_dir: str = "artifacts/folds_modeldata_plus_llmtext"
    folds: int = 5
    seed: int = 123
    val_ratio_in_pool: float = 0.2
    save_parquet: bool = True
    reset_out_dir: bool = False

    # merge keys
    id_cols: Tuple[str, ...] = ("stay_id", "subject_id", "hadm_id", "grid")
    # label / mask
    y_col: str = "y_death_nextgrid"
    y_mask_col: str = "mask_death_valid_aligned"

    # filters
    keep_first_stay_per_subject: bool = True   # ✅ NEW: subject-level de-dup (keep earliest stay only)
    min_icu_los_hours: float = 4.0
    drop_no_note_stays: bool = True

    # LOS source columns (any that exists will be used; else baseline_time/outtime)
    los_cols: Tuple[str, ...] = ("icu_los_hours", "los_hours", "icu_los_hr", "los_hr")

    # "first stay" time columns (first existing one will be used)
    # NOTE: baseline_time is often grid-start, but is still monotone and OK as a proxy for stay start.
    firststay_time_cols: Tuple[str, ...] = ("icu_intime", "intime", "admittime", "baseline_time")

    # non-UNK criterion
    unk_id: int = 0  # __UNK__
    debug: bool = True


def _safe_makedirs(p: str):
    os.makedirs(p, exist_ok=True)

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)

def _assert_unique_cols(df: pd.DataFrame, where: str):
    if not df.columns.is_unique:
        dup = df.columns[df.columns.duplicated()].tolist()
        raise RuntimeError(f"[{where}] duplicate column names: {dup[:50]} ...")

def _cast_int64_nullable(df: pd.DataFrame, cols: List[str], where: str, debug: bool) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if debug:
        bad = [c for c in cols if c in df.columns and df[c].isna().any()]
        if bad:
            print(f"[WARN] {where}: key(s) have NA after cast: {bad}")
    return df

def _compute_los_hours(df: pd.DataFrame, cfg: CFG) -> Optional[pd.Series]:
    for c in cfg.los_cols:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    if "baseline_time" in df.columns and "outtime" in df.columns:
        bt = pd.to_datetime(df["baseline_time"], errors="coerce")
        ot = pd.to_datetime(df["outtime"], errors="coerce")
        return (ot - bt).dt.total_seconds() / 3600.0
    return None

def _pick_first_existing_col(df: pd.DataFrame, cols: Tuple[str, ...]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None

def _keep_first_stay_per_subject(df_m: pd.DataFrame, cfg: CFG) -> Set[int]:
    """
    Keep only the earliest stay_id per subject_id.

    - If a start time column exists (cfg.firststay_time_cols), we use:
        stay_start = min(time_col) within stay
      and pick the stay with min(stay_start) per subject.
    - Otherwise, fallback to min(stay_id) per subject.

    Returns a set of stay_id (int).
    """
    if "subject_id" not in df_m.columns or "stay_id" not in df_m.columns:
        raise RuntimeError("keep_first_stay_per_subject requires both 'subject_id' and 'stay_id' columns.")

    time_col = _pick_first_existing_col(df_m, cfg.firststay_time_cols)

    base_cols = ["subject_id", "stay_id"] + ([time_col] if time_col else [])
    base = df_m[base_cols].copy()
    base = base.dropna(subset=["subject_id", "stay_id"])
    base["subject_id"] = base["subject_id"].astype("int64")
    base["stay_id"] = base["stay_id"].astype("int64")

    if time_col:
        base["_t"] = pd.to_datetime(base[time_col], errors="coerce")
        stay_start = base.groupby(["subject_id", "stay_id"], as_index=False)["_t"].min()

        # Some stays may have all-NaT; handle by setting far-future and relying on stay_id tie-break.
        stay_start["_is_nat"] = stay_start["_t"].isna().astype(np.int8)
        stay_start["_t_fill"] = stay_start["_t"].where(~stay_start["_t"].isna(), pd.Timestamp.max)

        stay_start = stay_start.sort_values(
            ["subject_id", "_is_nat", "_t_fill", "stay_id"],
            ascending=[True, True, True, True],
        )
        first = stay_start.groupby("subject_id", as_index=False).first()
        keep = set(first["stay_id"].astype("int64").tolist())

        if cfg.debug:
            n_subj = int(base["subject_id"].nunique())
            n_stays = int(base["stay_id"].nunique())
            print(f"[FILTER] first_stay_per_subject using time_col='{time_col}': subjects={n_subj:,} stays {n_stays:,} -> {len(keep):,}")
            nat_rate = float(stay_start["_t"].isna().mean())
            if nat_rate > 0:
                print(f"[WARN] first_stay: stay_start NaT rate={nat_rate:.6f} (fallback to stay_id for those)")
        return keep
    else:
        # fallback: choose minimal stay_id per subject
        first_stay = base.groupby("subject_id")["stay_id"].min()
        keep = set(first_stay.astype("int64").tolist())
        if cfg.debug:
            print(f"[FILTER] first_stay_per_subject (no time column found): stays {int(base['stay_id'].nunique()):,} -> {len(keep):,}")
        return keep


def _row_has_nonunk_from_npz(npz_path: str, unk_id: int, debug: bool) -> pd.Series:
    """Returns Series indexed by row_id (int64) with value 1 if that row has any token != unk_id."""
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    for k in ("tok_ids", "tok_offsets", "row_id"):
        if k not in z.files:
            raise RuntimeError(f"meta_bags.npz missing '{k}'. found={list(z.files)[:20]}")

    tok_ids = z["tok_ids"].astype(np.int32, copy=False)
    off = z["tok_offsets"].astype(np.int64, copy=False)
    rid = z["row_id"].astype(np.int64, copy=False)

    n_rows = int(off.shape[0] - 1)
    if rid.shape[0] != n_rows:
        raise RuntimeError(f"row_id length mismatch: len(row_id)={rid.shape[0]} vs n_rows={n_rows}")

    # Efficient per-row "any non-UNK" using prefix sums over token stream
    nz = (tok_ids != np.int32(unk_id)).astype(np.int8, copy=False)
    pref = np.empty((nz.shape[0] + 1,), dtype=np.int64)
    pref[0] = 0
    np.cumsum(nz, out=pref[1:])

    s = off[:-1]
    e = off[1:]
    has = (pref[e] - pref[s]) > 0

    if debug:
        print(f"[NPZ] rows={n_rows:,} total_tokens={len(tok_ids):,} nonUNK_row_rate={float(has.mean()):.6f}")

    return pd.Series(has.astype(np.int8), index=pd.Index(rid, name="row_id"), name="has_nonunk")


def build_folds(cfg: CFG):
    if cfg.reset_out_dir and os.path.exists(cfg.out_dir):
        shutil.rmtree(cfg.out_dir)
    _safe_makedirs(cfg.out_dir)

    df_m = _read_csv(cfg.model_data_csv)
    df_t = _read_csv(cfg.text_data_csv)

    _assert_unique_cols(df_m, "model_data")
    _assert_unique_cols(df_t, "text_data")

    # key dtype alignment
    df_m = _cast_int64_nullable(df_m, list(cfg.id_cols) + ["row_id"], "model_data", cfg.debug)
    df_t = _cast_int64_nullable(df_t, list(cfg.id_cols) + ["row_id"], "text_data", cfg.debug)

    if cfg.debug:
        print(f"[LOAD] model_data rows={len(df_m):,} stays={df_m['stay_id'].nunique() if 'stay_id' in df_m.columns else 'NA'} cols={len(df_m.columns)}")
        print(f"[LOAD] text_data  rows={len(df_t):,} stays={df_t['stay_id'].nunique() if 'stay_id' in df_t.columns else 'NA'} cols={len(df_t.columns)}")

    # (0) Keep only first stay per subject (before other filters)
    if cfg.keep_first_stay_per_subject:
        keep_first = _keep_first_stay_per_subject(df_m, cfg)
        df_m = df_m[df_m["stay_id"].isin(list(keep_first))].copy()
        df_t = df_t[df_t["stay_id"].isin(list(keep_first))].copy()
        if cfg.debug:
            print(f"[KEEP] after first_stay filter: model_data stays={df_m['stay_id'].nunique()} rows={len(df_m):,} | text_data stays={df_t['stay_id'].nunique()} rows={len(df_t):,}")

    # (A) LOS filter from model_data (stay-level)
    keep_stays: Optional[Set[int]] = None
    if cfg.min_icu_los_hours and cfg.min_icu_los_hours > 0:
        los = _compute_los_hours(df_m, cfg)
        if los is None:
            if cfg.debug:
                print("[WARN] cannot compute LOS hours; skipping LOS filter.")
        else:
            stay_los = los.groupby(df_m["stay_id"]).max()
            keep_los = set(stay_los[stay_los > float(cfg.min_icu_los_hours)].index.astype(int).tolist())
            if cfg.debug:
                n0 = int(df_m["stay_id"].nunique())
                print(f"[FILTER] min_icu_los_hours>{cfg.min_icu_los_hours}: stays {n0} -> {len(keep_los)} (dropped {n0-len(keep_los)})")
            keep_stays = keep_los if keep_stays is None else (keep_stays & keep_los)

    # (B) note filter using NON-UNK tokens (stay-level)
    if cfg.drop_no_note_stays:
        if "row_id" not in df_t.columns:
            raise RuntimeError("text_data_csv must include 'row_id' to map meta_bags -> stays.")
        row_has = _row_has_nonunk_from_npz(cfg.text_bags_npz, cfg.unk_id, cfg.debug)

        tmp = df_t[["stay_id", "row_id"]].copy()
        tmp["has_nonunk"] = tmp["row_id"].map(row_has).fillna(0).astype(np.int8)
        stay_has = tmp.groupby("stay_id")["has_nonunk"].max()
        keep_note = set(stay_has[stay_has > 0].index.astype(int).tolist())

        if cfg.debug:
            print(f"[FILTER] stays_with_any_nonUNK={len(keep_note)} / {int(df_t['stay_id'].nunique())}")

        keep_stays = keep_note if keep_stays is None else (keep_stays & keep_note)

    if keep_stays is not None:
        df_m = df_m[df_m["stay_id"].isin(list(keep_stays))].copy()
        df_t = df_t[df_t["stay_id"].isin(list(keep_stays))].copy()
        if cfg.debug:
            print(f"[KEEP] after stay-filters: model_data stays={df_m['stay_id'].nunique()} rows={len(df_m):,} | text_data stays={df_t['stay_id'].nunique()} rows={len(df_t):,}")

    # merge (drop exact overlaps from text except keys)
    keys = list(cfg.id_cols)
    overlap = [c for c in df_t.columns if (c in df_m.columns and c not in keys)]
    if overlap and cfg.debug:
        print(f"[WARN] exact overlap: dropping from text before merge: {overlap[:20]}{' ...' if len(overlap)>20 else ''}")
    if overlap:
        df_t = df_t.drop(columns=overlap, errors="ignore")

    df = df_m.merge(df_t, on=keys, how="left", validate="m:1")
    _assert_unique_cols(df, "merged")

    if cfg.debug:
        probe = "row_id" if "row_id" in df.columns else None
        if probe is not None:
            print(f"[DBG] merge_hit_rate({probe})={float(pd.Series(df[probe]).notna().mean()):.4f}")

    # stay-level label
    if cfg.y_col not in df.columns:
        raise RuntimeError(f"Missing y_col='{cfg.y_col}' in merged df.")

    tmp = df[["stay_id", cfg.y_col] + ([cfg.y_mask_col] if cfg.y_mask_col in df.columns else [])].copy()
    tmp[cfg.y_col] = pd.to_numeric(tmp[cfg.y_col], errors="coerce").fillna(0.0)

    if cfg.y_mask_col in tmp.columns:
        m = pd.to_numeric(tmp[cfg.y_mask_col], errors="coerce").fillna(0).astype(int) > 0
        tmp = tmp[m].copy()

    stay_y = tmp.groupby("stay_id")[cfg.y_col].max().reset_index()
    stay_y["y_stay"] = (stay_y[cfg.y_col] >= 0.5).astype(int)
    stay_ids = stay_y["stay_id"].to_numpy(np.int64)
    y_stay = stay_y["y_stay"].to_numpy(np.int64)

    if len(stay_ids) == 0:
        raise RuntimeError("No stays left after filtering/masking.")

    skf = StratifiedKFold(n_splits=int(cfg.folds), shuffle=True, random_state=int(cfg.seed))

    # copy meta assets
    if cfg.text_meta_json and os.path.exists(cfg.text_meta_json):
        shutil.copy2(cfg.text_meta_json, os.path.join(cfg.out_dir, os.path.basename(cfg.text_meta_json)))
    if cfg.text_bags_npz and os.path.exists(cfg.text_bags_npz):
        shutil.copy2(cfg.text_bags_npz, os.path.join(cfg.out_dir, "meta_bags.npz"))

    fold_summ = []
    for k, (tr_idx, te_idx) in enumerate(skf.split(stay_ids, y_stay)):
        fold_dir = os.path.join(cfg.out_dir, f"fold_{k}")
        _safe_makedirs(fold_dir)

        stays_tr = stay_ids[tr_idx]
        stays_te = stay_ids[te_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(cfg.val_ratio_in_pool),
                                     random_state=int(cfg.seed) + 10 + k)
        tr2_idx, va2_idx = next(sss.split(stays_tr, y_stay[tr_idx]))
        stays_tr2 = stays_tr[tr2_idx]
        stays_va2 = stays_tr[va2_idx]

        df_tr = df[df["stay_id"].isin(stays_tr2)].copy()
        df_va = df[df["stay_id"].isin(stays_va2)].copy()
        df_te = df[df["stay_id"].isin(stays_te)].copy()

        if cfg.debug:
            print(f"\n[FOLD {k}] stays: train={len(stays_tr2)} val={len(stays_va2)} test={len(stays_te)} rows: tr={len(df_tr):,} va={len(df_va):,} te={len(df_te):,}")

        if cfg.save_parquet:
            df_tr.to_parquet(os.path.join(fold_dir, "train.parquet"), index=False)
            df_va.to_parquet(os.path.join(fold_dir, "val.parquet"), index=False)
            df_te.to_parquet(os.path.join(fold_dir, "test.parquet"), index=False)
            paths = {"train": "train.parquet", "val": "val.parquet", "test": "test.parquet"}
        else:
            df_tr.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
            df_va.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
            df_te.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
            paths = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}

        idset = set(cfg.id_cols)
        num_cols = [c for c in df.columns if c not in idset and pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in df.columns if c not in idset and (not pd.api.types.is_numeric_dtype(df[c]))]

        meta = {
            "paths": {**paths, "meta_bags_npz": "../meta_bags.npz"},
            "cols": {
                "id_cols": list(cfg.id_cols),
                "y_col": cfg.y_col,
                "y_mask_col": cfg.y_mask_col if cfg.y_mask_col in df.columns else "",
                "feature_numeric": num_cols,
                "feature_categorical": cat_cols,
            },
            "cfg": asdict(cfg),
        }
        with open(os.path.join(fold_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        fold_summ.append({"fold": k, "n_stays_train": int(len(stays_tr2)), "n_stays_val": int(len(stays_va2)), "n_stays_test": int(len(stays_te))})

    with open(os.path.join(cfg.out_dir, "meta_folds.json"), "w", encoding="utf-8") as f:
        json.dump({"cfg": asdict(cfg), "folds": fold_summ}, f, indent=2, ensure_ascii=False)

    print("\n✅ DONE:", cfg.out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model_data_csv", type=str, default=CFG.model_data_csv)
    p.add_argument("--text_data_csv", type=str, default=CFG.text_data_csv)
    p.add_argument("--text_meta_json", type=str, default=CFG.text_meta_json)
    p.add_argument("--text_bags_npz", type=str, default=CFG.text_bags_npz)

    p.add_argument("--out_dir", type=str, default=CFG.out_dir)
    p.add_argument("--folds", type=int, default=CFG.folds)
    p.add_argument("--seed", type=int, default=CFG.seed)
    p.add_argument("--val_ratio_in_pool", type=float, default=CFG.val_ratio_in_pool)

    p.add_argument("--save_parquet", action="store_true", default=CFG.save_parquet)
    p.add_argument("--save_csv", action="store_false", dest="save_parquet")
    p.add_argument("--reset_out_dir", action="store_true", default=CFG.reset_out_dir)

    p.add_argument("--min_icu_los_hours", type=float, default=CFG.min_icu_los_hours)
    p.add_argument("--keep_no_note_stays", action="store_false", dest="drop_no_note_stays")

    # NEW: subject de-dup toggle (default ON)
    p.add_argument("--keep_all_stays", action="store_false", dest="keep_first_stay_per_subject",
                   help="Disable subject-level de-dup; keep all stays (NOT recommended).")

    p.add_argument("--unk_id", type=int, default=CFG.unk_id)
    p.add_argument("--debug", action="store_true", default=CFG.debug)
    p.add_argument("--no_debug", action="store_false", dest="debug")

    args = p.parse_args()
    cfg = CFG(**vars(args))
    build_folds(cfg)

