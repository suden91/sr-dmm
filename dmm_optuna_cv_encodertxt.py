"""
mru_joint_dmm_text_metabags_5foldcv_no_shap_text_to_encoder.py
============================================================
✅ Optuna 제거
✅ Nested CV 제거
✅ folds_dir/fold_k/{train,val,test}를 그대로 "평범한 5-fold CV"로 실행
  * 각 fold: train 학습 + val early-stop + test 1회 평가

✅ Text -> Encoder (default)
  - 시간그리드로 정렬된 임상노트 임베딩(txt__)로부터 (raw/GRU) 방식으로 txt_hidden feature를 만들고,
    DMM encoder 입력([x, x_mask, u, e])에 concat하여 h_t를 구성한다.
  - posterior(q)는 텍스트를 직접 받지 않는다 (posterior text 옵션 삭제).


✅ (C) posterior-only default
  - fuse_head는 DMM feature만 사용 (text를 fuse 입력에 넣지 않음)
  - 원하면 --fuse_use_text 켜서 fuse에도 text를 넣을 수 있음

Run:
  python mru_joint_dmm_text_metabags_5foldcv_no_shap_text_to_encoder.py \
    --folds_dir artifacts/folds_modeldata_plus_text2 \
    --out_dir artifacts/grumm_5fold_text_to_encoder \
    --device cuda:0

Optional:
  --no_text  : 텍스트/메타백 completely off (encoder/fuse 모두 OFF)
  --text_source {raw,gru} : 텍스트 feature 생성 방식 (raw=per-grid projection, gru=Text GRU)
  --encoder_use_text / --no_encoder_use_text         : DMM encoder에 text를 넣을지 (fuse-only text를 위해 OFF 가능)
  --encoder_txt_use_mask / --no_encoder_txt_use_mask : txt_present(0/1) concat 여부
  --encoder_txt_detach / --no_encoder_txt_detach     : encoder에 넣기 전에 text detach 여부
  --encoder_txt_drop_p <p>                           : encoder text dropout (train-time)
  --posterior_use_text / --no_posterior_use_text     : posterior(q)에 text feature concat 여부
  --posterior_txt_use_mask / --no_posterior_txt_use_mask : posterior text에 txt_present(0/1) concat 여부
  --posterior_txt_detach / --no_posterior_txt_detach : posterior에 넣기 전에 text detach 여부
  --posterior_txt_drop_p <p>                         : posterior text dropout (train-time)
"""

import os
import re
import json
import math
import random
import argparse
import copy
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

try:
    import optuna
except Exception as e:
    raise RuntimeError("optuna required: pip install optuna") from e



# -------------------------
# Config
# -------------------------
@dataclass
class CFG:
    # folds
    folds_dir: str = "artifacts/folds_modeldata_plus_llmtext"
    out_dir: str = "artifacts/result_all_in_one_fixed"
    folds: int = 5
    only_fold: int = -1  # set >=0 to run a single fold id

    # ============================================================
    # ✅ FIXED (True/False) switches — 이 파일에서는 CLI로 바꾸지 않음
    #   - Text: ON
    #   - Encoder: text OFF
    #   - Posterior(q): text ON (+ txt_present mask ON)
    #   - Fuse: text OFF
    #   - Fuse: x-skip ON (proj + mask)
    #   - Smoother(train): ON
    # ============================================================
    use_text: bool = True
    meta_bags_on: bool = True
    encoder_use_text: bool = False

    text_source: str = "raw"  # {"raw","gru"} 중 raw로 고정

    # posterior(q)에만 text를 넣는다
    posterior_use_text: bool = True
    posterior_txt_use_mask: bool = True
    posterior_txt_detach: bool = False
    posterior_txt_drop_p: float = 0.4

    # encoder text 옵션(고정: OFF) — 아래 3개는 의미 없지만 명시적으로 둠
    encoder_txt_use_mask: bool = False
    encoder_txt_detach: bool = False
    encoder_txt_drop_p: float = 0.0

    # fuse text 옵션(고정: OFF)
    fuse_use_text: bool = False

    # fuse x-skip (고정: ON)
    fuse_use_x: bool = True
    fuse_x_use_mask: bool = True
    fuse_x_proj_dim: int = 112
    fuse_x_detach: bool = False
    fuse_x_drop_p: float = 0.1
    fuse_x_no_proj: bool = False

    fuse_type: str = "mlp"  # {"mlp","lstm","gru"} 중 mlp로 고정

    # static usage in fuse head (고정: ON)
    death_use_static: bool = True

    # smoothing (train-time only) (고정: ON)
    use_smoother_train: bool = True

    # class imbalance (고정: ON)
    use_pos_weight: bool = True

    # scaling (고정)
    scale_x: bool = True
    scale_u: bool = True
    scale_s_num: bool = True
    scale_txt: bool = False

    # AMP (고정: ON)
    amp: bool = True

    # ============================================================
    # paths / non-bool hyperparams (CLI에서 조절 가능)
    # ============================================================
    meta_bags_npz: str = "artifacts/folds_modeldata_plus_llmtext/meta_bags.npz"

    # sequence
    batch_size: int = 256
    max_len: int = 64
    # --------------------------------------------
    # Drop long stays: exclude patients with sequence length > this
    # (length = number of grid rows per stay in the split)
    # NOTE: This is applied to train/val/test and to the pooled data used in inner CV.
    # --------------------------------------------
    stay_len_max_keep: int = 316
    truncate_mode: str = "tail"  # "tail" or "head"
    num_workers: int = 0
    pin_memory: bool = True
    prefetch_factor: int = 2

    # training
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed: int = 123
    epochs: int = 40
    # separate epoch caps for nested CV
    inner_epochs: int = -1  # <=0 => use epochs (used for inner CV splits)
    outer_epochs: int = -1  # <=0 => use epochs (used for final refit / outer)

    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 4.0  # 0 => no clip

    # fuse head (kept fixed unless you edit)
    fuse_hidden: int = 128
    fuse_layers: int = 2

    # -------------------------
    # Optuna + Nested CV
    # -------------------------
    use_optuna: bool = True
    n_trials: int = 30
    inner_folds: int = 3
    final_holdout_frac: float = 0.15  # final refit early-stop holdout on POOL=train+val

    # inner-CV parallelism (⚠ GPU면 보통 권장 X: single GPU에선 오히려 느리거나 OOM 가능)
    inner_parallel: bool = False
    inner_jobs: int = 0  # 0 => auto (min(inner_folds, len(inner_devices) or inner_folds))
    inner_devices: str = ""  # e.g. "cuda:0,cuda:1,cuda:2" (if empty, use cfg.device)

    optuna_seed: int = 123
    optuna_sampler: str = "tpe"   # {"tpe","random"}
    optuna_pruner: str = "median" # {"none","median"}
    optuna_timeout_sec: int = 0   # 0 => no timeout


    # DMM dims
    z_dim: int = 48
    s_emb_dim: int = 96
    enc_hidden: int = 160
    enc_layers: int = 1
    trans_hidden: int = 64
    dec_hidden: int = 128

    dropout: float = 0.2

    # text dims (txt_hidden is used as posterior text feature dim)
    txt_hidden: int = 224
    txt_layers: int = 2
    meta_emb_dim: int = 16

    # loss weights
    lambda_death: float = 1.0
    beta_kl: float = 1.0
    kl_warmup_epochs: int = 3

    # distillation (default off; 필요하면 CLI로 키되, smoother는 그대로 teacher)
    lambda_distill: float = 0.7
    distill_warmup_epochs: int = 3

    # class imbalance
    pos_weight_mult: float = 0.5
    pos_weight_clip_max: float = 500.0
    pos_weight_override: Optional[float] = None

    # statics
    static_cat_cols: Tuple[str, ...] = (
        "gender", "race", "insurance", "language", "marital_status", "anchor_year_group",
    )
    static_num_cols: Tuple[str, ...] = ("anchor_age",)

    # labels
    y_col_default: str = "y_death_nextgrid"

    # SOFA raw bases
    sofa_raw_bases: Tuple[str, ...] = (
        "pao2fio2ratio_vent",
        "platelet_min",
        "bilirubin_max",
        "meanbp_min",
        "gcs_min",
        "creatinine_max",
    )

    # event extras
    event_extras: Tuple[str, ...] = ("delta_input_hours", "delta_procedure_hours", "mask_input", "mask_procedure")

    # ---- feature exclusion ----
    exclude_cols: Tuple[str, ...] = ("race", "language")
    exclude_regex: Tuple[str, ...] = ()


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_any(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _dedup(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        x = str(x)
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def list_fold_dirs(base_dir: str) -> List[str]:
    subs = []
    for name in os.listdir(base_dir):
        if name.startswith("fold_") and os.path.isdir(os.path.join(base_dir, name)):
            subs.append(os.path.join(base_dir, name))
    subs.sort(key=lambda p: int(os.path.basename(p).split("_")[1]))
    if not subs:
        raise FileNotFoundError(f"No fold_* dirs found under: {base_dir}")
    return subs


def pick_first_existing(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def apply_excludes(cols: List[str], cfg: CFG) -> List[str]:
    cols = [str(c) for c in (cols or []) if c is not None]
    ex = set(map(str, cfg.exclude_cols or ()))
    regs = [re.compile(p) for p in (cfg.exclude_regex or ()) if p]

    out = []
    for c in cols:
        if c in ex:
            continue
        if regs and any(r.search(c) for r in regs):
            continue
        out.append(c)
    return out


def assert_disjoint(*named_lists: Tuple[str, List[str]]):
    all_seen: Dict[str, str] = {}
    for name, cols in named_lists:
        for c in cols:
            if c in all_seen:
                raise RuntimeError(f"[COL OVERLAP] '{c}' appears in BOTH '{all_seen[c]}' and '{name}'")
            all_seen[c] = name


def ensure_numeric(df: pd.DataFrame, col: str, fill: float = 0.0, dtype=np.float32) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = fill
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df[col] = s.fillna(fill).astype(dtype)
    return df


def ensure_int01(df: pd.DataFrame, col: str, default: int = 1) -> pd.DataFrame:
    if col not in df.columns:
        df[col] = default
    s = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df[col] = (s > 0.5).astype(np.int8)
    return df


def sorted_unique_stays(df: pd.DataFrame, stay_col: str = "stay_id") -> np.ndarray:
    s = df[stay_col].dropna().astype(np.int64).unique()
    return np.array(sorted(s.tolist()), dtype=np.int64)




def drop_long_stays_by_len(
    df: pd.DataFrame,
    stay_col: str,
    max_keep: int,
    split_name: str = "",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Drop stays with length (number of rows per stay) > max_keep.

    Returns (df_filtered, kept_stays).
    """
    if df is None or len(df) == 0:
        return df, np.array([], dtype=np.int64)

    # count rows per stay (this is the 'sequence length' used by the dataset)
    cnt = df.groupby(stay_col, sort=False).size()
    keep_stays = cnt.index[cnt.values <= int(max_keep)].to_numpy(dtype=np.int64)
    drop_n = int((cnt.values > int(max_keep)).sum())

    out = df[df[stay_col].isin(keep_stays)].reset_index(drop=True)

    if split_name:
        print(
            f"[LEN FILTER] {split_name}: keep<= {int(max_keep)} | stays {len(cnt)} -> {len(keep_stays)} (dropped {drop_n}) | rows {len(df)} -> {len(out)}"
        )

    return out, keep_stays

def add_static_ohe_by_stay(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    df_te: pd.DataFrame,
    stay_col: str,
    static_cat_cols: List[str],
) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], List[str]]:
    if not static_cat_cols:
        return (df_tr, df_va, df_te), []

    tr0 = df_tr.sort_values([stay_col]).groupby(stay_col, as_index=False).first()
    tr_ohe = tr0[[stay_col] + static_cat_cols].copy()
    for c in static_cat_cols:
        tr_ohe[c] = tr_ohe[c].astype("string").fillna("__NA__")

    dummies = pd.get_dummies(tr_ohe[static_cat_cols], prefix=static_cat_cols, prefix_sep="=", dtype=np.int8)
    ohe_cols = dummies.columns.tolist()

    def _merge(df: pd.DataFrame) -> pd.DataFrame:
        d0 = df.sort_values([stay_col]).groupby(stay_col, as_index=False).first()
        part = d0[[stay_col] + static_cat_cols].copy()
        for c in static_cat_cols:
            part[c] = part[c].astype("string").fillna("__NA__")
        dum = pd.get_dummies(part[static_cat_cols], prefix=static_cat_cols, prefix_sep="=", dtype=np.int8)
        for c in ohe_cols:
            if c not in dum.columns:
                dum[c] = 0
        dum = dum[ohe_cols]
        mp = pd.concat([part[[stay_col]], dum], axis=1)
        out = df.merge(mp, on=stay_col, how="left", validate="m:1")
        for c in ohe_cols:
            out[c] = out[c].fillna(0).astype(np.int8)
        return out

    return (_merge(df_tr), _merge(df_va), _merge(df_te)), ohe_cols


def compute_standardizer(df: pd.DataFrame, cols: List[str], mask_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for i, c in enumerate(cols):
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy()
        if mask_cols is not None:
            m = pd.to_numeric(df[mask_cols[i]], errors="coerce").fillna(0).to_numpy() > 0.5
            x = x[m]
        x = x[np.isfinite(x)]
        if x.size == 0:
            mu, sd = 0.0, 1.0
        else:
            mu = float(np.mean(x))
            sd = float(np.std(x))
            if not np.isfinite(sd) or sd < 1e-6:
                sd = 1.0
        means.append(mu)
        stds.append(sd)
    return np.array(means, dtype=np.float32), np.array(stds, dtype=np.float32)


def apply_standardizer(df: pd.DataFrame, cols: List[str], mean: np.ndarray, std: np.ndarray, mask_cols: Optional[List[str]] = None):
    for i, c in enumerate(cols):
        v = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if mask_cols is not None:
            m = pd.to_numeric(df[mask_cols[i]], errors="coerce").fillna(0).to_numpy() > 0.5
            vv = v.to_numpy()
            vv[~np.isfinite(vv)] = mean[i]
            vv = (vv - mean[i]) / max(std[i], 1e-6)
            vv[~m] = 0.0
            df[c] = vv.astype(np.float32)
        else:
            vv = v.fillna(mean[i]).to_numpy()
            vv = (vv - mean[i]) / max(std[i], 1e-6)
            df[c] = vv.astype(np.float32)


# -------------------------
# MetaBags loader (EmbeddingBag inputs)
# -------------------------
class MetaBagsTorch:
    def __init__(self, npz_path: str, device: str):
        self.npz_path = str(npz_path)
        self.device = device

        raw = np.load(self.npz_path, allow_pickle=False)
        keys = list(raw.keys())

        self.row_id = None
        for k in ("row_id", "row_ids", "rowid"):
            if k in keys:
                self.row_id = raw[k].astype(np.int64)
                break

        if self.row_id is not None:
            rid = self.row_id
            if rid.size and rid.min() >= 0 and rid.max() <= 10_000_000 and rid.max() < rid.size * 5:
                self._map_vec = np.full(int(rid.max()) + 1, -1, dtype=np.int64)
                self._map_vec[rid] = np.arange(rid.size, dtype=np.int64)
                self._map_dict = None
            else:
                self._map_vec = None
                self._map_dict = {int(r): int(i) for i, r in enumerate(rid.tolist())}
        else:
            self._map_vec = None
            self._map_dict = None

        bag_names = []
        for k in keys:
            if k.endswith("_values") and (k[:-7] + "_offsets") in keys:
                bag_names.append(k[:-7])
            if k.endswith("_ids") and (k[:-4] + "_offsets") in keys:
                bag_names.append(k[:-4])
        bag_names = sorted(set(bag_names))

        if not bag_names:
            raise RuntimeError(f"[MetaBagsTorch] No '*_values'/'*_offsets' pairs found in npz: {self.npz_path}")

        self.bag_names = bag_names
        self.bags: Dict[str, Dict[str, Any]] = {}

        for name in self.bag_names:
            v_key = name + "_values"
            if v_key not in keys:
                v_key = name + "_ids"
            v = raw[v_key].astype(np.int64)
            o = raw[name + "_offsets"].astype(np.int64)
            if o.ndim != 1 or o.size < 2:
                raise RuntimeError(f"[MetaBagsTorch] Bad offsets for '{name}': shape={o.shape}")
            n_rows = int(o.size - 1)

            n_key = f"n_{name}"
            if n_key in keys:
                vocab = int(raw[n_key])
            else:
                vocab = int(v.max()) + 1 if v.size else 1

            self.bags[name] = {
                "values": torch.from_numpy(v).long().to(self.device, non_blocking=True),
                "offsets": torch.from_numpy(o).long().to(self.device, non_blocking=True),
                "n_rows": n_rows,
                "vocab": vocab,
            }

        n_rows_all = {self.bags[n]["n_rows"] for n in self.bag_names}
        if len(n_rows_all) != 1:
            raise RuntimeError(f"[MetaBagsTorch] Inconsistent n_rows across bags: {n_rows_all}")
        self.n_rows = int(next(iter(n_rows_all)))

    def map_row_id_to_pos(self, row_id: torch.Tensor) -> torch.Tensor:
        if self.row_id is None:
            return row_id
        rid = row_id.detach().cpu().numpy().astype(np.int64, copy=False)
        out = np.full_like(rid, -1, dtype=np.int64)
        m = rid >= 0
        if m.any():
            if self._map_vec is not None:
                vv = rid[m]
                vv = np.clip(vv, 0, self._map_vec.size - 1)
                out[m] = self._map_vec[vv]
            else:
                for idx in np.argwhere(m).flatten():
                    out.flat[idx] = self._map_dict.get(int(rid.flat[idx]), -1)
        return torch.from_numpy(out).to(row_id.device)

    def build_bag_batch(self, name: str, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bag = self.bags[name]
        offsets0 = bag["offsets"]
        values0 = bag["values"]

        device = offsets0.device
        pos = pos.to(device)

        valid_idx = torch.nonzero(pos >= 0, as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return (
                torch.zeros(0, dtype=torch.long, device=device),
                torch.zeros(1, dtype=torch.long, device=device),
                valid_idx,
            )

        p = pos[valid_idx].clamp(0, bag["n_rows"] - 1)
        starts = offsets0[p]
        ends = offsets0[p + 1]
        lens = (ends - starts).clamp_min(0)

        prefix = torch.cat([torch.zeros(1, dtype=torch.long, device=device), torch.cumsum(lens, dim=0)[:-1]])
        total = int(lens.sum().item())
        if total == 0:
            offsets = torch.cat([prefix, lens.sum().view(1)], dim=0)
            return torch.zeros(0, dtype=torch.long, device=device), offsets, valid_idx

        rep_starts = torch.repeat_interleave(starts, lens)
        rep_prefix = torch.repeat_interleave(prefix, lens)
        local = torch.arange(total, device=device, dtype=torch.long) - rep_prefix
        gather_idx = rep_starts + local
        values_cat = values0[gather_idx]

        offsets = torch.cat([prefix, lens.sum().view(1)], dim=0)
        return values_cat, offsets, valid_idx


# -------------------------
# ArrayStore / Dataset
# -------------------------
class ArrayStore:
    def __init__(
        self,
        df: pd.DataFrame,
        x_cols: List[str],
        x_mask_cols: List[str],
        u_cols: List[str],
        e_cols: List[str],
        txt_cols: List[str],
        static_cols: List[str],
        y_col: str,
        y_mask_col: str,
        time_mask_col: str,
        stay_col: str = "stay_id",
        time_col: str = "grid",
        row_id_col: str = "row_id",
    ):
        assert stay_col in df.columns, f"Missing stay_col={stay_col}"
        assert time_col in df.columns, f"Missing time_col={time_col}"

        if row_id_col not in df.columns:
            df[row_id_col] = np.arange(len(df), dtype=np.int64)

        df = df.sort_values([stay_col, time_col, row_id_col]).reset_index(drop=True)

        self.arr_stay = df[stay_col].to_numpy(np.int64, copy=False)
        self.arr_time = df[time_col].to_numpy(np.int64, copy=False)
        self.arr_rowid = df[row_id_col].to_numpy(np.int64, copy=False)

        self.arr_x = np.ascontiguousarray(df[x_cols].to_numpy(np.float32, copy=False)) if x_cols else np.zeros((len(df), 0), np.float32)
        if x_mask_cols:
            self.arr_xm = np.ascontiguousarray(df[x_mask_cols].to_numpy(np.float32, copy=False))
        else:
            self.arr_xm = np.ones_like(self.arr_x, dtype=np.float32) if self.arr_x.shape[1] else np.zeros((len(df), 0), np.float32)

        self.arr_u = np.ascontiguousarray(df[u_cols].to_numpy(np.float32, copy=False)) if u_cols else np.zeros((len(df), 0), np.float32)
        self.arr_e = np.ascontiguousarray(df[e_cols].to_numpy(np.float32, copy=False)) if e_cols else np.zeros((len(df), 0), np.float32)
        self.arr_txt = np.ascontiguousarray(df[txt_cols].to_numpy(np.float32, copy=False)) if txt_cols else np.zeros((len(df), 0), np.float32)

        self.arr_s_row = np.ascontiguousarray(df[static_cols].to_numpy(np.float32, copy=False)) if static_cols else np.zeros((len(df), 0), np.float32)

        self.arr_y = np.ascontiguousarray(df[y_col].to_numpy(np.float32, copy=False))
        self.arr_ym = np.ascontiguousarray(df[y_mask_col].to_numpy(np.float32, copy=False))
        self.arr_tm = np.ascontiguousarray(df[time_mask_col].to_numpy(np.float32, copy=False))

        self.stay_to_idx: Dict[int, np.ndarray] = {}
        self.stay_to_static: Dict[int, np.ndarray] = {}

        uniq, idx, cnt = np.unique(self.arr_stay, return_index=True, return_counts=True)
        for sid, st, c in zip(uniq, idx, cnt):
            sid_i = int(sid)
            st_i = int(st)
            c_i = int(c)
            self.stay_to_idx[sid_i] = np.arange(st_i, st_i + c_i, dtype=np.int64)
            self.stay_to_static[sid_i] = self.arr_s_row[st_i].copy() if self.arr_s_row.shape[1] else np.zeros((0,), np.float32)


class StayDataset(Dataset):
    def __init__(self, store: ArrayStore, stay_ids: np.ndarray, max_len: int, truncate_mode: str):
        self.S = store
        self.stay_ids = stay_ids.astype(np.int64)
        self.max_len = int(max_len)
        assert truncate_mode in ("tail", "head")
        self.truncate_mode = truncate_mode

    def __len__(self):
        return len(self.stay_ids)

    def _slice_or_trunc(self, idx: np.ndarray) -> np.ndarray:
        if idx.size <= self.max_len:
            return idx
        return idx[-self.max_len:] if self.truncate_mode == "tail" else idx[:self.max_len]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        sid = int(self.stay_ids[i])
        idx0 = self.S.stay_to_idx[sid]
        idx = self._slice_or_trunc(idx0)

        out = {
            "x": torch.from_numpy(self.S.arr_x[idx]),
            "x_mask": torch.from_numpy(self.S.arr_xm[idx]),
            "u": torch.from_numpy(self.S.arr_u[idx]),
            "e": torch.from_numpy(self.S.arr_e[idx]),
            "txt": torch.from_numpy(self.S.arr_txt[idx]),
            "row_id": torch.from_numpy(self.S.arr_rowid[idx]),
            "s": torch.from_numpy(self.S.stay_to_static[sid]),
            "y": torch.from_numpy(self.S.arr_y[idx]),
            "y_mask": torch.from_numpy(self.S.arr_ym[idx]),
            "time_mask": torch.from_numpy(self.S.arr_tm[idx]),
            "length": torch.tensor(len(idx), dtype=torch.long),
            "stay_id": torch.tensor(sid, dtype=torch.long),
        }
        return out


def collate_pad(batch: List[Dict[str, torch.Tensor]], pad_to: int) -> Dict[str, torch.Tensor]:
    lengths = torch.stack([b["length"] for b in batch])
    out: Dict[str, torch.Tensor] = {}

    keys_3d = ["x", "x_mask", "u", "e", "txt"]
    for k in keys_3d:
        tensors = [b[k] for b in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        if padded.size(1) > pad_to:
            padded = padded[:, :pad_to]
        elif padded.size(1) < pad_to:
            diff = pad_to - padded.size(1)
            padded = F.pad(padded, (0, 0, 0, diff), value=0.0)
        out[k] = padded

    row_tensors = [b["row_id"].long() for b in batch]
    padded_row = pad_sequence(row_tensors, batch_first=True, padding_value=-1)
    if padded_row.size(1) > pad_to:
        padded_row = padded_row[:, :pad_to]
    elif padded_row.size(1) < pad_to:
        diff = pad_to - padded_row.size(1)
        padded_row = F.pad(padded_row, (0, diff), value=-1)
    out["row_id"] = padded_row

    for k in ["y", "y_mask", "time_mask"]:
        tensors = [b[k] for b in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        if padded.size(1) > pad_to:
            padded = padded[:, :pad_to]
        elif padded.size(1) < pad_to:
            diff = pad_to - padded.size(1)
            padded = F.pad(padded, (0, diff), value=0.0)
        out[k] = padded

    out["s"] = torch.stack([b["s"] for b in batch], dim=0)

    B = len(batch)
    T = int(pad_to)
    ar = torch.arange(T).unsqueeze(0).expand(B, T)
    out["seq_mask"] = (ar < lengths.unsqueeze(1)).float()
    out["lengths"] = lengths
    out["stay_id"] = torch.stack([b["stay_id"] for b in batch])
    return out


def make_loader(cfg: CFG, ds: Dataset, shuffle: bool, drop_last: bool):
    kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=lambda b: collate_pad(b, pad_to=cfg.max_len),
    )
    if cfg.num_workers and cfg.num_workers > 0:
        kwargs["prefetch_factor"] = cfg.prefetch_factor
        kwargs["persistent_workers"] = True
    return DataLoader(ds, **kwargs)


# -------------------------
# Loss helpers
# -------------------------
def kl_diag_gaussians(mu_q, logv_q, mu_p, logv_p):
    v_q = torch.exp(logv_q)
    v_p = torch.exp(logv_p)
    kl = 0.5 * (logv_p - logv_q + (v_q + (mu_q - mu_p) ** 2) / (v_p + 1e-8) - 1.0)
    return kl.sum(dim=-1)


def gaussian_nll(x, mu, logv, mask):
    """Gaussian NLL per timestep.
    Returns mean NLL over *observed* feature dims (mask-aware), NOT sum.
    """
    nll = 0.5 * (math.log(2 * math.pi) + logv + (x - mu) ** 2 / (torch.exp(logv) + 1e-8))
    nll = nll * mask
    den = mask.sum(dim=-1).clamp_min(1.0)
    return nll.sum(dim=-1) / den


def gaussian_nll_sum_den(x, mu, logv, mask):
    """
    Gaussian NLL per timestep (sum+den form).

    Returns (nll_sum, den) where
      - nll_sum: sum of NLL over observed feature dims
      - den    : number of observed feature dims (sum(mask))

    Shapes: x,mu,logv,mask: (B, Dx) -> nll_sum,den: (B,)
    """
    nll = 0.5 * (math.log(2 * math.pi) + logv + (x - mu) ** 2 / (torch.exp(logv) + 1e-8))
    nll = nll * mask
    nll_sum = nll.sum(dim=-1)
    den = mask.sum(dim=-1)
    return nll_sum, den



# -------------------------
# Model
# -------------------------
class DMMCore(nn.Module):
    """
    Dual-posterior DMM:
      - Filter posterior q_f(z_t | z_{t-1}, h_fwd_t, [s])  : used for ONLINE prediction (no future leakage)
      - Smoother posterior q_s(z_t | z_{t-1}, h_bwd_t, [s]): used for TRAIN-TIME ONLY generative learning (recon + KL)

    Training loss typically uses:
      recon+KL from q_s, death head from q_f, and distillation KL(q_s || q_f) to teach the filter.
    """
    def __init__(self, Dx: int, Du: int, De: int, Ds: int, Dte: int = 0, Dtp: int = 0, cfg: CFG = None):
        super().__init__()
        if cfg is None:
            raise ValueError("cfg must be provided")
        self.Dx, self.Du, self.De, self.Ds = Dx, Du, De, Ds
        self.Dte = int(Dte)
        self.Dtp = int(Dtp)
        self.z = int(cfg.z_dim)
        self.cfg = cfg

        self.s_mlp = nn.Sequential(
            nn.Linear(Ds, cfg.s_emb_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.s_emb_dim, cfg.s_emb_dim),
            nn.ReLU(),
        ) if Ds > 0 else None
        self.sdim = int(cfg.s_emb_dim) if Ds > 0 else 0

        enc_in_dim = Dx + Dx + Du + De + int(self.Dte)  # [x, x_mask, u, e, txt_enc]
        # Filter encoder (past -> present)
        self.enc_gru_fwd = nn.GRU(
            input_size=int(enc_in_dim),
            hidden_size=int(cfg.enc_hidden),
            num_layers=int(cfg.enc_layers),
            dropout=float(cfg.dropout) if int(cfg.enc_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        # Smoother encoder (future -> present), run on reversed sequence
        self.enc_gru_bwd = nn.GRU(
            input_size=int(enc_in_dim),
            hidden_size=int(cfg.enc_hidden),
            num_layers=int(cfg.enc_layers),
            dropout=float(cfg.dropout) if int(cfg.enc_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        # --- posterior text config (q uses text feature) ---
        self.use_post_text = bool(getattr(self.cfg, "posterior_use_text", False)) and (self.Dtp > 0)

        post_in_dim = int(cfg.enc_hidden) + self.z
        if self.use_post_text:
            post_in_dim += int(self.Dtp)

        # Separate posterior nets (filter vs smoother) to avoid dimension hacks and keep behavior explicit
        self.post_net_f = nn.Sequential(
            nn.Linear(int(post_in_dim), int(cfg.trans_hidden)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(int(cfg.trans_hidden), 2 * self.z),
        )
        self.post_net_s = nn.Sequential(
            nn.Linear(int(post_in_dim), int(cfg.trans_hidden)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(int(cfg.trans_hidden), 2 * self.z),
        )

        self.prior_init = nn.Sequential(
            nn.Linear(self.sdim if self.sdim > 0 else 1, int(cfg.trans_hidden)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(int(cfg.trans_hidden), 2 * self.z),
        )

        self.trans_net = nn.Sequential(
            nn.Linear(self.z + Du + De + self.sdim, int(cfg.trans_hidden)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(int(cfg.trans_hidden), 2 * self.z),
        )

        self.dec_net = nn.Sequential(
            nn.Linear(self.z, int(cfg.dec_hidden)),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(int(cfg.dec_hidden), int(cfg.dec_hidden)),
            nn.ReLU(),
            nn.Linear(int(cfg.dec_hidden), 2 * Dx),
        )

    @staticmethod
    def reparam(mu, logv, sample: bool):
        if not sample:
            return mu
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _make_q_in(
        self,
        ht: torch.Tensor,
        z_prev: torch.Tensor,
        txt_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build posterior net input."""
        parts = [ht, z_prev]
        if self.use_post_text and (txt_t is not None) and (txt_t.shape[-1] > 0):
            parts.append(txt_t)
        return torch.cat(parts, dim=-1)

    def _prior0(self, x: torch.Tensor, s_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.Ds > 0:
            p0 = self.prior_init(s_emb)
        else:
            p0 = self.prior_init(x.new_ones((x.size(0), 1)))
        mu_p0, logv_p0 = torch.chunk(p0, 2, dim=-1)
        logv_p0 = torch.clamp(logv_p0, -8.0, 8.0)
        return mu_p0, logv_p0

    def forward_dual(
        self,
        x, xm, u, e, s, seq_mask, time_mask,
        sample_latent: bool,
        txt_enc: Optional[torch.Tensor] = None,
        txt_post: Optional[torch.Tensor] = None,
        compute_recon: bool = True,
        compute_kl: bool = True,
        compute_distill: bool = True,
    ):
        """
        Returns (recon, kl, distill_kl, feat_filter, z_filter_seq, xhat_mu_smoother)
        - recon/kl computed from smoother posterior q_s
        - distill_kl: KL(q_s || q_f) (teach filter to imitate smoother)  [masked + averaged]
        - feat_filter: features built from (h_fwd, z_f) for prediction head
        """
        B, T, Dx = x.shape

        if self.Ds > 0:
            s_emb = self.s_mlp(s)
        else:
            s_emb = x.new_zeros((B, 0))

        if txt_enc is None:
            enc_in = torch.cat([x, xm, u, e], dim=-1)
        else:
            enc_in = torch.cat([x, xm, u, e, txt_enc], dim=-1)

        # filter encoder (past)
        h_fwd, _ = self.enc_gru_fwd(enc_in)

        # mask over valid timesteps
        tmask = (seq_mask * time_mask).float()
        lengths = tmask.sum(dim=1).long().clamp(min=1, max=T)

        # smoother encoder (future -> present)
        # IMPORTANT: do NOT torch.flip(enc_in, dim=1) directly (it moves padding to the front).
        # Reverse only the valid prefix per sample, then pack so GRU never sees padding.
        enc_in_rev = enc_in.new_zeros(enc_in.shape)
        for i in range(B):
            L = int(lengths[i].item())
            enc_in_rev[i, :L] = enc_in[i, :L].flip(dims=[0])

        packed = pack_padded_sequence(
            enc_in_rev, lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.enc_gru_bwd(packed)
        h_bwd_rev, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=T
        )

        h_bwd = h_bwd_rev.new_zeros(h_bwd_rev.shape)
        for i in range(B):
            L = int(lengths[i].item())
            h_bwd[i, :L] = h_bwd_rev[i, :L].flip(dims=[0])

        mu_p0, logv_p0 = self._prior0(x=x, s_emb=s_emb)

        z_prev_f = x.new_zeros((B, self.z))
        z_prev_s = x.new_zeros((B, self.z))

        recon_sum = x.new_tensor(0.0)
        recon_den = x.new_tensor(0.0)
        kl_sum = x.new_tensor(0.0)
        dist_sum = x.new_tensor(0.0)

        feat_filter_all, zf_all, xhat_mu_all = [], [], []

        for t in range(T):
            ht_f = h_fwd[:, t, :]
            ht_s = h_bwd[:, t, :]

            txt_t = txt_post[:, t, :] if (txt_post is not None) else None
            q_in_f = self._make_q_in(ht_f, z_prev_f, txt_t=txt_t)
            q_in_s = self._make_q_in(ht_s, z_prev_s, txt_t=txt_t)

            qf = self.post_net_f(q_in_f)
            mu_qf, logv_qf = torch.chunk(qf, 2, dim=-1)
            logv_qf = torch.clamp(logv_qf, -8.0, 8.0)
            z_f = self.reparam(mu_qf, logv_qf, sample=sample_latent)

            qs = self.post_net_s(q_in_s)
            mu_qs, logv_qs = torch.chunk(qs, 2, dim=-1)
            logv_qs = torch.clamp(logv_qs, -8.0, 8.0)
            z_s = self.reparam(mu_qs, logv_qs, sample=sample_latent)

            # prior for smoother ELBO uses z_prev_s
            if t == 0:
                mu_p, logv_p = mu_p0, logv_p0
            else:
                tr_in = torch.cat([z_prev_s, u[:, t, :], e[:, t, :], s_emb], dim=-1)
                p = self.trans_net(tr_in)
                mu_p, logv_p = torch.chunk(p, 2, dim=-1)
                logv_p = torch.clamp(logv_p, -8.0, 8.0)

            if compute_kl:
                kl_t = kl_diag_gaussians(mu_qs, logv_qs, mu_p, logv_p)
                kl_sum = kl_sum + (kl_t * tmask[:, t]).sum()

            if compute_distill:
                # teacher -> student (smoother -> filter)
                dkl_t = kl_diag_gaussians(mu_qs.detach(), logv_qs.detach(), mu_qf, logv_qf)  # teacher(detached)->student
                dist_sum = dist_sum + (dkl_t * tmask[:, t]).sum()

            if compute_recon:
                dec = self.dec_net(z_s)
                mu_x, logv_x = torch.chunk(dec, 2, dim=-1)
                logv_x = torch.clamp(logv_x, -8.0, 8.0)
                nll_sum_t, den_t = gaussian_nll_sum_den(x[:, t, :], mu_x, logv_x, xm[:, t, :])
                w = tmask[:, t]
                recon_sum = recon_sum + (nll_sum_t * w).sum()
                recon_den = recon_den + (den_t * w).sum()
                xhat_mu_all.append(mu_x)
            else:
                # keep shape for compatibility if someone wants outputs
                if len(xhat_mu_all) < t + 1:
                    xhat_mu_all.append(x.new_zeros((B, Dx)))

            # prediction features ALWAYS from FILTER path (no future)
            if self.cfg.death_use_static and s_emb.shape[1] > 0:
                feat_filter_all.append(torch.cat([ht_f, z_f, s_emb], dim=-1))
            else:
                feat_filter_all.append(torch.cat([ht_f, z_f], dim=-1))

            zf_all.append(z_f)

            z_prev_f = z_f
            z_prev_s = z_s

        t_den = tmask.sum().clamp_min(1.0)

        recon = recon_sum / recon_den.clamp_min(1.0) if compute_recon else x.new_tensor(0.0)
        kl = kl_sum / t_den if compute_kl else x.new_tensor(0.0)
        distill = dist_sum / t_den if compute_distill else x.new_tensor(0.0)

        feat_filter = torch.stack(feat_filter_all, dim=1)
        zf_seq = torch.stack(zf_all, dim=1)
        xhat_mu = torch.stack(xhat_mu_all, dim=1)
        return recon, kl, distill, feat_filter, zf_seq, xhat_mu

    def forward_filter(
        self,
        x, xm, u, e, s, seq_mask, time_mask,
        sample_latent: bool,
        txt_enc: Optional[torch.Tensor] = None,
        txt_post: Optional[torch.Tensor] = None,
    ):
        """
        Filter-only forward for ONLINE prediction / evaluation.
        Returns (feat_filter, z_filter_seq).
        """
        B, T, Dx = x.shape
        if self.Ds > 0:
            s_emb = self.s_mlp(s)
        else:
            s_emb = x.new_zeros((B, 0))

        if txt_enc is None:
            enc_in = torch.cat([x, xm, u, e], dim=-1)
        else:
            enc_in = torch.cat([x, xm, u, e, txt_enc], dim=-1)
        h_fwd, _ = self.enc_gru_fwd(enc_in)

        z_prev_f = x.new_zeros((B, self.z))
        feat_filter_all, zf_all = [], []

        for t in range(T):
            ht_f = h_fwd[:, t, :]
            txt_t = txt_post[:, t, :] if (txt_post is not None) else None
            q_in_f = self._make_q_in(ht_f, z_prev_f, txt_t=txt_t)
            qf = self.post_net_f(q_in_f)
            mu_qf, logv_qf = torch.chunk(qf, 2, dim=-1)
            logv_qf = torch.clamp(logv_qf, -8.0, 8.0)
            z_f = self.reparam(mu_qf, logv_qf, sample=sample_latent)

            if self.cfg.death_use_static and s_emb.shape[1] > 0:
                feat_filter_all.append(torch.cat([ht_f, z_f, s_emb], dim=-1))
            else:
                feat_filter_all.append(torch.cat([ht_f, z_f], dim=-1))

            zf_all.append(z_f)
            z_prev_f = z_f

        feat_filter = torch.stack(feat_filter_all, dim=1)
        zf_seq = torch.stack(zf_all, dim=1)
        return feat_filter, zf_seq


class TextBranch(nn.Module):
    """(optional) GRU text encoder. Used only when text_source='gru' or fuse_use_text=True with gru."""
    def __init__(self, Dt: int, cfg: CFG, meta_bags: Optional[MetaBagsTorch]):
        super().__init__()
        self.Dt = int(Dt)
        self.cfg = cfg
        self.meta_bags = meta_bags

        self.bag_names: List[str] = []
        self.bag_embs = nn.ModuleDict()
        self.bag_dim_total = 0

        if self.meta_bags is not None and cfg.meta_bags_on:
            self.bag_names = list(self.meta_bags.bag_names)
            for name in self.bag_names:
                vocab = int(self.meta_bags.bags[name]["vocab"])
                emb = nn.EmbeddingBag(
                    num_embeddings=vocab,
                    embedding_dim=int(cfg.meta_emb_dim),
                    mode="mean",
                    include_last_offset=True,
                )
                self.bag_embs[name] = emb
                self.bag_dim_total += int(cfg.meta_emb_dim)

        in_dim = self.Dt + self.bag_dim_total
        self.gru = nn.GRU(
            input_size=int(in_dim),
            hidden_size=int(cfg.txt_hidden),
            num_layers=int(cfg.txt_layers),
            dropout=float(cfg.dropout) if int(cfg.txt_layers) > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, txt: torch.Tensor, row_id: torch.Tensor) -> torch.Tensor:
        B, T, _ = txt.shape
        if self.meta_bags is None or not self.bag_names:
            h, _ = self.gru(txt)
            return h

        pos = self.meta_bags.map_row_id_to_pos(row_id)
        feats = [txt]
        pos_flat = pos.reshape(-1)

        for name in self.bag_names:
            values_cat, offsets, valid_idx = self.meta_bags.build_bag_batch(name, pos_flat)
            emb = txt.new_zeros((pos_flat.numel(), int(self.cfg.meta_emb_dim)))
            if valid_idx.numel() > 0:
                out = self.bag_embs[name](values_cat, offsets)
                emb[valid_idx] = out
            feats.append(emb.view(B, T, -1))

        inp = torch.cat(feats, dim=-1)
        h, _ = self.gru(inp)
        return h


class PosteriorTextAdapter(nn.Module):
    """
    text_source='raw'일 때 사용:
      raw txt embedding(+ optional meta_bags EmbeddingBag)을 concat 후,
      Linear projection으로 txt_hidden 차원으로 변환 (B,T,txt_hidden)
    """
    def __init__(self, Dt: int, cfg: CFG, meta_bags: Optional[MetaBagsTorch]):
        super().__init__()
        self.cfg = cfg
        self.meta_bags = meta_bags

        self.bag_names: List[str] = []
        self.bag_embs = nn.ModuleDict()
        self.bag_dim_total = 0

        if self.meta_bags is not None and cfg.meta_bags_on:
            self.bag_names = list(self.meta_bags.bag_names)
            for name in self.bag_names:
                vocab = int(self.meta_bags.bags[name]["vocab"])
                emb = nn.EmbeddingBag(
                    num_embeddings=vocab,
                    embedding_dim=int(cfg.meta_emb_dim),
                    mode="mean",
                    include_last_offset=True,
                )
                self.bag_embs[name] = emb
                self.bag_dim_total += int(cfg.meta_emb_dim)

        in_dim = int(Dt) + int(self.bag_dim_total)
        out_dim = int(cfg.txt_hidden)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, txt: torch.Tensor, row_id: torch.Tensor) -> torch.Tensor:
        B, T, _ = txt.shape
        feats = [txt]

        if self.meta_bags is not None and self.bag_names:
            pos = self.meta_bags.map_row_id_to_pos(row_id)
            pos_flat = pos.reshape(-1)

            for name in self.bag_names:
                values_cat, offsets, valid_idx = self.meta_bags.build_bag_batch(name, pos_flat)
                emb = txt.new_zeros((pos_flat.numel(), int(self.cfg.meta_emb_dim)))
                if valid_idx.numel() > 0:
                    out = self.bag_embs[name](values_cat, offsets)
                    emb[valid_idx] = out
                feats.append(emb.view(B, T, -1))

        inp = torch.cat(feats, dim=-1)  # (B,T,Dt+bag_dim)
        return self.proj(inp)           # (B,T,txt_hidden)


class MRUJointModel(nn.Module):
    def __init__(self, Dx: int, Du: int, De: int, Dt: int, Ds: int, cfg: CFG, meta_bags: Optional[MetaBagsTorch]):
        super().__init__()
        self.cfg = cfg
        self.use_text = bool(cfg.use_text and Dt > 0)
        self.Dx, self.Du, self.De, self.Dt, self.Ds = Dx, Du, De, Dt, Ds

        # Dual-posterior DMM core (filter for prediction, smoother for train-time ELBO)
        enc_txt_dim = 0
        if self.use_text and bool(getattr(self.cfg, "encoder_use_text", True)):
            enc_txt_dim = int(cfg.txt_hidden) + (1 if bool(getattr(self.cfg, "encoder_txt_use_mask", True)) else 0)
        post_txt_dim = 0
        if self.use_text and bool(getattr(self.cfg, "posterior_use_text", False)):
            post_txt_dim = int(cfg.txt_hidden) + (1 if bool(getattr(self.cfg, "posterior_txt_use_mask", True)) else 0)
        self.dmm = DMMCore(Dx=Dx, Du=Du, De=De, Ds=Ds, Dte=enc_txt_dim, Dtp=post_txt_dim, cfg=cfg)

        self.txt_gru = None
        self.txt_raw_adapter = None

        if self.use_text:
            src = str(getattr(self.cfg, "text_source", "raw") or "raw").lower()
            if src not in ("raw", "gru"):
                src = "raw"

            # text feature source에 맞춰 필요한 것만 생성
            if src == "gru":
                self.txt_gru = TextBranch(Dt=Dt, cfg=cfg, meta_bags=meta_bags)

            if src == "raw":
                self.txt_raw_adapter = PosteriorTextAdapter(Dt=Dt, cfg=cfg, meta_bags=meta_bags)

        dmm_feat_dim = int(cfg.enc_hidden) + int(cfg.z_dim)
        if cfg.death_use_static and Ds > 0:
            dmm_feat_dim += int(cfg.s_emb_dim)

        fuse_in = int(dmm_feat_dim)

        # (optional) text into fuse
        if self.use_text and bool(getattr(self.cfg, "fuse_use_text", False)):
            fuse_in += int(cfg.txt_hidden) + 1  # + txt_present mask

                # (optional) x skip connection into fuse
        self.use_x_skip = bool(getattr(self.cfg, "fuse_use_x", False)) and (Dx > 0)
        self.fuse_x_no_proj = bool(getattr(self.cfg, "fuse_x_no_proj", False))
        self.x_to_fuse: Optional[nn.Module] = None
        self._x_skip_in_dim: int = 0
        if self.use_x_skip:
            x_in_dim = int(Dx)
            if bool(getattr(self.cfg, "fuse_x_use_mask", True)):
                x_in_dim += int(Dx)  # x_mask
            self._x_skip_in_dim = int(x_in_dim)
            if self.fuse_x_no_proj:
                fuse_in += x_in_dim
            else:
                x_proj = int(getattr(self.cfg, "fuse_x_proj_dim", 64))
                fuse_in += x_proj
                self.x_to_fuse = nn.Sequential(
                    nn.Linear(x_in_dim, x_proj),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                )

# ---- fuse head (mlp / lstm / gru) ----
        self.fuse_type = str(getattr(self.cfg, "fuse_type", "mlp") or "mlp").lower()
        if self.fuse_type not in ("mlp", "lstm", "gru"):
            self.fuse_type = "mlp"

        self.fuse_head: Optional[nn.Module] = None
        self.fuse_rnn: Optional[nn.Module] = None
        self.fuse_out: Optional[nn.Module] = None

        hid = int(cfg.fuse_hidden)
        L = max(1, int(cfg.fuse_layers))

        if self.fuse_type == "mlp":
            layers: List[nn.Module] = []
            if L <= 1:
                layers += [nn.Linear(fuse_in, 1)]
            else:
                layers += [nn.Linear(fuse_in, hid), nn.ReLU(), nn.Dropout(cfg.dropout)]
                for _ in range(L - 2):
                    layers += [nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(cfg.dropout)]
                layers += [nn.Linear(hid, 1)]
            self.fuse_head = nn.Sequential(*layers)
        else:
            rnn_dropout = float(cfg.dropout) if L > 1 else 0.0
            if self.fuse_type == "lstm":
                self.fuse_rnn = nn.LSTM(
                    input_size=fuse_in,
                    hidden_size=hid,
                    num_layers=L,
                    dropout=rnn_dropout,
                    batch_first=True,
                    bidirectional=False,
                )
            else:
                self.fuse_rnn = nn.GRU(
                    input_size=fuse_in,
                    hidden_size=hid,
                    num_layers=L,
                    dropout=rnn_dropout,
                    batch_first=True,
                    bidirectional=False,
                )
            self.fuse_out = nn.Linear(hid, 1)

    def _make_text_present(self, txt: torch.Tensor) -> torch.Tensor:
        if txt.size(-1) > 0:
            return (txt.abs().sum(dim=-1, keepdim=True) > 0).float()
        return txt.new_zeros((txt.size(0), txt.size(1), 1))

    
    def _prepare_text_features(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          txt_enc     : (B,T,Dte) encoder text input (txt_feat [+ txt_present]) if encoder_use_text else None
          txt_post    : (B,T,Dtp) posterior(q) text input (txt_feat [+ txt_present]) if posterior_use_text else None
          feat_t      : (B,T,txt_hidden+1) fuse text feature if fuse_use_text else None
          txt_present : (B,T,1) presence mask (0/1)
        """
        if not self.use_text:
            return None, None, None, None

        src = str(getattr(self.cfg, "text_source", "raw") or "raw").lower()
        if src not in ("raw", "gru"):
            src = "raw"

        txt = batch["txt"]
        row_id = batch["row_id"]
        txt_present = self._make_text_present(txt)

        # text feature (B,T,txt_hidden)
        if src == "raw":
            if self.txt_raw_adapter is None:
                raise RuntimeError("text_source='raw' but txt_raw_adapter is None")
            txt_feat = self.txt_raw_adapter(txt=txt, row_id=row_id)
        else:
            if self.txt_gru is None:
                raise RuntimeError("text_source='gru' but txt_gru is None")
            txt_feat = self.txt_gru(txt=txt, row_id=row_id)

        # ----------------------
        # Encoder text injection
        # ----------------------
        txt_enc = None
        if bool(getattr(self.cfg, "encoder_use_text", True)):
            txt_for_enc = txt_feat.detach() if bool(getattr(self.cfg, "encoder_txt_detach", False)) else txt_feat
            txt_present_enc = txt_present

            p_drop = float(getattr(self.cfg, "encoder_txt_drop_p", 0.0) or 0.0)
            if self.training and p_drop > 0:
                keep = (torch.rand_like(txt_present) >= p_drop).float()
                txt_for_enc = txt_for_enc * keep
                txt_present_enc = txt_present_enc * keep

            if bool(getattr(self.cfg, "encoder_txt_use_mask", True)):
                txt_enc = torch.cat([txt_for_enc, txt_present_enc], dim=-1)
            else:
                txt_enc = txt_for_enc

        # ----------------------
        # Posterior text injection
        # ----------------------
        txt_post = None
        if bool(getattr(self.cfg, "posterior_use_text", False)):
            txt_for_post = txt_feat.detach() if bool(getattr(self.cfg, "posterior_txt_detach", False)) else txt_feat
            txt_present_post = txt_present

            p_drop = float(getattr(self.cfg, "posterior_txt_drop_p", 0.0) or 0.0)
            if self.training and p_drop > 0:
                keep = (torch.rand_like(txt_present) >= p_drop).float()
                txt_for_post = txt_for_post * keep
                txt_present_post = txt_present_post * keep

            if bool(getattr(self.cfg, "posterior_txt_use_mask", True)):
                txt_post = torch.cat([txt_for_post, txt_present_post], dim=-1)
            else:
                txt_post = txt_for_post

        # ----------------------
        # Fuse text feature
        # ----------------------
        feat_t = None
        if bool(getattr(self.cfg, "fuse_use_text", False)):
            feat_t = torch.cat([txt_feat, txt_present], dim=-1)

        return txt_enc, txt_post, feat_t, txt_present



    def _prepare_x_fuse(
        self,
        x: torch.Tensor,
        xm: torch.Tensor,
        seq_mask: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Optional skip features from observed x (and x_mask) to fuse head."""
        if not bool(getattr(self, "use_x_skip", False)):
            return None

        use_mask = bool(getattr(self.cfg, "fuse_x_use_mask", True))
        if use_mask:
            xin = torch.cat([x, xm], dim=-1)
        else:
            xin = x

        if bool(getattr(self.cfg, "fuse_x_detach", False)):
            xin = xin.detach()

        if bool(getattr(self, "fuse_x_no_proj", False)) or bool(getattr(self.cfg, "fuse_x_no_proj", False)):
            xfeat = xin
        else:
            if self.x_to_fuse is None:
                return None
            xfeat = self.x_to_fuse(xin)

        p_drop = float(getattr(self.cfg, "fuse_x_drop_p", 0.0) or 0.0)
        if self.training and p_drop > 0:
            keep = (torch.rand_like(xfeat[..., :1]) >= p_drop).float()
            xfeat = xfeat * keep

        # zero-out padded/invalid steps
        valid = (seq_mask * time_mask).unsqueeze(-1)
        xfeat = xfeat * valid
        return xfeat

    def forward_logits(self, batch: Dict[str, torch.Tensor], sample_latent: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ONLINE-safe forward:
          - Uses FILTER posterior only (no backward GRU, no future leakage)
          - Returns (logits, recon_dummy, kl_dummy) for API compatibility
        """
        x = batch["x"]
        xm = batch["x_mask"]
        u = batch["u"]
        e = batch["e"]
        s = batch["s"]
        seq_mask = batch["seq_mask"]
        time_mask = batch["time_mask"]

        txt_enc, txt_post, feat_t, _txt_present = self._prepare_text_features(batch)
        if txt_enc is not None and (not bool(getattr(self.cfg, 'encoder_use_text', True))):
            txt_enc = None

        feat_v, _zf = self.dmm.forward_filter(
            x=x, xm=xm, u=u, e=e, s=s,
            seq_mask=seq_mask, time_mask=time_mask,
            sample_latent=sample_latent,
            txt_enc=txt_enc,
            txt_post=txt_post,
        )


        fuse_parts: List[torch.Tensor] = [feat_v]
        if self.use_text and bool(getattr(self.cfg, "fuse_use_text", False)) and (feat_t is not None):
            fuse_parts.append(feat_t)

        x_fuse = self._prepare_x_fuse(x=x, xm=xm, seq_mask=seq_mask, time_mask=time_mask)
        if x_fuse is not None:
            fuse_parts.append(x_fuse)

        fuse_inp = torch.cat(fuse_parts, dim=-1) if len(fuse_parts) > 1 else fuse_parts[0]

        if getattr(self, "fuse_type", "mlp") == "mlp":

            logits = self.fuse_head(fuse_inp).squeeze(-1)

        else:

            out = self.fuse_rnn(fuse_inp)[0]

            logits = self.fuse_out(out).squeeze(-1)

        # recon/kl not computed in filter-only forward
        return logits, x.new_tensor(0.0), x.new_tensor(0.0)

    def forward(self, batch: Dict[str, torch.Tensor], bce_loss_fn: nn.Module, sample_latent: bool) -> Dict[str, torch.Tensor]:
        """
        TRAIN forward:
          - recon + KL from smoother posterior (backward GRU)  [train-time only]
          - death head uses FILTER posterior only (online-safe)
          - optional distillation: KL(q_s || q_f) teaches filter to match smoother
        """
        x = batch["x"]
        xm = batch["x_mask"]
        u = batch["u"]
        e = batch["e"]
        s = batch["s"]
        seq_mask = batch["seq_mask"]
        time_mask = batch["time_mask"]

        txt_enc, txt_post, feat_t, _txt_present = self._prepare_text_features(batch)
        if txt_enc is not None and (not bool(getattr(self.cfg, 'encoder_use_text', True))):
            txt_enc = None

        beta = float(getattr(self.cfg, "beta_kl_current", self.cfg.beta_kl))
        lam_dist = float(getattr(self.cfg, "lambda_distill_current", self.cfg.lambda_distill))

        # ---- DMM dual forward ----
        if bool(getattr(self.cfg, "use_smoother_train", True)):
            recon, kl, distill, feat_v, _zf, _xhat = self.dmm.forward_dual(
                x=x, xm=xm, u=u, e=e, s=s,
                seq_mask=seq_mask, time_mask=time_mask,
                sample_latent=sample_latent,
                txt_enc=txt_enc,
                txt_post=txt_post,
                compute_recon=True,
                compute_kl=True,
                compute_distill=(lam_dist > 0),
            )
        else:
            # fallback: no smoother (uses filter-only path; recon/kl=0)
            feat_v, _zf = self.dmm.forward_filter(
                x=x, xm=xm, u=u, e=e, s=s,
                seq_mask=seq_mask, time_mask=time_mask,
                sample_latent=sample_latent,
                txt_enc=txt_enc,
                txt_post=txt_post,
            )
            recon = x.new_tensor(0.0)
            kl = x.new_tensor(0.0)
            distill = x.new_tensor(0.0)

        # ---- fuse head on FILTER features ----

        fuse_parts: List[torch.Tensor] = [feat_v]
        if self.use_text and bool(getattr(self.cfg, "fuse_use_text", False)) and (feat_t is not None):
            fuse_parts.append(feat_t)

        x_fuse = self._prepare_x_fuse(x=x, xm=xm, seq_mask=seq_mask, time_mask=time_mask)
        if x_fuse is not None:
            fuse_parts.append(x_fuse)

        fuse_inp = torch.cat(fuse_parts, dim=-1) if len(fuse_parts) > 1 else fuse_parts[0]

        if getattr(self, "fuse_type", "mlp") == "mlp":

            logits = self.fuse_head(fuse_inp).squeeze(-1)

        else:

            out = self.fuse_rnn(fuse_inp)[0]

            logits = self.fuse_out(out).squeeze(-1)

        # ---- supervised death loss (ONLINE-safe) ----
        y = batch["y"]
        y_mask = batch["y_mask"]
        y_valid = (seq_mask * time_mask * y_mask).float()
        bce = bce_loss_fn(logits, y)
        death = (bce * y_valid).sum() / y_valid.sum().clamp_min(1.0)

        total = recon + beta * kl + float(self.cfg.lambda_death) * death
        if lam_dist > 0:
            total = total + lam_dist * distill

        return {
            "loss": total,
            "recon": recon.detach(),
            "kl": kl.detach(),
            "distill": distill.detach(),
            "death": death.detach(),
            "logits": logits,
            "y_valid": y_valid,
        }


# -------------------------
# Metrics / weights
# -------------------------
@torch.no_grad()
def eval_pr_auc(model: MRUJointModel, loader: DataLoader, device: str) -> float:
    model.eval()
    ys, ps = [], []
    for batch in loader:
        for k in ("x", "x_mask", "u", "e", "txt", "row_id", "s", "y", "y_mask", "time_mask", "seq_mask"):
            batch[k] = batch[k].to(device, non_blocking=True)

        logits, _, _ = model.forward_logits(batch, sample_latent=False)
        prob = torch.sigmoid(logits)
        m = (batch["seq_mask"] * batch["time_mask"] * batch["y_mask"]) > 0.5
        if m.any():
            ys.append(batch["y"][m].detach().cpu().numpy())
            ps.append(prob[m].detach().cpu().numpy())
    if not ys:
        return float("nan")
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return float(average_precision_score(y, p))


def compute_pos_weight_base_from_store(store: ArrayStore, train_stays: np.ndarray) -> float:
    stay = store.arr_stay
    m_rows = np.isin(stay, train_stays)
    if m_rows.sum() == 0:
        return 1.0
    y = store.arr_y[m_rows]
    ym = store.arr_ym[m_rows] > 0.5
    tm = store.arr_tm[m_rows] > 0.5
    m = ym & tm & np.isfinite(y)
    if m.sum() == 0:
        return 1.0
    pos = ((y >= 0.5) & m).sum()
    neg = ((y < 0.5) & m).sum()
    if pos <= 0:
        return 1.0
    return float(neg / max(1.0, float(pos)))


def compute_pos_weight_with_cfg_from_store(cfg: CFG, store: ArrayStore, train_stays: np.ndarray) -> float:
    if not cfg.use_pos_weight:
        return 1.0
    if cfg.pos_weight_override is not None:
        pw = float(cfg.pos_weight_override)
    else:
        pw = float(compute_pos_weight_base_from_store(store, train_stays=train_stays))
    pw *= float(cfg.pos_weight_mult)
    if cfg.pos_weight_clip_max is not None and cfg.pos_weight_clip_max > 0:
        pw = float(min(pw, float(cfg.pos_weight_clip_max)))
    if not np.isfinite(pw) or pw <= 0:
        pw = 1.0
    return float(pw)


# -------------------------
# Fold bundle loader (features)
# -------------------------
def filter_x_and_mask_pairs(x_cols: List[str], x_mask_cols: List[str], cfg: CFG) -> Tuple[List[str], List[str]]:
    assert len(x_cols) == len(x_mask_cols)
    keep_x, keep_m = [], []
    for x, m in zip(x_cols, x_mask_cols):
        x_ok = (x in apply_excludes([x], cfg))
        m_ok = (m in apply_excludes([m], cfg))
        if x_ok and m_ok:
            keep_x.append(x)
            keep_m.append(m)
    return keep_x, keep_m


def try_pick_meta_bags_path(cfg: CFG, fold_meta: Dict[str, Any], fold_dir: str) -> str:
    if cfg.meta_bags_npz:
        return cfg.meta_bags_npz

    for key in ("meta_bags_npz", "text_bags_npz", "meta_bags", "bags_npz", "meta_bags_path"):
        if key in fold_meta:
            p = fold_meta[key]
            if isinstance(p, str) and p:
                return p

    if "paths" in fold_meta:
        p = fold_meta["paths"]
        for key in ("meta_bags_npz", "text_bags_npz", "meta_bags"):
            if key in p and isinstance(p[key], str) and p[key]:
                cand = p[key]
                if not os.path.isabs(cand):
                    cand2 = os.path.join(os.path.dirname(fold_dir), cand)
                    if os.path.exists(cand2):
                        return cand2
                return cand
    return ""


def load_fold_bundle(cfg: CFG, fold_dir: str):
    meta_path = os.path.join(fold_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json in {fold_dir}")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    cols = meta.get("cols", {})

    if "paths" in meta:
        p = meta["paths"]
        tr_name = p.get("train", "train.parquet")
        va_name = p.get("val", "val.parquet")
        te_name = p.get("test", "test.parquet")
        tr_path = os.path.join(fold_dir, tr_name)
        va_path = os.path.join(fold_dir, va_name)
        te_path = os.path.join(fold_dir, te_name)
    else:
        def _pick(name: str):
            p1 = os.path.join(fold_dir, f"{name}.parquet")
            p2 = os.path.join(fold_dir, f"{name}.csv")
            if os.path.exists(p1):
                return p1
            if os.path.exists(p2):
                return p2
            raise FileNotFoundError(f"Cannot find {name}.parquet|csv in {fold_dir}")

        tr_path, va_path, te_path = _pick("train"), _pick("val"), _pick("test")

    df_tr = read_any(tr_path)
    df_va = read_any(va_path)
    df_te = read_any(te_path)

    id_cols = cols.get("id_cols", []) or []
    y_col = cols.get("y_col", cfg.y_col_default) or cfg.y_col_default
    y_mask_col = cols.get("y_mask_col", "mask_death_valid_aligned") or "mask_death_valid_aligned"
    time_mask_col = cols.get("time_mask_col", "time_mask") or "time_mask"

    time_col = "grid" if "grid" in df_tr.columns else pick_first_existing(df_tr, ("grid", "baseline_time"))
    if time_col is None:
        raise RuntimeError(f"[{fold_dir}] cannot find time column among ('grid','baseline_time')")

    stay_col = "stay_id" if "stay_id" in df_tr.columns else pick_first_existing(df_tr, ("stay_id", "icustay_id"))
    if stay_col is None:
        raise RuntimeError(f"[{fold_dir}] cannot find stay_id column")

    # ---- drop long stays (sequence length > cfg.stay_len_max_keep) ----
    max_keep_len = int(getattr(cfg, "stay_len_max_keep", 0) or 0)
    if max_keep_len > 0:
        df_tr, _ = drop_long_stays_by_len(df_tr, stay_col, max_keep_len, split_name="train")
        df_va, _ = drop_long_stays_by_len(df_va, stay_col, max_keep_len, split_name="val")
        df_te, _ = drop_long_stays_by_len(df_te, stay_col, max_keep_len, split_name="test")

    for df_ in (df_tr, df_va, df_te):
        ensure_int01(df_, y_mask_col, default=1)
        ensure_int01(df_, time_mask_col, default=1)

    feature_numeric = _dedup(cols.get("feature_numeric", []) or [])
    feature_categorical = _dedup(cols.get("feature_categorical", []) or [])
    feature_numeric = apply_excludes(feature_numeric, cfg)
    feature_categorical = apply_excludes(feature_categorical, cfg)

    banned = set(map(str, id_cols))
    banned |= {y_col, y_mask_col, time_mask_col}

    x_cols = [c for c in cfg.sofa_raw_bases if (c in feature_numeric and c in df_tr.columns)]
    if len(x_cols) == 0:
        raise RuntimeError(f"[{fold_dir}] no SOFA base columns found. expected among {cfg.sofa_raw_bases}.")

    x_mask_cols = []
    for b in x_cols:
        mcol = f"mask_{b}"
        for df_ in (df_tr, df_va, df_te):
            if mcol in df_.columns:
                ensure_int01(df_, mcol, default=1)
            else:
                raw = pd.to_numeric(df_[b], errors="coerce").replace([np.inf, -np.inf], np.nan)
                df_[mcol] = raw.notna().astype(np.int8)

            s = pd.to_numeric(df_[b], errors="coerce").replace([np.inf, -np.inf], np.nan)
            df_[b] = s.fillna(0.0).astype(np.float32)
        x_mask_cols.append(mcol)

    x_cols, x_mask_cols = filter_x_and_mask_pairs(x_cols, x_mask_cols, cfg)
    if len(x_cols) == 0:
        raise RuntimeError(f"[{fold_dir}] x_cols became empty after excludes.")

    u_vs = [c for c in feature_numeric if c.startswith("vs_") and c in df_tr.columns]
    u_lb = [c for c in feature_numeric if c.startswith("lb_") and c in df_tr.columns]
    u_cols = _dedup(
        [c for c in (u_vs + u_lb) if (c not in banned and c not in x_cols and not c.startswith("mask_"))]
    )
    u_cols = apply_excludes(u_cols, cfg)
    # ✅ drop *_obs flags from inputs (keep *_missing and *_last)
    #    (requested: remove obs variables completely from model inputs)
    u_cols = [c for c in u_cols if not str(c).endswith("_obs")]


    e_cols = [
        c
        for c in feature_numeric
        if (c.startswith("in_cat__") or c.startswith("pr_cat__")) and c in df_tr.columns and c not in banned
    ]
    for extra in cfg.event_extras:
        if extra in feature_numeric and extra in df_tr.columns and extra not in banned:
            e_cols.append(extra)
    e_cols = apply_excludes(_dedup(e_cols), cfg)

    txt_cols = [c for c in feature_numeric if c.startswith("txt__") and c in df_tr.columns and c not in banned]
    txt_cols = apply_excludes(_dedup(txt_cols), cfg)

    static_num = [c for c in cfg.static_num_cols if (c in feature_numeric and c in df_tr.columns)]
    static_num = apply_excludes(static_num, cfg)

    static_cat = [c for c in cfg.static_cat_cols if (c in feature_categorical and c in df_tr.columns)]
    static_cat = apply_excludes(static_cat, cfg)

    (df_tr, df_va, df_te), static_ohe_cols = add_static_ohe_by_stay(
        df_tr=df_tr, df_va=df_va, df_te=df_te, stay_col=stay_col, static_cat_cols=static_cat
    )
    static_cols = apply_excludes(_dedup(static_num + static_ohe_cols), cfg)

    assert_disjoint(
        ("x_cols(sofa6)", x_cols),
        ("x_mask_cols", x_mask_cols),
        ("u_cols(vitals)", u_cols),
        ("e_cols(events)", e_cols),
        ("txt_cols(text)", txt_cols),
        ("static_cols", static_cols),
    )

    for df_ in (df_tr, df_va, df_te):
        for c in u_cols + e_cols + txt_cols + static_cols:
            ensure_numeric(df_, c, fill=0.0, dtype=np.float32)
        ensure_numeric(df_, y_col, fill=0.0, dtype=np.float32)
        ensure_int01(df_, y_mask_col, default=1)
        ensure_int01(df_, time_mask_col, default=1)

    scaler = {}
    if cfg.scale_x:
        mx, sx = compute_standardizer(df_tr, x_cols, mask_cols=x_mask_cols)
        apply_standardizer(df_tr, x_cols, mx, sx, mask_cols=x_mask_cols)
        apply_standardizer(df_va, x_cols, mx, sx, mask_cols=x_mask_cols)
        apply_standardizer(df_te, x_cols, mx, sx, mask_cols=x_mask_cols)
        scaler["x_mean"] = mx.tolist()
        scaler["x_std"] = sx.tolist()

    if cfg.scale_u and u_cols:
        mu, su = compute_standardizer(df_tr, u_cols, mask_cols=None)
        apply_standardizer(df_tr, u_cols, mu, su, mask_cols=None)
        apply_standardizer(df_va, u_cols, mu, su, mask_cols=None)
        apply_standardizer(df_te, u_cols, mu, su, mask_cols=None)
        scaler["u_mean"] = mu.tolist()
        scaler["u_std"] = su.tolist()

    if cfg.scale_txt and txt_cols:
        mt, st = compute_standardizer(df_tr, txt_cols, mask_cols=None)
        apply_standardizer(df_tr, txt_cols, mt, st, mask_cols=None)
        apply_standardizer(df_va, txt_cols, mt, st, mask_cols=None)
        apply_standardizer(df_te, txt_cols, mt, st, mask_cols=None)
        scaler["txt_mean"] = mt.tolist()
        scaler["txt_std"] = st.tolist()

    if cfg.scale_s_num and static_num:
        ms, ss = compute_standardizer(df_tr, static_num, mask_cols=None)
        apply_standardizer(df_tr, static_num, ms, ss, mask_cols=None)
        apply_standardizer(df_va, static_num, ms, ss, mask_cols=None)
        apply_standardizer(df_te, static_num, ms, ss, mask_cols=None)
        scaler["snum_mean"] = ms.tolist()
        scaler["snum_std"] = ss.tolist()

    store_tr = ArrayStore(
        df_tr,
        x_cols,
        x_mask_cols,
        u_cols,
        e_cols,
        txt_cols,
        static_cols,
        y_col,
        y_mask_col,
        time_mask_col,
        stay_col=stay_col,
        time_col=time_col,
    )
    store_va = ArrayStore(
        df_va,
        x_cols,
        x_mask_cols,
        u_cols,
        e_cols,
        txt_cols,
        static_cols,
        y_col,
        y_mask_col,
        time_mask_col,
        stay_col=stay_col,
        time_col=time_col,
    )
    store_te = ArrayStore(
        df_te,
        x_cols,
        x_mask_cols,
        u_cols,
        e_cols,
        txt_cols,
        static_cols,
        y_col,
        y_mask_col,
        time_mask_col,
        stay_col=stay_col,
        time_col=time_col,
    )

    stays_tr = sorted_unique_stays(df_tr, stay_col=stay_col)
    stays_va = sorted_unique_stays(df_va, stay_col=stay_col)
    stays_te = sorted_unique_stays(df_te, stay_col=stay_col)

    return dict(
        meta=meta,
        y_col=y_col,
        y_mask_col=y_mask_col,
        time_mask_col=time_mask_col,
        stay_col=stay_col,
        time_col=time_col,
        x_cols=x_cols,
        x_mask_cols=x_mask_cols,
        u_cols=u_cols,
        e_cols=e_cols,
        txt_cols=txt_cols,
        static_cols=static_cols,
        scaler=scaler,
        store_tr=store_tr,
        store_va=store_va,
        store_te=store_te,
        stays_tr=stays_tr,
        stays_va=stays_va,
        stays_te=stays_te,
        df_tr=df_tr,
        df_va=df_va,
        df_te=df_te,
        meta_bags_npz=try_pick_meta_bags_path(cfg, meta, fold_dir),
    )


# -------------------------
# Training (single split: train vs val)
# -------------------------
def train_on_split(
    cfg: CFG,
    store_tr: ArrayStore,
    stays_train: np.ndarray,
    store_va: ArrayStore,
    stays_val: np.ndarray,
    Dx: int,
    Du: int,
    De: int,
    Dt: int,
    Ds: int,
    out_dir: str,
    tag: str,
    device: str,
    meta_bags: Optional[MetaBagsTorch],
    max_epochs: Optional[int] = None,
) -> Dict[str, Any]:
    ds_tr = StayDataset(store_tr, stays_train, cfg.max_len, cfg.truncate_mode)
    ds_va = StayDataset(store_va, stays_val, cfg.max_len, cfg.truncate_mode)
    ld_tr = make_loader(cfg, ds_tr, shuffle=True, drop_last=True)
    ld_va = make_loader(cfg, ds_va, shuffle=False, drop_last=False)

    model = MRUJointModel(Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds, cfg=cfg, meta_bags=meta_bags).to(device)

    pos_weight = compute_pos_weight_with_cfg_from_store(cfg, store_tr, train_stays=stays_train)
    if cfg.use_pos_weight:
        bce_loss_fn = nn.BCEWithLogitsLoss(
            reduction="none",
            pos_weight=torch.tensor(pos_weight, device=device, dtype=torch.float32),
        )
        posw_used = float(pos_weight)
    else:
        bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        posw_used = 1.0

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    amp_enabled = bool(cfg.amp and ("cuda" in str(device)))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"{tag}_best.pt")

    best_val_total = float("inf")
    best_epoch = -1
    bad = 0
    max_epochs_eff = int(max_epochs) if (max_epochs is not None and int(max_epochs) > 0) else int(cfg.epochs)

    for ep in range(1, max_epochs_eff + 1):
        if cfg.kl_warmup_epochs and cfg.kl_warmup_epochs > 0:
            beta_now = float(cfg.beta_kl) * min(1.0, ep / float(cfg.kl_warmup_epochs))
        else:
            beta_now = float(cfg.beta_kl)

        cfg.beta_kl_current = float(beta_now)
        model.cfg.beta_kl_current = float(beta_now)

        # distillation warmup (smoother -> filter)
        if getattr(cfg, "distill_warmup_epochs", 0) and cfg.distill_warmup_epochs > 0:
            lam_now = float(cfg.lambda_distill) * min(1.0, ep / float(cfg.distill_warmup_epochs))
        else:
            lam_now = float(getattr(cfg, "lambda_distill", 0.0))
        cfg.lambda_distill_current = float(lam_now)
        model.cfg.lambda_distill_current = float(lam_now)

        model.train()
        tr_total = 0.0
        tr_n = 0

        for batch in ld_tr:
            for k in ("x", "x_mask", "u", "e", "txt", "row_id", "s", "y", "y_mask", "time_mask", "seq_mask"):
                batch[k] = batch[k].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out = model(batch, bce_loss_fn=bce_loss_fn, sample_latent=True)
                loss = out["loss"]

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
            scaler.step(opt)
            scaler.update()

            tr_total += float(loss.detach().cpu().item())
            tr_n += 1
        tr_total /= max(1, tr_n)

        model.eval()
        # --- denom-weighted validation aggregation (sum(mask) cumulative) ---
        va_recon_num = 0.0
        va_recon_den = 0.0   # sum(x_mask) over valid timesteps
        va_kl_num = 0.0
        va_kl_den = 0.0      # sum(valid timesteps)
        va_dist_num = 0.0
        va_dist_den = 0.0    # sum(valid timesteps)
        va_death_num = 0.0
        va_death_den = 0.0   # sum(y_valid)

        with torch.no_grad():
            for batch in ld_va:
                for k in ("x", "x_mask", "u", "e", "txt", "row_id", "s", "y", "y_mask", "time_mask", "seq_mask"):
                    batch[k] = batch[k].to(device, non_blocking=True)

                model.cfg.beta_kl_current = float(beta_now)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    out = model(batch, bce_loss_fn=bce_loss_fn, sample_latent=False)

                # denoms
                tmask = (batch["seq_mask"] * batch["time_mask"]).float()
                t_den = float(tmask.sum().detach().cpu().item())
                x_den = float((batch["x_mask"].float() * tmask.unsqueeze(-1)).sum().detach().cpu().item())
                y_den = float(out["y_valid"].sum().detach().cpu().item())

                recon = float(out["recon"].detach().cpu().item())
                kl = float(out["kl"].detach().cpu().item())
                distill = float(out["distill"].detach().cpu().item())
                death = float(out["death"].detach().cpu().item())

                # recon is mean over observed x elements inside the batch
                if x_den > 0:
                    va_recon_num += recon * x_den
                    va_recon_den += x_den

                # kl/distill are mean over valid timesteps inside the batch
                if t_den > 0:
                    va_kl_num += kl * t_den
                    va_kl_den += t_den
                    va_dist_num += distill * t_den
                    va_dist_den += t_den

                # death is mean over y_valid inside the batch
                if y_den > 0:
                    va_death_num += death * y_den
                    va_death_den += y_den

        recon_g = va_recon_num / max(1.0, va_recon_den)
        kl_g = va_kl_num / max(1.0, va_kl_den)
        dist_g = va_dist_num / max(1.0, va_dist_den)
        death_g = va_death_num / max(1.0, va_death_den)

        va_total = recon_g + beta_now * kl_g + float(cfg.lambda_death) * death_g
        if lam_now > 0:
            va_total = va_total + lam_now * dist_g

        print(
            f"[{tag}] Ep {ep:03d} | "
            f"train_total={tr_total:.4f} | val_total={va_total:.4f} | "
            f"best_val={best_val_total:.4f} | beta_kl={beta_now:.3f} | lam_dist={lam_now:.3f} | pos_w={posw_used:.2f}"
        )

        if va_total < best_val_total - 5e-2:
            best_val_total = float(va_total)
            best_epoch = int(ep)
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg),
                    "best_epoch": int(best_epoch),
                    "best_val_total": float(best_val_total),
                    "beta_kl_at_best": float(beta_now),
                    "pos_weight_used": float(posw_used),
                },
                best_path,
            )
        else:
            bad += 1
            if bad >= int(cfg.patience):
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    val_pr = eval_pr_auc(model, ld_va, device=device)

    return dict(
        best_path=best_path,
        best_epoch=int(ckpt["best_epoch"]),
        best_val_total=float(ckpt["best_val_total"]),
        val_pr_auc=float(val_pr),
        pos_weight_used=float(ckpt.get("pos_weight_used", 1.0)),
    )


# -------------------------
# 5-fold CV runner per fold_dir
# --------------
def _stay_level_label_from_store(store: ArrayStore, sid: int) -> int:
    idx = store.stay_to_idx.get(int(sid))
    if idx is None or idx.size == 0:
        return 0
    y = store.arr_y[idx]
    ym = store.arr_ym[idx] > 0.5
    tm = store.arr_tm[idx] > 0.5
    m = ym & tm & np.isfinite(y)
    if m.sum() == 0:
        return 0
    return int((y[m].max() >= 0.5))


def stay_level_labels_from_store(store: ArrayStore, stay_ids: np.ndarray) -> np.ndarray:
    labels = np.zeros((len(stay_ids),), dtype=np.int64)
    for i, sid in enumerate(stay_ids.tolist()):
        labels[i] = _stay_level_label_from_store(store, int(sid))
    return labels


def clone_cfg(cfg: CFG) -> CFG:
    # deepcopy keeps tuple fields as tuples (asdict converts them to lists)
    return copy.deepcopy(cfg)


def apply_trial_params(cfg, params: dict):
    """Apply ONLY the selected Optuna parameters to cfg.

    This script is intentionally configured to tune a *small* set of hyperparameters:
      - lr (log)
      - weight_decay (log)
      - dropout (backbone)
      - posterior_txt_drop_p (posterior text regularization)
      - fuse_x_drop_p (fuse regularization)
      - fuse_x_proj_dim (ordered int grid: {96,112,128})
      - z_dim (ordered int grid: {32,48,64})

    Everything else remains as provided by CLI/defaults.
    """

    def _get(name: str, cast_fn, default):
        if name not in params:
            return default
        try:
            return cast_fn(params[name])
        except Exception:
            return default

    cfg.lr = _get("lr", float, cfg.lr)
    cfg.weight_decay = _get("weight_decay", float, cfg.weight_decay)

    cfg.dropout = _get("dropout", float, cfg.dropout)
    cfg.posterior_txt_drop_p = _get("posterior_txt_drop_p", float, cfg.posterior_txt_drop_p)
    cfg.fuse_x_drop_p = _get("fuse_x_drop_p", float, cfg.fuse_x_drop_p)

    # Ordered architecture dims (no categorical)
    cfg.fuse_x_proj_dim = _get("fuse_x_proj_dim", int, cfg.fuse_x_proj_dim)
    cfg.z_dim = _get("z_dim", int, cfg.z_dim)



def suggest_trial_params(trial):
    """Suggest Optuna trial parameters.

    IMPORTANT: We do NOT use suggest_categorical (ordered search only).
    Tuned params (and ONLY these):
      - lr (log=True)
      - weight_decay (log=True)
      - dropout
      - posterior_txt_drop_p
      - fuse_x_drop_p
      - fuse_x_proj_dim  (ordered int grid: {96,112,128})
      - z_dim           (ordered int grid: {32,48,64})
    """

    params = {}

    # Log-scale optimizer params (keep log=True as requested)
    params["lr"] = trial.suggest_float("lr", 1e-3, 3e-3, log=True)
    params["weight_decay"] = trial.suggest_float("weight_decay", 1e-5, 5e-4, log=True)

    # Ordered (step-based) regularization params
    params["dropout"] = trial.suggest_float("dropout", 0.0, 0.30, step=0.1)
    params["posterior_txt_drop_p"] = trial.suggest_float("posterior_txt_drop_p", 0.0, 0.50, step=0.1)
    params["fuse_x_drop_p"] = trial.suggest_float("fuse_x_drop_p", 0.0, 0.30, step=0.1)

    # Ordered architecture dims (no categorical)
    params["fuse_x_proj_dim"] = trial.suggest_int("fuse_x_proj_dim", 64, 128, step=16)
    params["z_dim"] = trial.suggest_int("z_dim", 32, 64, step=16)

    return params


def parse_device_list(devices: str, fallback: str) -> List[str]:
    """Parse comma-separated device list (e.g., 'cuda:0,cuda:1')."""
    s = str(devices or "").strip()
    if not s:
        return [str(fallback)]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return parts if parts else [str(fallback)]


def _inner_fold_worker(payload: Dict[str, Any]) -> float:
    """
    Multiprocessing worker for ONE inner split training/eval.
    NOTE: This reloads fold data inside the worker (safe for spawn, avoids pickling huge stores).
    """
    fold_dir: str = payload["fold_dir"]
    cfg: CFG = payload["cfg"]
    stays_tr = payload["stays_tr"]
    stays_va = payload["stays_va"]
    out_dir: str = payload["out_dir"]
    tag: str = payload.get("tag", "inner")
    device: str = payload.get("device", cfg.device)
    seed: int = int(payload.get("seed", cfg.seed))

    cfg_k = clone_cfg(cfg)
    cfg_k.device = device
    cfg_k.seed = seed
    set_seed(cfg_k.seed)

    # (re)load bundle inside worker
    bundle = load_fold_bundle(cfg_k, fold_dir)

    # POOL = train ∪ val
    df_pool = pd.concat([bundle["df_tr"], bundle["df_va"]], axis=0, ignore_index=True)
    store_pool = ArrayStore(
        df_pool,
        bundle["x_cols"],
        bundle["x_mask_cols"],
        bundle["u_cols"],
        bundle["e_cols"],
        bundle["txt_cols"],
        bundle["static_cols"],
        bundle["y_col"],
        bundle["y_mask_col"],
        bundle["time_mask_col"],
        stay_col=bundle["stay_col"],
        time_col=bundle["time_col"],
    )

    Dx, Du, De, Dt, Ds = (
        len(bundle["x_cols"]),
        len(bundle["u_cols"]),
        len(bundle["e_cols"]),
        len(bundle["txt_cols"]),
        len(bundle["static_cols"]),
    )

    # meta-bags (load on worker device)
    meta_bags = None
    if cfg_k.use_text and cfg_k.meta_bags_on:
        npz_path = bundle.get("meta_bags_npz", cfg_k.meta_bags_npz) or ""
        if npz_path and os.path.exists(npz_path):
            meta_bags = MetaBagsTorch(npz_path=npz_path, device=cfg_k.device)

    os.makedirs(out_dir, exist_ok=True)
    res = train_on_split(
        cfg=cfg_k,
        max_epochs=getattr(cfg_k, 'inner_epochs', cfg_k.epochs),
        store_tr=store_pool,
        stays_train=stays_tr,
        store_va=store_pool,
        stays_val=stays_va,
        Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds,
        out_dir=out_dir,
        tag=tag,
        device=cfg_k.device,
        meta_bags=meta_bags,
    )
    return float(res["val_pr_auc"])


def run_outer_fold_optuna_nested(fold_dir: str, cfg: CFG) -> Dict[str, Any]:
    """
    Outer fold is provided by folds_dir/fold_k/{train,val,test}.
    Nested CV:
      - POOL = train ∪ val
      - Inner CV: StratifiedKFold on stay-level label; objective = mean(inner val PR-AUC)
      - Final refit: POOL -> (train/holdout) early-stop, then evaluate once on OUTER test.
    """
    fold_id = int(os.path.basename(fold_dir).split("_")[1])
    out_fold = os.path.join(cfg.out_dir, f"fold_{fold_id}")
    os.makedirs(out_fold, exist_ok=True)

    bundle = load_fold_bundle(cfg, fold_dir)

    store_te = bundle["store_te"]
    stays_te = bundle["stays_te"]

    # ---- build POOL ----
    df_pool = pd.concat([bundle["df_tr"], bundle["df_va"]], axis=0, ignore_index=True)
    store_pool = ArrayStore(
        df_pool,
        bundle["x_cols"],
        bundle["x_mask_cols"],
        bundle["u_cols"],
        bundle["e_cols"],
        bundle["txt_cols"],
        bundle["static_cols"],
        bundle["y_col"],
        bundle["y_mask_col"],
        bundle["time_mask_col"],
        stay_col=bundle["stay_col"],
        time_col=bundle["time_col"],
    )
    stays_pool = sorted_unique_stays(df_pool, stay_col=bundle["stay_col"])
    y_pool = stay_level_labels_from_store(store_pool, stays_pool)

    meta_bags = None
    # NOTE: inner_parallel=True면 inner worker에서 meta-bags를 로드함 (부모 프로세스 GPU 점유/VRAM 낭비 방지)
    if cfg.use_text and cfg.meta_bags_on and (not cfg.inner_parallel):
        npz_path = bundle.get("meta_bags_npz", cfg.meta_bags_npz) or ""
        if npz_path and os.path.exists(npz_path):
            meta_bags = MetaBagsTorch(npz_path=npz_path, device=cfg.device)
            print(f"[FOLD {fold_id}] meta-bags ON: {npz_path} | bags={meta_bags.bag_names} | n_rows={meta_bags.n_rows}")
        else:
            print(f"[FOLD {fold_id}] meta-bags OFF (npz not found). path='{npz_path}'")

    Dx, Du, De, Dt, Ds = (
        len(bundle["x_cols"]),
        len(bundle["u_cols"]),
        len(bundle["e_cols"]),
        len(bundle["txt_cols"]),
        len(bundle["static_cols"]),
    )
    print(f"[FOLD {fold_id}] DIMS Dx={Dx} Du={Du} De={De} Dt={Dt} Ds={Ds} | POOL stays={len(stays_pool)} | TEST stays={len(stays_te)}")

    # ---- optuna study ----
    study_name = f"fold{fold_id}_nested_tune7"
    sampler = None
    if str(cfg.optuna_sampler).lower() == "random":
        sampler = optuna.samplers.RandomSampler(seed=int(cfg.optuna_seed))
    else:
        sampler = optuna.samplers.TPESampler(seed=int(cfg.optuna_seed), multivariate=True)

    # ✅ Pruning disabled (always run full trials)
    pruner = optuna.pruners.NopPruner()

    # Optional: resume storage
    storage = None
    storage_path = os.path.join(out_fold, "optuna_study_tune7.db")
    try:
        storage = optuna.storages.RDBStorage(url=f"sqlite:///{storage_path}")
    except Exception:
        storage = None  # fallback to in-memory

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_trial_params(trial)

        cfg_trial = clone_cfg(cfg)
        apply_trial_params(cfg_trial, params)

        # Inner CV
        skf = StratifiedKFold(n_splits=int(cfg_trial.inner_folds), shuffle=True, random_state=int(cfg_trial.optuna_seed) + int(trial.number))
        scores: List[float] = []

        if cfg_trial.inner_parallel:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor, as_completed

            # ⚠ GPU 1장만 쓰는 경우에는 inner fold 병렬이 대개 더 느리거나 OOM을 유발함.
            #    (각 fold가 '별도 학습'을 동시에 수행하므로 VRAM/SM을 나눠씀)
            devs = parse_device_list(cfg_trial.inner_devices, cfg_trial.device)
            max_workers = int(cfg_trial.inner_jobs) if int(cfg_trial.inner_jobs) > 0 else min(int(cfg_trial.inner_folds), len(devs))
            max_workers = max(1, max_workers)

            splits = list(skf.split(stays_pool, y_pool))
            futs = []
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context("spawn")) as ex:
                for k, (tr_idx, va_idx) in enumerate(splits):
                    stays_tr_k = stays_pool[tr_idx]
                    stays_va_k = stays_pool[va_idx]

                    out_trial = os.path.join(out_fold, "optuna_tune7", f"trial_{trial.number:05d}", f"inner_{k}")
                    os.makedirs(out_trial, exist_ok=True)

                    cfg_k = clone_cfg(cfg_trial)

                    dev = devs[k % len(devs)]
                    seed_k = int(cfg_k.seed) + int(trial.number) * 1000 + int(k)

                    payload = {
                        "fold_dir": fold_dir,
                        "cfg": cfg_k,
                        "stays_tr": stays_tr_k,
                        "stays_va": stays_va_k,
                        "out_dir": out_trial,
                        "tag": f"inner{k}",
                        "device": dev,
                        "seed": seed_k,
                    }
                    futs.append(ex.submit(_inner_fold_worker, payload))

                done = 0
                for fut in as_completed(futs):
                    score = float(fut.result())
                    scores.append(score)

                    trial.report(float(np.nanmean(scores)), step=done)
                    done += 1
                    if trial.should_prune():
                        raise optuna.TrialPruned()
        else:
            for k, (tr_idx, va_idx) in enumerate(skf.split(stays_pool, y_pool)):
                stays_tr_k = stays_pool[tr_idx]
                stays_va_k = stays_pool[va_idx]

                # unique dir per trial/split (avoid ckpt collisions)
                out_trial = os.path.join(out_fold, "optuna_tune7", f"trial_{trial.number:05d}", f"inner_{k}")
                os.makedirs(out_trial, exist_ok=True)

                cfg_k = clone_cfg(cfg_trial)
                # make inner folds quicker by allowing fewer epochs if you want (kept as cfg.epochs by default)

                res = train_on_split(
                    cfg=cfg_k,
                    max_epochs=getattr(cfg_k, 'inner_epochs', cfg_k.epochs),
                    store_tr=store_pool,
                    stays_train=stays_tr_k,
                    store_va=store_pool,
                    stays_val=stays_va_k,
                    Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds,
                    out_dir=out_trial,
                    tag=f"inner{k}",
                    device=cfg_k.device,
                    meta_bags=meta_bags,
                )
                score = float(res["val_pr_auc"])
                scores.append(score)

                trial.report(float(np.nanmean(scores)), step=k)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.nanmean(scores))

    timeout = int(cfg.optuna_timeout_sec) if int(cfg.optuna_timeout_sec) > 0 else None
    study.optimize(objective, n_trials=int(cfg.n_trials), timeout=timeout)

    best = study.best_trial
    best_params = dict(best.params)
    # txt_layers is fixed (not suggested) in many configs, so Optuna may not store it.
    best_params.setdefault("txt_layers", 2)

    with open(os.path.join(out_fold, "optuna_best_params_tune7.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"[FOLD {fold_id}] ✅ Optuna best value={best.value:.4f} | params saved -> optuna_best_params_tune7.json")

    # ---- final refit on POOL (early-stop with holdout) ----
    cfg_best = clone_cfg(cfg)
    # Some params may be fixed (not suggested), so they are absent from best_params.
    # Keep this robust for resumed/old studies.
    best_params.setdefault("txt_layers", getattr(cfg_best, "txt_layers", 2))
    apply_trial_params(cfg_best, best_params)

    # stratified holdout
    test_size = float(cfg_best.final_holdout_frac)
    if test_size <= 0 or test_size >= 0.5:
        test_size = 0.15

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=int(cfg_best.optuna_seed) + fold_id)
    tr_idx, va_idx = next(sss.split(stays_pool, y_pool))
    stays_final_tr = stays_pool[tr_idx]
    stays_final_va = stays_pool[va_idx]

    out_final = os.path.join(out_fold, "final_refit")
    os.makedirs(out_final, exist_ok=True)

    # meta-bags for final refit (if not loaded earlier)
    meta_bags_final = meta_bags
    if meta_bags_final is not None and getattr(meta_bags_final, "device", None) != cfg_best.device:
        meta_bags_final = None
    if meta_bags_final is None and cfg_best.use_text and cfg_best.meta_bags_on:
        npz_path = bundle.get("meta_bags_npz", cfg_best.meta_bags_npz) or ""
        if npz_path and os.path.exists(npz_path):
            meta_bags_final = MetaBagsTorch(npz_path=npz_path, device=cfg_best.device)

    res_final = train_on_split(
        cfg=cfg_best,
        max_epochs=getattr(cfg_best, 'outer_epochs', cfg_best.epochs),
        store_tr=store_pool,
        stays_train=stays_final_tr,
        store_va=store_pool,
        stays_val=stays_final_va,
        Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds,
        out_dir=out_final,
        tag="final",
        device=cfg_best.device,
        meta_bags=meta_bags_final,
    )

    # ---- load best final model and evaluate on OUTER test ----
    ckpt = torch.load(res_final["best_path"], map_location=cfg_best.device)
    cfg_loaded = CFG(**ckpt["cfg"])

    # enforce fixed boolean switches again (safety)
    cfg_loaded.use_text = True
    cfg_loaded.meta_bags_on = True
    cfg_loaded.text_source = "raw"
    cfg_loaded.encoder_use_text = True
    cfg_loaded.encoder_txt_use_mask = True
    cfg_loaded.posterior_use_text = False
    cfg_loaded.posterior_txt_use_mask = False
    cfg_loaded.fuse_use_text = False
    cfg_loaded.fuse_use_x = True
    cfg_loaded.fuse_x_use_mask = True
    cfg_loaded.fuse_x_no_proj = False
    cfg_loaded.fuse_type = "mlp"
    cfg_loaded.death_use_static = True
    cfg_loaded.use_smoother_train = True
    cfg_loaded.use_pos_weight = True
    cfg_loaded.scale_x = True
    cfg_loaded.scale_u = True
    cfg_loaded.scale_s_num = True
    cfg_loaded.scale_txt = False
    cfg_loaded.amp = True

    # meta-bags for evaluation (ensure device match)
    meta_bags_eval = meta_bags_final
    if meta_bags_eval is not None and getattr(meta_bags_eval, "device", None) != cfg_loaded.device:
        meta_bags_eval = None
    if meta_bags_eval is None and cfg_loaded.use_text and cfg_loaded.meta_bags_on:
        npz_path = bundle.get("meta_bags_npz", cfg_loaded.meta_bags_npz) or ""
        if npz_path and os.path.exists(npz_path):
            meta_bags_eval = MetaBagsTorch(npz_path=npz_path, device=cfg_loaded.device)

    model = MRUJointModel(Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds, cfg=cfg_loaded, meta_bags=meta_bags_eval).to(cfg_loaded.device)
    model.load_state_dict(ckpt["model"], strict=True)

    ds_te = StayDataset(store_te, stays_te, cfg_loaded.max_len, cfg_loaded.truncate_mode)
    ld_te = make_loader(cfg_loaded, ds_te, shuffle=False, drop_last=False)
    test_pr = eval_pr_auc(model, ld_te, device=cfg_loaded.device)

    summary = {
        "fold_id": int(fold_id),
        "dims": {"Dx": Dx, "Du": Du, "De": De, "Dt": Dt, "Ds": Ds},
        "pool_stays": int(len(stays_pool)),
        "test_stays": int(len(stays_te)),
        "optuna_best_value_inner_mean_pr_auc": float(best.value),
        "optuna_best_params": best_params,
        "final_best_epoch": int(res_final["best_epoch"]),
        "final_best_val_total": float(res_final["best_val_total"]),
        "final_holdout_pr_auc": float(res_final["val_pr_auc"]),
        "outer_test_pr_auc": float(test_pr),
        "final_ckpt_path": str(res_final["best_path"]),
    }
    with open(os.path.join(out_fold, "nested_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[FOLD {fold_id}] ✅ HOLDOUT PR-AUC={summary['final_holdout_pr_auc']:.4f} | OUTER TEST PR-AUC={summary['outer_test_pr_auc']:.4f}")
    return summary


def main(cfg: CFG):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    if not cfg.use_text:
        cfg.meta_bags_on = False
        cfg.fuse_use_text = False
        cfg.encoder_use_text = True
        cfg.posterior_use_text = False

    cfg.text_source = str(cfg.text_source or "raw").lower()
    if cfg.text_source not in ("raw", "gru"):
        cfg.text_source = "raw"

    fold_dirs = list_fold_dirs(cfg.folds_dir)
    if cfg.only_fold >= 0:
        fold_dirs = [d for d in fold_dirs if os.path.basename(d) == f"fold_{cfg.only_fold}"]
        if not fold_dirs:
            raise FileNotFoundError(f"only_fold={cfg.only_fold} not found under {cfg.folds_dir}")

    all_res = {}
    for fd in fold_dirs[: cfg.folds]:
        fid = int(os.path.basename(fd).split("_")[1])
        # Optuna + Nested CV (POOL=train+val; inner CV for tuning; refit; outer test once)
        res = run_outer_fold_optuna_nested(fd, cfg)
        all_res[f"fold_{fid}"] = res

        if "cuda" in str(cfg.device):
            torch.cuda.empty_cache()

    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    vals = [v.get("outer_test_pr_auc", float("nan")) for v in all_res.values()]
    vals = [x for x in vals if np.isfinite(x)]
    if vals:
        print(f"\n[SUMMARY] mean 5-fold OUTER TEST PR-AUC = {float(np.mean(vals)):.4f} over {len(vals)} folds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fixed-bool version: encoder(text)=ON, posterior(text)=OFF, fuse(text)=OFF, fuse(x-skip)=ON."
    )

    parser.add_argument("--folds_dir", type=str, default=CFG.folds_dir)
    parser.add_argument("--out_dir", type=str, default=CFG.out_dir)
    parser.add_argument("--folds", type=int, default=CFG.folds)
    parser.add_argument("--only_fold", type=int, default=CFG.only_fold)

    parser.add_argument("--meta_bags_npz", type=str, default=CFG.meta_bags_npz)

    parser.add_argument("--device", type=str, default=CFG.device)
    parser.add_argument("--seed", type=int, default=CFG.seed)
    parser.add_argument("--epochs", type=int, default=CFG.epochs)
    parser.add_argument('--inner_epochs', type=int, default=CFG.inner_epochs, help='Max epochs for INNER CV folds (<=0 uses --epochs)')
    parser.add_argument('--outer_epochs', type=int, default=CFG.outer_epochs, help='Max epochs for OUTER/final refit (<=0 uses --epochs)')
    parser.add_argument("--patience", type=int, default=CFG.patience)
    parser.add_argument("--batch_size", type=int, default=CFG.batch_size)
    parser.add_argument("--max_len", type=int, default=CFG.max_len)
    parser.add_argument("--truncate_mode", type=str, default=CFG.truncate_mode, choices=["tail", "head"])

    parser.add_argument("--lr", type=float, default=CFG.lr)
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay)
    parser.add_argument("--dropout", type=float, default=CFG.dropout)
    parser.add_argument("--grad_clip", type=float, default=CFG.grad_clip)

    parser.add_argument("--lambda_death", type=float, default=CFG.lambda_death)
    parser.add_argument("--beta_kl", type=float, default=CFG.beta_kl)
    parser.add_argument("--kl_warmup_epochs", type=int, default=CFG.kl_warmup_epochs)

    parser.add_argument("--lambda_distill", type=float, default=CFG.lambda_distill)
    parser.add_argument("--distill_warmup_epochs", type=int, default=CFG.distill_warmup_epochs)

    parser.add_argument("--pos_weight_mult", type=float, default=CFG.pos_weight_mult)
    parser.add_argument("--pos_weight_clip_max", type=float, default=CFG.pos_weight_clip_max)
    parser.add_argument("--pos_weight_override", type=float, default=None)

    # ---- fuse head dims (required by MRUJointModel)
    parser.add_argument("--fuse_hidden", type=int, default=CFG.fuse_hidden)
    parser.add_argument("--fuse_layers", type=int, default=CFG.fuse_layers)

    # ---- Optuna + Nested CV
    parser.add_argument("--n_trials", type=int, default=CFG.n_trials)
    parser.add_argument("--inner_folds", type=int, default=CFG.inner_folds)
    parser.add_argument("--inner_parallel", action="store_true")
    parser.add_argument("--inner_jobs", type=int, default=CFG.inner_jobs)
    parser.add_argument("--inner_devices", type=str, default=CFG.inner_devices)
    parser.add_argument("--final_holdout_frac", type=float, default=CFG.final_holdout_frac)
    parser.add_argument("--optuna_seed", type=int, default=CFG.optuna_seed)
    parser.add_argument("--optuna_sampler", type=str, default=CFG.optuna_sampler)
    parser.add_argument("--optuna_pruner", type=str, default=CFG.optuna_pruner)
    parser.add_argument("--optuna_timeout_sec", type=int, default=CFG.optuna_timeout_sec)

    parser.add_argument("--exclude_cols", type=str, default=",".join(CFG.exclude_cols))
    parser.add_argument("--exclude_regex", type=str, default=",".join(CFG.exclude_regex))

    args = parser.parse_args()

    exclude_cols = tuple([c for c in (args.exclude_cols.split(",") if args.exclude_cols else []) if c])
    exclude_regex = tuple([c for c in (args.exclude_regex.split(",") if args.exclude_regex else []) if c])

    cfg = CFG(
        folds_dir=args.folds_dir,
        out_dir=args.out_dir,
        folds=args.folds,
        only_fold=args.only_fold,
        meta_bags_npz=args.meta_bags_npz,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        inner_epochs=args.inner_epochs,
        outer_epochs=args.outer_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_len=args.max_len,
        truncate_mode=args.truncate_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        lambda_death=args.lambda_death,
        beta_kl=args.beta_kl,
        kl_warmup_epochs=args.kl_warmup_epochs,
        lambda_distill=args.lambda_distill,
        distill_warmup_epochs=args.distill_warmup_epochs,
        pos_weight_mult=args.pos_weight_mult,
        pos_weight_clip_max=args.pos_weight_clip_max,
        pos_weight_override=args.pos_weight_override,
        fuse_hidden=args.fuse_hidden,
        fuse_layers=args.fuse_layers,
        use_optuna=True,
        n_trials=args.n_trials,
        inner_folds=args.inner_folds,
        inner_parallel=bool(args.inner_parallel),
        inner_jobs=int(args.inner_jobs),
        inner_devices=str(args.inner_devices or ""),
        final_holdout_frac=args.final_holdout_frac,
        optuna_seed=args.optuna_seed,
        optuna_sampler=args.optuna_sampler,
        optuna_pruner=args.optuna_pruner,
        optuna_timeout_sec=args.optuna_timeout_sec,
        exclude_cols=exclude_cols,
        exclude_regex=exclude_regex,
    )

    # ---- FIX: lambda_death is fixed to 1.0 (ignore CLI/Optuna) ----
    if hasattr(args, "lambda_death") and float(args.lambda_death) != 1.0:
        print(f"[FIX] overriding lambda_death={args.lambda_death} -> 1.0 (fixed)")
    cfg.lambda_death = 1.0

    # ---- enforce fixed boolean switches (safety) ----
    cfg.use_text = True
    cfg.meta_bags_on = True
    cfg.text_source = "raw"
    cfg.encoder_use_text = True
    cfg.encoder_txt_use_mask = True

    cfg.posterior_use_text = False
    cfg.posterior_txt_use_mask = False

    cfg.fuse_use_text = False

    cfg.fuse_use_x = True
    cfg.fuse_x_use_mask = True
    cfg.fuse_x_proj_dim = 64
    cfg.fuse_x_detach = False
    cfg.fuse_x_drop_p = 0.2
    cfg.fuse_x_no_proj = False

    cfg.fuse_type = "mlp"

    cfg.death_use_static = True
    cfg.use_smoother_train = True

    cfg.use_pos_weight = True

    cfg.scale_x = True
    cfg.scale_u = True
    cfg.scale_s_num = True
    cfg.scale_txt = False

    cfg.amp = True

    # ---- nested-CV epoch caps ----
    if int(getattr(cfg, 'inner_epochs', -1)) <= 0:
        cfg.inner_epochs = int(cfg.epochs)
    if int(getattr(cfg, 'outer_epochs', -1)) <= 0:
        cfg.outer_epochs = int(cfg.epochs)
    print(f"[EPOCHS] inner_epochs={cfg.inner_epochs} | outer_epochs={cfg.outer_epochs} | base_epochs={cfg.epochs}")


    print(
        "[FIXED BOOL CFG] "
        f"use_text={cfg.use_text} | meta_bags_on={cfg.meta_bags_on} | text_source={cfg.text_source} | "
        f"encoder_use_text={cfg.encoder_use_text} | posterior_use_text={cfg.posterior_use_text} | "
        f"fuse_use_text={cfg.fuse_use_text} | fuse_use_x={cfg.fuse_use_x} | fuse_type={cfg.fuse_type} | "
        f"use_smoother_train={cfg.use_smoother_train} | amp={cfg.amp}"
    )

    main(cfg)



