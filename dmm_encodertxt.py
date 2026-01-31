#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Refit-only runner (NO Optuna) that reuses per-fold best_params (ENCODER TEXT ON, POSTERIOR TEXT OFF).

Fixes vs v1:
- Holdout split seed matches base script: random_state = optuna_seed + fold_id.
- Tries to restore the *original fixed cfg* by loading ckpt["cfg"] from:
    {best_params_root}/fold_k/final_refit/final_best.pt
  so max_len/truncate_mode/pos_weight/etc match the run that produced best_params.
- Writes cfg_used.json + sanity_overlap.json per fold.

Example:
  python refit_notxt_from_external_bestparams_v2.py \
    --base_py dmm_optuna_cv_notxt.py \
    --folds_dir artifacts/folds_modeldata_plus_llmtext \
    --best_params_root artifacts/dmm_optuna_cv_1h10 \
    --out_dir artifacts/notxt_refit_from_textparams_v2 \
    --device cuda:0
"""

import os
import json
import argparse
import importlib.util
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def load_module(py_path: str):
    py_path = os.path.abspath(py_path)
    if not os.path.exists(py_path):
        raise FileNotFoundError(py_path)
    spec = importlib.util.spec_from_file_location("base_dmm_module", py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def safe_json_load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON must be a dict: {path}")
    return obj


def load_best_params_for_fold(best_params_root: str, fold_id: int, filename: str) -> Dict[str, Any]:
    best_params_root = os.path.abspath(best_params_root)
    if os.path.isfile(best_params_root):
        obj = safe_json_load(best_params_root)
        key = f"fold_{fold_id}"
        if key in obj and isinstance(obj[key], dict):
            return obj[key]
        if "optuna_best_params" in obj and isinstance(obj["optuna_best_params"], dict):
            return obj["optuna_best_params"]
        return obj

    fold_dir = os.path.join(best_params_root, f"fold_{fold_id}")
    cands = [
        os.path.join(fold_dir, filename),
        os.path.join(fold_dir, "optuna_best_params_tune7.json"),
        os.path.join(fold_dir, "optuna_best_params.json"),
        os.path.join(fold_dir, "best_params.json"),
    ]
    for p in cands:
        if os.path.exists(p):
            return safe_json_load(p)
    raise FileNotFoundError(f"best_params not found for fold_{fold_id} under {best_params_root}")


def try_load_reference_cfg(best_params_root: str, fold_id: int) -> Optional[Dict[str, Any]]:
    """Load ckpt['cfg'] from original run's final_refit to reproduce fixed settings."""
    best_params_root = os.path.abspath(best_params_root)
    if not os.path.isdir(best_params_root):
        return None
    fold_final = os.path.join(best_params_root, f"fold_{fold_id}", "final_refit")
    cands = [
        os.path.join(fold_final, "final_best.pt"),
        os.path.join(fold_final, "final_best.pth"),
        os.path.join(fold_final, "final_best_model.pt"),
        os.path.join(fold_final, "best.pt"),
    ]
    for p in cands:
        if os.path.exists(p):
            try:
                import torch
                ckpt = torch.load(p, map_location="cpu")
                cfg = ckpt.get("cfg")
                if isinstance(cfg, dict):
                    return cfg
            except Exception:
                pass
    return None


def enforce_encodertxt_cfg(cfg: Any) -> None:
    """Force: text -> encoder only (posterior OFF, fuse OFF).
    Assumes meta_bags.npz exists under folds_dir.
    """
    cfg.use_text = True
    cfg.meta_bags_on = True
    cfg.text_source = "raw"
    cfg.encoder_use_text = True
    cfg.encoder_txt_use_mask = True  # recommended (mask/presence)
    cfg.posterior_use_text = False
    cfg.posterior_txt_use_mask = False
    cfg.fuse_use_text = False
    # scaling: keep cfg default unless you want to force off
    if hasattr(cfg, "scale_txt"):
        # keep as-is if already set; uncomment next line to force:
        # cfg.scale_txt = False
        pass


def merge_cfg_from_reference(mod: Any, cfg_template: Any, ref_cfg: Optional[Dict[str, Any]]) -> Any:
    if ref_cfg is None:
        return mod.clone_cfg(cfg_template)
    base = asdict(cfg_template)
    merged = dict(base)
    for k, v in ref_cfg.items():
        if k in base:
            merged[k] = v
    cfg_new = mod.CFG(**merged)
    # keep these from template
    cfg_new.folds_dir = cfg_template.folds_dir
    cfg_new.out_dir = cfg_template.out_dir
    cfg_new.folds = cfg_template.folds
    cfg_new.only_fold = cfg_template.only_fold
    cfg_new.device = cfg_template.device
    cfg_new.seed = cfg_template.seed
    return cfg_new


def overlap_stats(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    inter = np.intersect1d(a, b)
    denom = float(min(len(np.unique(a)), len(np.unique(b))) or 1)
    return int(len(inter)), float(len(inter) / denom)



def load_meta_bags_torch(mod, folds_dir: str, device: str):
    """Load meta_bags.npz and convert it to MetaBagsTorch (expected by base_py).

    Passing np.load(...) (NpzFile) will crash because it has no .bag_names attribute.
    """
    path = os.path.join(os.path.abspath(folds_dir), "meta_bags.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"meta_bags.npz not found under folds_dir: {path}")
    if not hasattr(mod, "MetaBagsTorch"):
        raise AttributeError("base_py has no MetaBagsTorch. Use a compatible base_py (the DMM scripts include it).")
    return mod.MetaBagsTorch(path, device)


def run_one_fold(mod: Any, cfg_template: Any, fold_dir: str, best_params_root: str, best_params_filename: str) -> Dict[str, Any]:
    fold_name = os.path.basename(os.path.abspath(fold_dir))
    fold_id = int(fold_name.split("_")[1])

    out_fold = os.path.join(cfg_template.out_dir, f"fold_{fold_id}")
    os.makedirs(out_fold, exist_ok=True)

    ref_cfg = try_load_reference_cfg(best_params_root, fold_id)
    cfg_base = merge_cfg_from_reference(mod, cfg_template, ref_cfg)
    enforce_encodertxt_cfg(cfg_base)

    bundle = mod.load_fold_bundle(cfg_base, fold_dir)

    # pool
    df_pool = mod.pd.concat([bundle["df_tr"], bundle["df_va"]], axis=0, ignore_index=True)
    store_pool = mod.ArrayStore(
        df_pool,
        bundle["x_cols"], bundle["x_mask_cols"],
        bundle["u_cols"], bundle["e_cols"], bundle["txt_cols"],
        bundle["static_cols"],
        bundle["y_col"], bundle["y_mask_col"], bundle["time_mask_col"],
        stay_col=bundle["stay_col"], time_col=bundle["time_col"],
    )
    stays_pool = mod.sorted_unique_stays(df_pool, stay_col=bundle["stay_col"])
    y_pool = mod.stay_level_labels_from_store(store_pool, stays_pool)

    store_te = bundle["store_te"]
    stays_te = bundle["stays_te"]

    Dx, Du, De, Dt, Ds = map(len, [bundle["x_cols"], bundle["u_cols"], bundle["e_cols"], bundle["txt_cols"], bundle["static_cols"]])

    # params
    best_params = load_best_params_for_fold(best_params_root, fold_id, best_params_filename)
    with open(os.path.join(out_fold, "external_best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    cfg_best = mod.clone_cfg(cfg_base)
    mod.apply_trial_params(cfg_best, dict(best_params))
    enforce_encodertxt_cfg(cfg_best)

    with open(os.path.join(out_fold, "cfg_used.json"), "w") as f:
        json.dump(asdict(cfg_best), f, indent=2)

    n_ov, r_ov = overlap_stats(stays_pool, stays_te)
    with open(os.path.join(out_fold, "sanity_overlap.json"), "w") as f:
        json.dump({
            "fold_id": fold_id,
            "pool_stays": int(len(stays_pool)),
            "test_stays": int(len(stays_te)),
            "pool_test_overlap_n": int(n_ov),
            "pool_test_overlap_rate_vs_minset": float(r_ov),
        }, f, indent=2)

    if n_ov > 0:
        print(f"[FOLD {fold_id}] ⚠️ POOL∩TEST overlap stays={n_ov} (rate={r_ov:.4f})")

    # holdout split (match base script)
    test_size = float(getattr(cfg_best, "final_holdout_frac", 0.15))
    if (not np.isfinite(test_size)) or test_size <= 0 or test_size >= 0.5:
        test_size = 0.15
    optuna_seed = int(getattr(cfg_best, "optuna_seed", getattr(cfg_best, "seed", 0)))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=optuna_seed + fold_id)
    tr_idx, va_idx = next(sss.split(stays_pool, y_pool))
    stays_tr = stays_pool[tr_idx]
    stays_va = stays_pool[va_idx]

    out_final = os.path.join(out_fold, "final_refit")
    os.makedirs(out_final, exist_ok=True)


    # meta-bags (EmbeddingBag ids/offsets) for text encoder
    meta_bags = load_meta_bags_torch(mod, cfg_best.folds_dir, cfg_best.device)

    res_final = mod.train_on_split(
        cfg=cfg_best,
        max_epochs=getattr(cfg_best, "outer_epochs", cfg_best.epochs),
        store_tr=store_pool,
        stays_train=stays_tr,
        store_va=store_pool,
        stays_val=stays_va,
        Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds,
        out_dir=out_final,
        tag="final",
        device=cfg_best.device,
        meta_bags=meta_bags,
    )

    ckpt = mod.torch.load(res_final["best_path"], map_location=cfg_best.device)
    cfg_loaded = mod.CFG(**ckpt["cfg"])
    enforce_encodertxt_cfg(cfg_loaded)
    # cfg_loaded.scale_txt kept as saved

    model = mod.MRUJointModel(Dx=Dx, Du=Du, De=De, Dt=Dt, Ds=Ds, cfg=cfg_loaded, meta_bags=meta_bags).to(cfg_loaded.device)
    model.load_state_dict(ckpt["model"], strict=True)

    ds_te = mod.StayDataset(store_te, stays_te, cfg_loaded.max_len, cfg_loaded.truncate_mode)
    ld_te = mod.make_loader(cfg_loaded, ds_te, shuffle=False, drop_last=False)
    test_pr = mod.eval_pr_auc(model, ld_te, device=cfg_loaded.device)

    summary = {
        "fold_id": fold_id,
        "dims": {"Dx": Dx, "Du": Du, "De": De, "Dt": Dt, "Ds": Ds},
        "pool_stays": int(len(stays_pool)),
        "test_stays": int(len(stays_te)),
        "pool_test_overlap_stays": int(n_ov),
        "pool_test_overlap_rate_vs_minset": float(r_ov),
        "reference_cfg_loaded_from_ckpt": bool(ref_cfg is not None),
        "final_holdout_pr_auc": float(res_final["val_pr_auc"]),
        "outer_test_pr_auc": float(test_pr),
        "final_ckpt_path": str(res_final["best_path"]),
        "text_mode": "encoder_on_posterior_off_fuse_off",
    }
    with open(os.path.join(out_fold, "refit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[FOLD {fold_id}] ✅ HOLDOUT PR-AUC={summary['final_holdout_pr_auc']:.4f} | OUTER TEST PR-AUC={summary['outer_test_pr_auc']:.4f}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_py", type=str, default="dmm_optuna_cv_notxt.py")
    ap.add_argument("--folds_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--best_params_root", type=str, required=True)
    ap.add_argument("--best_params_filename", type=str, default="optuna_best_params_tune7.json")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=-1)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    mod = load_module(args.base_py)

    cfg = mod.CFG()
    cfg.folds_dir = args.folds_dir
    cfg.out_dir = args.out_dir
    cfg.folds = int(args.folds)
    cfg.only_fold = int(args.only_fold)
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = int(args.seed)

    mod.set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    fold_dirs = mod.list_fold_dirs(cfg.folds_dir)
    if cfg.only_fold >= 0:
        fold_dirs = [d for d in fold_dirs if os.path.basename(d) == f"fold_{cfg.only_fold}"]
        if not fold_dirs:
            raise FileNotFoundError(f"only_fold={cfg.only_fold} not found under {cfg.folds_dir}")

    all_res: Dict[str, Any] = {}
    for fd in fold_dirs[: cfg.folds]:
        fid = int(os.path.basename(fd).split("_")[1])
        res = run_one_fold(mod, cfg, fd, args.best_params_root, args.best_params_filename)
        all_res[f"fold_{fid}"] = res
        if "cuda" in str(cfg.device):
            mod.torch.cuda.empty_cache()

    with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)

    vals = [v.get("outer_test_pr_auc", float("nan")) for v in all_res.values()]
    vals = [x for x in vals if np.isfinite(x)]
    if vals:
        print(f"\n[SUMMARY] mean OUTER TEST PR-AUC = {float(np.mean(vals)):.4f} over {len(vals)} folds")


if __name__ == "__main__":
    main()
