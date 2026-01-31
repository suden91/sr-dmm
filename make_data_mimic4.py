# make_data_mimic4.py
# ============================================================
# End-to-end builder (NO-LOCF / NO-CARRY-FORWARD):
# (A) df3 생성: SOFA(hourly) -> configurable grid_hours + death label horizon_hours
# (B) static features merge
# (C) vitals last + obs/missing indicators   (✅ LOCF 제거)
# (D) labs   last + obs/missing indicators   (✅ LOCF 제거)
# (E) inputevents/procedureevents: category-any features       (✅ carry-forward 제거)
# (F) 저장: model_data_mimic4.csv
#
# 핵심 업데이트:
#   - cfg.grid_hours: feature aggregation grid (e.g., 1h)
#   - cfg.horizon_hours: prediction window length for death label (e.g., 24h)
#   - 기존 "24" 고정들을 grid_hours/horizon_hours로 일반화
# ============================================================

from __future__ import annotations

import os
import re
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, List

# ============================================================
# 0) Config
# ============================================================
@dataclass
class CFG:
    # ---- MIMIC-IV root ----
    mimic_iv_dir: str = "mimiciv/3.1"   # 환경에 맞게 수정
    sofa_hourly_path: str = "sofa.csv"  # stay_id, hr(or timestamps), ... sofa_24hours 등
    first_day_sofa_path: str = "first_day_sofa.csv"

    # ---- output ----
    out_csv_path: str = "model_data_mimic4.csv"
    out_parquet_path: Optional[str] = None

    # ---- time/grid ----
    grid_hours: int = 1          # ✅ 1로 바꾸면 1시간 그리드
    horizon_hours: int = 24       # ✅ "24시간 내 사망"을 의미 (그리드와 독립)

    # ---- death label ----
    use_patient_dod: bool = False  # 기본 False 권장

    # ---- chunk ----
    chunksize: int = 2_000_000

    # ---- interventions(any) ----
    add_interventions: bool = True
    input_map_csv: str = "icu_inputevents_itemid_counts.csv"
    proc_map_csv: str = "icu_procedureevents_itemid_counts.csv"
    min_item_n: int = 0
    keep_proc_categories_regex: Optional[str] = None

    # ---- debug ----
    debug: bool = True


# ============================================================
# 1) Paths helper
# ============================================================
def _paths(cfg: CFG):
    hosp_dir = os.path.join(cfg.mimic_iv_dir, "hosp")
    icu_dir  = os.path.join(cfg.mimic_iv_dir, "icu")

    return {
        "HOSP_DIR": hosp_dir,
        "ICU_DIR": icu_dir,
        "ICUSTAYS_PATH": os.path.join(icu_dir, "icustays.csv.gz"),
        "ADMISSIONS_PATH": os.path.join(hosp_dir, "admissions.csv.gz"),
        "PATIENTS_PATH": os.path.join(hosp_dir, "patients.csv.gz"),
        "CHARTEVENTS_PATH": os.path.join(icu_dir, "chartevents.csv.gz"),
        "D_ITEMS_PATH": os.path.join(icu_dir, "d_items.csv.gz"),
        "LABEVENTS_PATH": os.path.join(hosp_dir, "labevents.csv.gz"),
        "D_LABITEMS_PATH": os.path.join(hosp_dir, "d_labitems.csv.gz"),
        "INPUTEVENTS_PATH": os.path.join(icu_dir, "inputevents.csv.gz"),
        "PROCEDUREEVENTS_PATH": os.path.join(icu_dir, "procedureevents.csv.gz"),
    }


# ============================================================
# 2) Common utils
# ============================================================
def _to_int_series(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def _safe_datetime(s):
    return pd.to_datetime(s, errors="coerce")

def _ensure_cols(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def _build_mask(series: pd.Series, patterns: list[str]) -> pd.Series:
    s = series.astype(str)
    mask = pd.Series(False, index=s.index)
    for p in patterns:
        mask |= s.str.contains(p, case=False, regex=True, na=False)
    return mask

def _filter_plausible_ranges(df, varname_col, value_col="valuenum"):
    v = pd.to_numeric(df[value_col], errors="coerce").astype(float)
    rules = {
        "lactate":    (0.0, 30.0),
        "creatinine": (0.0, 20.0),
        "bilirubin":  (0.0, 60.0),
        "wbc":        (0.0, 300.0),
        "ph":         (6.5, 8.0),

        "hr":   (0.0, 300.0),
        "rr":   (0.0, 120.0),
        "spo2": (0.0, 100.0),
        "temp": (25.0, 45.0),
        "sbp":  (0.0, 300.0),
        "dbp":  (0.0, 200.0),
        "map":  (0.0, 250.0),
    }
    keep = pd.Series(True, index=df.index)
    for name, (lo, hi) in rules.items():
        m = df[varname_col].eq(name)
        if m.any():
            keep.loc[m] = (v.loc[m] >= lo) & (v.loc[m] <= hi)
    return df[keep]

def _assert_unique_cols(df: pd.DataFrame, where: str):
    if not df.columns.is_unique:
        dup = df.columns[df.columns.duplicated()].tolist()
        raise RuntimeError(f"[{where}] Duplicate column names detected: {dup}")

def _merge_drop_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    on,
    how: str = "left",
    validate: Optional[str] = None,
    where: str = "merge",
) -> pd.DataFrame:
    on_set = set([on] if isinstance(on, str) else list(on))
    overlap = [c for c in right.columns if (c in left.columns) and (c not in on_set)]
    if overlap:
        right = right.drop(columns=overlap)
    out = left.merge(right, on=on, how=how, validate=validate)
    _assert_unique_cols(out, where)
    return out

def debug_death_and_label(df: pd.DataFrame, cfg: CFG, tag: str = ""):
    print(f"\n========== DEBUG REPORT {tag} ==========")
    if "stay_id" in df.columns:
        print("rows:", len(df), "stays:", df["stay_id"].nunique())
    else:
        print("rows:", len(df))

    if "y_death_nextgrid" in df.columns and "stay_id" in df.columns:
        y = pd.to_numeric(df["y_death_nextgrid"], errors="coerce").fillna(0)
        print("row-level y pos rate:", float((y == 1).mean()))
        s = df.assign(_y=y).groupby("stay_id")["_y"].max()
        print("stay-level any y=1:", int((s == 1).sum()), "/", len(s), f"({float((s==1).mean()):.4%})")

    for c in ["death_time_inhosp", "death_hr_inhosp", "death_time", "death_hr"]:
        if c in df.columns:
            if "time" in c:
                nn = int(pd.to_datetime(df[c], errors="coerce").notna().sum())
            else:
                nn = int(pd.to_numeric(df[c], errors="coerce").notna().sum())
            print(f"{c} non-null rows:", nn, "/", len(df))

    if all(c in df.columns for c in ["stay_id", "baseline_time"]):
        bt = pd.to_datetime(df["baseline_time"], errors="coerce")
        H = int(cfg.horizon_hours)

        def _count_within(time_col):
            if time_col not in df.columns:
                return None
            dt = pd.to_datetime(df[time_col], errors="coerce")
            within = dt.notna() & (dt >= bt) & (dt <= bt + pd.Timedelta(hours=H))
            return int(within.groupby(df["stay_id"]).max().sum())

        for tc in ["death_time_inhosp", "death_time"]:
            v = _count_within(tc)
            if v is not None:
                print(f"stays death within {H}h by {tc}:", v)

def build_columns_catalog(df: pd.DataFrame, cfg: CFG) -> dict:
    cols = list(df.columns)

    id_cols = [c for c in ["subject_id","hadm_id","stay_id","grid","hr","row_id"] if c in df.columns]
    time_cols = [c for c in ["baseline_time","intime","outtime","admittime","dischtime","edregtime","edouttime"] if c in df.columns]

    label_cols = [c for c in ["y_death_nextgrid"] if c in df.columns]
    label_aux_cols = [c for c in [
        "next_start_hr","next_end_hr",
        "icu_los_hours",
        "death_time_inhosp","death_hr_inhosp",
        "death_time","death_hr",
        "hospital_expire_flag",
    ] if c in df.columns]

    raw6 = getattr(cfg, "sofa_raw_vital_cols_fixed", None)
    if raw6 is None:
        raw6 = (
            "pao2fio2ratio_vent","platelet_min","bilirubin_max",
            "meanbp_min","gcs_min","creatinine_max",
        )
    raw6 = [c for c in list(raw6) if c in df.columns]
    raw6_masks = [f"mask_{c}" for c in raw6 if f"mask_{c}" in df.columns]

    sofa_score_like = [c for c in cols if (
        c in {
            "respiration","coagulation","liver","cardiovascular","cns","renal",
            "respiration_24hours","coagulation_24hours","liver_24hours",
            "cardiovascular_24hours","cns_24hours","renal_24hours",
            "sofa_24hours",
        }
    )]
    sofa_derived = [c for c in ["sofa_24hours_cummean","sofa_24hours_cummax"] if c in df.columns]
    sofa_firstday = [c for c in cols if c.startswith("fd_")]

    vit_last    = [c for c in cols if c.startswith("vs_") and c.endswith("_last")]
    vit_obs     = [c for c in cols if c.startswith("vs_") and c.endswith("_obs")]
    vit_missing = [c for c in cols if c.startswith("vs_") and c.endswith("_missing")]

    lab_last    = [c for c in cols if c.startswith("lb_") and c.endswith("_last")]
    lab_obs     = [c for c in cols if c.startswith("lb_") and c.endswith("_obs")]
    lab_missing = [c for c in cols if c.startswith("lb_") and c.endswith("_missing")]

    in_cols = [c for c in cols if c.startswith("in_cat__")]
    pr_cols = [c for c in cols if c.startswith("pr_cat__")]

    reserved = set(
        id_cols + time_cols + label_cols + label_aux_cols +
        raw6 + raw6_masks + sofa_score_like + sofa_derived + sofa_firstday +
        vit_last + vit_obs + vit_missing +
        lab_last + lab_obs + lab_missing +
        in_cols + pr_cols
    )

    static_candidates = [c for c in cols if c not in reserved]

    static_cat = []
    static_num = []
    static_time_extra = []
    for c in static_candidates:
        dt = df[c].dtype
        if np.issubdtype(dt, np.datetime64):
            static_time_extra.append(c)
        elif pd.api.types.is_bool_dtype(dt) or pd.api.types.is_object_dtype(dt) or pd.api.types.is_categorical_dtype(dt):
            static_cat.append(c)
        else:
            if pd.api.types.is_numeric_dtype(dt):
                static_num.append(c)
            else:
                static_cat.append(c)

    time_cols = sorted(set(time_cols + static_time_extra))

    grouped_all = set().union(
        id_cols, time_cols, label_cols, label_aux_cols,
        raw6, raw6_masks, sofa_score_like, sofa_derived, sofa_firstday,
        vit_last, vit_obs, vit_missing,
        lab_last, lab_obs, lab_missing,
        in_cols, pr_cols,
        static_cat, static_num
    )
    other_cols = [c for c in cols if c not in grouped_all]

    dtypes = {c: str(df[c].dtype) for c in cols}

    groups = {
        "id": id_cols,
        "time": time_cols,
        "label": label_cols,
        "label_aux": label_aux_cols,

        "sofa_raw6": raw6,
        "sofa_raw6_masks": raw6_masks,
        "sofa_scores": sofa_score_like,
        "sofa_derived": sofa_derived,
        "sofa_firstday": sofa_firstday,

        "vitals_last": vit_last,
        "vitals_obs": vit_obs,
        "vitals_missing": vit_missing,

        "labs_last": lab_last,
        "labs_obs": lab_obs,
        "labs_missing": lab_missing,

        "interventions_input": in_cols,
        "interventions_procedure": pr_cols,

        "static_categorical": static_cat,
        "static_numeric": static_num,

        "other": other_cols,
    }

    counts = {k: len(v) for k, v in groups.items()}
    return {
        "n_cols_total": len(cols),
        "groups": groups,
        "counts": counts,
        "dtypes": dtypes,
    }


# ============================================================
# 3) (A) df3 생성: SOFA hourly -> grid + label
# ============================================================
def build_df3_from_sofa(cfg: CFG, P: dict) -> pd.DataFrame:
    icu = pd.read_csv(
        P["ICUSTAYS_PATH"],
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
        low_memory=False,
    )
    adm = pd.read_csv(
        P["ADMISSIONS_PATH"],
        usecols=["subject_id", "hadm_id", "deathtime"],
        parse_dates=["deathtime"],
        low_memory=False,
    )
    cohort = icu.merge(adm, on=["subject_id", "hadm_id"], how="left")

    if cfg.use_patient_dod:
        pat = pd.read_csv(
            P["PATIENTS_PATH"],
            usecols=["subject_id", "dod"],
            parse_dates=["dod"],
            low_memory=False,
        )
        cohort = cohort.merge(pat, on=["subject_id"], how="left")
    else:
        cohort["dod"] = pd.NaT

    cohort["baseline_time"] = cohort["intime"]
    cohort["icu_los_hours"] = (cohort["outtime"] - cohort["baseline_time"]).dt.total_seconds() / 3600.0

    cohort["death_time_inhosp"] = cohort["deathtime"]
    cohort["death_hr_inhosp"] = (cohort["death_time_inhosp"] - cohort["baseline_time"]).dt.total_seconds() / 3600.0

    cohort["death_time"] = cohort["deathtime"].combine_first(cohort["dod"])
    cohort["death_hr"] = (cohort["death_time"] - cohort["baseline_time"]).dt.total_seconds() / 3600.0

    for c in ["death_hr_inhosp", "death_hr"]:
        cohort.loc[(cohort[c] < 0) | (cohort[c] > 1e6), c] = np.nan

    cohort_base = cohort[
        ["subject_id","hadm_id","stay_id","baseline_time","outtime","icu_los_hours",
         "death_time_inhosp","death_hr_inhosp","death_time","death_hr"]
    ].copy()

    # SOFA hourly load
    sofa = pd.read_parquet(cfg.sofa_hourly_path) if cfg.sofa_hourly_path.endswith(".parquet") else pd.read_csv(cfg.sofa_hourly_path, low_memory=False)
    if "stay_id" not in sofa.columns and "icustay_id" in sofa.columns:
        sofa = sofa.rename(columns={"icustay_id": "stay_id"})
    _ensure_cols(sofa, ["stay_id"])

    sofa = sofa.copy()
    sofa["stay_id"] = _to_int_series(sofa["stay_id"])
    sofa = sofa.dropna(subset=["stay_id"]).copy()
    sofa["stay_id"] = sofa["stay_id"].astype(int)

    # hr: either provided as integer hours from ICU intime, or inferred from timestamps
    if "hr" in sofa.columns:
        sofa["hr"] = _to_int_series(sofa["hr"])
        sofa = sofa.dropna(subset=["hr"]).copy()
        sofa["hr"] = sofa["hr"].astype(int)
    else:
        time_candidates = [c for c in ["endtime", "charttime", "starttime"] if c in sofa.columns]
        if not time_candidates:
            raise RuntimeError(
                "[DF3] sofa_hourly_path must contain either an integer 'hr' column "
                "or one of the timestamp columns: endtime/charttime/starttime."
            )
        tcol = time_candidates[0]
        sofa[tcol] = _safe_datetime(sofa[tcol])
        sofa = sofa.dropna(subset=[tcol]).copy()

        base = cohort_base[["stay_id", "baseline_time"]].drop_duplicates("stay_id").copy()
        sofa = sofa.merge(base, on="stay_id", how="left")
        sofa = sofa[sofa["baseline_time"].notna()].copy()

        sofa["hr"] = ((sofa[tcol] - sofa["baseline_time"]).dt.total_seconds() // 3600).astype("Int64")
        sofa = sofa.dropna(subset=["hr"]).copy()
        sofa["hr"] = sofa["hr"].astype(int)
        sofa = sofa.drop(columns=["baseline_time"])

    sofa = sofa[sofa["hr"] >= 0].copy()

    # ✅ configurable grid
    GH = int(cfg.grid_hours)
    sofa["grid"] = (sofa["hr"] // GH).astype(int)

    # raw6 required
    raw6 = getattr(cfg, "sofa_raw_vital_cols_fixed", None)
    if raw6 is None:
        raw6 = (
            "pao2fio2ratio_vent",
            "platelet_min",
            "bilirubin_max",
            "meanbp_min",
            "gcs_min",
            "creatinine_max",
        )
    raw6 = [str(c) for c in raw6]

    missing_raw = [c for c in raw6 if c not in sofa.columns]
    if missing_raw:
        raise RuntimeError(
            f"[DF3] Missing required SOFA raw vital columns in sofa_hourly: {missing_raw}. "
            f"Available examples={list(sofa.columns[:60])}"
        )

    sofa_score_cols = [
        "respiration","coagulation","liver","cardiovascular","cns","renal",
        "respiration_24hours","coagulation_24hours","liver_24hours","cardiovascular_24hours","cns_24hours","renal_24hours",
        "sofa_24hours",
    ]
    sofa_score_cols = [c for c in sofa_score_cols if c in sofa.columns]

    # grid별 마지막 hr 대표값
    keep_cols = ["stay_id", "grid", "hr"] + sofa_score_cols + raw6
    keep_cols = list(dict.fromkeys(keep_cols))
    idx_last = sofa.groupby(["stay_id", "grid"], sort=False)["hr"].idxmax()
    sofa_grid_last = sofa.loc[idx_last, keep_cols].copy()

    # raw: 0 fill + mask
    for c in raw6:
        sofa_grid_last[c] = pd.to_numeric(sofa_grid_last[c], errors="coerce")
        sofa_grid_last[f"mask_{c}"] = (~sofa_grid_last[c].isna()).astype(np.float32)
        sofa_grid_last[c] = sofa_grid_last[c].fillna(0.0).astype(np.float32)

    for c in sofa_score_cols:
        sofa_grid_last[c] = pd.to_numeric(sofa_grid_last[c], errors="coerce").astype(np.float32)

    df = sofa_grid_last.merge(cohort_base, on="stay_id", how="left")

    # ✅ label window: "next_start" is next grid boundary, "next_end" is + horizon_hours
    #    (grid_hours=1, horizon=24 => [h+1, h+25) )
    H = int(cfg.horizon_hours)
    df["next_start_hr"] = (df["grid"] + 1) * GH
    df["next_end_hr"]   = df["next_start_hr"] + H

    df["y_death_nextgrid"] = (
        df["death_hr_inhosp"].notna()
        & (df["death_hr_inhosp"] >= df["next_start_hr"])
        & (df["death_hr_inhosp"] <  df["next_end_hr"])
    ).astype("int8")

    # at-risk filter (must be in ICU at next_start)
    df = df[df["icu_los_hours"].notna() & (df["icu_los_hours"] >= df["next_start_hr"])].copy()
    df = df[df["death_hr_inhosp"].isna() | (df["death_hr_inhosp"] >= df["next_start_hr"])].copy()

    df = df.sort_values(["stay_id", "grid"]).reset_index(drop=True)

    if "sofa_24hours" in df.columns:
        g = df.groupby("stay_id", sort=False)
        df["sofa_24hours_cummean"] = g["sofa_24hours"].expanding().mean().reset_index(level=0, drop=True)
        df["sofa_24hours_cummax"]  = g["sofa_24hours"].cummax()

    # first-day sofa join
    if cfg.first_day_sofa_path:
        fd = pd.read_parquet(cfg.first_day_sofa_path) if cfg.first_day_sofa_path.endswith(".parquet") else pd.read_csv(cfg.first_day_sofa_path, low_memory=False)
        if "stay_id" not in fd.columns and "icustay_id" in fd.columns:
            fd = fd.rename(columns={"icustay_id": "stay_id"})
        _ensure_cols(fd, ["stay_id"])
        fd = fd.copy()
        fd["stay_id"] = _to_int_series(fd["stay_id"])
        fd = fd.dropna(subset=["stay_id"])
        fd["stay_id"] = fd["stay_id"].astype(int)

        fd_cols = [c for c in ["sofa","respiration","coagulation","liver","cardiovascular","cns","renal"] if c in fd.columns]
        fd = fd.sort_values("stay_id").drop_duplicates("stay_id", keep="first")
        fd = fd[["stay_id"] + fd_cols].rename(columns={c: f"fd_{c}" for c in fd_cols})

        df = df.merge(fd, on="stay_id", how="left")

    _assert_unique_cols(df, "df3:build_df3_from_sofa")
    return df


# ============================================================
# 4) (B) Static features
# ============================================================
def add_static_features(df: pd.DataFrame, P: dict) -> pd.DataFrame:
    patients = pd.read_csv(
        P["PATIENTS_PATH"],
        usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
        low_memory=False,
    )
    admissions = pd.read_csv(
        P["ADMISSIONS_PATH"],
        usecols=[
            "subject_id","hadm_id",
            "admission_type","admission_location","discharge_location",
            "insurance","language","marital_status","race",
            "hospital_expire_flag",
            "edregtime","edouttime","admittime","dischtime"
        ],
        parse_dates=["edregtime","edouttime","admittime","dischtime"],
        low_memory=False,
    )
    icustays = pd.read_csv(
        P["ICUSTAYS_PATH"],
        usecols=["subject_id","hadm_id","stay_id","first_careunit","last_careunit","los"],
        low_memory=False,
    )

    admissions["ed_los_hours"]  = (admissions["edouttime"] - admissions["edregtime"]).dt.total_seconds() / 3600.0
    admissions["hosp_los_days"] = (admissions["dischtime"] - admissions["admittime"]).dt.total_seconds() / (3600 * 24)

    out = _merge_drop_overlap(
        df, icustays,
        on=["subject_id","hadm_id","stay_id"],
        how="left",
        validate="m:1",
        where="static:icustays"
    )

    out = _merge_drop_overlap(
        out, admissions.drop(columns=["subject_id"]),
        on="hadm_id",
        how="left",
        validate="m:1",
        where="static:admissions"
    )

    out = _merge_drop_overlap(
        out, patients,
        on="subject_id",
        how="left",
        validate="m:1",
        where="static:patients"
    )

    _assert_unique_cols(out, "static:final")
    return out


# ============================================================
# 5) (C) Vitals last + indicators (NO-LOCF)
# ============================================================
def add_vitals_last_with_indicators(df: pd.DataFrame, cfg: CFG, P: dict) -> pd.DataFrame:
    ditems = pd.read_csv(P["D_ITEMS_PATH"], usecols=["itemid","label","linksto"], low_memory=False)
    ditems = ditems[ditems["linksto"].eq("chartevents")].copy()
    ditems["label"] = ditems["label"].astype(str)

    VITAL_PATTERNS = {
        "hr":   [r"\bHeart Rate\b"],
        "rr":   [r"\bRespiratory Rate\b"],
        "spo2": [r"\bSpO2\b", r"O2 Saturation"],
        "temp": [r"Temperature"],
        "sbp":  [r"Blood Pressure systolic"],
        "dbp":  [r"Blood Pressure diastolic"],
        "map":  [r"Blood Pressure mean", r"\bMean BP\b", r"\bMAP\b"],
    }

    itemid_to_vital = {}
    for vital, pats in VITAL_PATTERNS.items():
        mask = _build_mask(ditems["label"], pats)
        for itemid in ditems.loc[mask, "itemid"].astype(int).tolist():
            itemid_to_vital.setdefault(itemid, vital)

    selected_itemids = set(itemid_to_vital.keys())
    if not selected_itemids:
        raise ValueError("No vital itemids matched. Check patterns/d_items.")

    stay_time = (
        df[["stay_id","baseline_time","outtime"]]
        .drop_duplicates(subset=["stay_id"])
        .rename(columns={"baseline_time": "intime"})
        .copy()
    )
    stay_time["stay_id"] = pd.to_numeric(stay_time["stay_id"], errors="coerce")
    stay_time = stay_time.dropna(subset=["stay_id","intime"]).copy()
    stay_time["stay_id"] = stay_time["stay_id"].astype(int)

    stay_id_set = set(stay_time["stay_id"].tolist())
    max_grid_by_stay = df.groupby("stay_id")["grid"].max().to_dict()

    USECOLS = ["stay_id","charttime","itemid","valuenum","valueuom"]
    parts = []

    GH = int(cfg.grid_hours)

    for ch in pd.read_csv(
        P["CHARTEVENTS_PATH"], usecols=USECOLS, chunksize=cfg.chunksize, low_memory=False
    ):
        ch["stay_id"] = pd.to_numeric(ch["stay_id"], errors="coerce")
        ch = ch[ch["stay_id"].notna()]
        ch["stay_id"] = ch["stay_id"].astype(int)
        ch = ch[ch["stay_id"].isin(stay_id_set)]
        if ch.empty:
            continue

        ch["itemid"] = pd.to_numeric(ch["itemid"], errors="coerce")
        ch = ch[ch["itemid"].notna()]
        ch["itemid"] = ch["itemid"].astype(int)
        ch = ch[ch["itemid"].isin(selected_itemids)]
        if ch.empty:
            continue

        ch["charttime"] = _safe_datetime(ch["charttime"])
        ch["valuenum"] = pd.to_numeric(ch["valuenum"], errors="coerce")
        ch = ch[ch["charttime"].notna() & ch["valuenum"].notna()]
        if ch.empty:
            continue

        tmp = ch.merge(stay_time, on="stay_id", how="left")
        tmp = tmp[tmp["intime"].notna()]
        tmp = tmp[tmp["charttime"] >= tmp["intime"]]
        tmp = tmp[tmp["outtime"].isna() | (tmp["charttime"] <= tmp["outtime"])]
        if tmp.empty:
            continue

        hr = ((tmp["charttime"] - tmp["intime"]).dt.total_seconds() // 3600).astype("int64")
        tmp = tmp[hr >= 0].copy()
        tmp["hr"] = hr[hr >= 0]

        # ✅ configurable grid
        tmp["grid"] = (tmp["hr"] // GH).astype(int)

        tmp["max_grid"] = tmp["stay_id"].map(max_grid_by_stay)
        tmp = tmp[tmp["max_grid"].notna() & (tmp["grid"] <= tmp["max_grid"])].copy()
        if tmp.empty:
            continue

        tmp["vital"] = tmp["itemid"].map(itemid_to_vital)
        tmp = tmp[tmp["vital"].notna()].copy()
        if tmp.empty:
            continue

        # temp F->C
        is_temp = tmp["vital"].eq("temp")
        if is_temp.any() and "valueuom" in tmp.columns:
            uom = tmp.loc[is_temp, "valueuom"].astype(str).str.upper()
            v = tmp.loc[is_temp, "valuenum"].astype(float)
            is_f = uom.str.contains("F", na=False) | (v >= 79)
            tmp.loc[is_temp & is_f, "valuenum"] = (tmp.loc[is_temp & is_f, "valuenum"] - 32) * (5 / 9)

        tmp = _filter_plausible_ranges(tmp, varname_col="vital", value_col="valuenum")
        if tmp.empty:
            continue

        tmp = tmp.sort_values("charttime")
        key = ["stay_id","grid","vital"]
        last = tmp.groupby(key, as_index=False).tail(1)[key + ["charttime","valuenum"]]
        last = last.rename(columns={"valuenum":"last"})
        parts.append(last)

    if not parts:
        return df

    vitals_last = pd.concat(parts, ignore_index=True)
    vitals_last = vitals_last.sort_values("charttime")
    vitals_last = vitals_last.groupby(["stay_id","grid","vital"], as_index=False).tail(1)
    vitals_last = vitals_last.drop(columns=["charttime"])

    val_wide = vitals_last.pivot(index=["stay_id","grid"], columns="vital", values="last")
    val_wide.columns = [f"vs_{c}_last" for c in val_wide.columns]
    val_wide = val_wide.reset_index()

    obs = vitals_last.copy()
    obs["obs"] = 1
    obs_wide = obs.pivot_table(index=["stay_id","grid"], columns="vital", values="obs", aggfunc="max")
    obs_wide = obs_wide.fillna(0).astype("int8")
    obs_wide.columns = [f"vs_{c}_obs" for c in obs_wide.columns]
    obs_wide = obs_wide.reset_index()

    out = df.merge(val_wide, on=["stay_id","grid"], how="left").merge(obs_wide, on=["stay_id","grid"], how="left")

    out = out.sort_values(["stay_id","grid"]).reset_index(drop=True)
    vital_val_cols = [c for c in out.columns if c.startswith("vs_") and c.endswith("_last")]

    for c in vital_val_cols:
        out[c.replace("_last","_missing")] = out[c].isna().astype("int8")

    vital_obs_cols = [c for c in out.columns if c.startswith("vs_") and c.endswith("_obs")]
    if vital_obs_cols:
        out[vital_obs_cols] = out[vital_obs_cols].fillna(0).astype("int8")

    return out


# ============================================================
# 6) (D) Labs last + indicators (NO-LOCF)
# ============================================================
def add_labs_last_with_indicators(df: pd.DataFrame, cfg: CFG, P: dict) -> pd.DataFrame:
    dlab = pd.read_csv(P["D_LABITEMS_PATH"], usecols=["itemid","label","fluid","category"], low_memory=False)
    dlab["label"] = dlab["label"].astype(str)
    if "fluid" in dlab.columns:
        dlab["fluid"] = dlab["fluid"].astype(str)

    LAB_PATTERNS = {
        "lactate":    [r"\bLactate\b"],
        "creatinine": [r"\bCreatinine\b"],
        "bilirubin":  [r"\bBilirubin\b.*\bTotal\b", r"\bTotal Bilirubin\b"],
        "wbc":        [r"\bWBC\b", r"White Blood Cells?"],
        "ph":         [r"\bpH\b"],
    }

    itemid_to_lab = {}
    for lab, pats in LAB_PATTERNS.items():
        mask = _build_mask(dlab["label"], pats)
        if lab == "ph" and "fluid" in dlab.columns:
            mask = mask & dlab["fluid"].str.contains("blood|plasma|serum", case=False, regex=True, na=False)
        for itemid in dlab.loc[mask, "itemid"].astype(int).tolist():
            itemid_to_lab.setdefault(itemid, lab)

    selected_itemids = set(itemid_to_lab.keys())
    if not selected_itemids:
        raise ValueError("No lab itemids matched. Check patterns/d_labitems.")

    stay_table = (
        df[["subject_id","hadm_id","stay_id","baseline_time","outtime"]]
        .drop_duplicates(subset=["stay_id"])
        .rename(columns={"baseline_time":"intime"})
        .copy()
    )
    stay_table["subject_id"] = pd.to_numeric(stay_table["subject_id"], errors="coerce")
    stay_table["hadm_id"] = pd.to_numeric(stay_table["hadm_id"], errors="coerce")
    stay_table["stay_id"] = pd.to_numeric(stay_table["stay_id"], errors="coerce")
    stay_table = stay_table.dropna(subset=["subject_id","hadm_id","stay_id","intime"]).copy()
    stay_table["subject_id"] = stay_table["subject_id"].astype(int)
    stay_table["hadm_id"] = stay_table["hadm_id"].astype(int)
    stay_table["stay_id"] = stay_table["stay_id"].astype(int)

    hadm_set = set(stay_table["hadm_id"].unique().tolist())
    subj_set = set(stay_table["subject_id"].unique().tolist())
    max_grid_by_stay = df.groupby("stay_id")["grid"].max().to_dict()

    USECOLS = ["subject_id","hadm_id","itemid","charttime","valuenum","valueuom"]
    parts = []

    GH = int(cfg.grid_hours)

    for ch in pd.read_csv(
        P["LABEVENTS_PATH"], usecols=USECOLS, chunksize=cfg.chunksize, low_memory=False
    ):
        ch["subject_id"] = pd.to_numeric(ch["subject_id"], errors="coerce")
        ch["hadm_id"] = pd.to_numeric(ch["hadm_id"], errors="coerce")
        ch["itemid"] = pd.to_numeric(ch["itemid"], errors="coerce")
        ch = ch[ch["subject_id"].notna() & ch["hadm_id"].notna() & ch["itemid"].notna()]
        ch["subject_id"] = ch["subject_id"].astype(int)
        ch["hadm_id"] = ch["hadm_id"].astype(int)
        ch["itemid"] = ch["itemid"].astype(int)

        ch = ch[ch["subject_id"].isin(subj_set) & ch["hadm_id"].isin(hadm_set)]
        if ch.empty:
            continue

        ch = ch[ch["itemid"].isin(selected_itemids)]
        if ch.empty:
            continue

        ch["charttime"] = _safe_datetime(ch["charttime"])
        ch["valuenum"] = pd.to_numeric(ch["valuenum"], errors="coerce")
        ch = ch[ch["charttime"].notna() & ch["valuenum"].notna()]
        if ch.empty:
            continue

        tmp = ch.merge(stay_table, on=["subject_id","hadm_id"], how="inner")
        if tmp.empty:
            continue

        tmp = tmp[(tmp["charttime"] >= tmp["intime"]) & (tmp["outtime"].isna() | (tmp["charttime"] <= tmp["outtime"]))].copy()
        if tmp.empty:
            continue

        hr = ((tmp["charttime"] - tmp["intime"]).dt.total_seconds() // 3600).astype("int64")
        tmp = tmp[hr >= 0].copy()
        tmp["hr"] = hr[hr >= 0]
        tmp["grid"] = (tmp["hr"] // GH).astype(int)

        tmp["max_grid"] = tmp["stay_id"].map(max_grid_by_stay)
        tmp = tmp[tmp["max_grid"].notna() & (tmp["grid"] <= tmp["max_grid"])].copy()
        if tmp.empty:
            continue

        tmp["lab"] = tmp["itemid"].map(itemid_to_lab)
        tmp = tmp[tmp["lab"].notna()].copy()
        if tmp.empty:
            continue

        tmp = _filter_plausible_ranges(tmp, varname_col="lab", value_col="valuenum")
        if tmp.empty:
            continue

        tmp = tmp.sort_values("charttime")
        key = ["stay_id","grid","lab"]
        last = tmp.groupby(key, as_index=False).tail(1)[key + ["charttime","valuenum"]]
        last = last.rename(columns={"valuenum":"last"})
        parts.append(last)

    if not parts:
        return df

    labs_last = pd.concat(parts, ignore_index=True)
    labs_last = labs_last.sort_values("charttime")
    labs_last = labs_last.groupby(["stay_id","grid","lab"], as_index=False).tail(1)
    labs_last = labs_last.drop(columns=["charttime"])

    val_wide = labs_last.pivot(index=["stay_id","grid"], columns="lab", values="last")
    val_wide.columns = [f"lb_{c}_last" for c in val_wide.columns]
    val_wide = val_wide.reset_index()

    obs = labs_last.copy()
    obs["obs"] = 1
    obs_wide = obs.pivot_table(index=["stay_id","grid"], columns="lab", values="obs", aggfunc="max")
    obs_wide = obs_wide.fillna(0).astype("int8")
    obs_wide.columns = [f"lb_{c}_obs" for c in obs_wide.columns]
    obs_wide = obs_wide.reset_index()

    out = df.merge(val_wide, on=["stay_id","grid"], how="left").merge(obs_wide, on=["stay_id","grid"], how="left")

    out = out.sort_values(["stay_id","grid"]).reset_index(drop=True)
    lab_val_cols = [c for c in out.columns if c.startswith("lb_") and c.endswith("_last")]

    for c in lab_val_cols:
        out[c.replace("_last","_missing")] = out[c].isna().astype("int8")

    lab_obs_cols = [c for c in out.columns if c.startswith("lb_") and c.endswith("_obs")]
    if lab_obs_cols:
        out[lab_obs_cols] = out[lab_obs_cols].fillna(0).astype("int8")

    return out


# ============================================================
# 7) (E) ICU interventions by category any (NO-CARRY-FORWARD)
# ============================================================
def _sanitize_col(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s if s else "unknown"

def _load_itemid_to_category_map(counts_csv_path: str, min_n: int, n_col_guess: str):
    m = pd.read_csv(counts_csv_path, low_memory=False)
    if "itemid" not in m.columns or "category" not in m.columns:
        raise ValueError(f"{counts_csv_path} must contain columns ['itemid','category']")

    if n_col_guess in m.columns:
        m = m[m[n_col_guess].fillna(0) >= min_n].copy()

    m["category"] = m["category"].astype(str)
    m = m[m["category"].notna() & (m["category"].str.strip() != "")].copy()
    m = m.drop_duplicates("itemid")
    return dict(zip(m["itemid"].astype(int), m["category"]))

def _infer_time_col(cols: List[str], preferred=("starttime", "storetime", "charttime")) -> str:
    for c in preferred:
        if c in cols:
            return c
    raise ValueError(f"Could not infer time column. Available columns: {cols[:50]}")

def _aggregate_events_by_category(
    events_path: str,
    df3: pd.DataFrame,
    itemid_to_cat: dict,
    prefix: str,
    *,
    grid_hours: int,
    chunksize: int = 2_000_000,
):
    needed = {"stay_id", "grid", "baseline_time", "outtime"}
    missing = needed - set(df3.columns)
    if missing:
        raise ValueError(f"df3 must contain {sorted(needed)}; missing={sorted(missing)}")

    stay_time = df3[["stay_id", "baseline_time", "outtime"]].drop_duplicates("stay_id").copy()
    stay_time["baseline_time"] = pd.to_datetime(stay_time["baseline_time"], errors="coerce")
    stay_time["outtime"] = pd.to_datetime(stay_time["outtime"], errors="coerce")
    stay_time["stay_id"] = pd.to_numeric(stay_time["stay_id"], errors="coerce")
    stay_time = stay_time.dropna(subset=["stay_id", "baseline_time", "outtime"]).copy()
    stay_time["stay_id"] = stay_time["stay_id"].astype(np.int64)

    max_grid = df3.groupby("stay_id")["grid"].max().rename("max_grid").reset_index()
    max_grid["stay_id"] = pd.to_numeric(max_grid["stay_id"], errors="coerce").astype(np.int64)
    stay_time = stay_time.merge(max_grid, on="stay_id", how="inner")
    if stay_time.empty:
        return pd.DataFrame(columns=["stay_id", "grid"])

    selected_itemids = set(int(k) for k in itemid_to_cat.keys())
    if not selected_itemids:
        raise ValueError("itemid_to_cat is empty. Check mapping CSV.")

    header_cols = pd.read_csv(events_path, nrows=0).columns.tolist()
    time_col = _infer_time_col(header_cols)
    usecols = [c for c in ["stay_id", "itemid", time_col] if c in header_cols]

    GH = int(grid_hours)
    agg_parts = []
    for chunk in pd.read_csv(events_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        if "stay_id" not in chunk.columns or "itemid" not in chunk.columns or time_col not in chunk.columns:
            continue

        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk["itemid"] = pd.to_numeric(chunk["itemid"], errors="coerce")
        chunk = chunk.dropna(subset=["stay_id", "itemid"]).copy()
        if chunk.empty:
            continue

        chunk["stay_id"] = chunk["stay_id"].astype(np.int64)
        chunk["itemid"] = chunk["itemid"].astype(np.int64)

        chunk = chunk[chunk["itemid"].isin(selected_itemids)]
        if chunk.empty:
            continue

        chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")
        chunk = chunk.dropna(subset=[time_col]).copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(stay_time, on="stay_id", how="inner")
        if chunk.empty:
            continue

        chunk = chunk[(chunk[time_col] >= chunk["baseline_time"]) & (chunk[time_col] <= chunk["outtime"])].copy()
        if chunk.empty:
            continue

        hr = np.floor((chunk[time_col] - chunk["baseline_time"]).dt.total_seconds() / 3600.0)
        grid = np.floor(hr / float(GH)).astype(np.int32)
        grid = np.clip(grid, 0, chunk["max_grid"].astype(np.int32).values)
        chunk["grid"] = grid

        chunk["_cat"] = chunk["itemid"].map(itemid_to_cat)
        chunk = chunk[chunk["_cat"].notna()].copy()
        if chunk.empty:
            continue

        chunk["_val"] = 1
        g = chunk.groupby(["stay_id", "grid", "_cat"], as_index=False)["_val"].max()
        agg_parts.append(g)

    if not agg_parts:
        return pd.DataFrame(columns=["stay_id", "grid"])

    long = pd.concat(agg_parts, ignore_index=True)
    long = long.groupby(["stay_id", "grid", "_cat"], as_index=False)["_val"].max()

    wide = long.pivot_table(
        index=["stay_id", "grid"], columns="_cat", values="_val", aggfunc="max", fill_value=0
    )
    wide.columns = [f"{prefix}{_sanitize_col(c)}" for c in wide.columns]
    wide = wide.reset_index()

    for c in wide.columns:
        if c.startswith(prefix):
            wide[c] = wide[c].astype(np.int8)

    return wide

def add_icu_interventions_by_category_any(df3: pd.DataFrame, cfg: CFG, P: dict) -> pd.DataFrame:
    in_map = _load_itemid_to_category_map(cfg.input_map_csv, cfg.min_item_n, n_col_guess="inputevents_n")
    pr_map = _load_itemid_to_category_map(cfg.proc_map_csv,  cfg.min_item_n, n_col_guess="procedureevents_n")

    if cfg.keep_proc_categories_regex is not None:
        rgx = re.compile(cfg.keep_proc_categories_regex, flags=re.IGNORECASE)
        pr_map = {k: v for k, v in pr_map.items() if rgx.search(str(v))}

    in_wide = _aggregate_events_by_category(
        events_path=P["INPUTEVENTS_PATH"], df3=df3, itemid_to_cat=in_map,
        prefix="in_cat__", grid_hours=cfg.grid_hours, chunksize=cfg.chunksize
    )
    pr_wide = _aggregate_events_by_category(
        events_path=P["PROCEDUREEVENTS_PATH"], df3=df3, itemid_to_cat=pr_map,
        prefix="pr_cat__", grid_hours=cfg.grid_hours, chunksize=cfg.chunksize
    )

    out = df3.copy()

    if in_wide.shape[1] > 2:
        rename_in = {c: f"{c}_any" for c in in_wide.columns if c not in ["stay_id","grid"]}
        in_wide = in_wide.rename(columns=rename_in)
        out = out.merge(in_wide, on=["stay_id","grid"], how="left")

    if pr_wide.shape[1] > 2:
        rename_pr = {c: f"{c}_any" for c in pr_wide.columns if c not in ["stay_id","grid"]}
        pr_wide = pr_wide.rename(columns=rename_pr)
        out = out.merge(pr_wide, on=["stay_id","grid"], how="left")

    new_cols = [c for c in out.columns if c.startswith("in_cat__") or c.startswith("pr_cat__")]
    if new_cols:
        out[new_cols] = out[new_cols].fillna(0).astype("int8")

    print(f"[icu interventions by category] added_cols={len(new_cols)} "
          f"(input={sum(c.startswith('in_cat__') for c in new_cols)}, "
          f"procedure={sum(c.startswith('pr_cat__') for c in new_cols)})")

    return out


# ============================================================
# 8) Main pipeline
# ============================================================
def main(cfg: CFG):
    P = _paths(cfg)

    df3 = build_df3_from_sofa(cfg, P)
    print("[1] df3 built:", df3.shape)
    if cfg.debug:
        debug_death_and_label(df3, cfg, tag="after df3")

    df3 = add_static_features(df3, P)
    print("[2] +static:", df3.shape)

    df3 = add_vitals_last_with_indicators(df3, cfg, P)
    print("[3] +vitals (NO-LOCF):", df3.shape)

    df3 = add_labs_last_with_indicators(df3, cfg, P)
    print("[4] +labs   (NO-LOCF):", df3.shape)

    if cfg.add_interventions:
        df3 = add_icu_interventions_by_category_any(df3, cfg, P)
        print("[5] +interventions (NO-CARRY-FORWARD):", df3.shape)

    if cfg.debug:
        debug_death_and_label(df3, cfg, tag="final df3")

    df3.to_csv(cfg.out_csv_path, index=False)
    print(f"[SAVE] CSV: {cfg.out_csv_path}")

    if cfg.out_parquet_path:
        df3.to_parquet(cfg.out_parquet_path, index=False)
        print(f"[SAVE] Parquet: {cfg.out_parquet_path}")

    meta = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "mimic_iv_dir": cfg.mimic_iv_dir,
        "sofa_hourly_path": cfg.sofa_hourly_path,
        "first_day_sofa_path": cfg.first_day_sofa_path,
        "use_patient_dod": cfg.use_patient_dod,
        "chunksize": cfg.chunksize,
        "add_interventions": cfg.add_interventions,
        "min_item_n": cfg.min_item_n,
        "keep_proc_categories_regex": cfg.keep_proc_categories_regex,
        "grid_hours": cfg.grid_hours,
        "horizon_hours": cfg.horizon_hours,
        "note": "NO-LOCF; NO-carry-forward; configurable grid/horizon",
        "columns_catalog": build_columns_catalog(df3, cfg),
    }

    meta_path = os.path.join(os.path.dirname(cfg.out_csv_path) or ".", "model_data_mimic4_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[SAVE] meta: {meta_path}")

    return df3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mimic_iv_dir", type=str, default="mimiciv/3.1")
    ap.add_argument("--sofa_hourly_path", type=str, default="sofa.csv")
    ap.add_argument("--first_day_sofa_path", type=str, default="first_day_sofa.csv")
    ap.add_argument("--out_csv_path", type=str, default="model_data_mimic4.csv")
    ap.add_argument("--out_parquet_path", type=str, default="")
    ap.add_argument("--grid_hours", type=int, default=24)
    ap.add_argument("--horizon_hours", type=int, default=24)
    ap.add_argument("--use_patient_dod", action="store_true")
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    ap.add_argument("--no_interventions", action="store_true")
    ap.add_argument("--input_map_csv", type=str, default="icu_inputevents_itemid_counts.csv")
    ap.add_argument("--proc_map_csv", type=str, default="icu_procedureevents_itemid_counts.csv")
    ap.add_argument("--min_item_n", type=int, default=0)
    ap.add_argument("--keep_proc_categories_regex", type=str, default="")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = CFG(
        mimic_iv_dir=args.mimic_iv_dir,
        sofa_hourly_path=args.sofa_hourly_path,
        first_day_sofa_path=args.first_day_sofa_path,
        out_csv_path=args.out_csv_path,
        out_parquet_path=(args.out_parquet_path or None),
        grid_hours=int(args.grid_hours),
        horizon_hours=int(args.horizon_hours),
        use_patient_dod=bool(args.use_patient_dod),
        chunksize=int(args.chunksize),
        add_interventions=(not args.no_interventions),
        input_map_csv=args.input_map_csv,
        proc_map_csv=args.proc_map_csv,
        min_item_n=int(args.min_item_n),
        keep_proc_categories_regex=(args.keep_proc_categories_regex or None),
        debug=bool(args.debug),
    )
    _ = main(cfg)
