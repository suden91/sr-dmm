Markdown# SOFA-Reconstruction Deep Markov Model (SR-DMM) for Dynamic ICU Mortality Prediction with Clinical Notes


> **Anonymous repository for double-blind review.**  
> This repository contains code to reproduce the main experiments in the accompanying manuscript (submitted).  
> No patient-level data are included in this repository.

## Overview
We study **dynamic ICU mortality risk prediction** on **MIMIC-IV** with an hourly discretization of ICU stays.  
The proposed model is a **multimodal Deep Markov Model (DMM)** that:
- performs **latent-state inference** from structured time series,
- optionally incorporates **clinical note representations** aligned to the hourly grid (encoder-level fusion),
- predicts short-horizon mortality risk (e.g., next-24h / next-48h, depending on configuration),
- and can optionally reconstruct/forecast physiologic variables via the generative emission path.

This repository provides:
- preprocessing scripts (structured data + note tokenization/embedding),
- fold preparation utilities,
- training/evaluation code for the proposed model and baselines.

---

## Repository Structure (key scripts)
- `make_data_mimic4.py`  
  Build structured ICU time-series tables from MIMIC-IV (hourly grid).
- `make_token.py`  
  Tokenize and embed clinical notes, then export note-level artifacts aligned to ICU time bins.
- `prepare_folds_llm.py`  
  Merge structured + text artifacts and prepare patient-level splits (folds) for CV.
- `dmm_optuna_cv_encodertxt.py`  
  Proposed model: DMM with encoder-level text fusion + nested CV (Optuna).
- `dmm_optuna_cv_notxt.py`  
  Ablation: DMM without text.
- `gru_optuna_cv_compatible.py`  
  Baseline: GRU-style model with compatible I/O and evaluation pipeline.
- `using_clinical_optuna_cv.py`  
  Baseline inspired by prior multimodal ICU note + time-series integration (for comparison).
- `dmm_encodertxt.py`  
  Utility to refit the final model using best hyperparameters (e.g., per-fold refit).

> Note: Some scripts assume an `artifacts/` directory for outputs.

---

## Requirements
- Python >= 3.10 (recommended)
- PyTorch (CUDA optional but recommended for training)
- Common ML stack: `numpy`, `pandas`, `scikit-learn`, `tqdm`, etc.
- Optuna for hyperparameter search (when running nested CV)

Example (minimal):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch numpy pandas scikit-learn tqdm optuna
````

> Exact dependency versions used in the experiments will be provided in the camera-ready version (or a `requirements.txt`/`environment.yml` can be added if requested).

---

## Data Access (MIMIC)

This project uses **MIMIC-IV** and **MIMIC-IV-Note** from PhysioNet, which are **publicly available but require credentialed access** (training + data use agreement).
We do **not** redistribute any MIMIC data in this repository.

You must:

1. obtain PhysioNet credentialed access to MIMIC-IV and MIMIC-IV-Note,
2. download the datasets locally,
3. set paths in scripts or environment variables as needed.

---

## Reproduction Workflow (high level)

### Step 1) Build structured hourly-grid data

Run the structured preprocessing script to generate the model-ready table (CSV/Parquet depending on configuration):

```bash
python make_data_mimic4.py --help
# Example (arguments may differ by your local setup)
python make_data_mimic4.py \
  --mimic_root /path/to/mimic-iv \
  --out_dir artifacts/structured
```

### Step 2) Tokenize & embed notes (optional; for multimodal setting)

```bash
python make_token.py --help
# Example (LLM backend / caching may vary)
python make_token.py \
  --llm_backend ollama \
  --ollama_url http://127.0.0.1:11434 \
  --ollama_model qwen2.5-coder:7b \
  --llm_workers 4 \
  --cache_dir artifacts/note_cache \
  --out_dir artifacts/note_tokens \
  --use_cache
```

### Step 3) Prepare folds (patient-level split) and merged artifacts

```bash
python prepare_folds_llm.py --help
python prepare_folds_llm.py \
  --structured_path artifacts/structured/model_data.csv \
  --note_dir artifacts/note_tokens \
  --out_dir artifacts/folds_multimodal
```

### Step 4) Train / Evaluate (nested CV with Optuna)

**Proposed model (with text):**

```bash
python dmm_optuna_cv_encodertxt.py --help
python dmm_optuna_cv_encodertxt.py \
  --folds_dir artifacts/folds_multimodal \
  --meta_bags_npz artifacts/folds_multimodal/meta_bags.npz \
  --out_dir artifacts/results_dmm_encodertxt \
  --device cuda:0 \
  --folds 5 \
  --n_trials 30 \
  --inner_folds 3 \
  --inner_epochs 10 \
  --outer_epochs 30 \
  --batch_size 256 \
  --max_len 316
```

**Ablation (no text):**

```bash
python dmm_optuna_cv_notxt.py --help
python dmm_optuna_cv_notxt.py \
  --folds_dir artifacts/folds_multimodal \
  --out_dir artifacts/results_dmm_notxt \
  --device cuda:0
```

**Baseline (GRU):**

```bash
python gru_optuna_cv_compatible.py --help
python gru_optuna_cv_compatible.py \
  --folds_dir artifacts/folds_multimodal \
  --out_dir artifacts/results_gru \
  --device cuda:0
```

### Step 5) Final refit (optional)

After Optuna search, refit a final model per fold using best hyperparameters:

```bash
python dmm_encodertxt.py --help
python dmm_encodertxt.py \
  --base_py dmm_optuna_cv_encodertxt.py \
  --folds_dir artifacts/folds_multimodal \
  --best_params_root artifacts/results_dmm_encodertxt \
  --out_dir artifacts/refit_dmm_encodertxt \
  --device cuda:0
```

---

## Outputs

Typical outputs are written under `artifacts/` and may include:

* per-fold metrics (AUROC / AUPRC / calibration, depending on enabled evaluation),
* best hyperparameters from Optuna,
* final model checkpoints for refit runs,
* cached note artifacts (if enabled).

---

## Notes on Privacy & Licensing

* This repository contains **code only**.
* Do **not** upload MIMIC-derived tables, intermediate extracts, or any patient-level artifacts to public repos.
* Follow PhysioNet / MIMIC data use agreements and local IRB/ethics requirements as applicable.


