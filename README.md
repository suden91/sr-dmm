Markdown# SOFA-Reconstruction Deep Markov Model (SR-DMM) for Dynamic ICU Mortality Prediction with Clinical Notes

This repository contains the official implementation of the code used in the paper **"Deep Markov Model with LLM-Integrated Clinical Notes for Mortality Prediction"** (Anonymous Submission).

This framework integrates high-dimensional clinical time-series data (MIMIC-IV) with unstructured clinical notes processed via LLMs to predict ICU mortality using a Deep Markov Model (DMM).

## 📋 Table of Contents
- [Prerequisites](#prerequisites)
- [Data Preparation](#data-preparation)
  - [1. Clinical Features](#1-clinical-features)
  - [2. Text Embeddings (LLM)](#2-text-embeddings-llm)
  - [3. Fold Generation](#3-fold-generation)
- [Running Experiments](#running-experiments)
  - [Proposed Method (DMM + Text)](#proposed-method-dmm--text)
  - [Baselines](#baselines)
- [File Descriptions](#file-descriptions)

## 🛠 Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Pandas, NumPy, Scikit-learn
- [MIMIC-IV Database](https://physionet.org/content/mimiciv/) access (Credentialed access required)

```bash
# Example installation
pip install torch pandas numpy scikit-learn transformers
🚀 Data PreparationThe data pipeline consists of three steps: extracting clinical variables, processing text with LLMs, and creating cross-validation folds.1. Clinical FeaturesExtracts hourly grid features (Vitals, Labs) and static features from MIMIC-IV.Script: make_data_mimic4.pyOutput: model_data_mimic4.csvBashpython make_data_mimic4.py \
  --mimic_iv_dir /path/to/mimiciv/3.1 \
  --out_csv_path artifacts/model_data_mimic4.csv \
  --grid_hours 1 \
  --horizon_hours 24
2. Text Embeddings (LLM)Processes clinical notes using a locally hosted LLM (e.g., via Ollama or vLLM) to generate structured tokens and embeddings.Script: make_token.pyOutput: meta_bags.npz, text_llm_struct_tokens.csvBashpython make_token.py \
  --out_dir artifacts/text_features \
  --openai_base_url "http://localhost:11434/v1" \
  --openai_model "llama3"
3. Fold GenerationMerges clinical data and text features, applies exclusion criteria (e.g., LOS < min_hours), and splits data into 5 folds.Script: prepare_folds_llm.pyOutput: artifacts/folds_data/Bashpython prepare_folds_llm.py \
  --model_data_csv artifacts/model_data_mimic4.csv \
  --text_data_csv artifacts/text_features/text_llm_struct_tokens.csv \
  --text_bags_npz artifacts/text_features/meta_bags.npz \
  --out_dir artifacts/folds_data \
  --folds 5
🧪 Running ExperimentsAll training scripts utilize a 5-fold cross-validation scheme.Proposed Method (DMM + Text)Trains the Deep Markov Model where text representations are integrated into the Encoder (Text -> Encoder).Bashpython dmm_optuna_cv_encodertxt.py \
  --folds_dir artifacts/folds_data \
  --out_dir results/dmm_text_encoder \
  --device cuda:0 \
  --epochs 100
Baselines1. DMM (Clinical Variables Only)A standard DMM model without text integration.Bashpython dmm_optuna_cv_notxt.py \
  --folds_dir artifacts/folds_data \
  --out_dir results/dmm_baseline_no_text \
  --device cuda:0
2. GRU BaselineA standard GRU-based recurrent neural network.Bashpython gru_optuna_cv_compatible.py \
  --folds_dir artifacts/folds_data \
  --out_dir results/gru_baseline \
  --device cuda:0
📂 File DescriptionsFile NameDescriptionmake_data_mimic4.pyPreprocessing script for MIMIC-IV clinical data (Hourly grids, NO-LOCF).make_token.pyLLM-based text processor. Generates token IDs and embedding bags.prepare_folds_llm.pyCombines data sources and generates 5-fold CV splits (Train/Val/Test).dmm_optuna_cv_encodertxt.py(Main) Training script for the proposed DMM with text integration.dmm_optuna_cv_notxt.pyTraining script for the DMM baseline (Clinical features only).gru_optuna_cv_compatible.pyTraining script for the GRU baseline.dmm_encodertxt.pyRefit/Inference script designed to load best parameters and evaluate on test sets.using_clinical_optuna_cv.pyVariation of the training script focusing on clinical feature ablation.
