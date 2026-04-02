# ParkInsight: Parkinson's Screening Prototype

This repository contains a working end-to-end prototype for early Parkinson's risk screening using three modalities:

- gait (phone accelerometer and PhysioNet-derived model)
- finger tapping (mobile tapping rhythm analysis)
- voice (Oxford Parkinson's voice features and optional on-device recording pipeline)

It includes data processing, model training artifacts, backend APIs, a clinician-facing dashboard, and a mobile-friendly capture interface.

## What Has Been Implemented

### 1) End-to-End Product Flow

- Mobile user opens the phone capture page and performs gait, tapping, and voice tests.
- Backend receives raw data, extracts features, runs model/scoring logic, and stores latest results.
- Dashboard polls live results and visualizes per-test risk plus a combined weighted score.

### 2) Backend API and Serving Layer

Implemented Flask backend with CORS and static serving for dashboard + phone UI.

Main API capabilities include:

- gait analysis from live phone sensor JSON
- gait analysis from uploaded CSV data
- tapping analysis from timestamp sequences
- voice analysis from uploaded audio or provided feature vectors
- latest-results polling endpoint for dashboard
- reset endpoint for new patient/demo sessions
- server-info endpoint for QR generation and LAN/ngrok support
- demo scenario loading endpoint

### 3) Gait Feature Engineering and Inference

Gait pipeline includes robust feature extraction with:

- temporal features (stride mean/std/CV, cadence)
- variability and asymmetry features
- autocorrelation-based regularity features
- DFA and sample entropy style complexity features
- higher-order distributional statistics

Inference supports:

- XGBoost model bundle loading (model + scaler + feature names)
- fallback heuristic behavior when model or clean features are unavailable
- SHAP value computation (with safe fallback if SHAP import/runtime fails)

### 4) Tapping Feature Engineering and Risk Scoring

Implemented tapping feature extraction and continuous risk scoring with:

- interval mean/std/CV
- fatigue ratio and interval trend
- rhythm regularity and tap frequency
- sigmoid-based weighted risk scoring (Low/Medium/High bands)

### 5) Voice Processing and Inference

Implemented voice analysis path with:

- audio upload handling (WebM/WAV)
- optional Praat/parselmouth extraction flow (jitter, shimmer, HNR/NHR, etc.)
- model-based inference when voice model is available
- demo fallback scoring when extraction/model path is unavailable

### 6) Dashboard Implementation

Implemented a live monitoring dashboard with:

- QR code generation for phone test access
- live connectivity status and 2-second polling loop
- per-test status cards and risk bars
- combined risk gauge with modality-weighted breakdown
- gait waveform chart
- SHAP feature-impact chart
- gait feature cards against healthy reference ranges
- tapping rhythm chart and voice summary panels
- demo scenario triggers and CSV upload actions

### 7) Phone Capture Application

Implemented a single-page mobile capture interface with:

- gait capture via DeviceMotion APIs (including iOS permission handling)
- 30-second guided gait recording with progress and sample counts
- 10-second alternating tap test UI
- 5-second microphone recording and upload flow
- tabbed modality UX and server connectivity indicator

### 8) Training and Research Workflow

Implemented reproducible training scripts and model experiments for gait and voice:

- gait training with multiple table modes (subject_mean, trial, subject_hierarchical)
- strict grouping logic to reduce subject leakage risk
- robust/standard scaler comparisons
- manual hyperparameter search and report generation
- leave-one-study-out (LOSO) evaluation for cross-cohort robustness
- architecture comparison across XGBoost, CatBoost, LightGBM, HistGradientBoosting, RandomForest, ExtraTrees
- dedicated tuning script for top architectures
- voice model training on Oxford Parkinson's dataset with SHAP plots

### 9) Model and Experiment Artifacts

Repository includes saved artifacts such as:

- deployable gait and voice model bundles
- JSON training reports and benchmark summaries
- architecture comparison and tuning summaries
- SHAP summary plots for gait and voice

#### Classification Performance Summary (Gait)

| Experiment Setup | Model Family | Holdout AUC-ROC | Mean LOSO AUC-ROC | Notes |
|---|---|---:|---:|---|
| subject_mean + robust scaler | XGBoost | 0.8074 | 0.7850 | Deployable baseline bundle |
| subject_mean + standard scaler | XGBoost | 0.8259 | 0.7840 | Similar LOSO robustness to robust scaler |
| subject_hierarchical + robust scaler | XGBoost | 0.8111 | 0.7829 | Additional aggregated/hierarchical features |
| trial-level + robust scaler | XGBoost | 0.9199 | 0.7702 | Highest holdout, weaker cross-study generalization |
| architecture comparison (default params) | HistGradientBoosting | - | 0.7852 | Best default LOSO among tested families |
| architecture comparison (default params) | LightGBM | - | 0.7831 | Strong default cross-study behavior |
| LOSO-targeted tuned run | XGBoost | 0.7963 | 0.8035 | Best recorded LOSO AUC for XGBoost |
| tuned architecture search | LightGBM (tuned) | - | 0.7973 | Best tuned non-XGBoost candidate |
| tuned architecture search | HistGradientBoosting (tuned) | - | 0.7869 | Tuned improvement over XGBoost default baseline |

#### Tuning Activities Carried Out

| Activity | What Was Done | Output Artifact | Outcome |
|---|---|---|---|
| Multi-setup gait benchmarking | Compared table modes (`subject_mean`, `trial`, `subject_hierarchical`) and scaler choices (`robust`, `standard`) under grouped evaluation | `backend/models/gait_experiment_summary.json` | Established tradeoff: strongest holdout from trial-mode, strongest cross-study robustness from subject-level setups |
| Manual XGBoost hyperparameter search | Sampled parameter combinations over estimators, depth, learning rate, subsample, gamma, regularization | `backend/models/gait_training_report.json` | Produced LOSO-targeted XGBoost configuration reaching mean LOSO AUC 0.8035 |
| Architecture comparison | Evaluated XGBoost, CatBoost, LightGBM, HistGradientBoosting, RandomForest, ExtraTrees with LOSO protocol | `backend/models/gait_architecture_comparison.json` | HistGradientBoosting and LightGBM outperformed default XGBoost in LOSO mean AUC |
| Focused tuning of top architectures | Ran dedicated hyperparameter tuning for HistGradientBoosting and LightGBM | `backend/models/gait_architecture_tuning_results.json` | Tuned LightGBM reached LOSO AUC 0.7973; tuned HistGradientBoosting reached 0.7869 |
| Documentation of tuning phase | Consolidated ranking, study-level behavior, and recommendations | `backend/models/ARCHITECTURE_TUNING_SUMMARY.md` | Captured Phase 4a findings and next-step recommendations |

### 10) Demo and Presentation Support

Implemented demo data generation and scenarios:

- healthy
- early PD
- advanced PD

These scenarios can be loaded through API/UI to demonstrate full pipeline behavior without live collection.

### 11) Local Network, HTTPS, and iPhone Support

Implemented startup and connectivity support for real-device testing:

- simple launcher script for local dashboard startup
- optional ngrok tunnel flow for secure external phone access
- self-signed SAN certificate generation for iOS/Safari DeviceMotion requirements
- certificate download/install route and iPhone setup guidance

## Current Repository Layout (High Level)

- backend: Flask API, feature extraction, model artifacts, training scripts, cert utilities, demo data
- dashboard: clinician-facing web dashboard
- phone-capture: mobile capture UI
- gait-in-parkinsons-disease-1.0.0: gait dataset files used for training experiments
- parkinsight.ipynb: notebook-based experimentation
- start.py: convenience launcher for local/ngrok run modes

## How To Run

1. Install backend dependencies from the backend requirements file.
2. Start the app with:
   - `python start.py` for local LAN mode, or
   - `python start.py --ngrok` for HTTPS tunnel mode.
3. Open the dashboard URL shown in terminal.
4. Scan the QR code to open the phone capture page.
5. Run gait/tapping/voice tests and observe live dashboard updates.

## Important Note

This is a prototype screening system intended for research/demo use. It is not a medical diagnosis tool and should not be used as a substitute for clinical evaluation by qualified professionals.