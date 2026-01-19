# Nociceptive Event Detection using Transfer Learning versus Tree-based Classical Machine Learning
Harnessing deep learning versus classical machine learning for nociceptive event detection. Diagnosing high-importance features, best practices, and justifying model complexity and overhead for usage in clinical settings.

This repository contains the implementation of a nociception prediction model using physiological signals and anesthetic drug data. The core methodology leverages **Transfer Learning (TL)** using **Temporal Convolutional Networks (TCN)** to adapt a base model to individual patients with minimal data.

## Project Overview

The goal is to predict nociceptive stimuli (labeled as `noc_stim`) during surgery using high-frequency physiological data and drug infusion context. 

### Key Features
- **Physiological Signals (30 features):** Derived features from heart rate variability (HRV), electrodermal activity (EDA), etc.
- **Drug Context (18 features):** Time-since-infusion and cumulative doses for various anesthetic drug classes (e.g., Sedatives, Analgesics, Muscle Relaxants).
- **Total Features:** 48.

## Implementation Details

### Data Loading (`OR_data.mat`)
The pipeline loads clinical data from a `.mat` file, containing synchronized physiological and drug information from 101 surgeries. Please obtain the data through the proper channels on PhysioNet (https://physionet.org/content/multimodal-surgery-anesthesia/1.0/).

### Model Architectures
- **Temporal Convolutional Network (TCN):** Optimized for sequence modeling with 1D convolutions.
- **Baselines:** 
  - **Logistic Regression (L1-AIC):** Sparse baseline using physiological and drug features.
  - **Random Forest:** Bagging ensemble for robust classification.

### Methodology
- **Leave-One-Surgery-Out (LOSO) Cross-Validation:** Ensures evaluation is performed on unseen patients.
- **Transfer Learning (TL):**
  - **Pre-training:** A base TCN model is trained on multiple surgeries.
  - **Adaptation:** The model is fine-tuned on the first few minutes (e.g., 1, 3, 6, 10 minutes) of a new patient's data to improve individualized performance.

## Results Summary
The system evaluates performance using:
- **AUROC** (Area Under the Receiver Operating Characteristic curve)
- **AUPRC** (Area Under the Precision-Recall curve)
- **Bootstrap 95% Confidence Intervals** for robustness.

## Requirements
- `numpy`, `pandas`, `scipy`
- `torch` (PyTorch)
- `scikit-learn`
- `shap`, `matplotlib`, `seaborn` (Visualization)
- `tqdm` (Progress tracking)

## Usage
Open and run `TL_Nociception_v2.ipynb` in a Jupyter environment. The code is set up to utilize Apple Silicon (MPS) if available, falling back to CPU otherwise.

## Details
Preliminary manuscript on results can be reviewed on MedRxiv @ https://www.medrxiv.org/content/10.1101/2025.07.01.25330670v1
