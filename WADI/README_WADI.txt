================================================================================
README: 3-Prong Anomaly Detection Framework (WADI Dataset)
================================================================================

TITLE
-----
3-Prong Anomaly Detection Framework
Code Artifact & Benchmark Execution Package for the WADI Dataset

DESCRIPTION
-----------
This repository contains the official Python code implementation for the 
3-Prong Anomaly Detection Framework adapted for the Water Distribution (WADI) 
cyber-physical benchmark dataset. The system combines a bottleneck autoencoder 
with a three-pronged anomaly scoring engine (Reconstruction Error, Mahalanobis 
Distance over continuous context variables, and L1 Historical Differential 
Velocity) alongside Extreme Value Theory (EVT/POT) tail thresholding.

The package contains both point-in-time / zero-window (W=1) and spatio-temporal 
sliding window (W=5) execution pipelines to evaluate unadjusted and point-adjusted 
cyber-physical attack detection performance, complete with CART surrogate decision 
tree explainability.

DATASET INFORMATION
-------------------
1. Benchmark Name: Water Distribution (WADI) Dataset
2. Files Required: 
   - WADI_14days_normal.csv (Training set)
   - WADI_attackdata.csv (Testing / Attack set)
3. Target Schema & Leakage Removal:
   - Numerical Sensor Telemetry: Continuous pressure, flow, and level metrics.
   - Categorical / Discrete Actuator Columns: Discrete valve, pump, and switch states.
   - Metadata & Leak Column Removal: Drops 'Row', 'Date', 'Time', and direct PLC 
     state leaks ('2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS').
   - Label Mapping: Converts raw attack indicators (1 = Normal, -1 = Attack) into 
     standard binary flags (0 = Normal, 1 = Anomaly).
4. Dataset Access:
   The WADI benchmark dataset can be obtained via the iTrust Centre for Research 
   in Cyber Security at the Singapore University of Technology and Design (SUTD).

   Place the raw CSV files ('WADI_14days_normal.csv' and 'WADI_attackdata.csv') 
   directly in the project root directory prior to executing the scripts.

CODE INFORMATION
----------------
Main Execution Scripts:
- 'Contexual Learning-WADI_V0_nowindow.py': Point-in-time / Zero-window (W=1) pipeline.
- 'Contexual Learning-WADI_V0_window10.py': Spatio-temporal sliding window (W=5) pipeline.

Script Pipeline Workflow:
1. Data Preprocessing & Alignment: Performs forward-fill and backward-fill missing 
   value imputation, aligns columns across train and test sets, drops non-informative 
   or status-leakage attributes, and applies domain-shift Z-score guardrails (Z > 10) 
   to eliminate uncalibrated features.
2. Heterogeneous Feature Encoding: Transforms continuous numeric features using a custom 
   Guardrailed Robust Scaler and categorical features via OneHotEncoder.
3. Model Training: Trains a PyTorch progressive bottleneck autoencoder 
   (HighDimensionalBottleneckAE) on nominal sequence data using the AdamW optimizer 
   and Cosine Annealing learning rate schedule.
4. Tri-Metric Score Fusion: Computes MSE reconstruction loss, centered Mahalanobis 
   distance with pseudo-inverse regularization, and L1 historical moving-average 
   residual velocity. Standardizes scores via robust MAD Z-scores.
5. EVT Decision Gating: Fits tail excesses above high percentiles to a Generalized 
   Pareto Distribution (GPD) using Peaks-Over-Threshold (POT) to set dynamic decision bounds.
6. Evaluation Metrics: Computes raw unadjusted and Point-Adjusted (PA) Precision, 
   Recall, F1-Score, ROC-AUC, and PR-AUC.
7. Explainable AI: Fits a surrogate CART decision tree classifier on model predictions 
   to extract human-readable logic for triggered anomalies.

REQUIREMENTS
------------
System Hardware Requirements:
- Operating System: Linux, macOS, or Windows 10/11
- Python Version: Python 3.9 or higher
- System RAM: 16 GB minimum (32 GB recommended for multi-window tensor generation)
- Hardware Acceleration (Optional): NVIDIA GPU with CUDA support for PyTorch

Software Dependencies & Python Libraries:
- numpy
- pandas
- torch (PyTorch)
- scikit-learn
- scipy
- matplotlib

Install all Python dependencies via pip:
pip install numpy pandas torch scikit-learn scipy matplotlib

METHODOLOGY
-----------
The execution pipeline performs the following steps sequentially:
1. Windowing & Feature Assembly: Input features are scaled and optionally windowed 
   using tensor unfolding (W=1 or W=5).
2. Bottleneck Compression: Autoencoder compresses inputs through a hidden hierarchy 
   down to a low-dimensional bottleneck (LATENT_DIM = 8).
3. Tri-Metric Score Fusion: 
   Fused Score = (w1 * Reconstruction) + (w2 * Mahalanobis) + (w3 * Residual Velocity).
4. Automated Thresholding: Dynamic GPD fitting at the tail percentile threshold.
5. Explainability Extraction: DecisionTreeClassifier surrogate (max_depth = 4, 
   class_weight = 'balanced') to output rule hierarchies.

USAGE INSTRUCTIONS
------------------
1. Setup Repository Structure:
   Organize your directory as follows:

   project_root/
   ├── Contexual Learning-WADI_V0_nowindow.py
   ├── Contexual Learning-WADI_V0_window10.py
   ├── WADI_14days_normal.csv
   ├── WADI_attackdata.csv
   └── README.txt

2. Run Pipeline Scripts:
   Execute either the point-in-time or sliding-window script from your terminal:

   # To execute the Point-in-Time (Window Size = 1) pipeline:
   python "Contexual Learning-WADI_V0_nowindow.py"

   # To execute the Spatio-Temporal (Window Size = 5) pipeline:
   python "Contexual Learning-WADI_V0_window10.py"

3. Execution Outputs Generated:
   - Console Terminal: Logs dataset dimensions, feature drop reports, EVT tail 
     parameters, raw vs point-adjusted classification metrics, and extracted decision 
     tree surrogate rules.
   - Output Visualizations: Generates and saves evaluation performance curves and 
     anomaly tracking plots to the local workspace.

CITATIONS
---------
If you use this code, model architecture, or dataset setup in your research, 
please cite the associated manuscript:

[Insert Journal Reference / Manuscript Title here upon publication]

Dataset Reference:

https://www.kaggle.com/datasets/giovannimonco/wadi-data

Ahmed, C. M., Palleti, V. R., and Mathur, A. P. (2017). Wadi: A water distribution testbed for research756
in the design of secure cyber physical systems. In Proceedings of the 3rd international workshop on757
cyber-physical systems for smart water networks, pages 25–28

LICENSE & CONTRIBUTION GUIDELINES
---------------------------------
License:
Unlicensed. This codebase is provided "as is" for public academic evaluation, 
peer-review replication, and research purposes without express warranty.

Contribution Guidelines:
Pull requests and issue reports are welcome for bug fixes, performance improvements, 
or optimizations related to high-dimensional time-series data scaling and PyTorch pipelines.
================================================================================