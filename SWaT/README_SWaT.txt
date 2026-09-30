================================================================================
README: 3-Prong Anomaly Detection Framework
================================================================================

TITLE
-----
3-Prong Anomaly Detection Framework
Code Artifact & Benchmark Execution Package for the SWaT Dataset

DESCRIPTION
-----------
This repository contains the official Python code implementation for the 
3-Prong Anomaly Detection Framework. The system combines a bottleneck autoencoder with a three-pronged anomaly scoring 
engine (Reconstruction Error, Mahalanobis Distance over context variables, 
and L1 Historical Differential Velocity) and Extreme Value Theory (EVT/POT) 
tail thresholding.

The framework is designed to provide unadjusted, uninflated point-wise anomaly 
detection performance and includes explainability via CART decision tree surrogates.

DATASET INFORMATION
-------------------
1. Benchmark Name: Secure Water Treatment (SWaT) Dataset
2. File Required: SWaT_dataset.csv
3. Target Schema:
   - Numerical Sensor Columns (N1–N25): Continuous metric inputs.
   - Categorical Actuator Columns (C1–C20): Discrete control states.
   - Temporal Features: Transformed Timestamp, Month, Year mappings.
   - Label Column: 'AttackLabel' ('Normal' vs 'Attack').
4. Dataset Download & Reference Link:
   The dataset can be publicly accessed and downloaded from Kaggle at:
   https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system

   Place the downloaded 'SWaT_dataset.csv' file directly in the project root 
   directory prior to running the code script.

CODE INFORMATION
----------------
Main Execution Script:
- 'Contexual Learning-SWAT_V1.py': Complete end-to-end processing script.

Script Pipeline Workflow:
1. Data Preprocessing & Cleaning: Forward-fills missing values, removes zero-variance 
   features, sorts chronologically, and handles cyclical datetime transformations.
2. Heterogeneous Feature Encoding: Scaled via RobustScaler and OneHotEncoder.
3. Model Training: Trains a PyTorch progressive symmetric bottleneck autoencoder 
   (HighDimensionalBottleneckAE) on nominal sequences using AdamW optimizer.
4. Tri-Metric Score Fusion: Calculates MSE reconstruction loss, centered 
   Mahalanobis distance, and L1 moving-average residual velocity.
5. EVT Decision Gating: Fits tail excesses to a Generalized Pareto Distribution 
   (GPD) using Peaks-Over-Threshold (POT) to set statistical decision bounds.
6. Evaluation Metrics: Computes unadjusted and Point-Adjusted (PA) Precision, 
   Recall, F1-Score, ROC-AUC, and PR-AUC. Exports dual-panel evaluation plots.
7. Explainable AI: Trains a surrogate CART decision tree on model predictions 
   and prints human-readable decision rules.

REQUIREMENTS
------------
System Hardware Requirements:
- Operating System: Linux, macOS, or Windows 10/11
- Python Version: Python 3.9 or higher
- System RAM: 16 GB minimum
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
1. Feature Split: Chronological 68:32 train/test split on preprocessed SWaT data.
2. Latent Compression: Autoencoder bottleneck dimension set to LATENT_DIM = 8.
3. Scoring Weights: Fused Score = (w1 * Reconstruction) + (w2 * Mahalanobis) 
   + (w3 * Residual Velocity).
4. Automated Thresholding: Dynamic GPD fitting at 90th percentile.
5. Explainability Extraction: DecisionTreeRegressor/Classifier surrogate with 
   max_depth = 4.

USAGE INSTRUCTIONS
------------------
1. Setup Repository Structure:
   Organize your directory as follows:

   project_root/
   ├── Contexual Learning-SWAT_V1.py
   ├── SWaT_dataset.csv
   └── README.txt

2. Run Pipeline Script:
   Execute the main Python script from your command prompt or terminal:

   python "Contexual Learning-SWAT_V1.py"

3. Execution Outputs Generated:
   - Console Terminal: Logs dataset dimensions, training progress, EVT tail 
     parameters (c, scale, threshold), raw vs point-adjusted classification 
     reports, and decision tree surrogate rules.
   - Saved Plot Files:
     * evaluation_roc_curve.png
     * evaluation_pr_curve.png
     * swat_dual_evaluation_curves.png
     * swat_dual_evaluation_curves.pdf

CITATIONS
---------
If you use this code, model architecture, or dataset setup in your research, 
please cite the associated manuscript:

[Insert Journal Reference / Manuscript Title here upon publication]

Dataset Download and Reference:
https://www.kaggle.com/datasets/vishala28/swat-dataset-secure-water-treatment-system

Goh, J., Adepu, S., Junejo, K. N., and Mathur, A. (2016). A dataset to support research in the design of788
secure water treatment systems. In International conference on critical information infrastructures789
security, pages 88–99. Springer

LICENSE & CONTRIBUTION GUIDELINES
---------------------------------
License:
Unlicensed. This codebase is provided "as is" for public academic evaluation, 
peer-review replication, and research purposes without express warranty.

Contribution Guidelines:
Pull requests and issue reports are welcome for bug fixes or optimizations 
related to dataset preprocessing or PyTorch model scalability.
================================================================================