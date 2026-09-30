================================================================================
README: 3-Prong Anomaly Detection Framework (ULB Credit Card Fraud Dataset)
================================================================================

TITLE
-----
3-Prong Anomaly Detection Framework
Code Artifact & Benchmark Execution Package for the ULB Credit Card Fraud Dataset

DESCRIPTION
-----------
This repository contains the official Python code implementation for the 
3-Prong Anomaly Detection Framework adapted for the Université Libre de Bruxelles 
(ULB) Credit Card Fraud benchmark dataset. The system combines a progressive 
bottleneck autoencoder with a three-pronged anomaly scoring engine (Reconstruction 
Error, Mahalanobis Distance over continuous feature dimensions, and L1 Historical 
Differential Velocity) alongside Extreme Value Theory (EVT/POT) tail thresholding.

The package contains a point-in-time / zero-window pipeline to evaluate unadjusted 
and point-adjusted tabular/financial fraud detection performance, complete with CART 
surrogate decision tree explainability.

DATASET INFORMATION
-------------------
1. Benchmark Name: ULB Credit Card Fraud Detection Dataset
2. Files Required: 
   - creditcard.csv
3. Target Schema & Data Ingestion:
   - Numerical PCA Features: V1 through V28 (continuous latent features).
   - Transaction Telemetry: 'Time' (seconds elapsed) and 'Amount' (transaction value).
   - Target Label: 'Class' (0 = Normal/Legitimate transaction, 1 = Fraudulent anomaly).
   - Cleaning & Alignment: Deduplicates rows, handles missing/null values, and sorts 
     chronologically by 'Time'.
   - Train/Test Split: Splits pure normal records (80% train, 10% held-out test normal) 
     and combines held-out normal records with 100% of fraud anomalies into a realistic test set.
4. Dataset Access:
   The ULB Credit Card Fraud dataset is publicly available via Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

   Place the raw CSV file ('creditcard.csv') directly in the project root directory 
   prior to executing the script.

CODE INFORMATION
----------------
Main Execution Script:
- 'Contexual Learning-ULB_V1.py': Point-in-time / Zero-window execution pipeline.

Script Pipeline Workflow:
1. Data Preprocessing & Alignment: Performs duplicate removal, chronological sorting, 
   and splits normal baseline data from anomalous transactions.
2. Continuous Feature Encoding: Scales all continuous numerical variables using a custom 
   Guardrailed Robust Scaler / RobustScaler.
3. Model Training: Trains a PyTorch progressive bottleneck autoencoder 
   (HighDimensionalBottleneckAE, LATENT_DIM = 16) on nominal training sequence data using 
   the AdamW optimizer with Cosine Annealing learning rate scheduling.
4. Tri-Metric Score Fusion: Computes MSE reconstruction loss, centered Mahalanobis 
   distance with covariance precision matrix regularization, and L1 historical moving-average 
   residual velocity. Standardizes scores using robust Median Absolute Deviation (MAD) Z-scores.
5. EVT Decision Gating: Fits extreme tail excesses above high percentiles to a Generalized 
   Pareto Distribution (GPD) using Peaks-Over-Threshold (POT) to establish dynamic decision boundaries.
6. Evaluation Metrics: Computes raw unadjusted and Point-Adjusted (PA) Precision, 
   Recall, F1-Score, ROC-AUC, and PR-AUC.
7. Explainable AI: Fits a surrogate CART decision tree classifier on model predictions 
   to extract human-readable logic rules for triggered fraud events.

REQUIREMENTS
------------
System Hardware Requirements:
- Operating System: Linux, macOS, or Windows 10/11
- Python Version: Python 3.9 or higher
- System RAM: 16 GB minimum (32 GB recommended)
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
1. Feature Scaling & Data Partitioning: Continuous transaction attributes are scaled using 
   RobustScaler and converted into native PyTorch tensors.
2. Bottleneck Compression: Autoencoder compresses inputs through symmetric layers 
   down to a compact bottleneck representation (LATENT_DIM = 8).
3. Score Fusion: 
   Anomaly Score = (w1 * Reconstruction) + (w2 * Mahalanobis) + (w3 * Residual Velocity).
4. Automated Thresholding: Dynamic GPD parameter estimation via Peaks-Over-Threshold (POT).
5. Explainability Extraction: DecisionTreeClassifier surrogate (max_depth = 4) to output 
   human-interpretable rule hierarchies.

USAGE INSTRUCTIONS
------------------
1. Setup Repository Structure:
   Organize your directory as follows:

   project_root/
   ├── Contexual Learning-ULB_V1.py
   ├── creditcard.csv
   └── README.txt

2. Run Pipeline Script:
   Execute the pipeline from your terminal:

   # To execute the ULB Credit Card Fraud detection pipeline:
   python "Contexual Learning-ULB_V1.py"

3. Execution Outputs Generated:
   - Console Terminal: Logs dataset ingestion statistics, model architecture dimensions, 
     EVT tail parameters, raw vs point-adjusted classification metrics, and extracted decision 
     tree surrogate rules.
   - Output Visualizations: Generates and saves evaluation figures ('evaluation_roc_curve.png', 
     'evaluation_pr_curve.png', and 'evaluation_roc_pr_combined.png') to the local workspace.

CITATIONS
---------
If you use this code, model architecture, or dataset setup in your research, 
please cite the associated manuscript:

[Insert Journal Reference / Manuscript Title here upon publication]

Dataset Reference:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating Probability 
with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence 
and Data Mining (CIDM), IEEE.

LICENSE & CONTRIBUTION GUIDELINES
---------------------------------
License:
Unlicensed. This codebase is provided "as is" for public academic evaluation, 
peer-review replication, and research purposes without express warranty.

Contribution Guidelines:
Pull requests and issue reports are welcome for bug fixes, performance improvements, 
or optimizations related to high-dimensional tabular data scaling and PyTorch pipelines.
================================================================================