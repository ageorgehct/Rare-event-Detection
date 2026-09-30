#!/usr/bin/env python
# coding: utf-8

# In[92]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import genpareto
import warnings
from decimal import Decimal
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)


# In[93]:


# =========================================================================
# EXPERIMENTAL CONFIGURATION SETUP
# =========================================================================
SAMPLE_SIZE = 600000
ANOMALY_COUNT = 5000
LATENT_DIM = 8

import numpy as np

class GuardrailedRobustScaler:
    def __init__(self, eps=1.0): # Changed from 1e-5 to 1.0
        self.eps = eps
        self.medians = None
        self.iqrs = None

    def fit(self, X):
        # Convert to numpy array if it's a DataFrame
        X_arr = np.asarray(X)
        self.medians = np.median(X_arr, axis=0)
        q75, q25 = np.percentile(X_arr, [75, 25], axis=0)
        self.iqrs = q75 - q25
        
        # Guardrail: If IQR is 0, fall back to a scale of 1.0 (raw scale preservation)
        self.iqrs = np.where(self.iqrs == 0.0, self.eps, self.iqrs)
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        X_scaled = (X_arr - self.medians) / self.iqrs
        
        # CRITICAL REPAIR: Clip any infinite outliers to stable float limits
        X_scaled = np.clip(X_scaled, -1e5, 1e5)
        
        # Second safety wall: Replace any lingering NaNs/Infs with 0.0
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# In[94]:


def load_credit_card_fraud_data(path):
    
    # 2. Load into standard DataFrame
    df = pd.read_csv(path, low_memory=False)
    
    # 3. Basic Production Cleaning & Verification
    df.columns = df.columns.str.strip()
    initial_rows = len(df)
    
    # Drop rows where critical targets or values are missing
    df = df.dropna()
    
    # Remove strict duplicates to ensure the spatial mapping isn't artificially biased
    df = df.drop_duplicates()
    
    print("\n" + "="*50 + "\n DATASET INGESTION SUMMARY\n" + "="*50)
    print(f"Raw Samples Processed:     {initial_rows}")
    print(f"Cleaned Samples Retained:  {len(df)} (Dropped {initial_rows - len(df)} duplicates/nulls)")
    print(f"Total Spatial Features:    {df.shape[1] - 1} (Excluding Class target)")
    print(f"Total Fraud Anomalies:     {df['Class'].sum()} (Imbalance Ratio: {(df['Class'].sum() / len(df)) * 100:.4f}%)")
    print("="*50)
    
    return df

# Execute Ingestion
filepath = normal_file = r"creditcard.csv"
df_creditcard = load_credit_card_fraud_data(filepath)

# 1. Check which specific columns contain NaNs
nan_counts = df_creditcard.isna().sum()
print("Columns with missing values:\n", nan_counts[nan_counts > 0])

# Sort the DataFrame by the Timestamp column in ascending order
df_creditcard = df_creditcard.sort_values(by='Time', ascending=True)
# Reset the index to ensure sequential row numbers after sorting
df_creditcard = df_creditcard.reset_index(drop=True)

print(df_creditcard.columns.tolist())


# In[95]:


# 1. Separate features (X) and target labels (y)
X_raw = df_creditcard.drop(columns=['Class']).values
y_raw = df_creditcard['Class'].values

# 2. Isolate normal data from anomalous data
normal_mask = (y_raw == 0)
anomalous_mask = (y_raw == 1)

X_normal = X_raw[normal_mask]
X_anomalous = X_raw[anomalous_mask]

print (f"length of normal is {len(X_normal)} and length of anomalous is {len(X_anomalous)}");
# 3. Split the normal data (80% for training baseline, 10% held out for testing)
X_train, X_test_normal = train_test_split(X_normal, test_size=0.10, random_state=42)

# 4. Construct the realistic Test Set (20% normal data + 100% of the rare anomalies)
X_test = np.vstack([X_test_normal, X_anomalous])
#X_test = np.vstack([X_normal, X_anomalous])

# 5. Create the corresponding Ground Truth Target Arrays
y_train = np.zeros(len(X_train), dtype=int)
y_test = np.concatenate([np.zeros(len(X_test_normal), dtype=int), np.ones(len(X_anomalous), dtype=int)])
#y_test = np.concatenate([np.zeros(len(X_normal), dtype=int), np.ones(len(X_anomalous), dtype=int)])
print (f"Sum of y_test is {sum(y_test)} and len of y_test is {len(y_test)}")

print("\n" + "="*50 + "\n PROPOSED STRUCTURAL DATA SPLIT\n" + "="*50)
print(f"📊 X_train Shape (Pure Normal Baseline): {X_train.shape}")
print(f"📊 y_train Shape (All Zeros):             {y_train.shape}")
print(f"📊 X_test Shape  (Realistic Mixed Test):  {X_test.shape}")
print(f"📊 y_test Shape  (Anomalies + Normal):   {y_test.shape}")
print(f"📉 Test Set Imbalance (Fraud Ratio):     {(sum(y_test)/len(y_test))*100:.4f}%")
print("="*50)


# In[96]:


# =========================================================================
# STEP 1: HETEROGENEOUS ENCODING (CORRECTED BY INDEX SLICING)
# =========================================================================

# 1. Continuous Metric Normalization using GuardrailedRobustScaler
# Directly processes the arrays since they contain all numerical features
#scaler = GuardrailedRobustScaler()
scaler = RobustScaler()
X_train_processed = scaler.fit_transform(X_train)
print("X Train processed shape is:", X_train_processed.shape)

X_test_processed = scaler.transform(X_test)
print("X Test processed shape is:", X_test_processed.shape)

# 2. Convert Processed Matrix directly to Native PyTorch Tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
print("X Train tensor shape is:", X_train_tensor.shape)

X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
print("X Test tensor shape is:", X_test_tensor.shape)

# =========================================================================
# CONTEXT INDICES DEFINITION
# =========================================================================
# Dynamically maps to all available features in your input array (e.g., all 30 columns)
context_indices = list(range(0, X_train_processed.shape[1]))
print(f"Context indices initialized for {len(context_indices)} features.")

print("-----Done-----------")


# In[97]:


# =========================================================================
# STEP 2: REPRESENTATION LEARNING (FIXED LATENT SIZE = 8)
# =========================================================================

# =========================================================================
# STEP 0: DEVICE ALLOCATION (TARGETING THE RTX 4000)
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Utilizing execution target device: {device}")
if device.type == 'cuda':
    print(f"   -> Active GPU Profile: {torch.cuda.get_device_name(0)}")
    
import torch
import torch.nn as nn

class HighDimensionalBottleneckAE(nn.Module):
    """
    Generic, Dataset-Agnostic Bottleneck Autoencoder.
    Uses continuous smooth activations (GELU) and Batch Normalization
    to preserve feature-wise physical scales without dataset-specific assumptions.
    """
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        
        # Progressive symmetric dimension scaling based solely on input dimension
        hidden_dim_1 = max(input_dim // 2, latent_dim * 4)
        hidden_dim_2 = max(input_dim // 4, latent_dim * 2)
        
        # Encoder (NO Dropout, GELU activation for smooth gradients)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.GELU(),
            
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.GELU(),
            
            nn.Linear(hidden_dim_2, latent_dim)  # Linear latent projection
        )
        
        # Decoder (Mirroring structure, clean linear output)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.GELU(),
            
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.GELU(),
            
            nn.Linear(hidden_dim_1, input_dim)  # Pure linear reconstruction
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


# Clear the old model from memory
try:
    del model
except NameError:
    pass  # Model wasn't initialized, move on silently

input_feature_dimension = X_train_processed.shape[1]
print (" Input Dimension is ",input_feature_dimension)

model = HighDimensionalBottleneckAE(input_dim=input_feature_dimension, latent_dim=LATENT_DIM)
model = model.to(device)

# 1. Initialize Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=1e-3,             # Slightly higher initial learning rate for faster convergence
    weight_decay=1e-4,   # L2 weight decay
    eps=1e-8
)

# 2. Add Learning Rate Scheduler (Smooth decay across epochs)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=150,           # Total training epochs
    eta_min=1e-5         # Minimum learning rate target
)


# 2. Push your primary training tensor onto the device memory
# (Do this here if the entire matrix fits easily inside your VRAM)
X_train_tensor = X_train_tensor.to(device)

# Train on the expanded baseline subset
# OPTIMIZED TRAINING LOOP
model.train()
for epoch in range(200):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    predictions = model(X_train_tensor)
    loss = torch.mean((X_train_tensor - predictions) ** 2)
    
    loss.backward()
    optimizer.step()
    scheduler.step()    # Step scheduler per epoch
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/200] - Loss: {loss.item():.6f}")

# Clean up local references immediately after the loop ends
del predictions, loss
gc.collect()
torch.cuda.empty_cache()


# In[56]:


print(f"Shape of X_train_tensor  is {X_test_tensor .shape}")


# In[98]:


# =========================================================================
# SCRIPT REPAIR: FORCE SLIDING WINDOW TENSORS TO RTX 4000 VRAM
# =========================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)

model = model.to(device)

# =========================================================================
# STEP 3: THREE-PRONGED MULTI-DIMENSIONAL ANOMALY SCORING (GPU-OPTIMIZED)
# =========================================================================
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

# --- 1. CONTINUOUS-ONLY MAHALANOBIS MEAN & PRECISION CALCULATIONS ---
# CHANGED: Ensure context_indices strictly contains continuous numeric features 
# to prevent matrix instabilities from discrete categorical switches

print("------------------------", type(X_train_processed))
print(context_indices);
v_train_baseline = X_train_processed[:, context_indices]

mu_v = torch.tensor(np.mean(v_train_baseline, axis=0), dtype=torch.float32)
cov_matrix = np.cov(v_train_baseline, rowvar=False) + 1e-5 * np.eye(len(context_indices))
precision_matrix_v = torch.tensor(np.linalg.inv(cov_matrix), dtype=torch.float32)

def extract_scores(X_raw, X_hat):
    num_samples = X_raw.size(0)
    
    # --- PRONG A: MEAN SQUARED ERROR (MSE) ---
    R = torch.mean((X_raw - X_hat) ** 2, dim=-1)

    # --- PRONG B: CORRECTED MAHALANOBIS DISTANCE ---
    mu_v_cuda = mu_v.to(X_raw.device)
    precision_matrix_v_cuda = precision_matrix_v.to(X_raw.device)
    
    # Slice only the continuous numeric sub-space
    v_centered = (X_raw[:, context_indices] - mu_v_cuda).unsqueeze(1)
    prec_expanded = precision_matrix_v_cuda.unsqueeze(0).expand(num_samples, -1, -1)
    
    MD = torch.bmm(torch.bmm(v_centered, prec_expanded), v_centered.transpose(1, 2)).squeeze(-1).squeeze(-1)
    MD = torch.sqrt(torch.clamp(MD, min=0.0))

    # --- PRONG C: VECTORIZED L1 HISTORICAL DIFFERENTIAL (CORRECTED) ---
    K = 30
    D = X_raw - X_hat  
    
    # CHANGED: Replaced 'replicate' with value=0.0 to guarantee true causal histories
    D_padded = torch.nn.functional.pad(D.t().unsqueeze(0), (K-1, 0), mode='constant', value=0.0)
    historical_mean_tensor = torch.nn.functional.avg_pool1d(D_padded, kernel_size=K, stride=1).squeeze(0).t()
    
    L1_delta = torch.norm(D - historical_mean_tensor, p=1, dim=-1)

    return R.cpu().numpy(), MD.cpu().numpy(), L1_delta.cpu().numpy()

print(f"length of train_tensor is {len(X_train_tensor)} and length of train_preds is {len(train_preds)}")
print("Train Tensor:", torch.isnan(X_train_tensor).any().item())
print("Test Tensor:", torch.isnan(X_test_tensor).any().item())

R_tr, MD_tr, L1_tr = extract_scores(X_train_tensor, train_preds)
R_ts, MD_ts, L1_ts = extract_scores(X_test_tensor, test_preds)

# --- 2. ROBUST MIN-MAX UNIFICATION MATRIX ---
'''
stats = {
    'R_min': R_tr.min(), 'R_max': R_tr.max(),
    'MD_min': MD_tr.min(), 'MD_max': MD_tr.max(),
    'L1_min': L1_tr.min(), 'L1_max': L1_tr.max()
}

r_range = stats['R_max'] - stats['R_min']
md_range = stats['MD_max'] - stats['MD_min']
l1_range = stats['L1_max'] - stats['L1_min']


r_denom = r_range if r_range > 0 else 1.0
md_denom = md_range if md_range > 0 else 1.0
l1_denom = l1_range if l1_range > 0 else 1.0

print("\n" + "="*60 + "\n Training Data set\n" + "="*60)
R_tr_norm = (R_tr - stats['R_min']) / r_denom
print(f"Max of R_tr is {np.amax(R_tr_norm)},range is {np.ptp(R_tr_norm)}, median is {np.median(R_tr_norm)}, r_denom is {r_denom}")

MD_tr_norm = (MD_tr - stats['MD_min']) / md_denom
print(f"Max of MD_tr is {np.amax(MD_tr_norm)},range is {np.ptp(MD_tr_norm)}, median is {np.median(MD_tr_norm)}, md_denom is {md_denom}")

L1_tr_norm = (L1_tr - stats['L1_min']) / l1_denom
print(f"Max of L1_tr is {np.amax(L1_tr_norm)},range is {np.ptp(L1_tr_norm)}, median is {np.median(L1_tr_norm)}, l1_denom is {l1_denom}")

A_train = (0.35 * R_tr_norm) + (0.30 * MD_tr_norm) + (0.35 * L1_tr_norm)
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")

print("\n" + "="*60 + "\n Test Data set\n" + "="*60)
R_ts_norm = np.clip((R_ts - stats['R_min']) / r_denom, 0, None)
print(f"Max of R_ts is {np.amax(R_ts_norm)},range is {np.ptp(R_ts_norm)}, median is {np.median(R_ts_norm)}, r_denom is {r_denom}")

MD_ts_norm = np.clip((MD_ts - stats['MD_min']) / md_denom, 0, None)
print(f"Max of MD_ts is {np.amax(MD_ts_norm)},range is {np.ptp(MD_ts_norm)}, median is {np.median(MD_ts_norm)}, md_denom is {md_denom}")

L1_ts_norm = np.clip((L1_ts - stats['L1_min']) / l1_denom, 0, None)
print(f"Max of L1_ts is {np.amax(L1_ts_norm)},range is {np.ptp(L1_ts_norm)}, median is {np.median(L1_ts_norm)}, l1_denom is {l1_denom}")

A_test = (0.35 * R_ts_norm) + (0.30 * MD_ts_norm) + (0.35 * L1_ts_norm)
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(f"Scoring pipeline complete. Total test anomaly indices mapped: {len(A_test)}")

'''
# =========================================================================
# UPGRADED STEP 4: ROBUST MEDIAN-ABS-DEVIATION (MAD) UNIFICATION 
# =========================================================================
# Min-Max is highly sensitive to WADI training spikes. 
# Robust Z-Score scales features using non-parametric medians.

def compute_robust_z_score(tr_metric, ts_metric):
    median = np.median(tr_metric)
    # Calculate Median Absolute Deviation (MAD)
    mad = np.median(np.abs(tr_metric - median))
    # Prevent divide-by-zero for hyper-stable channels
    mad = mad if mad > 1e-6 else 1e-6
    
    # Scale both using training invariants (Standardized Distance)
    tr_norm = (tr_metric - median) / (1.4826 * mad)
    ts_norm = (ts_metric - median) / (1.4826 * mad)
    return tr_norm, ts_norm

# 2. Apply a Log-Transform to compress variance and stabilize the training baseline
R_tr_log, R_ts_log   = np.log1p(R_tr), np.log1p(R_ts)
MD_tr_log, MD_ts_log = np.log1p(MD_tr), np.log1p(MD_ts)
L1_tr_log, L1_ts_log = np.log1p(L1_tr), np.log1p(L1_ts)

# 3. Apply your Robust Scaling function to the log-transformed variables
R_tr_norm, R_ts_norm   = compute_robust_z_score(R_tr, R_ts)
MD_tr_norm, MD_ts_norm = compute_robust_z_score(MD_tr, MD_ts)
L1_tr_norm, L1_ts_norm = compute_robust_z_score(L1_tr, L1_ts)

print("\n" + "="*60 + "\n Robust Standardized Training Dataset\n" + "="*60)
print(f"MSE  Tr -> Median: {np.median(R_tr_norm):.4f}, Max: {R_tr_norm.max():.4f}, Std: {np.std(R_tr_norm):.4f}, Range is {np.ptp(R_tr_norm):.2f}, 99p is {np.percentile(R_tr_norm, 99):.2f}")
print(f"MD   Tr -> Median: {np.median(MD_tr_norm):.4f}, Max: {MD_tr_norm.max():.4f}, Std: {np.std(MD_tr_norm):.4f}, Range is {np.ptp(MD_tr_norm):.2f}")
print(f"L1   Tr -> Median: {np.median(L1_tr_norm):.4f}, Max: {L1_tr_norm.max():.4f}, Std: {np.std(L1_tr_norm):.4f}, Range is {np.ptp(L1_tr_norm):.2f}")

# Re-combine using balanced weights
A_train = (0.30 * R_tr_norm) + (0.45 * MD_tr_norm) + (0.25 * L1_tr_norm)
print(f"Unified A_train Median: {np.median(A_train):.4f}, Max: {A_train.max():.4f}, Std: {np.std(A_train):.4f}, Range is {np.ptp(A_train):.2f}, 99p is {np.percentile(A_train, 99):.2f} ")

print("\n" + "="*60 + "\n Robust Standardized Test Dataset\n" + "="*60)
print(f"MSE  Ts -> Median: {np.median(R_ts_norm):.4f}, Max: {R_ts_norm.max():.4f}, Std: {np.std(R_ts_norm):.4f}, Range is {np.ptp(R_ts_norm):.2f}, 99p is {np.percentile(R_ts_norm, 99):.2f}")
print(f"MD   Ts -> Median: {np.median(MD_ts_norm):.4f}, Max: {MD_ts_norm.max():.4f}, Std: {np.std(MD_ts_norm):.4f}, Range is {np.ptp(MD_ts_norm):.2f}")
print(f"L1   Ts -> Median: {np.median(L1_ts_norm):.4f}, Max: {L1_ts_norm.max():.4f}, Std: {np.std(L1_ts_norm):.4f}, Range is {np.ptp(L1_ts_norm):.2f}")

A_test = (0.40 * R_ts_norm) + (0.30 * MD_ts_norm) + (0.30 * L1_ts_norm)
print(f"Unified A_test  Median: {np.median(A_test):.4f}, Max: {A_test.max():.4f}, Std: {np.std(A_test):.4f}, Range is {np.ptp(A_test):.2f},99p is {np.percentile(A_test, 99):.2f}")



# =========================================================================
# SYSTEM DEBUG BOUNDARY: TRACKING NAN INF CORRUPTION
# =========================================================================
print("🔍 === STARTING PIPELINE INTEGRITY CHECK ===")

# 1. Check Input Tensors on GPU
print(f"X_train_tensor -> Is NaN? {torch.isnan(X_train_tensor).any().item()} | Is Inf? {torch.isinf(X_train_tensor).any().item()} | Shape: {X_train_tensor.shape}")
print(f"X_test_tensor  -> Is NaN? {torch.isnan(X_test_tensor).any().item()} | Is Inf? {torch.isinf(X_test_tensor).any().item()} | Shape: {X_test_tensor.shape}")

# 2. Check Model Predictions
print(f"train_preds    -> Is NaN? {torch.isnan(train_preds).any().item()} | Is Inf? {torch.isinf(train_preds).any().item()} | Shape: {train_preds.shape}")
print(f"test_preds     -> Is NaN? {torch.isnan(test_preds).any().item()} | Is Inf? {torch.isinf(test_preds).any().item()} | Shape: {test_preds.shape}")

# 3. Check Covariance Matrix & Precision Matrix Invariants for Mahalanobis
print(f"cov_matrix     -> Is NaN? {np.isnan(cov_matrix).any()} | Is Inf? {np.isinf(cov_matrix).any()}")
print(f"precision_v    -> Is NaN? {torch.isnan(precision_matrix_v).any()} | Is Inf? {torch.isinf(precision_matrix_v).any()}")

# 4. Locate the exact column indices if inputs are corrupted
if torch.isnan(X_train_tensor).any():
    nan_cols = torch.where(torch.isnan(X_train_tensor))[1].unique().cpu().numpy()
    print(f"⚠️ Crucial: Training NaN values found in processed feature indices: {nan_cols}")
    # Map back to names if possible
    # print([num_cols + list(encoder.get_feature_names_out())][i] for i in nan_cols)

print("🔍 === END OF PIPELINE INTEGRITY CHECK ===\n")


# In[100]:


print("Train Tensor:", torch.isnan(X_train_tensor).any().item())
print("Test Tensor:", torch.isnan(X_test_tensor).any().item())

num_nans = torch.isnan(X_train_tensor).sum().item()
print("Number of NaNs:", num_nans)


# In[101]:


# =========================================================================
# STEP 4: AUTOMATED GATING (EVT/POT METHOD)
# =========================================================================
gamma_threshold = np.percentile(A_train, 99) # Tightened slightly to fit the 0.4% density profile
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(" Gamma Threshold is ", gamma_threshold);
result = [x for x in A_train if x > gamma_threshold]  # one-liner
print(len(result))  # [6, 7, 7]
y_gate = (A_test > gamma_threshold).astype(int)
print(len(y_gate))
total_ones = np.sum(y_gate == 1)
print(total_ones)


# In[102]:


gamma_threshold = np.percentile(A_train, 99) # Tightened slightly to fit the 0.4% density profile
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(" Gamma Threshold is ", gamma_threshold);

# =========================================================================
# STEP 4: REVISED AUTOMATED GATING (EVT/POT METHOD)
# =========================================================================
# 1. Set an initial loose threshold (t) to capture the tail candidate pool
# Typically 98th percentile is standard to isolate extreme values for fitting
t = np.percentile(A_train, 99)
tail_excesses = A_train[A_train > t] - t

# 2. Fit the tail excesses to the Generalized Pareto Distribution (GPD)
# c = shape parameter, loc = location parameter, scale = scale parameter
c, loc, scale = genpareto.fit(tail_excesses, floc=0)
print(f"Pareto distribution: c is {c}, loc is {loc}, scale is {scale}")
print(f"Shape parameter (c): {c:.4f}  ({'Heavy-tailed' if c > 0 else 'Light/Bounded-tailed'})")

# 3. Define your target risk probability (q) 
#q=1e-4 #means you accept a 0.01% theoretical probability of a false alarm
#q = 2e-3
#q = 0.018
#q = 0.0135
q = 0.0085
N = len(A_train)
N_t = len(tail_excesses)

# 4. Mathematically compute the dynamic EVT Threshold (gamma)
# Stabilized using log-space mapping to protect the floating-point registry from exploding
epsilon = (N * q) / N_t
print(f"N is {N}, N_t is {N_t}, epsilon is {epsilon}")

if abs(c) < 1e-12:
    gamma_threshold = t - scale * np.log(epsilon)
    print("c==0 (or converging to 0)")
else:
    print("c!=0")
    try:
        # Numerically stable equivalent of: t + (scale / c) * (epsilon**(-c) - 1)
        gamma_threshold = t + (scale / c) * (np.exp(-c * np.log(epsilon)) - 1.0)
    except OverflowError:
        print("Warning: Floating-point overflow detected. Falling back to log asymptotic limit.")
        gamma_threshold = t - scale * np.log(epsilon)
    print(f"Gamma threshold before correction is {gamma_threshold}")

# =========================================================================
# MATHEMATICALLY VALIDATED GATING GUARDRAIL (Q1 JOURNAL COMPLIANT)
# =========================================================================
is_valid_evt = True

# Check A: Ensure the maximum likelihood optimizer didn't fail or yield non-positive scale
if scale <= 0 or np.isnan(gamma_threshold) or np.isinf(gamma_threshold) or gamma_threshold > max(A_train):
    is_valid_evt = False

# Check B: If c < 0 (light-tailed regime), the distribution has a strict upper limit.
# The calculated threshold cannot physically exceed this asymptote.
elif c < 0:
    theoretical_upper_bound = t - (scale / c)
    if gamma_threshold > theoretical_upper_bound:
        is_valid_evt = False

# Fallback occurs ONLY if the GPD distribution constraints are mathematically broken
if not is_valid_evt:
    print("Warning: GPD parameter estimation violated distribution support. Applying empirical tail safeguard.")
    gamma_threshold = np.percentile(A_train, 99.9)
    print(f"Gamma threshold after guardrail correction is {gamma_threshold}")
else:
    print("EVT threshold successfully validated against statistical distribution constraints.")

print(f"Initial Tail Threshold (t): {t:.4f}")
print(f"Calculated GPD Shape (c): {c:.4f}, Scale: {scale:.4f}, Tail Excess samples: {N_t}, Total samples: {N}")
print(f"Final Mathematically Defended EVT Threshold (gamma): {gamma_threshold:.4f}")

# Apply EWMA smoothing to the unified test score
# span=5 acts as a soft filter over roughly 5 timesteps
#A_test_series = pd.Series(A_test)
#A_test_smoothed = A_test_series.ewm(span=5, adjust=False).mean().values
#A_test = A_test_smoothed

# 5. Evaluate the testing data against the mathematically robust threshold
y_gate = (A_test > gamma_threshold).astype(int)
y_gate_train = (A_train > gamma_threshold).astype(int)
y_gate_full = np.concatenate([y_gate_train, y_gate])

total_ones = np.sum(y_gate_full == 1)
print("Anomaly count is ",total_ones)

# Diagnostic Prints
print(f"Total Test Samples: {len(y_gate)}")
print(f"Total Anomalies Flagged (y_gate == 1): {np.sum(y_gate == 1)}")


# In[23]:


# =========================================================================
# STEP 5: EXPLAINABLE AI SURROGATE EXTRACTION (50-FEATURE MATRIX alignment)
# =========================================================================
df_test_aligned = df_test.copy()

numerical_features = [f'N{i}' for i in range(1, 26)]
categorical_features = [f'C{j}' for j in range(1, 21)] # Keeps original names C1-C8
temporal_features = ['MONTH', 'YEAR']

# Overwrite the text with numbers ONLY inside the isolated dataframe copy
for col_name in categorical_features:
    df_test_aligned[col_name] = df_test_aligned[col_name].astype('category').cat.codes

# Now the feature names perfectly match your original dataset columns
surrogate_features = numerical_features + categorical_features + temporal_features

cart_surrogate = DecisionTreeClassifier(max_depth=4, random_state=42)
cart_surrogate.fit(df_test_aligned[surrogate_features], y_gate)

print(export_text(cart_surrogate, feature_names=surrogate_features))
print("-------Complete--------------------------------")


# In[125]:


# =========================================================================
# STEP 5 (REVISED): GLOBAL SURROGATE EXTRACTION ON FULL DATASET
# =========================================================================
# 1. Combine training and testing data back into a single unified matrix
df_full = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

numerical_features = [f'N{i}' for i in range(1, 26)]
categorical_features = [f'C{j}' for j in range(1, 21)] # Keeps original names C1-C8
temporal_features = ['MONTH', 'YEAR']

total_ones = np.sum(y_gate_full == 1)
print("Anomaly count is ",total_ones)

# Now the feature names perfectly match your original dataset columns
surrogate_features = numerical_features + categorical_features + temporal_features

#global_cart_surrogate = DecisionTreeClassifier(max_depth=4, random_state=42)
#global_cart_surrogate.fit(df_full[surrogate_features], y_gate_full)

print("\n" + "="*60 + "\nEXTRACTED GLOBAL HUMAN-READABLE INTERPRETATION RULES (FULL SET)\n" + "="*60)
print(export_text(global_cart_surrogate, feature_names=surrogate_features))


# In[103]:


# =========================================================================
# STEP 6: RAW UNADJUSTED PERFORMANCE TRACKING
# =========================================================================
y_test_aligned = (y_test> 0).astype(int)

# use full data set
#y_test_aligned_full = (y_binary > 0).astype(int)

#Merge A_test and A_train
print(type(A_train),A_train.shape)
print(type(A_test),A_test.shape)

#A_full = np.concatenate((A_train, A_test), axis=0)


# In[104]:


#print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
#print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")

# use full data set
print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(classification_report(y_test_aligned_full, y_gate_full, target_names=["Normal Matrix State", "Rare Event Target"]))
print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")


# In[89]:


# 1. Compute the raw unadjusted ROC-AUC (for comparison)
#roc_auc = roc_auc_score(y_test_aligned_full, A_full)
roc_auc = roc_auc_score(y_test_aligned, A_test)

# 2. Generate the Precision and Recall arrays across all possible thresholds
# Note: precision_recall_curve expects true labels and continuous scores/probabilities
#precisions, recalls, thresholds = precision_recall_curve(y_test_aligned_full, A_full)
precisions, recalls, thresholds = precision_recall_curve(y_test_aligned, A_test)

# 3. Calculate the Area Under the Curve (AUC) using the precision and recall coordinates
pr_auc = auc(recalls, precisions)

print("============================================================")
print(f"Framework Global ROC-AUC Bounds: {roc_auc:.4f}")
print(f"Framework Rigorous PR-AUC Bounds: {pr_auc:.4f}")
print("============================================================")


# In[27]:


import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def optimize_anomaly_threshold(y_true, anomaly_scores, num_steps=1000):
    """
    Sweeps through potential thresholds to find the cutoff that maximizes the F1-score.
    
    Parameters:
    - y_true: Ground truth binary labels (0 for normal, 1 for anomaly)
    - anomaly_scores: The composite anomaly scores generated by your 3-prong framework
    - num_steps: Resolution of the threshold sweep
    """
    best_f1 = 0.0
    best_threshold = 0.0
    best_metrics = None
    
    # Generate linear steps between the minimum and maximum scores observed in the test set
    min_score = float(np.min(anomaly_scores))
    max_score = float(np.max(anomaly_scores))
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    print("🔍 Optimizing threshold boundaries...")
    print(f"Score Range: Min = {min_score:.6f} | Max = {max_score:.6f}")
    
    for t in thresholds:
        # If score > t, classify as anomaly (1)
        y_pred = (anomaly_scores > t).astype(int)
        
        # Calculate precision, recall, and F1 for the rare event class (pos_label=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', pos_label=1, zero_division=0
        )
        
        # Track the peak operational F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1
            }
            
    print("\n" + "="*50 + "\n THRESHOLD OPTIMIZATION COMPLETE\n" + "="*50)
    print(f"🎯 Optimal Threshold:        {best_threshold:.6f}")
    print(f"📈 Peak Rare Event Precision: {best_metrics['precision']:.4f}")
    print(f"📈 Peak Rare Event Recall:    {best_metrics['recall']:.4f}")
    print(f"🔥 Maximum Achievable F1:    {best_metrics['f1-score']:.4f}")
    print("="*50)
    
    return best_threshold

# =========================================================================
# EXECUTION
# =========================================================================
# Pass your test ground truth labels and your 3-prong composite test scores
target_threshold = optimize_anomaly_threshold(y_test_aligned, A_test)


# In[327]:


import numpy as np

def calibrate_evt_parameters(A_train, target_percentile=99.8):
    """
    Sweeps EVT parameters on training data to align the mathematical threshold
    with the distribution bounds of the clean baseline data.
    
    Parameters:
    - A_train: Anomaly scores generated on the normal training dataset
    - target_percentile: The target percentile of normal data we want to wrap 
                         (e.g., 99.8% means we tolerate a 0.2% normal outlier tail)
    """
    # Calculate what the empirical threshold *should* look like on training data
    empirical_target = np.percentile(A_train, target_percentile)
    
    # Define standard search spaces for EVT
    q_space = [0.97, 0.98, 0.985, 0.99, 0.995]  # Initial filtering quantile
    epsilon_space = [1e-2, 1e-3, 5e-4, 1e-4, 1e-5]  # Extreme risk parameters
    
    best_q = None
    best_eps = None
    min_diff = float('inf')
    calibrated_gamma = None

    print(f"📋 Training Target Score (at {target_percentile}th percentile): {empirical_target:.6f}")
    print("🔄 Simulating EVT Peak-Over-Threshold combinations...")

    for q in q_space:
        # Step 1: Calculate the initial threshold t based on quantile q
        t = np.percentile(A_train, q * 100)
        peaks = A_train[A_train > t] - t
        
        if len(peaks) < 10: 
            continue # Skip if there aren't enough points to fit the GPD tail Safely
            
        # Step 2: Fit the Generalized Pareto Distribution (GPD) parameters
        # (This mirrors standard POT/SPOT mathematical logic)
        mean_peaks = np.mean(peaks)
        var_peaks = np.var(peaks)
        
        # Method of Moments fit for GPD shape (gamma/k) and scale (sigma)
        sigma = 0.5 * mean_peaks * (1 + mean_peaks**2 / (var_peaks + 1e-8))
        gamma_gpd = 0.5 * (1 - mean_peaks**2 / (var_peaks + 1e-8))
        
        for eps in epsilon_space:
            # Step 3: Run the EVT Gamma calculation equation
            n = len(A_train)
            N_t = len(peaks)
            
            # Avoid math errors if shape parameter is exactly zero
            if abs(gamma_gpd) < 1e-5:
                calculated_gamma = t - sigma * np.log((n * eps) / N_t)
            else:
                calculated_gamma = t + (sigma / gamma_gpd) * (((n * eps) / N_t)**(-gamma_gpd) - 1)
                
            # Step 4: Track which parameters match our distribution targets
            diff = abs(calculated_gamma - empirical_target)
            if diff < min_diff:
                min_diff = diff
                best_q = q
                best_eps = eps
                calibrated_gamma = calculated_gamma

    print("\n" + "="*50 + "\nEVT CALIBRATION COMPLETE\n" + "="*50)
    print(f"🎯 Recommended Initial Quantile (q):  {best_q}")
    print(f"🎯 Recommended Risk Parameter (eps): {best_eps}")
    print(f"📊 Projected Training Gamma:          {calibrated_gamma:.6f} (Diff: {min_diff:.6f})")
    print("="*50)
    
    return best_q, best_eps

# =========================================================================
# RUN CALIBRATION BEFORE TESTING
# =========================================================================
# Pass your training anomaly scores to get the updated hyperparameters
recommended_q, recommended_eps = calibrate_evt_parameters(A_train, target_percentile=99.8)


# In[107]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# =========================================================================
# STEP 7: GENERATE PUBLICATION-READY EVALUATION PLOTS
# =========================================================================

# -------------------------------------------------------------------------
# 1. GENERATE AND SAVE THE ROC CURVE
# -------------------------------------------------------------------------
# Compute true positive and false positive rates across all thresholds
fpr, tpr, _ = roc_curve(y_test_aligned, A_test)
roc_auc = auc(fpr, tpr)

# Plot directly to the canvas axis
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Proposed Architecture (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--') # Random baseline reference line

# Fine-tune visual layouts for peer-review presentation
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (FPR)', fontsize=11)
plt.ylabel('True Positive Rate (TPR)', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
plt.legend(loc="lower right", frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

# Optimize spacing and export at high-resolution DPI
plt.tight_layout()
plt.show()
plt.savefig('evaluation_roc_curve.png', dpi=300)
plt.clf()  # Clear canvas cleanly to prevent figure blending

# -------------------------------------------------------------------------
# 2. GENERATE AND SAVE THE PRECISION-RECALL (PR) CURVE
# -------------------------------------------------------------------------
# Compute precision and recall dimensions across all thresholds
precision, recall, _ = precision_recall_curve(y_test_aligned, A_test)
pr_auc = auc(recall, precision)

# Plot the PR trajectory
plt.plot(recall, precision, color='purple', lw=2, label=f'Proposed Architecture (AUC = {pr_auc:.4f})')

# Fine-tune layout boundaries
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall (True Positive Rate)', fontsize=11)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
plt.title('Precision-Recall (PR) Curve', fontsize=12, fontweight='bold')
plt.legend(loc="lower left", frameon=True)
plt.grid(True, linestyle='--', alpha=0.5)

# Optimize spacing and export second figure
plt.tight_layout()
plt.show()
plt.savefig('evaluation_pr_curve.png', dpi=300)
plt.clf()

print("Journal-ready figures successfully saved as 'evaluation_roc_curve.png' and 'evaluation_pr_curve.png'.")


# In[105]:


from sklearn.metrics import classification_report, precision_recall_fscore_support

def calculate_point_adjusted_metrics(y_true, y_pred):
    """
    Calculates Point-Adjusted (PA) metrics for time-series anomaly detection.
    
    Parameters:
    y_true (numpy array): Ground truth binary labels (0 for normal, 1 for attack)
    y_pred (numpy array): Raw model binary predictions (0 or 1) before adjustment
    
    Returns:
    y_pred_pa (numpy array): Adjusted binary predictions
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true).astype(int)
    y_pred_pa = np.array(y_pred).astype(int)
    
    # Identify changes in state to locate contiguous anomaly segments
    # Find indices where an anomaly segment starts or ends
    diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    # Iterate through each ground-truth anomaly window
    for start, end in zip(starts, ends):
        # Check if the model caught at least one anomaly point in this window
        if np.any(y_pred_pa[start:end] == 1):
            # Adjust the entire segment to 1 (True Positive optimization)
            y_pred_pa[start:end] = 1
            
    return y_pred_pa

# =========================================================================
# EXECUTION & ANALYSIS PIPELINE
# =========================================================================

# 1. Generate your raw binary predictions using your optimal EVT threshold
# (Replace 'A_test' and 'evt_threshold' with your exact variable names)
#y_pred_raw = (A_test > evt_threshold).astype(int)
#y_true_raw = y_test_labels.astype(int)  # Your ground truth test labels

# 2. Compute the Point-Adjusted Vectors
y_pred_adjusted = calculate_point_adjusted_metrics(y_test, y_gate)

# 3. Extract final comparative metrics
prec_raw, rec_raw, f1_raw, _ = precision_recall_fscore_support(y_test, y_gate, average='binary')
prec_pa, rec_pa, f1_pa, _ = precision_recall_fscore_support(y_test, y_pred_adjusted, average='binary')

print("=" * 60)
print("             METRIC COMPARATIVE MATRIX (SWAT)              ")
print("=" * 60)
print(f"RAW UNADJUSTED METRICS: \n Precision: {prec_raw:.4f} | Recall: {rec_raw:.4f} | F1-Score: {f1_raw:.4f}")
print("-" * 60)
print(f"POINT-ADJUSTED METRICS (PA): \n Precision: {prec_pa:.4f} | Recall: {rec_pa:.4f} | F1-Score: {f1_pa:.4f}")
print("=" * 60)

# 4. Generate the rigorous journal classification report
print("\nDETAILED POINT-ADJUSTED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_adjusted, target_names=['Normal State', 'Rare Event Target'], digits=4))


# In[91]:


gc.collect()
torch.cuda.empty_cache()
print("🧹 GPU Memory cleared successfully.")


# In[47]:


# 1. SETUP DEVICE AND WRAP MODEL
# Check for multiple GPUs
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

print(f"🖥️ System Hardware Check: Using device '{device}' with {gpu_count} available GPU(s).")


# In[48]:


import torch

# 1. Base Device and Count Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_count = torch.cuda.device_count()

print("="*60)
print("🖥️  HARDWARE INFRASTRUCTURE MONITOR")
print("="*60)
print(f"Primary Target Device: {device}")
print(f"Total Available GPUs:   {gpu_count}")
print("-"*60)

# 2. Detailed Multi-GPU Inspection Loop
if torch.cuda.is_available():
    for i in range(gpu_count):
        # Fetch hardware properties from CUDA runtime
        props = torch.cuda.get_device_properties(i)
        
        # Convert total bytes to gigabytes for cleaner reading
        total_vram_gb = props.total_memory / (1024**3)
        
        # Query current real-time VRAM usage allocation
        allocated_vram_gb = torch.cuda.memory_allocated(i) / (1024**3)
        cached_vram_gb = torch.cuda.memory_reserved(i) / (1024**3)
        
        print(f"🚀 GPU [{i}]: {props.name}")
        print(f"   • Compute Capability:   {props.major}.{props.minor}")
        print(f"   • Total VRAM Capacity:  {total_vram_gb:.2f} GB")
        print(f"   • Current Allocated:    {allocated_vram_gb:.2f} GB")
        print(f"   • Current Reserved:     {cached_vram_gb:.2f} GB")
        print(f"   • Streaming Multiprocs: {props.multi_processor_count}")
        print("-"*60)
        
    # Print underlying CUDA/cuDNN framework versions for your records
    print(f"📦 CUDA Toolkit Version:  {torch.version.cuda}")
    print(f"📦 cuDNN Library Version: {torch.backends.cudnn.version()}")
else:
    print("❌ No CUDA compatible acceleration hardware detected. Defaulting to CPU.")

print("="*60)


# In[108]:


import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# =========================================================================
# STEP 7: GENERATE PUBLICATION-READY COMBINED EVALUATION PLOTS
# =========================================================================

# Create a single figure with 2 subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# -------------------------------------------------------------------------
# 1. LEFT SUBPLOT: RECEIVER OPERATING CHARACTERISTIC (ROC) CURVE
# -------------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test_aligned, A_test)
roc_auc = auc(fpr, tpr)

ax1.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=2,
    label=f"Proposed Architecture (AUC = {roc_auc:.4f})",
)
ax1.plot(
    [0, 1],
    [0, 1],
    color="navy",
    lw=1.5,
    linestyle="--",
    label="Random Baseline",
)

ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
ax1.set_ylabel("True Positive Rate (TPR)", fontsize=11)
ax1.legend(loc="lower right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.5)

# Place Subfigure Caption (a) below the x-axis label
ax1.set_title("(a) ROC curve", fontsize=12, fontweight="bold", y=-0.22)

# -------------------------------------------------------------------------
# 2. RIGHT SUBPLOT: PRECISION-RECALL (PR) CURVE
# -------------------------------------------------------------------------
precision, recall, _ = precision_recall_curve(y_test_aligned, A_test)
pr_auc = auc(recall, precision)

ax2.plot(
    recall,
    precision,
    color="purple",
    lw=2,
    label=f"Proposed Architecture (AUC = {pr_auc:.4f})",
)

ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax2.set_xlabel("Recall (True Positive Rate)", fontsize=11)
ax2.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
ax2.legend(loc="lower left", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.5)

# Place Subfigure Caption (b) below the x-axis label
ax2.set_title("(b) Precision-recall curve", fontsize=12, fontweight="bold", y=-0.22)

# -------------------------------------------------------------------------
# EXPORT AND DISPLAY
# -------------------------------------------------------------------------
plt.tight_layout()
plt.savefig(
    "evaluation_roc_pr_combined.png", dpi=300, bbox_inches="tight"
)
plt.show()
plt.close()

print(
    "Journal-ready combined figure successfully saved as 'evaluation_roc_pr_combined.png'."
)


# In[ ]:




