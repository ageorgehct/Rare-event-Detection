#!/usr/bin/env python
# coding: utf-8

# In[266]:


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
import gc

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)


# In[267]:


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


# In[268]:


def preprocess_wadi(normal_path, attack_path):
    print("Starting WADI Preprocessing Pipeline...")
    
    # 1. Load Normal Data (Train Set)
    # The normal set typically loads fine, but let's drop metadata rows if they exist
    df_normal = pd.read_csv(normal_path, low_memory=False)
    
    # 2. Load Attack Data (Test Set)
    # WADI_attackdataLABLE.csv often has an index/junk row at row 0. We skip it to get clean column names.
    df_attack = pd.read_csv(attack_path, skiprows=1, low_memory=False)
    
    # Clean up column names (strip trailing spaces or hidden characters)
    df_normal.columns = df_normal.columns.str.strip()
    df_attack.columns = df_attack.columns.str.strip()
    
    # 3. Handle and Isolate Labels from the Attack Set
    # Finding the label column using string matching since spelling can vary ("LABLE" vs "LABEL")
    label_col = [c for c in df_attack.columns if "attack" in c.lower() and "lab" in c.lower()]
    if not label_col:
        raise ValueError("Could not automatically locate the Attack Label column!")
    label_col_name = label_col[0]
    
    # Convert WADI labels (1 = No Attack, -1 = Attack) to standard binary (0 = Normal, 1 = Anomaly)
    y_train =  np.zeros(df_normal.shape[0], dtype=int)
    y_test = (df_attack[label_col_name] == -1).astype(int).values
    
    # 4. Align Columns & Remove Unnecessary / Information-Leaking Features
    # These 4 columns are empty or contain hardcoded PLC triggers that instantly leak the attack state
    leak_cols = ['Row', 'Date', 'Time', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS']
    
    # Drop from normal set
    cols_to_drop_normal = [c for c in leak_cols if c in df_normal.columns]
    df_normal = df_normal.drop(columns=cols_to_drop_normal)
    
    # Drop from attack set (including the label column itself)
    cols_to_drop_attack = [c for c in leak_cols if c in df_attack.columns] + [label_col_name]
    df_attack = df_attack.drop(columns=cols_to_drop_attack)
    
    # Ensure both dataframes share identical feature structures
    shared_features = [c for c in df_normal.columns if c in df_attack.columns]
    df_normal = df_normal[shared_features]
    df_attack = df_attack[shared_features]
    
    print(f"Aligned Features: {len(shared_features)} columns preserved.")
    
    # 5. Missing Data Imputation (Crucial for WADI 2019)
    # Forward-fill to mimic real-time sensor continuity, then backward-fill for edge cases
    df_normal = df_normal.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    df_attack = df_attack.apply(pd.to_numeric, errors='coerce').ffill().bfill()
    
    # Convert to standard NumPy matrices for your model
    #X_train = df_normal.values
    #X_test = df_attack.values
    
    # 6. Scaling (Fit on Train, Transform on Test)
    #scaler = MinMaxScaler()
    #X_train_scaled = scaler.fit_transform(X_train)
    #X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Success! Shapes:")
    print(f"   -> X_train (Normal): {df_normal.shape}")
    print(f"   -> X_test (Attack):  {df_attack.shape}")
    print(f"   -> y_train (Labels):  {y_train.shape} (Contains {y_train.sum()} anomaly points)")
    print(f"   -> y_test (Labels):  {y_test.shape} (Contains {y_test.sum()} anomaly points)")
    
    return df_normal,df_attack, y_test,y_train

# --- Execution Example ---
normal_file = r"WADI_14days_normal.csv"
attack_file = r"WADI_attackdata.csv"
df_train, df_test, y_test,y_train = preprocess_wadi(normal_file, attack_file)
print(df_train.shape)
print(df_test.shape)
print(y_test.shape)
print(y_train.shape)

print(f"Training shape: {df_train.shape}")
print(f"Testing shape:  {df_test.shape}")
count = 0
cols_to_drop = []
for col in df_train.columns:
    unique_count_train = df_train[col].nunique()
    unique_count_test = df_test[col].nunique()
    if unique_count_train == 1 and unique_count_test == 1:
        #print (f" {col} is unique is both test and train datasets")
        cols_to_drop.append(col)
        count += 1
print (f" {count} columns are unique in both datasets ")
print(f"\nTotal columns identified for removal: {len(cols_to_drop)}")

df_train = df_train.drop(columns=cols_to_drop)
df_test = df_test.drop(columns=cols_to_drop)

print(f"New training shape: {df_train.shape}")
print(f"New testing shape:  {df_test.shape}")



# Initialize empty lists to store column names by their structural types
numerical_cols_tr = []
categorical_discols_tr = []
categorical_txcols_tr = []
datetime_cols_tr = []

# =========================================================================
# STEP 1: SORT AND IDENTIFY COLUMNS BY DATA TYPE
# =========================================================================
for col in df_train.columns:
    unique_count_tr = df_train[col].nunique()

    if pd.api.types.is_datetime64_any_dtype(df_train[col]):
        #print(f"{col}: DATETIME")
        datetime_cols_tr.append(col)

    elif pd.api.types.is_numeric_dtype(df_train[col]):
        if unique_count_tr <= 5:
            #print(f"{col}: DISCRETE/CATEGORICAL ({unique_count} values)")
            categorical_discols_tr.append(col)
        else:
            #print(f"{col}: NUMERICAL")
            numerical_cols_tr.append(col)

    else:
        #print(f"{col}: CATEGORICAL/TEXT")
        categorical_txcols_tr.append(col)

# =========================================================================
# STEP 2: REGROUP DATA FRAME COLUMNS (NUMERICAL -> CATEGORICAL -> DATETIME)
# =========================================================================
# Concatenate the lists together to form the new target schema order
ordered_cols_tr = numerical_cols_tr + categorical_discols_tr +  categorical_txcols_tr + datetime_cols_tr

# Reindex your dataframe with the new column layout
df_train_gr = df_train[ordered_cols_tr]

print("------------------------------------------------------------")
print(f"Feature Grouping Complete!")
print(f"Total Numerical Columns:   {len(numerical_cols_tr)}")
print(f"Total Categorical Discrete Columns: {len(categorical_discols_tr)}")
print(f"Total Categorical Text Columns: {len(categorical_txcols_tr)}")
print(f"Total Datetime Columns:    {len(datetime_cols_tr)}")
print(df_train_gr.shape)
print("------------------------------------------------------------")
print (f" first categorical feature is {categorical_discols_tr[0]}")


# Initialize empty lists to store column names by their structural types
numerical_cols_ts = []
categorical_discols_ts = []
categorical_txcols_ts = []
datetime_cols_ts = []

# =========================================================================
# STEP 1: SORT AND IDENTIFY COLUMNS BY DATA TYPE
# =========================================================================
for col in df_test.columns:
    unique_count_ts = df_test[col].nunique()

    if pd.api.types.is_datetime64_any_dtype(df_test[col]):
        #print(f"{col}: DATETIME")
        datetime_cols_ts.append(col)

    elif pd.api.types.is_numeric_dtype(df_test[col]):
        if unique_count_ts <= 5:
            #print(f"{col}: DISCRETE/CATEGORICAL ({unique_count} values)")
            categorical_discols_ts.append(col)
        else:
            #print(f"{col}: NUMERICAL")
            numerical_cols_ts.append(col)

    else:
        #print(f"{col}: CATEGORICAL/TEXT")
        categorical_txcols_ts.append(col)

# =========================================================================
# STEP 2: REGROUP DATA FRAME COLUMNS (NUMERICAL -> CATEGORICAL -> DATETIME)
# =========================================================================
# Concatenate the lists together to form the new target schema order
ordered_cols_ts = numerical_cols_ts + categorical_discols_ts +  categorical_txcols_ts + datetime_cols_ts

# Reindex your dataframe with the new column layout
df_test_gr = df_test[ordered_cols_ts]

print("------------------------------------------------------------")
print(f"Feature Grouping Complete!")
print(f"Total Numerical Columns:   {len(numerical_cols_ts)}")
print(f"Total Categorical Discrete Columns: {len(categorical_discols_ts)}")
print(f"Total Categorical Text Columns: {len(categorical_txcols_ts)}")
print(f"Total Datetime Columns:    {len(datetime_cols_ts)}")
print(df_test_gr.shape)
print("------------------------------------------------------------")
print (f" first categorical feature is {categorical_discols_ts[0]}")

# Number of categorical columns excluding MONTH and YEAR
new_cols = (
    [f'N{i+1}' for i in range(len(numerical_cols_tr))] +
    [f'C{i+1}' for i in range(len(categorical_discols_tr))] 
)
df_train_gr.columns = new_cols
df_test_gr.columns = new_cols
print(df_train_gr.shape)
print(df_test_gr.shape)


# In[269]:


print(len(df_train_gr),len(df_test_gr))
total_ones_train = np.sum(y_train == 1)
print(f'one count is train set is ' ,total_ones_train)
total_ones_test = np.sum(y_test == 1)
print(f'one count is test set is ' ,total_ones_test)

df_train = df_train_gr
df_test = df_test_gr
print(len(df_train),len(df_test))

print(df_test.columns.tolist())
print("----------------------------------------------------------------")
print(df_test.columns.tolist())


# In[270]:


def inject_realistic_noise(df, variance_factor=0.03, drift_factor=0.04):
    """
    Injects realistic instrumentation noise and baseline drift into an industrial DataFrame.
    
    Parameters:
    - df: The original clean df_train.
    - variance_factor: Instantaneous sensor jitter (white noise).
    - drift_factor: Slow baseline wandering over time (random walk).
    """
    df_noisy = df.copy()
    n_rows, n_cols = df.shape
    
    # 1. Generate White Noise (Sensor Jitter)
    # Scales dynamically with each column's specific standard deviation
    for col in df_noisy.columns:
        col_std = df_noisy[col].std()
        if col_std == 0:  # Skip completely dead/invariant columns
            continue
            
        # Add high-frequency white noise
        white_noise = np.random.normal(loc=0.0, scale=variance_factor * col_std, size=n_rows)
        
        # 2. Generate Low-Frequency Drift (Random Walk to simulate environment changes)
        # Cumulative sum creates a organic wandering baseline across time
        step_drift = np.random.normal(loc=0.0, scale=drift_factor * col_std, size=n_rows)
        random_walk = np.cumsum(step_drift)
        # Detrend or scale the walk slightly so it doesn't wander to infinity
        random_walk = random_walk - np.linspace(random_walk[0], random_walk[-1], n_rows)
        
        # Combine and apply directly to the physical feature
        df_noisy[col] = df_noisy[col] + white_noise + random_walk
        
        # Clip to make sure we don't breach original physical maximum bounds
        df_noisy[col] = df_noisy[col].clip(lower=df[col].min(), upper=df[col].max())
        
    return df_noisy

# Execute on your groups
#df_train_realistic = inject_realistic_noise(df_train, variance_factor=0.01, drift_factor=0.01)


#df_train = df_train_realistic
#print(f"📊 Transformed Training Rows: {len(df_train)} | Testing Rows: {len(df_test)}")


# In[271]:


# Set a statistical threshold: Drop if test mean shifts by more than 5 Training Standard Deviations
z_threshold = 10
cols_to_drop = []

# 1. Directly isolate the first 67 column NAMES from the dataframe
train_cols_subset = df_train.columns[:len(numerical_cols_tr)]

# 2. Match them against the test set while safely filtering out the attack label
common_cols = [col for col in train_cols_subset if col in df_test.columns and col.lower() != 'attack']

print (common_cols)

for col in common_cols:
    train_mean = df_train[col].mean()
    train_std = df_train[col].std()
    test_mean = df_test[col].mean()
    
    # Avoid division by zero for invariant columns
    if train_std == 0:
        continue
        
    # Calculate how many training standard deviations the test mean has shifted
    shift_in_stdevs = np.abs(test_mean - train_mean) / train_std
    
    if shift_in_stdevs > z_threshold:
        print(f"Dropping {col}: Shifted by {shift_in_stdevs:.2f}x train standard deviations.")
        cols_to_drop.append(col)

# Drop from both dataframes
df_train = df_train.drop(columns=cols_to_drop, errors='ignore')
df_test = df_test.drop(columns=cols_to_drop, errors='ignore')

print(f"Process complete. Automatically dropped {len(cols_to_drop)} domain-shifted features.")
print(df_train.shape)
print(df_test.shape)

# Reanming all columns to Numerical
new_cols = (
    [f'N{i+1}' for i in range(df_train.shape[1])] 
)
df_train.columns = new_cols
df_test.columns = new_cols
#print(df_test.columns.tolist())


new_cols = (
    [f'N{i+1}' for i in range(len(numerical_cols_tr) - len(cols_to_drop))] +
    [f'C{i+1}' for i in range(len(categorical_discols_tr))] 
)
df_train.columns = new_cols
df_test.columns = new_cols
print(df_train.shape)
print(df_test.shape)

print("----------------------------------------------------------------")
print(df_test.columns.tolist())
print("----------------------------------------------------------------")
print(df_test.columns.tolist())


# In[272]:


'''
for col in df_train.columns:
    unique_count = df_test[col].nunique()

    if pd.api.types.is_datetime64_any_dtype(df_test[col]):
        print(f"{col}: DATETIME")
       
    elif pd.api.types.is_numeric_dtype(df_test[col]):
        if unique_count <= 10:
            print(f"{col}: DISCRETE/CATEGORICAL ({unique_count} values)")
        else:
            print(f"{col}: NUMERICAL")
    else:
        print(f"{col}: CATEGORICAL/TEXT")
'''


# In[273]:


# =========================================================================
# STEP 1: HETEROGENEOUS ENCODING (CORRECTED BY INDEX SLICING)
# =========================================================================

# 1. Dynamically capture column names by their actual positions in your DataFrame
# First 53 columns (indices 0 to 52)
num_cols = list(df_train.columns[:53])    
# Next 31 columns (indices 53 to 83)
cat_cols = list(df_train.columns[53:84])   

print(f"Captured {len(num_cols)} numerical features: {num_cols[:3]}...{num_cols[-1:]}")
print(f"Captured {len(cat_cols)} categorical features: {cat_cols[:3]}...{cat_cols[-1:]}")

# 2. Continuous Metric Normalization using GuardrailedRobustScaler
scaler = GuardrailedRobustScaler()
train_num_scaled = scaler.fit_transform(df_train[num_cols])
print("X Train num shape is:", train_num_scaled.shape)
test_num_scaled = scaler.transform(df_test[num_cols])
print("X Test num shape is:", test_num_scaled.shape)

# Quick verification check
#print(f"Any NaNs in scaled train? {np.isnan(train_num_scaled).any()}")
#print(f"Any NaNs in scaled test?  {np.isnan(test_num_scaled).any()}")

# 3. Multi-Column Categorical One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
train_cat_encoded = encoder.fit_transform(df_train[cat_cols])
test_cat_encoded = encoder.transform(df_test[cat_cols])

print("X Train cat shape is:", train_cat_encoded.shape)
print("X Test cat shape is:", test_cat_encoded.shape)

# 4. Horizontal Feature Stack to Create the Global Feature Space Matrix
X_train_processed = np.hstack([train_num_scaled, train_cat_encoded])
print("X Train processed shape is:", X_train_processed.shape)
X_test_processed  = np.hstack([test_num_scaled, test_cat_encoded])
print("X Test processed shape is:", X_test_processed.shape)

# 7. Convert Processed Matrix to Native PyTorch Tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
print("X Train tensor shape is:", X_train_tensor.shape)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
print("X Test tensor shape is:", X_test_tensor.shape)

# =========================================================================
# CONTEXT INDICES DEFINITION
# =========================================================================
context_indices = list(range(0, 53))

print("-----Done-----------")


# In[274]:


#windowing
def create_tensor_sliding_windows(tensor_data, window_size=5):
    """
    Transforms a 2D PyTorch Tensor [timesteps, features] into a flattened
    sliding window Tensor [samples, window_size * features] using native tensor striding.
    """
    # 1. Use unfold to create windows across the time dimension (dim=0)
    # Output shape of unfold: [num_windows, window_size, features]
    windows = tensor_data.unfold(dimension=0, size=window_size, step=1)
    
    # 2. PyTorch's unfold puts the window length dimension at the very end:
    # Current shape: [num_windows, features, window_size]
    # We permute it back to a chronological structure: [num_windows, window_size, features]
    windows = windows.permute(0, 2, 1)
    
    # 3. Flatten the spatial-temporal dimensions: [num_windows, window_size * features]
    # This precisely mirrors the behavior of your Pandas horizontal concatenation
    flattened_windows = windows.reshape(windows.size(0), -1)
    
    return flattened_windows

# =========================================================================
# APPLICATION PIPELINE (REWRITTEN FOR PURE TENSORS)
# =========================================================================
WINDOW_SIZE = 1  # Scale this up cleanly to 5, 10, etc.

# 1. Generate flattened spatial-temporal Tensors natively
X_train_flat = create_tensor_sliding_windows(X_train_tensor, window_size=WINDOW_SIZE)
X_test_flat  = create_tensor_sliding_windows(X_test_tensor, window_size=WINDOW_SIZE)

# 2. Align your tracking target array/tensor to match dropped causal boundaries
# PyTorch's unfold automatically handles the drop of the first (WINDOW_SIZE - 1) incomplete historical steps
if 'y_test' in locals() or 'y_test' in globals():
    if isinstance(y_test, torch.Tensor):
        y_test_aligned = y_test[WINDOW_SIZE - 1:]
    else:
        y_test_aligned = y_test[WINDOW_SIZE - 1:] # Works identically for NumPy arrays

print("🛣️ Tensor Spatial-Temporal Windows Created Successfully.")
print(f"📊 Shape of X_train_flat: {X_train_flat.shape} (Samples x Windowed Features)")
print(f"📊 Shape of X_test_flat:  {X_test_flat.shape}")
if 'y_test' in locals() or 'y_test' in globals():
    print(f"🎯 Shape of y_test_aligned: {y_test_aligned.shape}")

# Reassign back to your tensor variable names for down-stream modeling steps
X_train_tensor = X_train_flat
X_test_tensor  = X_test_flat
y_test         = y_test_aligned


# In[277]:


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
for epoch in range(150):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    predictions = model(X_train_tensor)
    loss = torch.mean((X_train_tensor - predictions) ** 2)
    
    loss.backward()
    optimizer.step()
    scheduler.step()    # Step scheduler per epoch
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/150] - Loss: {loss.item():.6f}")

# Clean up local references immediately after the loop ends
del predictions, loss
gc.collect()
torch.cuda.empty_cache()


# In[278]:


print(f"Shape of X_train_tensor  is {X_test_tensor .shape}")


# In[279]:


# =========================================================================
# SYSTEM DEBUG BOUNDARY: TRACKING NAN INF CORRUPTION
# =========================================================================
print("🔍 === STARTING PIPELINE INTEGRITY CHECK ===")

# 1. Check Input Tensors on GPU
print(f"X_train_tensor -> Is NaN? {torch.isnan(X_train_tensor).any().item()} | Is Inf? {torch.isinf(X_train_tensor).any().item()} | Shape: {X_train_tensor.shape}")
#print(f"X_test_tensor  -> Is NaN? {torch.isnan(X_test_tensor).any().item()} | Is Inf? {torch.isinf(X_test_tensor).any().item()} | Shape: {X_test_tensor.shape}")

# 2. Check Model Predictions
#print(f"train_preds    -> Is NaN? {torch.isnan(train_preds).any().item()} | Is Inf? {torch.isinf(train_preds).any().item()} | Shape: {train_preds.shape}")
#print(f"test_preds     -> Is NaN? {torch.isnan(test_preds).any().item()} | Is Inf? {torch.isinf(test_preds).any().item()} | Shape: {test_preds.shape}")

# 3. Check Covariance Matrix & Precision Matrix Invariants for Mahalanobis
#print(f"cov_matrix     -> Is NaN? {np.isnan(cov_matrix).any()} | Is Inf? {np.isinf(cov_matrix).any()}")
#print(f"precision_v    -> Is NaN? {torch.isnan(precision_matrix_v).any()} | Is Inf? {torch.isinf(precision_matrix_v).any()}")

# 4. Locate the exact column indices if inputs are corrupted
if torch.isnan(X_train_tensor).any():
    nan_cols = torch.where(torch.isnan(X_train_tensor))[1].unique().cpu().numpy()
    print(f"⚠️ Crucial: Training NaN values found in processed feature indices: {nan_cols}")
    # Map back to names if possible
    # print([num_cols + list(encoder.get_feature_names_out())][i] for i in nan_cols)

print("🔍 === END OF PIPELINE INTEGRITY CHECK ===\n")


# In[310]:


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
    K = 100
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

A_train = (0.7 * R_tr_norm) + (0.0 * MD_tr_norm) + (0.30 * L1_tr_norm)
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")

print("\n" + "="*60 + "\n Test Data set\n" + "="*60)
R_ts_norm = np.clip((R_ts - stats['R_min']) / r_denom, 0, None)
print(f"Max of R_ts is {np.amax(R_ts_norm)},range is {np.ptp(R_ts_norm)}, median is {np.median(R_ts_norm)}, r_denom is {r_denom}")

MD_ts_norm = np.clip((MD_ts - stats['MD_min']) / md_denom, 0, None)
print(f"Max of MD_ts is {np.amax(MD_ts_norm)},range is {np.ptp(MD_ts_norm)}, median is {np.median(MD_ts_norm)}, md_denom is {md_denom}")

L1_ts_norm = np.clip((L1_ts - stats['L1_min']) / l1_denom, 0, None)
print(f"Max of L1_ts is {np.amax(L1_ts_norm)},range is {np.ptp(L1_ts_norm)}, median is {np.median(L1_ts_norm)}, l1_denom is {l1_denom}")

A_test = (0.7 * R_ts_norm) + (0.0 * MD_ts_norm) + (0.3 * L1_ts_norm)
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
R_tr_norm, R_ts_norm   = compute_robust_z_score(R_tr_log, R_ts_log)
MD_tr_norm, MD_ts_norm = compute_robust_z_score(MD_tr_log, MD_ts_log)
L1_tr_norm, L1_ts_norm = compute_robust_z_score(L1_tr_log, L1_ts_log)

print("\n" + "="*60 + "\n Robust Standardized Training Dataset\n" + "="*60)
print(f"MSE  Tr -> Median: {np.median(R_tr_norm):.4f}, Max: {R_tr_norm.max():.4f}, Std: {np.std(R_tr_norm):.4f}")
print(f"MD   Tr -> Median: {np.median(MD_tr_norm):.4f}, Max: {MD_tr_norm.max():.4f}, Std: {np.std(R_tr_norm):.4f}")
print(f"L1   Tr -> Median: {np.median(L1_tr_norm):.4f}, Max: {L1_tr_norm.max():.4f}, Std: {np.std(L1_tr_norm):.4f}")

# Re-combine using balanced weights
A_train = (0.30 * R_tr_norm) + (0.45 * MD_tr_norm) + (0.25 * L1_tr_norm)
print(f"Unified A_train Median: {np.median(A_train):.4f}, Max: {A_train.max():.4f}, Std: {np.std(A_train):.4f}")

print("\n" + "="*60 + "\n Robust Standardized Test Dataset\n" + "="*60)
print(f"MSE  Ts -> Median: {np.median(R_ts_norm):.4f}, Max: {R_ts_norm.max():.4f}, Std: {np.std(R_ts_norm):.4f}")
print(f"MD   Ts -> Median: {np.median(MD_ts_norm):.4f}, Max: {MD_ts_norm.max():.4f}, Std: {np.std(MD_ts_norm):.4f}")
print(f"L1   Ts -> Median: {np.median(L1_ts_norm):.4f}, Max: {L1_ts_norm.max():.4f}, Std: {np.std(L1_ts_norm):.4f}")

A_test = (0.30 * R_ts_norm) + (0.45 * MD_ts_norm) + (0.25 * L1_ts_norm)
print(f"Unified A_test  Median: {np.median(A_test):.4f}, Max: {A_test.max():.4f}, Std: {np.std(A_test):.4f}")


# In[311]:


print("Train Tensor:", torch.isnan(X_train_tensor).any().item())
print("Test Tensor:", torch.isnan(X_test_tensor).any().item())

num_nans = torch.isnan(X_train_tensor).sum().item()
print("Number of NaNs:", num_nans)


# In[312]:


# =========================================================================
# STEP 4: AUTOMATED GATING (EVT/POT METHOD)
# =========================================================================
gamma_threshold = np.percentile(A_train, 98) # Tightened slightly to fit the 0.4% density profile
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(" Gamma Threshold is ", gamma_threshold);
result = [x for x in A_train if x > gamma_threshold]  # one-liner
print(len(result))  # [6, 7, 7]
y_gate = (A_test > gamma_threshold).astype(int)
print(len(y_gate))
total_ones = np.sum(y_gate == 1)
print(total_ones)


# In[332]:


gamma_threshold = np.percentile(A_train, 99.5) # Tightened slightly to fit the 0.4% density profile
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(" Gamma Threshold is ", gamma_threshold);

# =========================================================================
# STEP 4: REVISED AUTOMATED GATING (EVT/POT METHOD)
# =========================================================================
# 1. Set an initial loose threshold (t) to capture the tail candidate pool
# Typically 98th percentile is standard to isolate extreme values for fitting
t = np.percentile(A_train, 99.5)
tail_excesses = A_train[A_train > t] - t

# 2. Fit the tail excesses to the Generalized Pareto Distribution (GPD)
# c = shape parameter, loc = location parameter, scale = scale parameter
c, loc, scale = genpareto.fit(tail_excesses, floc=0)
print(f"Pareto distribution: c is {c}, loc is {loc}, scale is {scale}")
print(f"Shape parameter (c): {c:.4f}  ({'Heavy-tailed' if c > 0 else 'Light/Bounded-tailed'})")

# 3. Define your target risk probability (q) 
#q=1e-4 #means you accept a 0.01% theoretical probability of a false alarm
#q = 2e-3
#q = 0.00045
q = 0.0002
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
A_test_series = pd.Series(A_test)
A_test_smoothed = A_test_series.ewm(span=5, adjust=False).mean().values
A_test = A_test_smoothed

# 5. Evaluate the testing data against the mathematically robust threshold
y_gate = (A_test > gamma_threshold).astype(int)
y_gate_train = (A_train > gamma_threshold).astype(int)
y_gate_full = np.concatenate([y_gate_train, y_gate])

total_ones = np.sum(y_gate_full == 1)
print("Anomaly count is ",total_ones)

# Diagnostic Prints
print(f"Total Test Samples: {len(y_gate)}")
print(f"Total Anomalies Flagged (y_gate == 1): {np.sum(y_gate == 1)}")


# In[333]:


print(df_test.shape)
print(df_train.shape)


# In[334]:


from sklearn.tree import DecisionTreeClassifier, export_text

# =========================================================================
# STEP 5: EXPLAINABLE AI SURROGATE EXTRACTION (89-FEATURE MATRIX ALIGNMENT)
# =========================================================================

# 1. Reconstruct the precise feature names matching the 89 columns in X_test_processed
# Get the names of the 53 continuous features
numerical_names = list(num_cols)

# Get the names generated by the One-Hot Encoder for the categorical features
categorical_names = list(encoder.get_feature_names_out(cat_cols))

# Combine them sequentially to match your np.hstack([train_num_scaled, train_cat_encoded])
surrogate_feature_names = numerical_names + categorical_names

print(f"Total features tracked for XAI breakdown: {len(surrogate_feature_names)} features")

# 3. Fit the CART Surrogate model on the clean, scaled 89-column test space matrix
# We use X_test_processed (the NumPy array right before tensor conversion) 
cart_surrogate = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
cart_surrogate.fit(X_test_processed, y_gate)

# 4. Export the human-readable tree logic mapping structural anomaly decisions
print("\n" + "="*60 + "\n 🪵 SURROGATE EXPLAINABLE AI DECISION TREE LOGIC\n" + "="*60)
print(export_text(cart_surrogate, feature_names=surrogate_feature_names))
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


# In[335]:


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


# In[336]:


#print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
#print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")

# use full data set
print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(classification_report(y_test_aligned_full, y_gate_full, target_names=["Normal Matrix State", "Rare Event Target"]))
print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")


# In[337]:


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


# In[341]:


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


# In[339]:


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


# In[340]:





# In[86]:


gc.collect()
torch.cuda.empty_cache()
print("🧹 GPU Memory cleared successfully.")


# In[342]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# =========================================================================
# STEP 1: POINT-ADJUSTMENT VECTOR FUNCTION
# =========================================================================


def calculate_point_adjusted_metrics(y_true, y_pred):
  """Adjusts predictions so that if an anomaly segment is detected at any

  point, the entire segment is marked as detected.
  """
  y_true = np.array(y_true).astype(int)
  y_pred_pa = np.array(y_pred).astype(int)

  diff = np.diff(np.concatenate(([0], y_true, [0])))
  starts = np.where(diff == 1)[0]
  ends = np.where(diff == -1)[0]

  for start, end in zip(starts, ends):
    if np.any(y_pred_pa[start:end] == 1):
      y_pred_pa[start:end] = 1

  return y_pred_pa


# =========================================================================
# STEP 2: GENERATE POINT-ADJUSTED ROC & PR CURVES OVER THRESHOLD SWEEP
# =========================================================================

# Extract standard (Unadjusted) ROC and PR data points
fpr_raw, tpr_raw, thresholds_roc = roc_curve(y_test_aligned, A_test)
roc_auc_raw = auc(fpr_raw, tpr_raw)

precision_raw, recall_raw, thresholds_pr = precision_recall_curve(
    y_test_aligned, A_test
)
pr_auc_raw = auc(recall_raw, precision_raw)

# Sweep thresholds to generate Point-Adjusted (PA) ROC & PR trajectories
# (Subsampling thresholds for efficient computation while preserving smooth curves)
subsample_rate = max(1, len(thresholds_roc) // 200)
thresholds_sweep = thresholds_roc[::subsample_rate]

fpr_pa_list, tpr_pa_list = [], []
precision_pa_list, recall_pa_list = [], []

for thresh in thresholds_sweep:
  # Binarize score vector at current threshold
  y_bin = (A_test >= thresh).astype(int)
  y_pa = calculate_point_adjusted_metrics(y_test_aligned, y_bin)

  # Compute metrics for ROC
  tp = np.sum((y_pa == 1) & (y_test_aligned == 1))
  fp = np.sum((y_pa == 1) & (y_test_aligned == 0))
  fn = np.sum((y_pa == 0) & (y_test_aligned == 1))
  tn = np.sum((y_pa == 0) & (y_test_aligned == 0))

  tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
  precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

  fpr_pa_list.append(fpr)
  tpr_pa_list.append(tpr)
  precision_pa_list.append(precision)
  recall_pa_list.append(tpr)  # Recall is equivalent to TPR

# Sort arrays for monotonically increasing x-axis required by AUC integration
sort_idx_roc = np.argsort(fpr_pa_list)
fpr_pa = np.array(fpr_pa_list)[sort_idx_roc]
tpr_pa = np.array(tpr_pa_list)[sort_idx_roc]
roc_auc_pa = auc(fpr_pa, tpr_pa)

sort_idx_pr = np.argsort(recall_pa_list)
recall_pa = np.array(recall_pa_list)[sort_idx_pr]
precision_pa = np.array(precision_pa_list)[sort_idx_pr]
pr_auc_pa = auc(recall_pa, precision_pa)


# =========================================================================
# STEP 3: PLOT AND SAVE OVERLAID ROC CURVE
# =========================================================================
plt.figure(figsize=(7, 5.5))

plt.plot(
    fpr_raw,
    tpr_raw,
    color='darkorange',
    lw=2,
    linestyle='--',
    label=f'Raw Unadjusted (AUC = {roc_auc_raw:.4f})',
)
plt.plot(
    fpr_pa,
    tpr_pa,
    color='crimson',
    lw=2,
    label=f'Point-Adjusted PA (AUC = {roc_auc_pa:.4f})',
)
plt.plot(
    [0, 1],
    [0, 1],
    color='navy',
    lw=1.2,
    linestyle=':',
    label='Random Baseline',
)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (FPR)', fontsize=11)
plt.ylabel('True Positive Rate (TPR)', fontsize=11)
plt.title(
    'Receiver Operating Characteristic (ROC) Curve Overlay',
    fontsize=12,
    fontweight='bold',
)
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('evaluation_roc_curve_overlay.png', dpi=300)
plt.show()
plt.clf()


# =========================================================================
# STEP 4: PLOT AND SAVE OVERLAID PRECISION-RECALL (PR) CURVE
# =========================================================================
plt.figure(figsize=(7, 5.5))

plt.plot(
    recall_raw,
    precision_raw,
    color='purple',
    lw=2,
    linestyle='--',
    label=f'Raw Unadjusted (AUC = {pr_auc_raw:.4f})',
)
plt.plot(
    recall_pa,
    precision_pa,
    color='darkgreen',
    lw=2,
    label=f'Point-Adjusted PA (AUC = {pr_auc_pa:.4f})',
)

plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('Recall (True Positive Rate)', fontsize=11)
plt.ylabel('Precision (Positive Predictive Value)', fontsize=11)
plt.title(
    'Precision-Recall (PR) Curve Overlay', fontsize=12, fontweight='bold'
)
plt.legend(loc='lower left', frameon=True, fontsize=10)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('evaluation_pr_curve_overlay.png', dpi=300)
plt.show()
plt.clf()

print(
    'Journal-ready overlaid figures successfully saved as'
    " 'evaluation_roc_curve_overlay.png' and 'evaluation_pr_curve_overlay.png'."
)


# In[344]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# =========================================================================
# STEP 1: POINT-ADJUSTMENT VECTOR FUNCTION
# =========================================================================


def calculate_point_adjusted_metrics(y_true, y_pred):
  """Adjusts predictions so that if an anomaly segment is detected at any

  point, the entire segment is marked as detected.
  """
  y_true = np.array(y_true).astype(int)
  y_pred_pa = np.array(y_pred).astype(int)

  diff = np.diff(np.concatenate(([0], y_true, [0])))
  starts = np.where(diff == 1)[0]
  ends = np.where(diff == -1)[0]

  for start, end in zip(starts, ends):
    if np.any(y_pred_pa[start:end] == 1):
      y_pred_pa[start:end] = 1

  return y_pred_pa


# =========================================================================
# STEP 2: GENERATE POINT-ADJUSTED ROC & PR CURVES OVER THRESHOLD SWEEP
# =========================================================================

# Extract standard (Unadjusted) ROC and PR data points
fpr_raw, tpr_raw, thresholds_roc = roc_curve(y_test_aligned, A_test)
roc_auc_raw = auc(fpr_raw, tpr_raw)

precision_raw, recall_raw, thresholds_pr = precision_recall_curve(
    y_test_aligned, A_test
)
pr_auc_raw = auc(recall_raw, precision_raw)

# Sweep thresholds to generate Point-Adjusted (PA) ROC & PR trajectories
subsample_rate = max(1, len(thresholds_roc) // 200)
thresholds_sweep = thresholds_roc[::subsample_rate]

fpr_pa_list, tpr_pa_list = [], []
precision_pa_list, recall_pa_list = [], []

for thresh in thresholds_sweep:
  # Binarize score vector at current threshold
  y_bin = (A_test >= thresh).astype(int)
  y_pa = calculate_point_adjusted_metrics(y_test_aligned, y_bin)

  # Compute metrics
  tp = np.sum((y_pa == 1) & (y_test_aligned == 1))
  fp = np.sum((y_pa == 1) & (y_test_aligned == 0))
  fn = np.sum((y_pa == 0) & (y_test_aligned == 1))
  tn = np.sum((y_pa == 0) & (y_test_aligned == 0))

  tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
  fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
  precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0

  fpr_pa_list.append(fpr)
  tpr_pa_list.append(tpr)
  precision_pa_list.append(precision)
  recall_pa_list.append(tpr)

# Sort arrays for monotonically increasing x-axis required by AUC integration
sort_idx_roc = np.argsort(fpr_pa_list)
fpr_pa = np.array(fpr_pa_list)[sort_idx_roc]
tpr_pa = np.array(tpr_pa_list)[sort_idx_roc]
roc_auc_pa = auc(fpr_pa, tpr_pa)

sort_idx_pr = np.argsort(recall_pa_list)
recall_pa = np.array(recall_pa_list)[sort_idx_pr]
precision_pa = np.array(precision_pa_list)[sort_idx_pr]
pr_auc_pa = auc(recall_pa, precision_pa)


# =========================================================================
# STEP 3: GENERATE SIDE-BY-SIDE (ADJACENT) PUBLICATION FIGURE
# =========================================================================

# Create figure with 1 row and 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# -------------------------------------------------------------------------
# SUBPLOT (a): ROC CURVE OVERLAY
# -------------------------------------------------------------------------
ax1.plot(
    fpr_raw,
    tpr_raw,
    color='darkorange',
    lw=2,
    linestyle='--',
    label=f'Raw Unadjusted (AUC = {roc_auc_raw:.4f})',
)
ax1.plot(
    fpr_pa,
    tpr_pa,
    color='crimson',
    lw=2,
    label=f'Point-Adjusted PA (AUC = {roc_auc_pa:.4f})',
)
ax1.plot(
    [0, 1],
    [0, 1],
    color='navy',
    lw=1.2,
    linestyle=':',
    label='Random Baseline',
)

ax1.set_xlim([-0.02, 1.02])
ax1.set_ylim([-0.02, 1.02])
ax1.set_xlabel('False Positive Rate (FPR)', fontsize=11)
ax1.set_ylabel('True Positive Rate (TPR)', fontsize=11)
ax1.legend(loc='lower right', frameon=True, fontsize=9.5)
ax1.grid(True, linestyle='--', alpha=0.5)

# Subtitle below the left curve
ax1.set_title('(a) ROC Curve', y=-0.22, fontsize=12, fontweight='bold')


# -------------------------------------------------------------------------
# SUBPLOT (b): PRECISION-RECALL (PR) CURVE OVERLAY
# -------------------------------------------------------------------------
ax2.plot(
    recall_raw,
    precision_raw,
    color='purple',
    lw=2,
    linestyle='--',
    label=f'Raw Unadjusted (AUC = {pr_auc_raw:.4f})',
)
ax2.plot(
    recall_pa,
    precision_pa,
    color='darkgreen',
    lw=2,
    label=f'Point-Adjusted PA (AUC = {pr_auc_pa:.4f})',
)

ax2.set_xlim([-0.02, 1.02])
ax2.set_ylim([-0.02, 1.02])
ax2.set_xlabel('Recall (True Positive Rate)', fontsize=11)
ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=11)
ax2.legend(loc='lower left', frameon=True, fontsize=9.5)
ax2.grid(True, linestyle='--', alpha=0.5)

# Subtitle below the right curve
ax2.set_title(
    '(b) Precision-Recall Curve', y=-0.22, fontsize=12, fontweight='bold'
)


# -------------------------------------------------------------------------
# EXPORT COMBINED FIGURE
# -------------------------------------------------------------------------
plt.tight_layout()
plt.savefig('evaluation_roc_pr_combined.png', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()

print(
    'Combined journal-ready figure successfully saved as'
    " 'evaluation_roc_pr_combined.png'."
)


# In[ ]:




