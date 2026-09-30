#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from scipy.stats import genpareto
from sklearn.metrics import classification_report, precision_recall_fscore_support
import warnings
from decimal import Decimal
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support, auc

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)


# In[3]:


# =========================================================================
# EXPERIMENTAL CONFIGURATION SETUP
# =========================================================================
SAMPLE_SIZE = 600000
ANOMALY_COUNT = 5000
LATENT_DIM = 8


# In[4]:


#Read SWAT file
df = pd.read_csv(r"SWaT_dataset.csv", low_memory=False)
print(df.shape)
df.columns = df.columns.str.strip()
df.rename(columns={"Normal/Attack": "AttackLabel"}, inplace=True)
# df_copy = df

# Forward fill missing values and drop zero-variance features
df.ffill(inplace=True)
df = df.loc[:, df.nunique() > 1]

# 1. Check which specific columns contain NaNs
nan_counts = df.isna().sum()
print("Columns with missing values:\n", nan_counts[nan_counts > 0])

# 2. Check if the NaNs are concentrated in the 'Normal/Attack' label column or the sensors
print(f"Total rows in dataframe: {len(df)}")

# Print all column names sorted to check for hidden duplicates
#print(df.columns.tolist())

# Remove leading/trailing spaces if any
df['Timestamp'] = df['Timestamp'].astype(str).str.strip()
# Convert to datetime
df['Timestamp_dt'] = pd.to_datetime(
    df['Timestamp'],
    format='%d/%m/%Y %I:%M:%S %p',
    dayfirst=True,
    errors='coerce'
)
# Sort dataframe by timestamp
df = df.sort_values(by='Timestamp_dt',ascending=True).reset_index(drop=True)
print(df.shape)
# Optional: remove helper column
df = df.drop('Timestamp_dt',axis=1)
print(df.shape)

# Filter
normal_df = df[df["AttackLabel"] == "Normal"]
attack_df = df[df["AttackLabel"] == "Attack"]

# Counts
print(f"Normal events: {len(normal_df)}")
print(f"Attack events: {len(attack_df)}")

#Regroup columns to make like column together

# Initialize empty lists to store column names by their structural types
numerical_cols = []
categorical_discols = []
categorical_txcols = []
datetime_cols = []

# =========================================================================
# STEP 1: SORT AND IDENTIFY COLUMNS BY DATA TYPE
# =========================================================================
for col in df.columns:
    unique_count = df[col].nunique()

    if pd.api.types.is_datetime64_any_dtype(df[col]):
        #print(f"{col}: DATETIME")
        datetime_cols.append(col)

    elif pd.api.types.is_numeric_dtype(df[col]):
        if unique_count <= 5:
            #print(f"{col}: DISCRETE/CATEGORICAL ({unique_count} values)")
            categorical_discols.append(col)
        else:
            #print(f"{col}: NUMERICAL")
            numerical_cols.append(col)

    else:
        #print(f"{col}: CATEGORICAL/TEXT")
        categorical_txcols.append(col)

# =========================================================================
# STEP 2: REGROUP DATA FRAME COLUMNS (NUMERICAL -> CATEGORICAL -> DATETIME)
# =========================================================================
# Concatenate the lists together to form the new target schema order
ordered_cols = numerical_cols + categorical_discols +  categorical_txcols + datetime_cols

# Reindex your dataframe with the new column layout
df_grouped = df[ordered_cols]

print("------------------------------------------------------------")
print(f"Feature Grouping Complete!")
print(f"Total Numerical Columns:   {len(numerical_cols)}")
print(f"Total Categorical Discrete Columns: {len(categorical_discols)}")
print(f"Total Categorical Text Columns: {len(categorical_txcols)}")
print(f"Total Datetime Columns:    {len(datetime_cols)}")
print(df_grouped.shape)
print("------------------------------------------------------------")

num_f = 0
cat_f = 0
text_f = 0
date_f = 0
for col in df_grouped.columns:
    unique_count = df_grouped[col].nunique()

    if pd.api.types.is_datetime64_any_dtype(df[col]):
        #print(f"{col}: DATETIME")
        date_f += 1

    elif pd.api.types.is_numeric_dtype(df[col]):
        if unique_count <= 5:
            #print(f"{col}: DISCRETE/CATEGORICAL ({unique_count} values)")
            cat_f +=1
        else:
            #print(f"{col}: NUMERICAL")
            num_f +=1

    else:
        #print(f"{col}: CATEGORICAL/TEXT")
        text_f += 1

print(f"Numerical feature count is {num_f}, categorical feature count is {cat_f}, text feature is {text_f}")

#add month,year, day/night columns
df_grouped['Timestamp'] = df_grouped['Timestamp'].astype(str).str.strip()
df_grouped['Timestamp'] = pd.to_datetime(
    df_grouped['Timestamp'],
    format='%d/%m/%Y %I:%M:%S %p'
)
# Create new columns
df_grouped['MONTH'] = df_grouped['Timestamp'].dt.month
df_grouped['YEAR'] = df_grouped['Timestamp'].dt.year
df_grouped['AM_PM'] = df_grouped['Timestamp'].dt.strftime('%p')

y_true = df_grouped['AttackLabel'].to_numpy()
y_binary = np.where(y_true == 'Normal', 0, 1)

df_grouped = df_grouped.drop('Timestamp', axis = 1)
df_grouped = df_grouped.drop('AttackLabel', axis = 1)
df_grouped = df_grouped.drop('AM_PM', axis = 1)

#Rename columns
# Number of categorical columns excluding MONTH and YEAR
n_cat = len(df_grouped.columns) - 25 - 2
new_cols = (
    [f'N{i+1}' for i in range(25)] +
    [f'C{i+1}' for i in range(n_cat)] +
    ['MONTH', 'YEAR']
)
df_grouped.columns = new_cols
print(list(df_grouped.columns))


# In[5]:


# 60:40 Split Matrix
split_idx = int(len(df_grouped) * 0.68)
df_train, df_test = df_grouped.iloc[:split_idx].copy(), df_grouped.iloc[split_idx:].copy()
print(len(df_train),len(df_test))
y_train_true, y_test_true = y_binary[:split_idx], y_binary[split_idx:]
total_ones_train = np.sum(y_train_true == 1)
print(f'one count is train set is ' ,total_ones_train)
total_ones_test = np.sum(y_test_true == 1)
print(f'one count is test set is ' ,total_ones_test)


# In[6]:


from sklearn.preprocessing import RobustScaler, OneHotEncoder
import torch
# =========================================================================
# STEP 1: HETEROGENEOUS ENCODING (REVISED FOR 50-FEATURE MATRIX)
# =========================================================================
# 1. Programmatically define feature column lists to avoid dictionaries or walls of text
num_cols = [f'N{i}' for i in range(1, 26)]    # N1 to N40
cat_cols = [f'C{j}' for j in range(1, 21)]     # C1 to C8
month_col = 'MONTH'                              # Month
year_col = 'YEAR'                               # Year

# 2. Continuous Metric Normalization using RobustScaler
scaler = RobustScaler()
train_num_scaled = scaler.fit_transform(df_train[num_cols])
test_num_scaled = scaler.transform(df_test[num_cols])

# 3. Multi-Column Categorical One-Hot Encoding
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
train_cat_encoded = encoder.fit_transform(df_train[cat_cols])
test_cat_encoded = encoder.transform(df_test[cat_cols])

# 4. Cyclical Trigonometric Mapping for Month Feature (T1)
train_month_sin = np.sin(2 * np.pi * df_train[month_col].values / 12.0)[:, None]
train_month_cos = np.cos(2 * np.pi * df_train[month_col].values / 12.0)[:, None]
test_month_sin = np.sin(2 * np.pi * df_test[month_col].values / 12.0)[:, None]
test_month_cos = np.cos(2 * np.pi * df_test[month_col].values / 12.0)[:, None]

# 5. Temporal Baseline Alignment for Year Feature (T2)
# Re-centered around 2024 to keep trend scales balanced near the origin
train_year_scaled = ((df_train[year_col].values - 2015) / 2.0)[:, None]
test_year_scaled = ((df_test[year_col].values - 2015) / 2.0)[:, None]

# 6. Horizontal Feature Stack to Create the Global Feature Space Matrix
X_train_processed = np.hstack([
    train_num_scaled, 
    train_cat_encoded, 
    train_month_sin, 
    train_month_cos, 
    train_year_scaled
])

X_test_processed = np.hstack([
    test_num_scaled, 
    test_cat_encoded, 
    test_month_sin, 
    test_month_cos, 
    test_year_scaled
])

'''
print("-------------------------------------")
print("Sample Row from Processed NumPy Array:")
print(X_train_processed[100])
print("-------------------------------------")
'''
# 7. Convert Processed Matrix to Native PyTorch Tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
'''
print("-------------------------------------")
print("Sample Row from Final PyTorch Tensor:")
print(X_train_tensor[100])
print("-------------------------------------")
'''
# =========================================================================
# CONTEXT INDICES DEFINITION
# =========================================================================
# Selecting the first 10 numerical features (N1 to N10) as your context variables.
# These represent your continuous, sinusoidal cyclic metrics (e.g., Temperature, Pressure).
context_indices = list(range(5,22))                         # Year


'''
#print(X_train_processed[100])
'''

# 7. Convert Processed Matrix to Native PyTorch Tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)

print("-----Done-----------")


# In[7]:


# =========================================================================
# STEP 2: REPRESENTATION LEARNING (FIXED LATENT SIZE = 8)
# =========================================================================
'''
class HighDimensionalBottleneckAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        hidden_dim = max(input_dim // 2, latent_dim * 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

'''
class HighDimensionalBottleneckAE(nn.Module):
    def __init__(self, input_dim, latent_dim=LATENT_DIM): # Example default latent dim
        super().__init__()
        
        # Define a progressive symmetric bottleneck
        hidden_dim_1 = max(input_dim // 2, latent_dim * 4)
        hidden_dim_2 = max(input_dim // 4, latent_dim * 2)
        
        # Encoder with LeakyReLU to preserve tail data variation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim_2, latent_dim) # Latent coordinates can now be negative
        )
        
        # Decoder mirroring the encoder structure
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim_1, input_dim)  # Linear output to match scaled continuous ranges
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))


# Clear the old model from memory
#del model

input_feature_dimension = X_train_processed.shape[1]
model = HighDimensionalBottleneckAE(input_dim=input_feature_dimension, latent_dim=LATENT_DIM)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

# Train on the expanded baseline subset
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    loss = torch.mean((X_train_tensor - model(X_train_tensor)) ** 2)
    loss.backward()
    optimizer.step()
print("****Done**********")


# In[8]:


# =========================================================================
# STEP 3: THREE-PRONGED MULTI-DIMENSIONAL ANOMALY SCORING (CORRECTED)
# =========================================================================
model.eval()
with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds = model(X_test_tensor)

# --- 1. MAHALANOBIS MEAN & PRECISION CALCULATIONS ---
# Isolate context features from the processed array
v_train_baseline = X_train_processed[:, context_indices]

# Compute mean vector (mu) to properly center Mahalanobis space
mu_v = torch.tensor(np.mean(v_train_baseline, axis=0), dtype=torch.float32)

# Compute precision matrix with standard ridge regularization
cov_matrix = np.cov(v_train_baseline, rowvar=False) + 1e-5 * np.eye(len(context_indices))
precision_matrix_v = torch.tensor(np.linalg.inv(cov_matrix), dtype=torch.float32)

def extract_scores(X_raw, X_hat):
    num_samples = X_raw.size(0)
    
    # --- PRONG A: MEAN SQUARED ERROR (MSE) ---
    R = torch.mean((X_raw - X_hat) ** 2, dim=-1)

    # --- PRONG B: CORRECTED MAHALANOBIS DISTANCE ---
    # Center the vectors by subtracting the training mean (v - mu)
    # Force mu_v to match the hardware device and precision of X_raw dynamically
    v_centered = (X_raw[:, context_indices] - mu_v.to(X_raw.device)).unsqueeze(1)
    
    # Inside extract_scores function, make sure precision matrix matches the GPU device
    prec_expanded = precision_matrix_v.to(X_raw.device).unsqueeze(0).expand(num_samples, -1, -1)
    
    # Compute: (v-mu) * Inv_Cov * (v-mu)^T
    MD = torch.bmm(torch.bmm(v_centered, prec_expanded), v_centered.transpose(1, 2)).squeeze(-1).squeeze(-1)
    # Journal Standard: Take the square root to align with the formal mathematical definition
    MD = torch.sqrt(torch.clamp(MD, min=0.0))

    # --- PRONG C: VECTORIZED L1 HISTORICAL DIFFERENTIAL ---
    K = 100
    D = X_raw - X_hat  # Current residual matrix [N, num_features]
    
    # Use causal 1D average pooling to compute moving averages in parallel
    # Transpose to [1, num_features, N] to meet Conv1d / AvgPool1d specifications
    D_padded = torch.nn.functional.pad(D.t().unsqueeze(0), (K-1, 0), mode='replicate')
    
    # Compute sliding window means instantly across all rows simultaneously
    historical_mean_tensor = torch.nn.functional.avg_pool1d(D_padded, kernel_size=K, stride=1).squeeze(0).t()
    
    # Compute L1 norm across the feature dimension
    L1_delta = torch.norm(D - historical_mean_tensor, p=1, dim=-1)

    return R.cpu().numpy(), MD.cpu().numpy(), L1_delta.cpu().numpy()

# Execute scoring
print(f"length of train_tensor is {len(X_train_tensor)} and length of train_preds is {len(X_test_tensor)}")
print("Train Tensor:", torch.isnan(X_train_tensor).any().item())
print("Test Tensor:", torch.isnan(X_test_tensor).any().item())
R_tr, MD_tr, L1_tr = extract_scores(X_train_tensor, train_preds)
R_ts, MD_ts, L1_ts = extract_scores(X_test_tensor, test_preds)

# --- 2. ROBUST MIN-MAX UNIFICATION MATRIX ---
# Store reference training statistics
stats = {
    'R_min': R_tr.min(), 'R_max': R_tr.max(),
    'MD_min': MD_tr.min(), 'MD_max': MD_tr.max(),
    'L1_min': L1_tr.min(), 'L1_max': L1_tr.max()
}

# Calculate exact delta ranges to prevent zero-division magnification errors
r_range = stats['R_max'] - stats['R_min']
md_range = stats['MD_max'] - stats['MD_min']
l1_range = stats['L1_max'] - stats['L1_min']

# Safeguard ranges: if range is 0, default to 1.0 to keep values at zero cleanly
r_denom = r_range if r_range > 0 else 1.0
md_denom = md_range if md_range > 0 else 1.0
l1_denom = l1_range if l1_range > 0 else 1.0

# Normalize training data
R_tr_norm = (R_tr - stats['R_min']) / r_denom
MD_tr_norm = (MD_tr - stats['MD_min']) / md_denom
print(f"Max of MD_tr is {np.amax(MD_tr_norm)},range is {np.ptp(MD_tr_norm)}, md_denom is {md_denom}")

L1_tr_norm = (L1_tr - stats['L1_min']) / l1_denom
A_train = (0.15 * R_tr_norm) + (0.70 * MD_tr_norm) + (0.15 * L1_tr_norm)

# Normalize testing data with a clip guardrail
# Extreme test spikes could exceed training maximums; clipping prevents negative weights or infinity distortions
R_ts_norm = np.clip((R_ts - stats['R_min']) / r_denom, 0, None)
MD_ts_norm = np.clip((MD_ts - stats['MD_min']) / md_denom, 0, None)
print(f"Max of MD_ts is {np.amax(MD_ts_norm)},range is {np.ptp(MD_ts_norm)}, md_denom is {md_denom}")

L1_ts_norm = np.clip((L1_ts - stats['L1_min']) / l1_denom, 0, None)
A_test = (0.15 * R_ts_norm) + (0.70 * MD_ts_norm) + (0.15 * L1_ts_norm)

print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Max ATest is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(f"Scoring pipeline complete. Total test anomaly indices mapped: {len(A_test)}")



# In[9]:


print("Train Tensor:", torch.isnan(X_train_tensor).any().item())
print("Test Tensor:", torch.isnan(X_test_tensor).any().item())

num_nans = torch.isnan(X_train_tensor).sum().item()
print("Number of NaNs:", num_nans)


# In[10]:


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


# In[11]:


gamma_threshold = np.percentile(A_train, 95) # Tightened slightly to fit the 0.4% density profile
print(f"Max ATrain is {np.amax(A_train)}, Min is {np.amin(A_train)}, std is {np.std(A_train)}, range is {np.ptp(A_train)}, median is {np.median(A_train)}")
print(f"Min ATrain is {np.amax(A_test)}, Min is {np.amin(A_test)}, std is {np.std(A_test)}, range is {np.ptp(A_test)}, median is {np.median(A_test)}")
print(" Gamma Threshold is ", gamma_threshold);

# =========================================================================
# STEP 4: REVISED AUTOMATED GATING (EVT/POT METHOD)
# =========================================================================
# 1. Set an initial loose threshold (t) to capture the tail candidate pool
# Typically 98th percentile is standard to isolate extreme values for fitting
t = np.percentile(A_train, 95)
tail_excesses = A_train[A_train > t] - t

# 2. Fit the tail excesses to the Generalized Pareto Distribution (GPD)
# c = shape parameter, loc = location parameter, scale = scale parameter
c, loc, scale = genpareto.fit(tail_excesses, floc=0)
print(f"Pareto distribution: c is {c}, loc is {loc}, scale is {scale}")
print(f"Shape parameter (c): {c:.4f}  ({'Heavy-tailed' if c > 0 else 'Light/Bounded-tailed'})")

# 3. Define your target risk probability (q) 
#q=1e-4 #means you accept a 0.01% theoretical probability of a false alarm
#q = 1e-4
q = 0.009
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

# 5. Evaluate the testing data against the mathematically robust threshold
y_gate = (A_test > gamma_threshold).astype(int)
y_gate_train = (A_train > gamma_threshold).astype(int)
y_gate_full = np.concatenate([y_gate_train, y_gate])

total_ones = np.sum(y_gate_full == 1)
print("Anomaly count is ",total_ones)

# Diagnostic Prints
print(f"Total Test Samples: {len(y_gate)}")
print(f"Total Anomalies Flagged (y_gate == 1): {np.sum(y_gate == 1)}")


# In[12]:


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


# In[339]:


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


# In[13]:


# =========================================================================
# STEP 6: RAW UNADJUSTED PERFORMANCE TRACKING
# =========================================================================
y_test_aligned = (y_test_true > 0).astype(int)

# use full data set
y_test_aligned_full = (y_binary > 0).astype(int)

#Merge A_test and A_train
print(type(A_train),A_train.shape)
print(type(A_test),A_test.shape)

A_full = np.concatenate((A_train, A_test), axis=0)


# In[14]:


#print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
#print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")

# use full data set
print("\n" + "="*60 + "\nRAW UNADJUSTED METRIC TRACKING MATRIX\n" + "="*60)
print(classification_report(y_test_aligned, y_gate, target_names=["Normal Matrix State", "Rare Event Target"]))
#print(classification_report(y_test_aligned_full, y_gate_full, target_names=["Normal Matrix State", "Rare Event Target"]))
print(f"Framework Unadjusted ROC-AUC Tracking Bounds: {roc_auc_score(y_test_aligned, A_test):.4f}")


# In[15]:


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


# In[16]:


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


# In[17]:


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
y_pred_adjusted = calculate_point_adjusted_metrics(y_test_true, y_gate)

# 3. Extract final comparative metrics
prec_raw, rec_raw, f1_raw, _ = precision_recall_fscore_support(y_test_true, y_gate, average='binary')
prec_pa, rec_pa, f1_pa, _ = precision_recall_fscore_support(y_test_true, y_pred_adjusted, average='binary')

print("=" * 60)
print("             METRIC COMPARATIVE MATRIX (SWAT)              ")
print("=" * 60)
print(f"RAW UNADJUSTED METRICS: \n Precision: {prec_raw:.4f} | Recall: {rec_raw:.4f} | F1-Score: {f1_raw:.4f}")
print("-" * 60)
print(f"POINT-ADJUSTED METRICS (PA): \n Precision: {prec_pa:.4f} | Recall: {rec_pa:.4f} | F1-Score: {f1_pa:.4f}")
print("=" * 60)

# 4. Generate the rigorous journal classification report
print("\nDETAILED POINT-ADJUSTED CLASSIFICATION REPORT:")
print(classification_report(y_test_true, y_pred_adjusted, target_names=['Normal State', 'Rare Event Target'], digits=4))


# In[18]:

# Set journal typography and rendering defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5

def calculate_point_adjusted_metrics(y_true, y_pred):
    """
    Calculates Point-Adjusted (PA) metrics for time-series anomaly detection.
    """
    y_true = np.array(y_true).astype(int)
    y_pred_pa = np.array(y_pred).copy().astype(int)
    
    diff = np.diff(np.concatenate(([0], y_true, [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if np.any(y_pred_pa[start:end] == 1):
            y_pred_pa[start:end] = 1
            
    return y_pred_pa

def compute_pa_roc_pr_curves(y_true, scores, num_thresholds=200):
    """
    Computes Point-Adjusted ROC and PR curves across dynamic threshold sweeps.
    """
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    
    tpr_list, fpr_list = [], []
    precision_list, recall_list = [], []
    
    total_positives = np.sum(y_true == 1)
    total_negatives = np.sum(y_true == 0)
    
    for th in reversed(thresholds):
        raw_pred = (scores >= th).astype(int)
        pa_pred = calculate_point_adjusted_metrics(y_true, raw_pred)
        
        tp = np.sum((pa_pred == 1) & (y_true == 1))
        fp = np.sum((pa_pred == 1) & (y_true == 0))
        
        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tpr
        
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        precision_list.append(precision)
        recall_list.append(recall)
        
    return np.array(fpr_list), np.array(tpr_list), np.array(precision_list), np.array(recall_list)

# =========================================================================
# PIPELINE EXECUTION
# =========================================================================

# 1. Compute Raw Continuous Curves
fpr_raw, tpr_raw, _ = roc_curve(y_test_true, A_test)
roc_auc_raw = auc(fpr_raw, tpr_raw)

precision_raw, recall_raw, _ = precision_recall_curve(y_test_true, A_test)
pr_auc_raw = auc(recall_raw, precision_raw)

# 2. Compute Point-Adjusted Continuous Curves
fpr_pa, tpr_pa, prec_pa_curve, rec_pa_curve = compute_pa_roc_pr_curves(y_test_true, A_test, num_thresholds=200)
roc_auc_pa = auc(fpr_pa, tpr_pa)
pr_auc_pa = auc(rec_pa_curve, prec_pa_curve)

# =========================================================================
# GENERATE DUAL-PANEL PUBLICATION FIGURE WITH SUB-TITLES BELOW
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), dpi=300)

# --- PANEL A: ROC CURVE ---
axes[0].plot(fpr_raw, tpr_raw, color='#1f77b4', lw=2, linestyle='-', label=f'Raw Unadjusted (AUC = {roc_auc_raw:.4f})')
axes[0].plot(fpr_pa, tpr_pa, color='#d62728', lw=2, linestyle='--', label=f'Point-Adjusted (AUC = {roc_auc_pa:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle=':', label='Random Baseline')

axes[0].set_xlim([-0.02, 1.02])
axes[0].set_ylim([-0.02, 1.02])
axes[0].set_xlabel('False Positive Rate (FPR)')
axes[0].set_ylabel('True Positive Rate (TPR)')
axes[0].legend(loc="lower right", frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
axes[0].grid(True, linestyle='--', alpha=0.4)

# Move Sub-caption below Subplot A
axes[0].text(0.5, -0.22, '(a) ROC Curve', transform=axes[0].transAxes,
             fontsize=12, fontweight='bold', ha='center', va='top')

# --- PANEL B: PRECISION-RECALL CURVE ---
axes[1].plot(recall_raw, precision_raw, color='#2ca02c', lw=2, linestyle='-', label=f'Raw Unadjusted (AUC = {pr_auc_raw:.4f})')
axes[1].plot(rec_pa_curve, prec_pa_curve, color='#9467bd', lw=2, linestyle='--', label=f'Point-Adjusted (AUC = {pr_auc_pa:.4f})')

axes[1].set_xlim([-0.02, 1.02])
axes[1].set_ylim([-0.02, 1.02])
axes[1].set_xlabel('Recall (True Positive Rate)')
axes[1].set_ylabel('Precision (Positive Predictive Value)')
axes[1].legend(loc="lower left", frameon=True, facecolor='white', edgecolor='none', framealpha=0.9)
axes[1].grid(True, linestyle='--', alpha=0.4)

# Move Sub-caption below Subplot B
axes[1].text(0.5, -0.22, '(b) Precision-recall curve', transform=axes[1].transAxes,
             fontsize=12, fontweight='bold', ha='center', va='top')

# Adjust layout to accommodate bottom sub-labels
plt.tight_layout()
plt.subplots_adjust(bottom=0.20)

# Save high-resolution outputs
plt.savefig('swat_dual_evaluation_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('swat_dual_evaluation_curves.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("Journal-ready figures successfully exported with sub-captions positioned below.")


# In[ ]:




