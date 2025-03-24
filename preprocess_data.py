import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# **Step 1: Load the dataset**
df = pd.read_csv("MIT-BIH Arrhythmia Database.csv", low_memory=False)

# **Step 2: Drop unnecessary columns**
df.drop(columns=["record"], inplace=True, errors="ignore")  # Ignore error if 'record' is missing

# **Step 3: Handle Missing Values**
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill NaN with column mean

# Create missing value indicator columns (optional)
missing_indicators = df.isna().astype(int).add_suffix('_missing')
df = pd.concat([df, missing_indicators], axis=1)

# **Step 4: Encode the "type" column as Normal (0) or Abnormal (1)**
type_mapping = {
    "N": 0,  # Normal
    "F": 1,  # Abnormal
    "Q": 1,  # Abnormal
    "SVEB": 1,  # Abnormal
    "VEB": 1   # Abnormal
}
df["type"] = df["type"].map(type_mapping)

# Remove any rows where 'type' could not be mapped
df = df.dropna(subset=["type"])
df["type"] = df["type"].astype(int)  # Ensure type column is integer

# **Step 5: Normalize Numeric Columns**
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.difference(["type"])
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# **Step 6: Split into Features (X) and Labels (y)**
X = df.drop(columns=["type"]).values  # Features
y = df["type"].values  # Labels

# **Step 7: Convert Labels to Binary Format (No One-Hot Encoding)**
y = y.astype(np.float32)  # Convert labels to float for binary classification

# **Step 8: Reshape X for CNN Input (samples, features, channels)**
X = X.reshape((X.shape[0], X.shape[1], 1))

# **Step 9: Save Preprocessed Data**
np.save("X_mit.npy", X)
np.save("y_mit.npy", y)

print("Preprocessing completed successfully.")
print("X shape:", X.shape)
print("y shape:", y.shape)
