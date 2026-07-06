import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_predict
import shutil

INPUT_CSV = "features_table_enhanced.csv"
DATASET_DIR = "dataset_cleaned_final"
QUARANTINE_DIR = "quarantine_filtered"

df = pd.read_csv(INPUT_CSV)
feature_cols = [c for c in df.columns if c not in ['class', 'filename']]
X = df[feature_cols].values
y_str = df['class'].values

le = LabelEncoder()
y = le.fit_transform(y_str)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
pred = cross_val_predict(model, X_scaled, y, cv=5)

bad_idx = np.where(pred != y)[0]
print(f"Total: {len(df)}  -  Wrong: {len(bad_idx)}")

os.makedirs(QUARANTINE_DIR, exist_ok=True)
for idx in bad_idx:
    row = df.iloc[idx]
    cls, fname = row['class'], row['filename']
    src = os.path.join(DATASET_DIR, cls, fname)
    dst_dir = os.path.join(QUARANTINE_DIR, cls)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Quarantined: {cls}/{fname}")
    else:
        print(f"File not found: {src}")

print("Filtering completed.")