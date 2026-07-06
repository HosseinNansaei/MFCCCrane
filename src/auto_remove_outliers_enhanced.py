import pandas as pd
import numpy as np
import os
from scipy.spatial.distance import cdist

INPUT_CSV = "features_table_enhanced.csv"
DATASET_DIR = "dataset_cleaned_final"
OUTLIER_PERCENT = 5.0
REMOVED_REPORT = "removed_files_report_enhanced.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Total samples: {len(df)}")

class_col = 'class'
filename_col = 'filename'
feature_cols = [c for c in df.columns if c not in [class_col, filename_col]]

def delete_audio(cls, fname):
    path = os.path.join(DATASET_DIR, cls, fname)
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

removed = []
for cls in df[class_col].unique():
    df_cls = df[df[class_col] == cls].copy()
    X = df_cls[feature_cols].values.astype(np.float64)
    mean_vec = np.mean(X, axis=0)
    std_vec = np.std(X, axis=0)
    std_vec[std_vec < 1e-8] = 1.0
    X_norm = (X - mean_vec) / std_vec
    centroid = np.mean(X_norm, axis=0)
    distances = cdist(X_norm, [centroid], metric='euclidean').flatten()
    threshold = np.percentile(distances, 100 - OUTLIER_PERCENT)
    outlier_mask = distances > threshold
    for _, row in df_cls[outlier_mask].iterrows():
        fname = row[filename_col]
        if delete_audio(cls, fname):
            print(f"Removed: {cls}/{fname}")
            removed.append({'class': cls, 'filename': fname})
        else:
            print(f"Not found: {cls}/{fname}")

if removed:
    pd.DataFrame(removed).to_csv(REMOVED_REPORT, index=False)
    print(f"Report saved: {REMOVED_REPORT}")
else:
    print("No files removed.")