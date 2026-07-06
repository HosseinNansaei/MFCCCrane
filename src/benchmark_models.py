"""
================================================================================
BENCHMARK: Full Model Comparison with Live Testing
================================================================================
This script:
1. Compares all classification methods (Theoretical Benchmark)
2. Tests each method with 10 sample commands (Practical Test)
3. Generates two separate tables: Theoretical & Practical
4. Creates comparison charts for reports
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings("ignore")

# =================================================================================
# CONFIGURATION
# =================================================================================
plt.rcParams['font.family'] = 'DejaVu Sans'
CSV_FILE = None
for f in ["features_table_enhanced.csv", "features_table.csv"]:
    if os.path.exists(f):
        CSV_FILE = f
        break

if CSV_FILE is None:
    print("ERROR: No CSV file found! Run extract_features_enhanced.py first.")
    exit(1)

NUM_TEST_SAMPLES = 10  # Number of live test commands

# =================================================================================
# LOAD DATA
# =================================================================================
print("=" * 70)
print("BENCHMARK: Full Model Comparison with Live Testing")
print("=" * 70)

df = pd.read_csv(CSV_FILE)
feature_cols = [c for c in df.columns if c not in ['class', 'filename']]
X = df[feature_cols].values.astype(np.float64)
y_str = df['class'].values

le = LabelEncoder()
y = le.fit_transform(y_str)
class_names = le.classes_.tolist()

print(f"Dataset: {CSV_FILE}")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {class_names}")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("=" * 70)

# =================================================================================
# METHOD 1: Simple KNN (Project Implementation)
# =================================================================================
def knn_simple_predict(X_train, y_train, X_test, k=3):
    preds = []
    n_classes = len(np.unique(y_train))
    for test_pt in X_test:
        distances = np.sqrt(np.sum((X_train - test_pt) ** 2, axis=1))
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        counts = np.bincount(nearest_labels, minlength=n_classes)
        preds.append(np.argmax(counts))
    return np.array(preds)

# =================================================================================
# METHOD 2: Weighted KNN (Project Implementation)
# =================================================================================
def weighted_knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    n_classes = len(np.unique(y_train))
    for test_pt in X_test:
        distances = np.sqrt(np.sum((X_train - test_pt) ** 2, axis=1))
        nearest_idx = np.argsort(distances)[:k]
        nearest_distances = distances[nearest_idx]
        nearest_labels = y_train[nearest_idx]
        weights = 1.0 / (nearest_distances + 1e-10)
        weights = weights / weights.sum()
        scores = np.zeros(n_classes)
        for label, weight in zip(nearest_labels, weights):
            scores[label] += weight
        preds.append(np.argmax(scores))
    return np.array(preds)

# =================================================================================
# METHOD 3: Random Forest (sklearn)
# =================================================================================
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# =================================================================================
# METHOD 4: SVM (sklearn)
# =================================================================================
svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)

# =================================================================================
# METHOD 5: KNN (sklearn)
# =================================================================================
knn_sk = KNeighborsClassifier(n_neighbors=3, metric='euclidean', n_jobs=-1)

# =================================================================================
# METHOD 6: Random Forest with GridSearch
# =================================================================================
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_gs = RandomForestClassifier(random_state=42, n_jobs=-1)
grid = GridSearchCV(rf_gs, param_grid, cv=3, n_jobs=-1, verbose=0)

# =================================================================================
# THEORETICAL BENCHMARK (On test dataset)
# =================================================================================
print("\n" + "=" * 70)
print("SECTION 1: THEORETICAL BENCHMARK (On Test Dataset)")
print("=" * 70)

theoretical_results = []

def evaluate_method(name, predictor, train_data, train_labels, test_data, test_labels, has_train=True):
    """Evaluate a method and return metrics"""
    results = {}
    results['Method'] = name
    
    # Training time (if applicable)
    train_time = 0
    if has_train:
        start = time.time()
        predictor.fit(train_data, train_labels)
        train_time = time.time() - start
    results['Train Time (s)'] = train_time
    
    # Prediction time
    start = time.time()
    if has_train:
        pred = predictor.predict(test_data)
    else:
        pred = predictor(train_data, train_labels, test_data)
    pred_time = time.time() - start
    results['Predict Time (s)'] = pred_time
    
    # Accuracy
    acc = np.mean(pred == test_labels) * 100
    results['Accuracy (%)'] = acc
    
    # Cross-Validation (5-fold)
    if has_train:
        cv_scores = cross_val_score(predictor, train_data, train_labels, cv=5)
        results['CV Mean (%)'] = np.mean(cv_scores) * 100
        results['CV Std (%)'] = np.std(cv_scores) * 100
    else:
        results['CV Mean (%)'] = acc
        results['CV Std (%)'] = 0
    
    return results, pred

# Evaluate all methods
print("\nEvaluating methods...")

# 1. Simple KNN
print("  [1/6] Simple KNN (Project)...")
res, pred = evaluate_method('Simple KNN (Project)', knn_simple_predict, 
                            X_train_scaled, y_train, X_test_scaled, y_test, has_train=False)
theoretical_results.append(res)

# 2. Weighted KNN
print("  [2/6] Weighted KNN (Project)...")
res, pred = evaluate_method('Weighted KNN (Project)', weighted_knn_predict,
                            X_train_scaled, y_train, X_test_scaled, y_test, has_train=False)
theoretical_results.append(res)

# 3. Random Forest
print("  [3/6] Random Forest (sklearn)...")
res, pred = evaluate_method('Random Forest (sklearn)', rf,
                            X_train_scaled, y_train, X_test_scaled, y_test, has_train=True)
theoretical_results.append(res)

# 4. SVM
print("  [4/6] SVM (sklearn)...")
res, pred = evaluate_method('SVM (sklearn)', svm,
                            X_train_scaled, y_train, X_test_scaled, y_test, has_train=True)
theoretical_results.append(res)

# 5. KNN (sklearn)
print("  [5/6] KNN (sklearn)...")
res, pred = evaluate_method('KNN (sklearn)', knn_sk,
                            X_train_scaled, y_train, X_test_scaled, y_test, has_train=True)
theoretical_results.append(res)

# 6. Random Forest (GridSearch)
print("  [6/6] Random Forest (GridSearch)...")
start = time.time()
grid.fit(X_train_scaled, y_train)
train_time = time.time() - start
start = time.time()
pred = grid.predict(X_test_scaled)
pred_time = time.time() - start
acc = np.mean(pred == y_test) * 100
cv_scores = cross_val_score(grid.best_estimator_, X_train_scaled, y_train, cv=5)
res = {
    'Method': 'RF (GridSearch)',
    'Train Time (s)': train_time,
    'Predict Time (s)': pred_time,
    'Accuracy (%)': acc,
    'CV Mean (%)': np.mean(cv_scores) * 100,
    'CV Std (%)': np.std(cv_scores) * 100
}
theoretical_results.append(res)

# Create DataFrame
df_theoretical = pd.DataFrame(theoretical_results)
df_theoretical = df_theoretical.sort_values('Accuracy (%)', ascending=False)

print("\n" + "-" * 70)
print("THEORETICAL RESULTS TABLE")
print("-" * 70)
print(df_theoretical.to_string(index=False))
print("-" * 70)

# Save theoretical results
df_theoretical.to_csv("benchmark_theoretical.csv", index=False, encoding='utf-8-sig')
print("✓ Saved: benchmark_theoretical.csv")

# =================================================================================
# PRACTICAL TEST: Live Testing with 10 Commands
# =================================================================================
print("\n" + "=" * 70)
print("SECTION 2: PRACTICAL TEST (Live Testing)")
print("=" * 70)
print(f"Testing with {NUM_TEST_SAMPLES} sample commands...")
print("")

# Select 10 random test samples (or all if less)
test_indices = np.random.choice(len(X_test), min(NUM_TEST_SAMPLES, len(X_test)), replace=False)
practical_results = []

# Retrain all models on full training data for practical test
print("Retraining models for practical test...")

# Simple KNN (no training needed)
# Weighted KNN (no training needed)

# Train sklearn models
rf.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
knn_sk.fit(X_train_scaled, y_train)
grid.fit(X_train_scaled, y_train)

print("\nTesting each sample...\n")

# Test each sample with all methods
for idx, test_idx in enumerate(test_indices, 1):
    sample = X_test_scaled[test_idx:test_idx+1]
    true_label = class_names[y_test[test_idx]]
    
    # Get predictions from all methods
    methods = [
        ('Simple KNN', lambda x: knn_simple_predict(X_train_scaled, y_train, x, k=3)),
        ('Weighted KNN', lambda x: weighted_knn_predict(X_train_scaled, y_train, x, k=3)),
        ('Random Forest', lambda x: rf.predict(x)),
        ('SVM', lambda x: svm.predict(x)),
        ('KNN (sklearn)', lambda x: knn_sk.predict(x)),
        ('RF (GridSearch)', lambda x: grid.predict(x))
    ]
    
    results = {'Sample #': idx, 'True Label': true_label}
    correct_count = 0
    
    for name, predictor in methods:
        pred = predictor(sample)[0]
        pred_label = class_names[pred]
        is_correct = (pred_label == true_label)
        results[f'{name}'] = pred_label
        results[f'{name}_Correct'] = is_correct
        if is_correct:
            correct_count += 1
    
    results['Correct Count'] = correct_count
    results['Accuracy (%)'] = (correct_count / len(methods)) * 100
    practical_results.append(results)
    
    # Print progress
    status = "✓" if correct_count == len(methods) else "⚠" if correct_count >= 4 else "✗"
    print(f"  Sample {idx}: True={true_label:6} | {correct_count}/{len(methods)} correct {status}")

# Create practical results DataFrame
df_practical = pd.DataFrame(practical_results)

print("\n" + "-" * 70)
print("PRACTICAL RESULTS TABLE (10 Samples)")
print("-" * 70)

# Select columns for display
display_cols = ['Sample #', 'True Label'] + [m for m, _ in methods]
df_display = df_practical[display_cols]
print(df_display.to_string(index=False))
print("-" * 70)

# Calculate per-method accuracy from practical test
print("\nPer-Method Accuracy (Practical Test):")
practical_summary = []
for name, _ in methods:
    correct_col = f'{name}_Correct'
    if correct_col in df_practical.columns:
        acc = df_practical[correct_col].sum() / len(df_practical) * 100
        practical_summary.append({'Method': name, 'Practical Accuracy (%)': acc})

df_practical_summary = pd.DataFrame(practical_summary)
df_practical_summary = df_practical_summary.sort_values('Practical Accuracy (%)', ascending=False)
print(df_practical_summary.to_string(index=False))

# Save practical results
df_practical.to_csv("benchmark_practical.csv", index=False, encoding='utf-8-sig')
df_practical_summary.to_csv("benchmark_practical_summary.csv", index=False, encoding='utf-8-sig')
print("\n✓ Saved: benchmark_practical.csv")
print("✓ Saved: benchmark_practical_summary.csv")

# =================================================================================
# GENERATE COMPARISON CHARTS
# =================================================================================
print("\n" + "=" * 70)
print("SECTION 3: GENERATING CHARTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Comparison: Theoretical vs Practical', fontsize=16, fontweight='bold')

# Chart 1: Theoretical Accuracy
ax1 = axes[0, 0]
methods = df_theoretical['Method'].tolist()
acc = df_theoretical['Accuracy (%)'].tolist()
colors = ['#2ecc71' if 'Project' in m else '#e67e22' for m in methods]
bars = ax1.barh(methods, acc, color=colors, edgecolor='#2c3e50', linewidth=1.5)
ax1.set_xlabel('Accuracy (%)', fontsize=11)
ax1.set_title('Theoretical Accuracy (Test Dataset)', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 105)
for bar, val in zip(bars, acc):
    ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
ax1.axvline(x=85, color='red', linestyle='--', alpha=0.5, label='Target (85%)')
ax1.legend()

# Chart 2: Practical Accuracy
ax2 = axes[0, 1]
prac_methods = df_practical_summary['Method'].tolist()
prac_acc = df_practical_summary['Practical Accuracy (%)'].tolist()
bars = ax2.barh(prac_methods, prac_acc, color=['#2ecc71' if 'Project' in m else '#3498db' for m in prac_methods],
                edgecolor='#2c3e50', linewidth=1.5)
ax2.set_xlabel('Accuracy (%)', fontsize=11)
ax2.set_title('Practical Accuracy (10 Sample Test)', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 105)
for bar, val in zip(bars, prac_acc):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Chart 3: Training Time Comparison
ax3 = axes[1, 0]
train_times = df_theoretical['Train Time (s)'].tolist()
bars = ax3.barh(methods, train_times, color=['#3498db' if t > 0 else '#95a5a6' for t in train_times],
                edgecolor='#2c3e50', linewidth=1.5)
ax3.set_xlabel('Training Time (seconds)', fontsize=11)
ax3.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
for bar, val in zip(bars, train_times):
    if val > 0:
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f'{val:.2f}s', va='center', fontsize=9)

# Chart 4: CV vs Test Accuracy
ax4 = axes[1, 1]
cv_mean = df_theoretical['CV Mean (%)'].tolist()
test_acc = df_theoretical['Accuracy (%)'].tolist()
x_pos = np.arange(len(methods))
width = 0.35
bars1 = ax4.bar(x_pos - width/2, cv_mean, width, label='CV Mean', color='#3498db')
bars2 = ax4.bar(x_pos + width/2, test_acc, width, label='Test Accuracy', color='#e67e22')
ax4.set_xlabel('Method', fontsize=11)
ax4.set_ylabel('Accuracy (%)', fontsize=11)
ax4.set_title('CV vs Test Accuracy', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
ax4.legend()
ax4.axhline(y=85, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_full_chart.png', dpi=300, bbox_inches='tight')
plt.show()

print("✓ Saved: benchmark_full_chart.png")

# =================================================================================
# GENERATE LATEX TABLES FOR REPORT
# =================================================================================
print("\n" + "=" * 70)
print("SECTION 4: LATEX TABLES FOR REPORT")
print("=" * 70)

# Theoretical Table (LaTeX format)
print("\n--- THEORETICAL TABLE (LaTeX) ---")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Theoretical Benchmark Results}")
print("\\begin{tabular}{|l|c|c|c|c|}")
print("\\hline")
print("\\textbf{Method} & \\textbf{Accuracy (\\%)} & \\textbf{CV Mean (\\%)} & \\textbf{Train Time (s)} & \\textbf{Predict Time (s)} \\\\")
print("\\hline")
for _, row in df_theoretical.iterrows():
    print(f"{row['Method']} & {row['Accuracy (%)']:.2f} & {row['CV Mean (%)']:.2f} & {row['Train Time (s)']:.3f} & {row['Predict Time (s)']:.3f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

# Practical Table (LaTeX format)
print("\n--- PRACTICAL TABLE (LaTeX) ---")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Practical Test Results (10 Samples)}")
print("\\begin{tabular}{|l|c|}")
print("\\hline")
print("\\textbf{Method} & \\textbf{Practical Accuracy (\\%)} \\\\")
print("\\hline")
for _, row in df_practical_summary.iterrows():
    print(f"{row['Method']} & {row['Practical Accuracy (%)']:.2f} \\\\")
print("\\hline")
print("\\end{tabular}")
print("\\end{table}")

# =================================================================================
# SUMMARY REPORT
# =================================================================================
print("\n" + "=" * 70)
print("SECTION 5: SUMMARY REPORT")
print("=" * 70)

print("\nBEST METHOD (Theoretical):")
best_theoretical = df_theoretical.iloc[0]
print(f"  📌 {best_theoretical['Method']}")
print(f"  📊 Accuracy: {best_theoretical['Accuracy (%)']:.2f}%")
print(f"  ⏱️ Train Time: {best_theoretical['Train Time (s)']:.3f}s")
print(f"  ⏱️ Predict Time: {best_theoretical['Predict Time (s)']:.3f}s")

print("\nBEST METHOD (Practical):")
best_practical = df_practical_summary.iloc[0]
print(f"  📌 {best_practical['Method']}")
print(f"  📊 Accuracy: {best_practical['Practical Accuracy (%)']:.2f}%")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nOUTPUT FILES:")
print("  1. benchmark_theoretical.csv     - Theoretical results table")
print("  2. benchmark_practical.csv       - Detailed practical test results")
print("  3. benchmark_practical_summary.csv - Practical accuracy per method")
print("  4. benchmark_full_chart.png      - Comparison charts")
print("\n✅ All files ready for your report!")