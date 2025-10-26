#!/usr/bin/env python3
"""
Comprehensive EEG Emotion Classification Analysis
Analyzes classification performance, feature importance, pairwise separability,
and arousal-valence space for EEG emotion data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import warnings
warnings.filterwarnings('ignore')

# For XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except (ImportError, Exception) as e:
    HAS_XGBOOST = False
    print(f"Warning: XGBoost not available. Continuing without it.")

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("EEG EMOTION CLASSIFICATION ANALYSIS")
print("="*80)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("\n[1] Loading and exploring data...")
df = pd.read_csv('features.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nEmotion distribution:")
emotion_counts = df['emotion'].value_counts()
print(emotion_counts)
print(f"\nEmotion percentages:")
print(df['emotion'].value_counts(normalize=True) * 100)

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Prepare features and target
X = df.drop('emotion', axis=1)
y = df['emotion']
feature_names = X.columns.tolist()

print(f"\nNumber of features: {len(feature_names)}")
print(f"Number of samples: {len(df)}")

# ============================================================================
# 2. CLASSIFICATION PERFORMANCE - MULTI-CLASS
# ============================================================================
print("\n" + "="*80)
print("[2] MULTI-CLASS CLASSIFICATION PERFORMANCE")
print("="*80)

# Initialize models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=200, max_depth=15,
                                          min_samples_split=10, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42)
}

if HAS_XGBOOST:
    models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6,
                                     learning_rate=0.1, random_state=42,
                                     eval_metric='mlogloss')

# Use StratifiedKFold to handle class imbalance
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Scale features for models that need it
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
predictions = {}

for name, model in models.items():
    print(f"\n{'-'*60}")
    print(f"Training {name}...")

    # Use scaled data for SVM and Logistic Regression
    if name in ['SVM (RBF)', 'Logistic Regression']:
        y_pred = cross_val_predict(model, X_scaled, y, cv=skf)
    else:
        y_pred = cross_val_predict(model, X, y, cv=skf)

    predictions[name] = y_pred

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    results[name] = accuracy

    print(f"Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred, labels=['happy', 'sad', 'calm'])
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              happy    sad      calm")
    for i, emotion in enumerate(['happy', 'sad', 'calm']):
        print(f"Actual {emotion:6s}  {cm[i,0]:5d}    {cm[i,1]:5d}    {cm[i,2]:5d}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=['happy', 'sad', 'calm'], zero_division=0
    )

    print(f"\nPer-Class Metrics:")
    print(f"{'Emotion':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 56)
    for i, emotion in enumerate(['happy', 'sad', 'calm']):
        print(f"{emotion:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")

print(f"\n{'='*60}")
print("SUMMARY - Model Comparison:")
print("="*60)
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name:<25} Accuracy: {acc:.4f}")

# Save best model predictions for plotting
best_model_name = max(results, key=results.get)
best_predictions = predictions[best_model_name]

# ============================================================================
# 3. FEATURE IMPORTANCE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[3] FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=200, max_depth=15,
                            min_samples_split=10, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 Most Important Features:")
print("-" * 60)
print(f"{'Rank':<6} {'Feature':<45} {'Importance':<12}")
print("-" * 60)
for i, row in feature_importance.head(15).iterrows():
    print(f"{feature_importance.index.get_loc(i)+1:<6} {row['feature']:<45} {row['importance']:<12.6f}")

# Analyze by feature type
print("\n" + "-"*60)
print("Feature Importance by Type:")
print("-" * 60)

# Categorize features
power_features = [f for f in feature_names if 'power_' in f]
de_features = [f for f in feature_names if 'de_' in f]
asymmetry_features = [f for f in feature_names if 'asymmetry' in f]
ratio_features = [f for f in feature_names if 'ratio' in f]

# Calculate average importance by type
power_importance = feature_importance[feature_importance['feature'].isin(power_features)]['importance'].mean()
de_importance = feature_importance[feature_importance['feature'].isin(de_features)]['importance'].mean()
asymmetry_importance = feature_importance[feature_importance['feature'].isin(asymmetry_features)]['importance'].mean()
ratio_importance = feature_importance[feature_importance['feature'].isin(ratio_features)]['importance'].mean()

print(f"Band Powers (avg):        {power_importance:.6f}")
print(f"Differential Entropy:     {de_importance:.6f}")
print(f"Asymmetry:                {asymmetry_importance:.6f}")
print(f"Ratios:                   {ratio_importance:.6f}")

# Identify low-importance features
threshold = 0.005
low_importance = feature_importance[feature_importance['importance'] < threshold]
print(f"\n{len(low_importance)} features have importance < {threshold}:")
print(low_importance['feature'].tolist())

# ============================================================================
# 4. PAIRWISE BINARY CLASSIFICATION
# ============================================================================
print("\n" + "="*80)
print("[4] PAIRWISE BINARY CLASSIFICATION")
print("="*80)

emotion_pairs = [
    ('happy', 'sad'),
    ('happy', 'calm'),
    ('sad', 'calm')
]

pairwise_results = {}

for emo1, emo2 in emotion_pairs:
    print(f"\n{'-'*60}")
    print(f"Binary Classification: {emo1.upper()} vs {emo2.upper()}")
    print("-"*60)

    # Filter data for this pair
    mask = df['emotion'].isin([emo1, emo2])
    X_pair = X[mask]
    y_pair = y[mask]

    print(f"Samples: {emo1}={sum(y_pair==emo1)}, {emo2}={sum(y_pair==emo2)}")

    # Test with multiple classifiers
    pair_models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'LogisticReg': LogisticRegression(max_iter=1000, random_state=42)
    }

    if HAS_XGBOOST:
        pair_models['XGBoost'] = XGBClassifier(n_estimators=200, random_state=42,
                                              eval_metric='logloss')

    pair_accuracies = {}

    for name, model in pair_models.items():
        skf_pair = StratifiedKFold(n_splits=min(5, min(sum(y_pair==emo1), sum(y_pair==emo2))),
                                   shuffle=True, random_state=42)

        if name in ['SVM', 'LogisticReg']:
            X_pair_scaled = scaler.fit_transform(X_pair)
            y_pred = cross_val_predict(model, X_pair_scaled, y_pair, cv=skf_pair)
        else:
            y_pred = cross_val_predict(model, X_pair, y_pair, cv=skf_pair)

        acc = accuracy_score(y_pair, y_pred)
        pair_accuracies[name] = acc

    pairwise_results[f"{emo1}_vs_{emo2}"] = pair_accuracies

    print(f"\nResults:")
    for name, acc in sorted(pair_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<15} Accuracy: {acc:.4f}")

print(f"\n{'='*60}")
print("PAIRWISE CLASSIFICATION SUMMARY:")
print("="*60)
for pair_name, accuracies in pairwise_results.items():
    best_acc = max(accuracies.values())
    worst_acc = min(accuracies.values())
    avg_acc = np.mean(list(accuracies.values()))
    print(f"{pair_name:<20} Best: {best_acc:.4f}  Avg: {avg_acc:.4f}  Worst: {worst_acc:.4f}")

# ============================================================================
# 5. AROUSAL-VALENCE SPACE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[5] AROUSAL-VALENCE SPACE ANALYSIS")
print("="*80)

# Extract band power and asymmetry features
alpha_cols = [c for c in X.columns if 'power_alpha' in c]
beta_cols = [c for c in X.columns if 'power_beta' in c]
gamma_cols = [c for c in X.columns if 'power_gamma' in c]

# Calculate arousal: high beta/gamma, low alpha
arousal = (X[beta_cols].mean(axis=1) + X[gamma_cols].mean(axis=1) - X[alpha_cols].mean(axis=1))

# Valence: frontal alpha asymmetry (right - left)
# Positive asymmetry = more right activity = positive valence
if 'frontal_alpha_asymmetry' in X.columns:
    valence = X['frontal_alpha_asymmetry']
else:
    # Calculate from AF7 (left frontal) and AF8 (right frontal)
    if 'AF8_power_alpha' in X.columns and 'AF7_power_alpha' in X.columns:
        valence = X['AF8_power_alpha'] - X['AF7_power_alpha']
    else:
        valence = np.zeros(len(X))

# Normalize
arousal = (arousal - arousal.mean()) / arousal.std()
valence = (valence - valence.mean()) / valence.std()

# Add to dataframe
df_av = df.copy()
df_av['arousal'] = arousal
df_av['valence'] = valence

print("\nArousal-Valence Statistics by Emotion:")
print("-" * 60)
print(f"{'Emotion':<10} {'Arousal (mean±std)':<25} {'Valence (mean±std)':<25}")
print("-" * 60)
for emotion in ['happy', 'sad', 'calm']:
    emo_data = df_av[df_av['emotion'] == emotion]
    arousal_mean = emo_data['arousal'].mean()
    arousal_std = emo_data['arousal'].std()
    valence_mean = emo_data['valence'].mean()
    valence_std = emo_data['valence'].std()
    print(f"{emotion:<10} {arousal_mean:>7.3f} ± {arousal_std:<7.3f}      "
          f"{valence_mean:>7.3f} ± {valence_std:<7.3f}")

# Test classification in arousal-valence space
X_av = df_av[['arousal', 'valence']].values
y_av = df_av['emotion']

rf_av = RandomForestClassifier(n_estimators=200, random_state=42)
y_pred_av = cross_val_predict(rf_av, X_av, y_av, cv=skf)
acc_av = accuracy_score(y_av, y_pred_av)

print(f"\nClassification using only Arousal-Valence features:")
print(f"Accuracy: {acc_av:.4f}")
print(f"Baseline (all 52 features): {results['RandomForest']:.4f}")
print(f"Difference: {results['RandomForest'] - acc_av:.4f}")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("[6] RECOMMENDATIONS")
print("="*80)

# Analyze confusion patterns
best_model = list(models.keys())[0]  # Use RandomForest
cm_best = confusion_matrix(y, predictions[best_model], labels=['happy', 'sad', 'calm'])

# Calculate per-class accuracy
class_accuracies = {}
for i, emotion in enumerate(['happy', 'sad', 'calm']):
    class_acc = cm_best[i, i] / cm_best[i].sum()
    class_accuracies[emotion] = class_acc

print("\nPer-Class Reliability:")
print("-" * 60)
for emotion in sorted(class_accuracies, key=class_accuracies.get, reverse=True):
    acc = class_accuracies[emotion]
    reliability = "HIGHLY RELIABLE" if acc > 0.8 else "RELIABLE" if acc > 0.6 else "MODERATE" if acc > 0.4 else "UNRELIABLE"
    print(f"{emotion.capitalize():<10} {acc:.4f} ({reliability})")

# Calculate confusion pairs
print("\nMost Confused Emotion Pairs:")
print("-" * 60)
confusions = []
for i in range(len(['happy', 'sad', 'calm'])):
    for j in range(len(['happy', 'sad', 'calm'])):
        if i != j:
            confusion_rate = cm_best[i, j] / cm_best[i].sum()
            confusions.append((
                ['happy', 'sad', 'calm'][i],
                ['happy', 'sad', 'calm'][j],
                confusion_rate
            ))

confusions.sort(key=lambda x: x[2], reverse=True)
for emo1, emo2, rate in confusions[:3]:
    print(f"{emo1.capitalize()} → {emo2.capitalize()}: {rate:.4f} ({rate*100:.1f}% of {emo1} samples)")

# Recommendations
print("\n" + "="*60)
print("STRATEGIC RECOMMENDATIONS:")
print("="*60)

# 1. Binary vs 3-class
avg_pairwise = np.mean([max(accs.values()) for accs in pairwise_results.values()])
three_class_acc = results[best_model_name]

print(f"\n1. Binary vs 3-Class Classification:")
print(f"   - 3-class accuracy: {three_class_acc:.4f}")
print(f"   - Average best pairwise accuracy: {avg_pairwise:.4f}")
print(f"   - Improvement with binary: {avg_pairwise - three_class_acc:.4f}")

if three_class_acc >= 0.70:
    print(f"   → RECOMMENDATION: Stick with 3-class (accuracy is good)")
elif avg_pairwise - three_class_acc > 0.15:
    print(f"   → RECOMMENDATION: Consider binary classification (significant improvement)")
else:
    print(f"   → RECOMMENDATION: 3-class is acceptable, but binary may help for specific use cases")

# 2. Adding 4th emotion
print(f"\n2. Adding a 4th Emotion:")
print(f"   For maximum separability, the new emotion should have:")

# Calculate arousal-valence centroids
centroids = {}
for emotion in ['happy', 'sad', 'calm']:
    emo_data = df_av[df_av['emotion'] == emotion]
    centroids[emotion] = (emo_data['arousal'].mean(), emo_data['valence'].mean())

print(f"   Current emotion positions (arousal, valence):")
for emotion, (a, v) in centroids.items():
    print(f"   - {emotion.capitalize()}: ({a:.3f}, {v:.3f})")

# Suggest position for 4th emotion
arousal_vals = [c[0] for c in centroids.values()]
valence_vals = [c[1] for c in centroids.values()]

# Find the quadrant that's least occupied
suggestions = []
if all(a < 0 for a in arousal_vals) or all(a > 0 for a in arousal_vals):
    suggestions.append("HIGH AROUSAL" if all(a < 0 for a in arousal_vals) else "LOW AROUSAL")
if all(v < 0 for v in valence_vals) or all(v > 0 for v in valence_vals):
    suggestions.append("POSITIVE VALENCE" if all(v < 0 for v in valence_vals) else "NEGATIVE VALENCE")

if suggestions:
    print(f"   → SUGGESTION: Add an emotion with {' and '.join(suggestions)}")
    print(f"      Examples: 'excited' (high arousal, positive valence)")
    print(f"                'fear/anxiety' (high arousal, negative valence)")
else:
    print(f"   → Current emotions cover arousal-valence space reasonably well")
    print(f"      Consider: 'excited', 'anxious', 'relaxed', or 'bored'")

# 3. Most/least reliable
print(f"\n3. Emotion Reliability:")
most_reliable = max(class_accuracies, key=class_accuracies.get)
least_reliable = min(class_accuracies, key=class_accuracies.get)
print(f"   - MOST reliable: {most_reliable.capitalize()} ({class_accuracies[most_reliable]:.4f})")
print(f"   - LEAST reliable: {least_reliable.capitalize()} ({class_accuracies[least_reliable]:.4f})")

# 4. Feature reduction
print(f"\n4. Feature Optimization:")
print(f"   - Current features: {len(feature_names)}")
print(f"   - Low importance features (<{threshold}): {len(low_importance)}")
print(f"   → RECOMMENDATION: Can potentially remove {len(low_importance)} features")
print(f"     to reduce complexity with minimal accuracy loss")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

# ============================================================================
# GENERATE VISUALIZATIONS
# ============================================================================
print("\nGenerating visualizations...")

fig = plt.figure(figsize=(20, 12))

# 1. Model Comparison
ax1 = plt.subplot(2, 3, 1)
model_names = list(results.keys())
accuracies = list(results.values())
bars = ax1.barh(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(model_names)])
ax1.set_xlabel('Accuracy', fontsize=11)
ax1.set_title('Multi-Class Classification Performance', fontsize=12, fontweight='bold')
ax1.set_xlim([0, 1])
for i, (name, acc) in enumerate(zip(model_names, accuracies)):
    ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)
ax1.grid(axis='x', alpha=0.3)

# 2. Best Model Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm_normalized = cm_best.astype('float') / cm_best.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=['happy', 'sad', 'calm'],
            yticklabels=['happy', 'sad', 'calm'], ax=ax2,
            cbar_kws={'label': 'Proportion'})
ax2.set_ylabel('True Emotion', fontsize=11)
ax2.set_xlabel('Predicted Emotion', fontsize=11)
ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12, fontweight='bold')

# 3. Top 15 Features
ax3 = plt.subplot(2, 3, 3)
top_features = feature_importance.head(15)
ax3.barh(range(15), top_features['importance'].values[::-1],
         color=plt.cm.viridis(np.linspace(0.3, 0.9, 15)))
ax3.set_yticks(range(15))
ax3.set_yticklabels(top_features['feature'].values[::-1], fontsize=9)
ax3.set_xlabel('Importance', fontsize=11)
ax3.set_title('Top 15 Most Important Features', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# 4. Feature Type Importance
ax4 = plt.subplot(2, 3, 4)
feature_types = ['Band Powers', 'Differential\nEntropy', 'Asymmetry', 'Ratios']
importances = [power_importance, de_importance, asymmetry_importance, ratio_importance]
bars = ax4.bar(feature_types, importances, color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
ax4.set_ylabel('Average Importance', fontsize=11)
ax4.set_title('Feature Importance by Type', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
for i, (ft, imp) in enumerate(zip(feature_types, importances)):
    ax4.text(i, imp + 0.0001, f'{imp:.5f}', ha='center', va='bottom', fontsize=9)

# 5. Pairwise Classification Results
ax5 = plt.subplot(2, 3, 5)
pair_names = list(pairwise_results.keys())
pair_best_accs = [max(accs.values()) for accs in pairwise_results.values()]
bars = ax5.barh(pair_names, pair_best_accs,
                color=['#9b59b6', '#1abc9c', '#e67e22'])
ax5.set_xlabel('Best Accuracy', fontsize=11)
ax5.set_title('Pairwise Binary Classification', fontsize=12, fontweight='bold')
ax5.set_xlim([0, 1])
for i, (name, acc) in enumerate(zip(pair_names, pair_best_accs)):
    ax5.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=10)
ax5.grid(axis='x', alpha=0.3)

# 6. Arousal-Valence Space
ax6 = plt.subplot(2, 3, 6)
colors = {'happy': '#e74c3c', 'sad': '#3498db', 'calm': '#2ecc71'}
for emotion in ['happy', 'sad', 'calm']:
    mask = df_av['emotion'] == emotion
    ax6.scatter(df_av[mask]['valence'], df_av[mask]['arousal'],
                c=colors[emotion], label=emotion.capitalize(), alpha=0.5, s=30)

    # Plot centroid
    centroid = centroids[emotion]
    ax6.scatter(centroid[1], centroid[0], c=colors[emotion],
                marker='*', s=500, edgecolors='black', linewidths=2)

ax6.set_xlabel('Valence (Negative ← → Positive)', fontsize=11)
ax6.set_ylabel('Arousal (Low ← → High)', fontsize=11)
ax6.set_title('Arousal-Valence Space Distribution', fontsize=12, fontweight='bold')
ax6.legend(loc='best')
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax6.axvline(x=0, color='k', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('eeg_emotion_analysis.png', dpi=150, bbox_inches='tight')
print("Visualization saved as 'eeg_emotion_analysis.png'")

print("\n" + "="*80)
print("All analyses complete! Check the output above for detailed results.")
print("="*80)
