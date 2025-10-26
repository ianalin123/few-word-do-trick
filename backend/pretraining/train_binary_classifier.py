"""
Binary Emotion Classification Training Script (FIXED)
Trains a RandomForest classifier to distinguish between happy and sad emotions.
Uses GroupKFold with actual trial_id column to prevent data leakage.
"""

import os
import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import warnings

warnings.filterwarnings('ignore')


class BinaryEmotionClassifier:
    """Binary emotion classifier with proper cross-validation."""

    def __init__(self, features_file='features.csv'):
        """
        Initialize classifier.

        Args:
            features_file: Path to features CSV file
        """
        self.features_file = features_file
        self.df = None
        self.X = None
        self.y = None
        self.trial_groups = None
        self.scaler = None
        self.model = None
        self.label_map = None

    def load_and_prepare_data(self):
        """Load features and prepare for binary classification."""
        print("="*70)
        print("LOADING DATA")
        print("="*70)

        if not os.path.exists(self.features_file):
            print(f"✗ Features file '{self.features_file}' not found!")
            return False

        # Load data
        self.df = pd.read_csv(self.features_file)
        print(f"✓ Loaded {len(self.df)} total samples")
        print(f"  Emotions found: {sorted(self.df['emotion'].unique())}")

        # Check if trial_id column exists
        if 'trial_id' not in self.df.columns:
            print(f"\n✗ ERROR: 'trial_id' column not found in features.csv!")
            print(f"  Please run the FIXED preprocessing script (preprocess_FIXED.py)")
            print(f"  The old preprocessing script doesn't track trial IDs.")
            return False

        # Filter to only happy and sadness
        binary_df = self.df[self.df['emotion'].isin(['happy', 'sadness'])].copy()
        print(f"\n✓ Filtered to binary classification:")
        print(f"  Total samples: {len(binary_df)}")

        # Count samples per emotion
        emotion_counts = binary_df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"  - {emotion}: {count} samples")

        # Separate features and labels
        feature_cols = [c for c in binary_df.columns if c not in ['emotion', 'trial_id']]
        self.X = binary_df[feature_cols].values
        self.y = binary_df['emotion'].values
        
        # Extract trial groups (CRITICAL: this is the actual trial ID)
        self.trial_groups = binary_df['trial_id'].values

        # Create label mapping (0 = happy, 1 = sadness)
        self.label_map = {'happy': 0, 'sadness': 1}
        self.y_encoded = np.array([self.label_map[label] for label in self.y])

        print(f"\n✓ Prepared features:")
        print(f"  X shape: {self.X.shape}")
        print(f"  Label mapping: {self.label_map}")
        
        # Show trial info
        print(f"\n✓ Trial information:")
        unique_trials = np.unique(self.trial_groups)
        print(f"  Total unique trials: {len(unique_trials)}")
        
        # Count trials per emotion
        for emotion in ['happy', 'sadness']:
            emotion_mask = self.y == emotion
            emotion_trials = np.unique(self.trial_groups[emotion_mask])
            print(f"  {emotion} trials: {len(emotion_trials)}")
            # Show trial IDs
            print(f"    {', '.join(emotion_trials[:10])}", end="")
            if len(emotion_trials) > 10:
                print(f" ... ({len(emotion_trials)} total)")
            else:
                print()

        return True

    def train_with_group_kfold(self, n_splits=5):
        """
        Train model using GroupKFold cross-validation.
        This prevents data leakage by keeping all windows from the same trial together.

        Args:
            n_splits: Number of folds for cross-validation
        """
        print("\n" + "="*70)
        print("TRAINING WITH GROUP K-FOLD CROSS-VALIDATION")
        print("="*70)

        # Adjust n_splits if we have fewer trials than requested folds
        n_trials = len(np.unique(self.trial_groups))
        if n_trials < n_splits:
            print(f"⚠ WARNING: Only {n_trials} trials available!")
            print(f"  Adjusting n_splits from {n_splits} to {n_trials}")
            print(f"  For better cross-validation, collect more trials!\n")
            n_splits = n_trials
        else:
            print(f"Using {n_splits}-fold cross-validation on {n_trials} trials")

        print("\nCRITICAL: All windows from the same trial stay together")
        print("This prevents data leakage from overlapping windows!\n")

        # Initialize GroupKFold
        gkf = GroupKFold(n_splits=n_splits)

        # Store metrics for each fold
        fold_metrics = {
            'accuracy': [],
            'precision_happy': [],
            'recall_happy': [],
            'f1_happy': [],
            'precision_sadness': [],
            'recall_sadness': [],
            'f1_sadness': []
        }

        # Iterate through folds
        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(self.X, self.y_encoded, self.trial_groups), 1):
            print(f"\n{'─'*70}")
            print(f"FOLD {fold_idx}/{n_splits}")
            print(f"{'─'*70}")

            # Split data
            X_train, X_test = self.X[train_idx], self.X[test_idx]
            y_train, y_test = self.y_encoded[train_idx], self.y_encoded[test_idx]

            # Get trial information
            train_trials = np.unique(self.trial_groups[train_idx])
            test_trials = np.unique(self.trial_groups[test_idx])

            print(f"Train: {len(train_idx)} samples from {len(train_trials)} trials")
            print(f"  Trials: {', '.join(sorted(train_trials)[:5])}", end="")
            if len(train_trials) > 5:
                print(f" ... ({len(train_trials)} total)")
            else:
                print()
                
            print(f"Test:  {len(test_idx)} samples from {len(test_trials)} trials")
            print(f"  Trials: {', '.join(sorted(test_trials)[:5])}", end="")
            if len(test_trials) > 5:
                print(f" ... ({len(test_trials)} total)")
            else:
                print()

            # Verify no trial overlap (sanity check)
            overlap = set(train_trials) & set(test_trials)
            if overlap:
                print(f"⚠ WARNING: Trial overlap detected: {overlap}")
            else:
                print("✓ No trial overlap (good!)")

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)

            # Predict
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average=None, labels=[0, 1], zero_division=0
            )

            # Store metrics
            fold_metrics['accuracy'].append(accuracy)
            fold_metrics['precision_happy'].append(precision[0])
            fold_metrics['recall_happy'].append(recall[0])
            fold_metrics['f1_happy'].append(f1[0])
            fold_metrics['precision_sadness'].append(precision[1])
            fold_metrics['recall_sadness'].append(recall[1])
            fold_metrics['f1_sadness'].append(f1[1])

            # Print fold results
            print(f"\nFold {fold_idx} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"\n  Happy (class 0):")
            print(f"    Precision: {precision[0]:.4f}")
            print(f"    Recall:    {recall[0]:.4f}")
            print(f"    F1-Score:  {f1[0]:.4f}")
            print(f"\n  Sadness (class 1):")
            print(f"    Precision: {precision[1]:.4f}")
            print(f"    Recall:    {recall[1]:.4f}")
            print(f"    F1-Score:  {f1[1]:.4f}")

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            print(f"\n  Confusion Matrix:")
            print(f"                 Predicted")
            print(f"                 Happy  Sadness")
            print(f"    Actual Happy   {cm[0,0]:4d}   {cm[0,1]:4d}")
            print(f"    Actual Sad     {cm[1,0]:4d}   {cm[1,1]:4d}")

        # Print average metrics
        print(f"\n{'='*70}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"\nAverage Accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
        print(f"\nHappy (class 0):")
        print(f"  Precision: {np.mean(fold_metrics['precision_happy']):.4f} ± {np.std(fold_metrics['precision_happy']):.4f}")
        print(f"  Recall:    {np.mean(fold_metrics['recall_happy']):.4f} ± {np.std(fold_metrics['recall_happy']):.4f}")
        print(f"  F1-Score:  {np.mean(fold_metrics['f1_happy']):.4f} ± {np.std(fold_metrics['f1_happy']):.4f}")
        print(f"\nSadness (class 1):")
        print(f"  Precision: {np.mean(fold_metrics['precision_sadness']):.4f} ± {np.std(fold_metrics['precision_sadness']):.4f}")
        print(f"  Recall:    {np.mean(fold_metrics['recall_sadness']):.4f} ± {np.std(fold_metrics['recall_sadness']):.4f}")
        print(f"  F1-Score:  {np.mean(fold_metrics['f1_sadness']):.4f} ± {np.std(fold_metrics['f1_sadness']):.4f}")

        return fold_metrics

    def train_final_model(self, use_calibration=True):
        """
        Train final model on all data.

        Args:
            use_calibration: Whether to use probability calibration
        """
        print("\n" + "="*70)
        print("TRAINING FINAL MODEL ON ALL DATA")
        print("="*70)

        # Standardize all features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(self.X)

        # Train base model
        print("\nTraining RandomForestClassifier...")
        base_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        base_model.fit(X_scaled, self.y_encoded)
        print("✓ Base model trained")

        # Apply calibration
        if use_calibration:
            print("\nApplying probability calibration...")
            print("This fixes overconfident predictions!")
            self.model = CalibratedClassifierCV(
                base_model,
                method='sigmoid',
                cv=5
            )
            self.model.fit(X_scaled, self.y_encoded)
            print("✓ Calibrated model ready")
        else:
            self.model = base_model

        # Evaluate on training data (just for reference)
        y_pred = self.model.predict(X_scaled)
        train_accuracy = accuracy_score(self.y_encoded, y_pred)
        print(f"\nTraining accuracy: {train_accuracy:.4f}")
        print("(Note: Real performance is from cross-validation above)")

        return True

    def save_model(self, model_path='emotion_model_binary.joblib', label_map_path='label_map.json'):
        """
        Save trained model and label mapping.

        Args:
            model_path: Path to save model
            label_map_path: Path to save label mapping
        """
        print("\n" + "="*70)
        print("SAVING MODEL")
        print("="*70)

        # Save model
        feature_cols = [c for c in self.df.columns if c not in ['emotion', 'trial_id']]
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_map': self.label_map,
            'feature_names': feature_cols
        }
        joblib.dump(model_data, model_path)
        print(f"✓ Model saved to: {model_path}")

        # Save label map separately (for easy reading)
        with open(label_map_path, 'w') as f:
            json.dump(self.label_map, f, indent=2)
        print(f"✓ Label map saved to: {label_map_path}")

        # Print model info
        print(f"\nModel Information:")
        print(f"  Type: RandomForest + Calibration")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Classes: {list(self.label_map.keys())}")

        return True

    def print_classification_report(self):
        """Print detailed classification report."""
        print("\n" + "="*70)
        print("DETAILED CLASSIFICATION REPORT (on all data)")
        print("="*70)

        # Predict on all data
        X_scaled = self.scaler.transform(self.X)
        y_pred = self.model.predict(X_scaled)

        # Get label names for report
        inverse_label_map = {v: k for k, v in self.label_map.items()}
        target_names = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]

        # Print report
        report = classification_report(
            self.y_encoded,
            y_pred,
            target_names=target_names,
            digits=4
        )
        print(report)

        # Confusion matrix
        cm = confusion_matrix(self.y_encoded, y_pred)
        print("Confusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Happy  Sadness")
        print(f"  Actual Happy   {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"  Actual Sad     {cm[1,0]:4d}   {cm[1,1]:4d}")

        return report


def main():
    """Main training workflow."""
    print("="*70)
    print("BINARY EMOTION CLASSIFICATION TRAINING (FIXED)")
    print("Happy vs Sadness - With Proper Trial Tracking")
    print("="*70)

    # Initialize classifier
    classifier = BinaryEmotionClassifier()

    # Load and prepare data
    if not classifier.load_and_prepare_data():
        print("\n✗ Failed to load data. Exiting.")
        return

    # Train with GroupKFold cross-validation
    # This gives us HONEST metrics without data leakage
    fold_metrics = classifier.train_with_group_kfold(n_splits=5)

    # Train final model on all data
    classifier.train_final_model(use_calibration=True)

    # Print detailed classification report
    classifier.print_classification_report()

    # Save model
    classifier.save_model()

    # Final summary
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print(f"  ✓ Used GroupKFold with actual trial_id column")
    print(f"  ✓ Cross-validation accuracy: {np.mean(fold_metrics['accuracy']):.4f} ± {np.std(fold_metrics['accuracy']):.4f}")
    print(f"  ✓ Applied probability calibration")
    print(f"  ✓ Model saved as 'emotion_model_binary.joblib'")
    print("\nThis is your HONEST expected performance on new trials!")
    print("\nNext steps:")
    print("  1. Use this model for real-time prediction")
    print("  2. If accuracy is low, collect more diverse trials")
    print("  3. Try feature engineering or different classifiers")
    print("="*70)


if __name__ == "__main__":
    main()