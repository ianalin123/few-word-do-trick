"""
EEG Emotion Recognition - PCA Visualization (Phase 3)
Visualizes feature separability using PCA and other methods.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'

# Color scheme for emotions (supports both old and new labels)
EMOTION_COLORS = {
    # New labels
    'irritation': '#FF6B6B',    # Red (irritation/anger)
    'happy': '#FFD93D',         # Yellow (happy/comfort)
    'sadness': '#4A90E2',       # Blue (sadness)
    'neutral': '#95E1D3',       # Light green (calm/neutral)
    'calm': '#95E1D3',          # Light green (calm/neutral)
    'disgusted': '#9B59B6',     # Purple (disgust)
    # Old labels (for backwards compatibility)
    'lemon': '#FF6B6B',
    'blanket': '#FFD93D',
    'coldpool': '#4A90E2',
    'windynight': '#95E1D3'
}


class EEGVisualizer:
    """Handles visualization of EEG features."""

    def __init__(self, features_file='features.csv'):
        """
        Initialize visualizer.

        Args:
            features_file: Path to features CSV file
        """
        self.features_file = features_file
        self.df = None
        self.X = None
        self.y = None
        self.X_scaled = None
        self.pca = None
        self.X_pca = None

    def load_features(self):
        """Load features from CSV file."""
        if not os.path.exists(self.features_file):
            print(f"✗ Features file '{self.features_file}' not found!")
            print("  Please run 'python preprocess.py' first.")
            return False

        self.df = pd.read_csv(self.features_file)

        # Separate features and labels
        self.X = self.df.drop('emotion', axis=1).values
        self.y = self.df['emotion'].values

        print(f"✓ Loaded features: {self.X.shape[0]} samples × {self.X.shape[1]} features")
        print(f"  Emotions: {np.unique(self.y)}")

        return True

    def perform_pca(self, n_components=3):
        """
        Perform PCA on features.

        Args:
            n_components: Number of principal components

        Returns:
            Transformed data
        """
        # Standardize features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        # Apply PCA
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)

        # Print explained variance
        print("\n" + "="*60)
        print("PCA RESULTS")
        print("="*60)
        print(f"Explained variance ratio:")
        for i, var in enumerate(self.pca.explained_variance_ratio_, 1):
            print(f"  PC{i}: {var:.4f} ({var*100:.2f}%)")

        cumulative_var = np.cumsum(self.pca.explained_variance_ratio_)
        print(f"\nCumulative variance explained:")
        for i, cum_var in enumerate(cumulative_var, 1):
            print(f"  PC1-PC{i}: {cum_var:.4f} ({cum_var*100:.2f}%)")

        return self.X_pca

    def plot_pca_2d(self, save_path='plots/pca_2d.png'):
        """
        Create 2D PCA scatter plot.

        Args:
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot each emotion
        emotions = np.unique(self.y)
        for emotion in emotions:
            mask = self.y == emotion
            ax.scatter(
                self.X_pca[mask, 0],
                self.X_pca[mask, 1],
                c=EMOTION_COLORS.get(emotion, 'gray'),
                label=emotion,
                alpha=0.8,
                s=80,
                edgecolors='white',
                linewidth=0.3
            )

        # Labels and title
        var1 = self.pca.explained_variance_ratio_[0] * 100
        var2 = self.pca.explained_variance_ratio_[1] * 100

        ax.set_xlabel(f'PC1 ({var1:.2f}% variance)', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.2f}% variance)', fontsize=12, fontweight='bold')
        ax.set_title('PCA: Emotion Clustering', fontsize=14, fontweight='bold', pad=20)

        ax.legend(title='Emotion', fontsize=10, title_fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_pca_3d(self, save_path='plots/pca_3d.png'):
        """
        Create 3D PCA scatter plot.

        Args:
            save_path: Path to save plot
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each emotion
        emotions = np.unique(self.y)
        for emotion in emotions:
            mask = self.y == emotion
            ax.scatter(
                self.X_pca[mask, 0],
                self.X_pca[mask, 1],
                self.X_pca[mask, 2],
                c=EMOTION_COLORS.get(emotion, 'gray'),
                label=emotion,
                alpha=0.8,
                s=80,
                edgecolors='white',
                linewidth=0.3
            )

        # Labels and title
        var1 = self.pca.explained_variance_ratio_[0] * 100
        var2 = self.pca.explained_variance_ratio_[1] * 100
        var3 = self.pca.explained_variance_ratio_[2] * 100

        ax.set_xlabel(f'PC1 ({var1:.2f}%)', fontsize=10, fontweight='bold')
        ax.set_ylabel(f'PC2 ({var2:.2f}%)', fontsize=10, fontweight='bold')
        ax.set_zlabel(f'PC3 ({var3:.2f}%)', fontsize=10, fontweight='bold')
        ax.set_title('PCA: 3D Emotion Clustering', fontsize=14, fontweight='bold', pad=20)

        ax.legend(title='Emotion', fontsize=9, title_fontsize=10, framealpha=0.9)

        # Set viewing angle for better separation
        ax.view_init(elev=20, azim=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_pca_2d_interactive(self, save_path='plots/pca_2d_interactive.html'):
        """
        Create interactive 2D PCA scatter plot using Plotly.

        Args:
            save_path: Path to save interactive HTML plot
        """
        # Prepare data
        emotions = np.unique(self.y)
        var1 = self.pca.explained_variance_ratio_[0] * 100
        var2 = self.pca.explained_variance_ratio_[1] * 100

        fig = go.Figure()

        # Plot each emotion
        for emotion in emotions:
            mask = self.y == emotion
            fig.add_trace(go.Scatter(
                x=self.X_pca[mask, 0],
                y=self.X_pca[mask, 1],
                mode='markers',
                name=emotion,
                marker=dict(
                    size=10,
                    color=EMOTION_COLORS.get(emotion, 'gray'),
                    opacity=0.8,
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate=(
                    f'<b>{emotion}</b><br>' +
                    'PC1: %{x:.3f}<br>' +
                    'PC2: %{y:.3f}<br>' +
                    '<extra></extra>'
                )
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive PCA: Emotion Clustering',
                font=dict(size=18, color='#2c3e50', family='Arial Black')
            ),
            xaxis_title=f'PC1 ({var1:.2f}% variance)',
            yaxis_title=f'PC2 ({var2:.2f}% variance)',
            font=dict(size=12),
            hovermode='closest',
            plot_bgcolor='white',
            width=1000,
            height=700,
            legend=dict(
                title=dict(text='Emotion', font=dict(size=14)),
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1
            )
        )

        # Add grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='gray'
        )

        # Save
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")

    def plot_pca_3d_interactive(self, save_path='plots/pca_3d_interactive.html'):
        """
        Create interactive 3D PCA scatter plot using Plotly.

        Args:
            save_path: Path to save interactive HTML plot
        """
        # Prepare data
        emotions = np.unique(self.y)
        var1 = self.pca.explained_variance_ratio_[0] * 100
        var2 = self.pca.explained_variance_ratio_[1] * 100
        var3 = self.pca.explained_variance_ratio_[2] * 100

        fig = go.Figure()

        # Plot each emotion
        for emotion in emotions:
            mask = self.y == emotion
            fig.add_trace(go.Scatter3d(
                x=self.X_pca[mask, 0],
                y=self.X_pca[mask, 1],
                z=self.X_pca[mask, 2],
                mode='markers',
                name=emotion,
                marker=dict(
                    size=6,
                    color=EMOTION_COLORS.get(emotion, 'gray'),
                    opacity=0.8,
                    line=dict(color='white', width=0.5)
                ),
                hovertemplate=(
                    f'<b>{emotion}</b><br>' +
                    'PC1: %{x:.3f}<br>' +
                    'PC2: %{y:.3f}<br>' +
                    'PC3: %{z:.3f}<br>' +
                    '<extra></extra>'
                )
            ))

        # Update layout
        fig.update_layout(
            title=dict(
                text='Interactive 3D PCA: Emotion Clustering',
                font=dict(size=18, color='#2c3e50', family='Arial Black')
            ),
            scene=dict(
                xaxis=dict(
                    title=f'PC1 ({var1:.2f}%)',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True
                ),
                yaxis=dict(
                    title=f'PC2 ({var2:.2f}%)',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True
                ),
                zaxis=dict(
                    title=f'PC3 ({var3:.2f}%)',
                    backgroundcolor='white',
                    gridcolor='lightgray',
                    showbackground=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            width=1100,
            height=800,
            font=dict(size=12),
            hovermode='closest',
            legend=dict(
                title=dict(text='Emotion', font=dict(size=14)),
                font=dict(size=12),
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='#ccc',
                borderwidth=1,
                x=0.02,
                y=0.98
            )
        )

        # Save
        fig.write_html(save_path)
        print(f"✓ Saved: {save_path}")

    def plot_feature_correlation(self, save_path='plots/feature_correlation.png'):
        """
        Create feature correlation heatmap.

        Args:
            save_path: Path to save plot
        """
        # Compute correlation matrix
        feature_names = self.df.drop('emotion', axis=1).columns
        corr_matrix = np.corrcoef(self.X_scaled.T)

        # Select top features by variance for readability
        feature_vars = np.var(self.X_scaled, axis=0)
        top_features_idx = np.argsort(feature_vars)[-20:]  # Top 20 features
        top_feature_names = [feature_names[i] for i in top_features_idx]
        corr_subset = corr_matrix[np.ix_(top_features_idx, top_features_idx)]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))

        sns.heatmap(
            corr_subset,
            xticklabels=top_feature_names,
            yticklabels=top_feature_names,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            ax=ax
        )

        ax.set_title('Feature Correlation Heatmap (Top 20 Features by Variance)',
                     fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_feature_distributions(self, save_path='plots/feature_distributions.png'):
        """
        Plot distributions of top features across emotions.

        Args:
            save_path: Path to save plot
        """
        # Select top 6 features by variance
        feature_names = self.df.drop('emotion', axis=1).columns
        feature_vars = np.var(self.X_scaled, axis=0)
        top_features_idx = np.argsort(feature_vars)[-6:]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, feat_idx in enumerate(top_features_idx):
            ax = axes[i]
            feature_name = feature_names[feat_idx]

            # Plot distribution for each emotion
            emotions = np.unique(self.y)
            for emotion in emotions:
                mask = self.y == emotion
                values = self.X_scaled[mask, feat_idx]

                ax.hist(
                    values,
                    bins=20,
                    alpha=0.6,
                    label=emotion,
                    color=EMOTION_COLORS.get(emotion, 'gray'),
                    edgecolor='white',
                    linewidth=0.3
                )

            ax.set_xlabel('Standardized Value', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_title(feature_name, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle('Feature Distributions by Emotion (Top 6 Features)',
                     fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def plot_emotion_statistics(self, save_path='plots/emotion_statistics.png'):
        """
        Plot basic statistics per emotion.

        Args:
            save_path: Path to save plot
        """
        emotions = np.unique(self.y)

        # Compute mean feature values per emotion
        mean_features = []
        for emotion in emotions:
            mask = self.y == emotion
            mean_features.append(np.mean(self.X_scaled[mask], axis=0))

        mean_features = np.array(mean_features)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(14, 5))

        # Select top 30 features for visualization
        feature_names = self.df.drop('emotion', axis=1).columns
        feature_vars = np.var(self.X_scaled, axis=0)
        top_features_idx = np.argsort(feature_vars)[-30:]
        top_feature_names = [feature_names[i] for i in top_features_idx]

        sns.heatmap(
            mean_features[:, top_features_idx].T,
            xticklabels=emotions,
            yticklabels=top_feature_names,
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Mean Standardized Value', 'shrink': 0.8},
            linewidths=0.5,
            ax=ax
        )

        ax.set_title('Mean Feature Values per Emotion (Top 30 Features)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
        ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
        plt.xticks(rotation=0, fontsize=10)
        plt.yticks(fontsize=7)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()

    def analyze_separability(self):
        """
        Analyze and report on emotion separability.

        Returns:
            Dictionary with separability metrics
        """
        print("\n" + "="*60)
        print("SEPARABILITY ANALYSIS")
        print("="*60)

        emotions = np.unique(self.y)

        # Compute pairwise distances between emotion centroids in PCA space
        centroids = []
        for emotion in emotions:
            mask = self.y == emotion
            centroid = np.mean(self.X_pca[mask, :2], axis=0)  # Use first 2 PCs
            centroids.append(centroid)

        centroids = np.array(centroids)

        print("\nPairwise centroid distances (PC1-PC2 space):")
        for i, emotion1 in enumerate(emotions):
            for j, emotion2 in enumerate(emotions):
                if i < j:
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    print(f"  {emotion1:12s} <-> {emotion2:12s}: {dist:.3f}")

        # Compute within-class vs between-class scatter
        within_class_scatter = 0
        for emotion in emotions:
            mask = self.y == emotion
            centroid = np.mean(self.X_pca[mask, :2], axis=0)
            distances = np.linalg.norm(self.X_pca[mask, :2] - centroid, axis=1)
            within_class_scatter += np.mean(distances)

        within_class_scatter /= len(emotions)

        between_class_scatter = 0
        overall_centroid = np.mean(centroids, axis=0)
        for centroid in centroids:
            between_class_scatter += np.linalg.norm(centroid - overall_centroid)

        between_class_scatter /= len(emotions)

        print(f"\nWithin-class scatter: {within_class_scatter:.3f}")
        print(f"Between-class scatter: {between_class_scatter:.3f}")
        print(f"Separability ratio: {between_class_scatter / within_class_scatter:.3f}")

        if between_class_scatter / within_class_scatter > 1.0:
            print("\n✓ GOOD: Between-class scatter > within-class scatter")
            print("  Emotions show some separability!")
        else:
            print("\n⚠ WARNING: Within-class scatter > between-class scatter")
            print("  Emotions may be difficult to distinguish.")
            print("  Consider: more data, better emotion induction, or different features")

        return {
            'within_class_scatter': within_class_scatter,
            'between_class_scatter': between_class_scatter,
            'separability_ratio': between_class_scatter / within_class_scatter
        }


def main():
    """Main visualization workflow."""
    print("="*60)
    print("EEG EMOTION RECOGNITION - PCA VISUALIZATION")
    print("="*60)

    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Initialize visualizer
    visualizer = EEGVisualizer()

    # Load features
    if not visualizer.load_features():
        return

    # Perform PCA
    visualizer.perform_pca(n_components=3)

    # Generate all visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)

    # Static plots
    visualizer.plot_pca_2d()
    visualizer.plot_pca_3d()
    visualizer.plot_feature_correlation()
    visualizer.plot_feature_distributions()
    visualizer.plot_emotion_statistics()

    # Interactive plots
    print("\nGenerating interactive plots...")
    visualizer.plot_pca_2d_interactive()
    visualizer.plot_pca_3d_interactive()

    # Analyze separability
    metrics = visualizer.analyze_separability()

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print("Generated plots:")
    print("\nStatic plots:")
    print("  - plots/pca_2d.png                  (2D PCA scatter)")
    print("  - plots/pca_3d.png                  (3D PCA scatter)")
    print("  - plots/feature_correlation.png     (correlation heatmap)")
    print("  - plots/feature_distributions.png   (feature histograms)")
    print("  - plots/emotion_statistics.png      (mean values heatmap)")
    print("\nInteractive plots:")
    print("  - plots/pca_2d_interactive.html     (Interactive 2D PCA)")
    print("  - plots/pca_3d_interactive.html     (Interactive 3D PCA)")
    print("\n  Open the .html files in your browser for interactive exploration!")

    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)

    if metrics['separability_ratio'] > 1.0:
        print("✓ Emotions show separability in PCA space!")
        print("\nRecommended next steps:")
        print("  1. Train a classifier (e.g., RandomForest, SVM)")
        print("  2. Perform cross-validation to assess accuracy")
        print("  3. Test with new data")
    else:
        print("⚠ Low separability detected")
        print("\nRecommended next steps:")
        print("  1. Collect more trials (aim for 5-10 per emotion)")
        print("  2. Try stronger emotion induction methods")
        print("  3. Check data quality (headset placement, artifacts)")
        print("  4. Consider additional features or preprocessing")

    print("="*60)


if __name__ == "__main__":
    main()
