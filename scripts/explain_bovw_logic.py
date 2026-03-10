"""
BoVW Explainability Script for Driver Pulse
=============================================
This script generates visual explanations of our motion detection pipeline
for the Punjabi Bagh → Kurukshetra trip (TRIP_DRV003_03).

Outputs:
1. PCA Reorientation Plot - Shows how phone tilt is corrected
2. K-Means Codeword Histogram - The "Visual Fingerprint" of a maneuver
3. Feature Importance Plot - Which codewords trigger safety flags
4. Cluster Visualization - 2D projection of all codewords

Run this AFTER training the model using pipeline1_motion_bovw.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
TARGET_TRIP = "TRIP_DRV003_03"  # Punjabi Bagh → Kurukshetra
SIM_DATA_PATH = "simulation_data/sensor_data/accelerometer_data.csv"
TRAIN_DATA_PATH = "driver_pulse_data/sensor_data/accelerometer_data.csv"
MODEL_PATH = "outputs/models/bovw_motion_models.pkl"
OUTPUT_DIR = "outputs/explainability"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_models():
    """Load trained BoVW models."""
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Train the model first using pipeline1_motion_bovw.py!")
        print(f"   Expected model at: {MODEL_PATH}")
        sys.exit(1)
    
    print(f"Loading models from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    
    print(f"  ✓ K-Means Codebook: {models['codebook'].n_clusters} codewords")
    print(f"  ✓ Random Forest: {models['classifier'].n_estimators} trees")
    print(f"  ✓ Segment length: {models.get('segment_length', 3)}")
    
    return models


def load_data():
    """Load simulation and training data."""
    print(f"\nLoading simulation data for {TARGET_TRIP}...")
    sim_df = pd.read_csv(SIM_DATA_PATH)
    trip_df = sim_df[sim_df['trip_id'] == TARGET_TRIP].sort_values('elapsed_seconds').copy()
    print(f"  ✓ {len(trip_df)} samples for Punjabi Bagh → Kurukshetra trip")
    
    print("\nLoading training data for comparison...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    print(f"  ✓ {len(train_df)} total training samples")
    
    return trip_df, train_df


def plot_pca_reorientation(trip_df, output_path):
    """
    Generate PCA Reorientation visualization.
    Shows how phone tilt noise is corrected.
    """
    print("\n[1/4] Generating PCA Reorientation plot...")
    
    raw_accel = trip_df[['accel_x', 'accel_y', 'accel_z']].values
    
    # Apply PCA
    pca = PCA(n_components=3)
    rotated_accel = pca.fit_transform(raw_accel)
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: Raw Accelerometer Space (3D)
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(
        raw_accel[:, 0], raw_accel[:, 1], raw_accel[:, 2],
        c=np.arange(len(raw_accel)), cmap='Reds', alpha=0.3, s=5
    )
    ax1.set_xlabel('Accel X (g)')
    ax1.set_ylabel('Accel Y (g)')
    ax1.set_zlabel('Accel Z (g)')
    ax1.set_title('Raw Accelerometer Space\n(Phone Tilt Noise)', fontsize=12, fontweight='bold')
    
    # Plot 2: PCA-Reoriented Space (3D)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(
        rotated_accel[:, 0], rotated_accel[:, 1], rotated_accel[:, 2],
        c=np.arange(len(rotated_accel)), cmap='Greens', alpha=0.3, s=5
    )
    ax2.set_xlabel('PC1: Longitudinal (g)')
    ax2.set_ylabel('PC2: Lateral (g)')
    ax2.set_zlabel('PC3: Vertical (g)')
    ax2.set_title('PCA-Reoriented Space\n(Vehicle-Aligned Axes)', fontsize=12, fontweight='bold')
    
    # Plot 3: Explained Variance
    ax3 = fig.add_subplot(133)
    variance_ratio = pca.explained_variance_ratio_ * 100
    bars = ax3.bar(['PC1\n(Forward/Back)', 'PC2\n(Left/Right)', 'PC3\n(Up/Down)'], 
                   variance_ratio, color=['#2ecc71', '#3498db', '#9b59b6'])
    ax3.set_ylabel('Explained Variance (%)')
    ax3.set_title('PCA Component Importance', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 100)
    for bar, val in zip(bars, variance_ratio):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_bovw_histogram(trip_df, models, output_path):
    """
    Generate BoVW Histogram visualization.
    Shows the "visual fingerprint" of different driving segments.
    """
    print("\n[2/4] Generating BoVW Codeword Histogram...")
    
    codebook = models['codebook']
    scaler = models['scaler']
    seg_len = models.get('segment_length', 3)
    
    # Get features
    feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    available_cols = [c for c in feature_cols if c in trip_df.columns]
    
    # Take 3 different windows to show how histograms differ
    total_samples = len(trip_df)
    windows = [
        ("Start (Normal Driving)", trip_df.iloc[0:100]),
        ("Middle (Highway)", trip_df.iloc[total_samples//2:total_samples//2+100]),
        ("End (Urban)", trip_df.iloc[-100:])
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (label, window) in enumerate(windows):
        features = window[available_cols].values
        
        # Segment the data
        segments = []
        for i in range(0, len(features) - seg_len + 1):
            segments.append(features[i:i + seg_len].flatten())
        
        if len(segments) == 0:
            continue
        
        # Normalize and predict clusters
        segments_norm = scaler.transform(np.array(segments))
        codewords = codebook.predict(segments_norm)
        
        # Create histogram
        histogram = np.zeros(codebook.n_clusters)
        for cw in codewords:
            histogram[cw] += 1
        histogram = histogram / histogram.sum()  # Normalize
        
        # Plot
        ax = axes[idx]
        colors = plt.cm.viridis(histogram / histogram.max())
        ax.bar(range(64), histogram, color=colors, edgecolor='none')
        ax.set_xlabel('Codeword Index')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label}', fontsize=11, fontweight='bold')
        ax.set_xlim(-1, 65)
        
        # Highlight top codewords
        top_3 = np.argsort(histogram)[-3:]
        for t in top_3:
            ax.annotate(f'CW{t}', xy=(t, histogram[t]), 
                       xytext=(t, histogram[t] + 0.02),
                       ha='center', fontsize=8, color='red', fontweight='bold')
    
    plt.suptitle('BoVW Histograms: "Visual Fingerprints" of Different Trip Segments\n'
                 f'(Punjabi Bagh → Kurukshetra Trip)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_feature_importance(models, output_path):
    """
    Generate Feature Importance visualization.
    Shows which codewords are most important for classification.
    """
    print("\n[3/4] Generating Feature Importance plot...")
    
    classifier = models['classifier']
    if classifier is None:
        print("  ⚠ No classifier found, skipping...")
        return
    
    importances = classifier.feature_importances_
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: All codeword importances
    ax1 = axes[0]
    colors = plt.cm.RdYlGn(importances / importances.max())
    ax1.bar(range(64), importances, color=colors, edgecolor='none')
    ax1.set_xlabel('Codeword Index')
    ax1.set_ylabel('Importance Score')
    ax1.set_title('All 64 Codeword Importances', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1, 65)
    
    # Plot 2: Top 15 most important codewords
    ax2 = axes[1]
    top_indices = np.argsort(importances)[-15:]
    top_importances = importances[top_indices]
    
    bars = ax2.barh(range(15), top_importances, color=plt.cm.Oranges(top_importances / top_importances.max()))
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([f'Codeword {i}' for i in top_indices])
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Top 15 Most Important Codewords\nfor Safety Classification', fontsize=12, fontweight='bold')
    
    # Add importance values
    for bar, val in zip(bars, top_importances):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Random Forest Feature Importance: Which Codewords Detect Unsafe Driving?', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_cluster_visualization(train_df, models, output_path):
    """
    Generate 2D visualization of all clusters/codewords.
    Shows how different event types cluster together.
    """
    print("\n[4/4] Generating Cluster Visualization (this may take a moment)...")
    
    codebook = models['codebook']
    scaler = models['scaler']
    seg_len = models.get('segment_length', 3)
    
    # Sample data from different event classes
    feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    event_classes = train_df['event_label_gt'].unique() if 'event_label_gt' in train_df.columns else []
    
    all_segments = []
    all_labels = []
    
    for event_class in event_classes:
        class_data = train_df[train_df['event_label_gt'] == event_class].head(500)
        features = class_data[feature_cols].values
        
        for i in range(0, min(100, len(features) - seg_len + 1)):
            segment = features[i:i + seg_len].flatten()
            all_segments.append(segment)
            all_labels.append(event_class)
    
    if len(all_segments) < 100:
        print("  ⚠ Not enough data for cluster visualization, skipping...")
        return
    
    # Normalize and get cluster assignments
    segments_matrix = np.array(all_segments)
    segments_norm = scaler.transform(segments_matrix)
    cluster_assignments = codebook.predict(segments_norm)
    
    # Reduce to 2D using PCA (faster than t-SNE for large data)
    pca_2d = PCA(n_components=2)
    segments_2d = pca_2d.fit_transform(segments_norm)
    
    # Also get centroids in 2D
    centroids_2d = pca_2d.transform(codebook.cluster_centers_)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Points colored by event class
    ax1 = axes[0]
    unique_labels = list(set(all_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = np.array(all_labels) == label
        ax1.scatter(segments_2d[mask, 0], segments_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.5, s=20)
    
    ax1.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
               c='black', marker='X', s=100, label='Centroids', zorder=5)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title('Data Points Colored by Event Class', fontsize=12, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    # Plot 2: Points colored by cluster assignment
    ax2 = axes[1]
    scatter = ax2.scatter(segments_2d[:, 0], segments_2d[:, 1], 
                         c=cluster_assignments, cmap='viridis', alpha=0.5, s=20)
    ax2.scatter(centroids_2d[:, 0], centroids_2d[:, 1], 
               c='red', marker='X', s=100, edgecolors='white', linewidths=2, 
               label='Cluster Centroids', zorder=5)
    
    # Label some centroids
    for i in range(0, 64, 8):
        ax2.annotate(f'{i}', xy=(centroids_2d[i, 0], centroids_2d[i, 1]),
                    fontsize=8, fontweight='bold', color='red')
    
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.set_title('Data Points Colored by Cluster Assignment\n(K=64 Codewords)', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')
    
    plt.suptitle('K-Means Clustering: How BoVW Quantizes Motion Patterns', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def generate_summary_figure(models, output_path):
    """
    Generate a summary figure showing the complete pipeline.
    """
    print("\n[BONUS] Generating Pipeline Summary figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Raw → PCA transformation concept
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.8, "STAGE 1: PCA Reorientation", fontsize=14, fontweight='bold', 
             ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.5, "Phone in cupholder?\nTilted on dashboard?\n→ PCA aligns axes to vehicle frame", 
             fontsize=11, ha='center', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax1.axis('off')
    
    # 2. Segmentation concept
    ax2 = axes[0, 1]
    ax2.text(0.5, 0.8, "STAGE 2: Segmentation", fontsize=14, fontweight='bold', 
             ha='center', transform=ax2.transAxes)
    ax2.text(0.5, 0.5, "Break trip into 0.12s micro-windows\n→ Each window = 1 data point\n"
             "→ 6 channels × 3 timesteps = 18-dim vector", 
             fontsize=11, ha='center', transform=ax2.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    ax2.axis('off')
    
    # 3. K-Means concept
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.8, "STAGE 3: K-Means Quantization", fontsize=14, fontweight='bold', 
             ha='center', transform=ax3.transAxes)
    ax3.text(0.5, 0.5, f"Map each window to nearest of K={models['codebook'].n_clusters} centroids\n"
             "→ Create histogram of codeword frequencies\n→ This is the 'Visual Fingerprint'", 
             fontsize=11, ha='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    ax3.axis('off')
    
    # 4. Random Forest concept
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.8, "STAGE 4: Random Forest Classification", fontsize=14, fontweight='bold', 
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.5, f"Train {models['classifier'].n_estimators} decision trees on histograms\n"
             "→ Each tree votes on event class\n→ Output: AGGRESSIVE_BRAKING, NORMAL, etc.", 
             fontsize=11, ha='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.5))
    ax4.axis('off')
    
    plt.suptitle('BoVW Pipeline: From Raw Accelerometer to Event Classification', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("DRIVER PULSE - BoVW EXPLAINABILITY REPORT")
    print("=" * 70)
    print(f"\nTarget Trip: {TARGET_TRIP} (Punjabi Bagh → Kurukshetra)")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Load models and data
    models = load_models()
    trip_df, train_df = load_data()
    
    # Generate all plots
    plot_pca_reorientation(trip_df, os.path.join(OUTPUT_DIR, "1_pca_reorientation.png"))
    plot_bovw_histogram(trip_df, models, os.path.join(OUTPUT_DIR, "2_bovw_histogram.png"))
    plot_feature_importance(models, os.path.join(OUTPUT_DIR, "3_feature_importance.png"))
    plot_cluster_visualization(train_df, models, os.path.join(OUTPUT_DIR, "4_cluster_visualization.png"))
    generate_summary_figure(models, os.path.join(OUTPUT_DIR, "5_pipeline_summary.png"))
    
    print("\n" + "=" * 70)
    print("EXPLAINABILITY REPORT COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated {5} visualizations in: {OUTPUT_DIR}/")
    print("""
📊 Files generated:
   1_pca_reorientation.png    - Shows phone tilt correction
   2_bovw_histogram.png       - Visual fingerprints of driving segments
   3_feature_importance.png   - Which codewords matter most
   4_cluster_visualization.png - How K-Means groups motion patterns
   5_pipeline_summary.png     - High-level pipeline diagram

📝 Suggested README caption:
   "Unlike basic threshold models, Driver Pulse uses a Bag of Visual Words 
   (BoVW) pipeline. We compress high-frequency motion into 64 discrete 
   'codewords'. By analyzing the frequency distribution of these codewords, 
   our Random Forest classifier can distinguish between a sharp turn and 
   aggressive lane change even if the G-force magnitudes are identical."
""")


if __name__ == '__main__':
    main()
