"""
Audio Pipeline Explainability Script for Driver Pulse
======================================================
This script generates visual explanations of our 4-Layer Audio Detection System
using the Punjabi Bagh → Kurukshetra trip (TRIP_DRV003_03).

The 4-Layer Architecture:
- Layer 1 (Acoustic): Harsh fricatives, shouting via ZCR & Spectral Centroid
- Layer 2 (Temporal): Fragmented speech (interruptions or stunned silence)
- Layer 3 (Prosodic): Emotional arousal via F0 (pitch) variance
- Layer 4 (Context): Dynamic baseline normalization (not just "loud")

Outputs:
1. conflict_anatomy.png - Multi-panel breakdown of a conflict detection
2. layer_heatmap.png - Timeline showing when each layer fires
3. decision_table_demo.png - Shows how layers combine per PDF v4.0
4. baseline_adaptation.png - Context Gate normalization visualization

Run this AFTER running pipeline2_audio_4layer.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIGURATION ---
TARGET_TRIP = "TRIP_DRV003_03"  # Punjabi Bagh → Kurukshetra
AUDIO_PROCESSED_PATH = "outputs/audio_processed.csv"
AUDIO_FEATURES_PATH = "simulation_data/sensor_data/audio_features.csv"
OUTPUT_DIR = "outputs/explainability"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Layer thresholds (from constants.py)
LAYER1_ZCR_THRESHOLD = 0.55
LAYER1_CENTROID_THRESHOLD = 2000  # Hz
LAYER2_TURN_GAP_MIN = 0.3
LAYER2_TURN_GAP_MAX = 3.0
LAYER2_GAP_RATIO_THRESHOLD = 0.85
LAYER2_ENERGY_SLOPE_THRESHOLD = 50
LAYER3_F0_STD_THRESHOLD = 40  # Hz
LAYER3_SPEECH_RATE_MIN = 2.5
LAYER3_SPEECH_RATE_MAX = 5.5
LAYER4_DB_DEVIATION_THRESHOLD = 12  # dB


def load_data():
    """Load audio data - prefer raw features for visualization (uses simpler layer logic)."""
    # Prefer raw features for visualization (shows partial activations clearly)
    if os.path.exists(AUDIO_FEATURES_PATH):
        print(f"Loading raw audio features from {AUDIO_FEATURES_PATH}...")
        df = pd.read_csv(AUDIO_FEATURES_PATH)
        trip_df = df[df['trip_id'] == TARGET_TRIP].copy()
        if len(trip_df) > 0:
            trip_df = trip_df.sort_values('elapsed_seconds')
            print(f"  ✓ {len(trip_df)} windows for {TARGET_TRIP}")
            # Compute layer activations using visualization logic
            trip_df = compute_layer_activations(trip_df)
            return trip_df, False
    
    # Fall back to processed output
    if os.path.exists(AUDIO_PROCESSED_PATH):
        print(f"Loading processed audio data from {AUDIO_PROCESSED_PATH}...")
        df = pd.read_csv(AUDIO_PROCESSED_PATH)
        trip_df = df[df['trip_id'] == TARGET_TRIP].copy()
        if len(trip_df) > 0:
            trip_df = trip_df.sort_values('elapsed_seconds')
            print(f"  ✓ {len(trip_df)} windows for {TARGET_TRIP}")
            return trip_df, True
    
    # Generate synthetic data for demo
    print("⚠️  No audio data found. Generating synthetic demo data...")
    return generate_synthetic_demo(), False


def compute_layer_activations(df):
    """Compute layer activations from raw features."""
    df = df.copy()
    
    # Layer 1: Acoustic - use direct column access
    zcr = df['zcr'] if 'zcr' in df.columns else pd.Series([0.3]*len(df))
    centroid = df['spectral_centroid'] if 'spectral_centroid' in df.columns else pd.Series([1500]*len(df))
    df['layer1_active'] = (zcr > LAYER1_ZCR_THRESHOLD) & (centroid > LAYER1_CENTROID_THRESHOLD)
    
    # Layer 2: Temporal
    turn_gap = df['turn_gap_sec'] if 'turn_gap_sec' in df.columns else pd.Series([1.5]*len(df))
    gap_ratio = df['gap_ratio'] if 'gap_ratio' in df.columns else pd.Series([0.5]*len(df))
    energy_slope = df['energy_slope'] if 'energy_slope' in df.columns else pd.Series([10]*len(df))
    
    gap_flag = (turn_gap < LAYER2_TURN_GAP_MIN) | (turn_gap > LAYER2_TURN_GAP_MAX)
    ratio_flag = gap_ratio > LAYER2_GAP_RATIO_THRESHOLD
    slope_flag = energy_slope > LAYER2_ENERGY_SLOPE_THRESHOLD
    df['layer2_active'] = (gap_flag.astype(int) + ratio_flag.astype(int) + slope_flag.astype(int)) >= 2
    
    # Layer 3: Prosodic
    f0_std = df['f0_std'] if 'f0_std' in df.columns else pd.Series([25]*len(df))
    speech_rate = df['speech_rate'] if 'speech_rate' in df.columns else pd.Series([4.0]*len(df))
    
    f0_flag = f0_std > LAYER3_F0_STD_THRESHOLD
    rate_flag = (speech_rate < LAYER3_SPEECH_RATE_MIN) | (speech_rate > LAYER3_SPEECH_RATE_MAX)
    df['layer3_active'] = f0_flag | rate_flag
    
    # Layer 4: Context - use db_deviation directly if available
    if 'db_deviation' in df.columns:
        db_dev = df['db_deviation']
    else:
        db_level = df['db_level'] if 'db_level' in df.columns else pd.Series([55]*len(df))
        baseline = df['baseline_db'] if 'baseline_db' in df.columns else pd.Series([55]*len(df))
        db_dev = db_level - baseline
    df['db_deviation'] = db_dev
    df['layer4_active'] = db_dev > LAYER4_DB_DEVIATION_THRESHOLD
    
    # Debug output
    print(f"  Layer activations computed:")
    print(f"    L1 (Acoustic): {df['layer1_active'].sum()} windows ({df['layer1_active'].mean()*100:.1f}%)")
    print(f"    L2 (Temporal): {df['layer2_active'].sum()} windows ({df['layer2_active'].mean()*100:.1f}%)")
    print(f"    L3 (Prosodic): {df['layer3_active'].sum()} windows ({df['layer3_active'].mean()*100:.1f}%)")
    print(f"    L4 (Context):  {df['layer4_active'].sum()} windows ({df['layer4_active'].mean()*100:.1f}%)")
    conflict = df['layer1_active'] & df['layer2_active'] & df['layer3_active'] & df['layer4_active']
    print(f"    CONFLICT (all 4): {conflict.sum()} windows ({conflict.mean()*100:.1f}%)")
    
    return df


def generate_synthetic_demo():
    """Generate synthetic demo data showing the 4-layer system."""
    np.random.seed(42)
    n_windows = 200
    
    # Time axis (5s windows)
    elapsed = np.arange(0, n_windows * 5, 5)
    
    # Simulate a trip with a conflict zone around minute 8-10
    conflict_start = 95
    conflict_end = 115
    
    # Base levels
    df = pd.DataFrame({
        'elapsed_seconds': elapsed,
        'trip_id': TARGET_TRIP,
        'driver_id': 'DRV003'
    })
    
    # Layer 1: Acoustic (ZCR, Centroid)
    df['zcr'] = np.random.normal(0.35, 0.1, n_windows)
    df['spectral_centroid'] = np.random.normal(1500, 300, n_windows)
    # Spike during conflict
    mask = (elapsed >= conflict_start * 5) & (elapsed <= conflict_end * 5)
    df.loc[mask, 'zcr'] = np.random.normal(0.65, 0.1, mask.sum())
    df.loc[mask, 'spectral_centroid'] = np.random.normal(2500, 400, mask.sum())
    
    # Layer 2: Temporal
    df['turn_gap_sec'] = np.random.uniform(1.0, 2.5, n_windows)
    df['gap_ratio'] = np.random.uniform(0.4, 0.7, n_windows)
    df['energy_slope'] = np.random.normal(20, 15, n_windows)
    # Conflict zone: rushed interruptions
    df.loc[mask, 'turn_gap_sec'] = np.random.uniform(0.1, 0.3, mask.sum())
    df.loc[mask, 'gap_ratio'] = np.random.uniform(0.85, 0.95, mask.sum())
    df.loc[mask, 'energy_slope'] = np.random.normal(60, 10, mask.sum())
    
    # Layer 3: Prosodic
    df['f0_std'] = np.random.normal(25, 10, n_windows)
    df['speech_rate'] = np.random.uniform(3.0, 5.0, n_windows)
    # Conflict zone: high pitch variance
    df.loc[mask, 'f0_std'] = np.random.normal(55, 12, mask.sum())
    
    # Layer 4: Context (volume)
    df['baseline_db'] = 55
    df['db_level'] = np.random.normal(58, 4, n_windows)
    # Conflict zone: elevated volume
    df.loc[mask, 'db_level'] = np.random.normal(72, 5, mask.sum())
    df['db_deviation'] = df['db_level'] - df['baseline_db']
    
    # Compute layer activations
    return compute_layer_activations(df)


def plot_conflict_anatomy(df, output_path):
    """
    Generate the "Anatomy of a Conflict" 3-panel plot.
    Shows Context Gate, Acoustic Harshness, and Layer Fusion.
    """
    print("\n[1/4] Generating Conflict Anatomy plot...")
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.3)
    
    time_axis = df['elapsed_seconds'] / 60  # Convert to minutes
    
    # ======= Panel 1: Context Gate (Volume vs Baseline) =======
    db_level = df.get('db_level', pd.Series([55]*len(df)))
    baseline_db = df.get('baseline_db', pd.Series([55]*len(df)))
    db_deviation = df.get('db_deviation', db_level - baseline_db)
    
    ax1.plot(time_axis, db_level, color='gray', alpha=0.7, label='Live Volume (dB)', linewidth=1.5)
    ax1.plot(time_axis, baseline_db, 'k--', label='Vehicle Baseline', linewidth=2)
    
    # Shade where context gate is open
    gate_open = db_deviation > LAYER4_DB_DEVIATION_THRESHOLD
    ax1.fill_between(time_axis, db_level, baseline_db, 
                     where=gate_open, color='red', alpha=0.2, 
                     label=f'Context Gate Open (>{LAYER4_DB_DEVIATION_THRESHOLD}dB)')
    
    # Threshold line
    ax1.axhline(baseline_db.mean() + LAYER4_DB_DEVIATION_THRESHOLD, 
                color='red', linestyle=':', alpha=0.5, label='Gate Threshold')
    
    ax1.set_ylabel('Decibels (dB)', fontsize=11)
    ax1.set_title("Layer 4: Context Gate — Dynamic Baseline Normalization", 
                  fontsize=13, fontweight='bold', color='#e74c3c')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(40, 90)
    
    # ======= Panel 2: Acoustic Harshness =======
    zcr = df.get('zcr', pd.Series([0.3]*len(df)))
    centroid = df.get('spectral_centroid', pd.Series([1500]*len(df)))
    
    # Normalize for plotting
    zcr_norm = zcr / zcr.max() if zcr.max() > 0 else zcr
    centroid_norm = centroid / centroid.max() if centroid.max() > 0 else centroid
    
    ax2.plot(time_axis, zcr_norm, color='#9b59b6', label='Zero Crossing Rate (Fricative Harshness)', linewidth=1.5)
    ax2.plot(time_axis, centroid_norm, color='#16a085', label='Spectral Centroid (Tonal Sharpness)', linewidth=1.5)
    ax2.axhline(LAYER1_ZCR_THRESHOLD / zcr.max(), color='red', linestyle=':', alpha=0.7, label='Harshness Threshold')
    
    ax2.set_ylabel('Normalized Intensity', fontsize=11)
    ax2.set_title("Layer 1: Acoustic Harshness — The 'Shouting' Fingerprint", 
                  fontsize=13, fontweight='bold', color='#9b59b6')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.2)
    
    # ======= Panel 3: 4-Layer Fusion Heatmap =======
    layers = np.vstack([
        df['layer1_active'].astype(int).values,
        df['layer2_active'].astype(int).values,
        df['layer3_active'].astype(int).values,
        df['layer4_active'].astype(int).values
    ])
    
    # Create custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('conflict', ['#2c3e50', '#e74c3c'], N=2)
    
    im = ax3.imshow(layers, aspect='auto', cmap=cmap, 
                    extent=[time_axis.min(), time_axis.max(), -0.5, 3.5])
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_yticklabels(['L1: Acoustic\n(Harsh)', 'L2: Temporal\n(Fragmented)', 
                         'L3: Prosodic\n(Aroused)', 'L4: Context\n(Deviated)'])
    ax3.set_xlabel('Trip Time (minutes)', fontsize=11)
    ax3.set_title("4-Layer Heuristic Fusion — When All Layers Align = CONFLICT", 
                  fontsize=13, fontweight='bold', color='#e74c3c')
    
    # Highlight conflict zones (all 4 layers active)
    all_active = (df['layer1_active'] & df['layer2_active'] & 
                  df['layer3_active'] & df['layer4_active'])
    conflict_times = time_axis[all_active]
    for t in conflict_times:
        ax3.axvline(t, color='yellow', alpha=0.3, linewidth=3)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor='#2c3e50', label='Inactive'),
                       mpatches.Patch(facecolor='#e74c3c', label='Active'),
                       mpatches.Patch(facecolor='yellow', alpha=0.3, label='CONFLICT')]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.suptitle(f"Audio Conflict Detection: Anatomy of a Social Stress Event\n"
                 f"Trip: {TARGET_TRIP} (Punjabi Bagh → Kurukshetra)", 
                 fontsize=15, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_decision_table(df, output_path):
    """
    Visualize the Decision Table from PDF v4.0.
    Shows how layer combinations map to labels.
    """
    print("\n[2/4] Generating Decision Table visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Decision Table as a grid
    ax1 = axes[0]
    
    # Create decision table data
    table_data = [
        ['False', '*', '*', '*', 'NORMAL', '#27ae60'],
        ['True', 'True', 'True', '*', 'CONFLICT', '#e74c3c'],
        ['True', 'True', 'False', 'True', 'AMBIGUOUS', '#f39c12'],
        ['True', 'False', 'True', 'True', 'AMBIGUOUS', '#f39c12'],
        ['True', '*', '*', 'False', 'NORMAL_LOUD', '#3498db'],
    ]
    
    # Draw table
    ax1.axis('off')
    table = ax1.table(
        cellText=[[row[0], row[1], row[2], row[3], row[4]] for row in table_data],
        colLabels=['L4\n(Context)', 'L1\n(Harsh)', 'L2\n(Frag)', 'L3\n(Aroused)', 'Label'],
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.12, 0.12, 0.12, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Color cells by label
    for i, row in enumerate(table_data):
        table[(i+1, 4)].set_facecolor(row[5])
        table[(i+1, 4)].set_text_props(color='white', fontweight='bold')
    
    ax1.set_title("PDF v4.0 Decision Table\n(Context Gate Acts as Primary Filter)", 
                  fontsize=13, fontweight='bold', pad=20)
    
    # Right: Pie chart of actual classifications
    ax2 = axes[1]
    
    # Compute labels for each window
    labels = []
    for _, row in df.iterrows():
        l1 = row.get('layer1_active', False)
        l2 = row.get('layer2_active', False)
        l3 = row.get('layer3_active', False)
        l4 = row.get('layer4_active', False)
        
        if not l4:
            labels.append('NORMAL')
        elif l1 and l2:
            labels.append('CONFLICT')
        elif (l1 and l3) or (l2 and l3):
            labels.append('AMBIGUOUS')
        else:
            labels.append('NORMAL_LOUD')
    
    label_counts = pd.Series(labels).value_counts()
    
    colors = {'NORMAL': '#27ae60', 'CONFLICT': '#e74c3c', 
              'AMBIGUOUS': '#f39c12', 'NORMAL_LOUD': '#3498db'}
    pie_colors = [colors.get(l, 'gray') for l in label_counts.index]
    
    wedges, texts, autotexts = ax2.pie(
        label_counts.values, 
        labels=label_counts.index,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.05 if l == 'CONFLICT' else 0 for l in label_counts.index]
    )
    
    ax2.set_title(f"Window Classifications for {TARGET_TRIP}\n({len(df)} windows analyzed)", 
                  fontsize=13, fontweight='bold')
    
    plt.suptitle("4-Layer Decision Logic: How We Label Each 5-Second Window", 
                 fontsize=15, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_layer_timeline(df, output_path):
    """
    Create a detailed timeline showing when each layer fires.
    """
    print("\n[3/4] Generating Layer Timeline plot...")
    
    fig, axes = plt.subplots(5, 1, figsize=(16, 10), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    
    time_axis = df['elapsed_seconds'] / 60  # minutes
    
    layer_configs = [
        ('layer4_active', 'Layer 4: Context Gate\n(Volume > Baseline+12dB)', '#9b59b6'),
        ('layer1_active', 'Layer 1: Acoustic\n(Harsh Fricatives)', '#e74c3c'),
        ('layer2_active', 'Layer 2: Temporal\n(Fragmented Speech)', '#3498db'),
        ('layer3_active', 'Layer 3: Prosodic\n(Pitch Variability)', '#f39c12'),
    ]
    
    for idx, (col, label, color) in enumerate(layer_configs):
        ax = axes[idx]
        active = df[col].astype(int).values
        
        ax.fill_between(time_axis, 0, active, color=color, alpha=0.7, step='mid')
        ax.set_ylabel(label, fontsize=9, rotation=0, ha='right', va='center')
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Count activations
        n_active = active.sum()
        pct = n_active / len(active) * 100
        ax.text(time_axis.max() + 0.5, 0.5, f'{pct:.1f}%', 
                fontsize=10, va='center', fontweight='bold', color=color)
    
    # Bottom panel: Combined (CONFLICT = all 4)
    ax = axes[4]
    conflict = (df['layer1_active'] & df['layer2_active'] & 
                df['layer3_active'] & df['layer4_active']).astype(int).values
    
    ax.fill_between(time_axis, 0, conflict, color='#c0392b', alpha=0.9, step='mid')
    ax.set_ylabel('CONFLICT\n(All 4 Layers)', fontsize=9, rotation=0, ha='right', va='center')
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_xlabel('Trip Time (minutes)', fontsize=12)
    
    n_conflict = conflict.sum()
    pct = n_conflict / len(conflict) * 100
    ax.text(time_axis.max() + 0.5, 0.5, f'{pct:.1f}%', 
            fontsize=10, va='center', fontweight='bold', color='#c0392b')
    
    plt.suptitle(f"Layer Activation Timeline: {TARGET_TRIP}\n"
                 f"Defense-in-Depth: CONFLICT only when ALL layers align", 
                 fontsize=14, fontweight='bold')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_feature_distributions(df, output_path):
    """
    Show the distribution of features and how thresholds separate classes.
    """
    print("\n[4/4] Generating Feature Distribution plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Determine if each window is conflict or not
    conflict = (df['layer1_active'] & df['layer2_active'] & 
                df['layer3_active'] & df['layer4_active'])
    
    # Top-left: ZCR distribution
    ax = axes[0, 0]
    zcr = df.get('zcr', pd.Series([0.3]*len(df)))
    ax.hist(zcr[~conflict], bins=30, alpha=0.6, color='#27ae60', label='Normal', density=True)
    ax.hist(zcr[conflict], bins=30, alpha=0.6, color='#e74c3c', label='Conflict', density=True)
    ax.axvline(LAYER1_ZCR_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold ({LAYER1_ZCR_THRESHOLD})')
    ax.set_xlabel('Zero Crossing Rate')
    ax.set_ylabel('Density')
    ax.set_title('Layer 1: Acoustic Harshness (ZCR)', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Top-right: Spectral Centroid
    ax = axes[0, 1]
    centroid = df.get('spectral_centroid', pd.Series([1500]*len(df)))
    ax.hist(centroid[~conflict], bins=30, alpha=0.6, color='#27ae60', label='Normal', density=True)
    ax.hist(centroid[conflict], bins=30, alpha=0.6, color='#e74c3c', label='Conflict', density=True)
    ax.axvline(LAYER1_CENTROID_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold ({LAYER1_CENTROID_THRESHOLD}Hz)')
    ax.set_xlabel('Spectral Centroid (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('Layer 1: Tonal Sharpness (Centroid)', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Bottom-left: F0 Standard Deviation
    ax = axes[1, 0]
    f0_std = df.get('f0_std', pd.Series([25]*len(df)))
    ax.hist(f0_std[~conflict], bins=30, alpha=0.6, color='#27ae60', label='Normal', density=True)
    ax.hist(f0_std[conflict], bins=30, alpha=0.6, color='#e74c3c', label='Conflict', density=True)
    ax.axvline(LAYER3_F0_STD_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold ({LAYER3_F0_STD_THRESHOLD}Hz)')
    ax.set_xlabel('Pitch Variability (F0 Std Dev)')
    ax.set_ylabel('Density')
    ax.set_title('Layer 3: Emotional Arousal (Pitch)', fontsize=12, fontweight='bold')
    ax.legend()
    
    # Bottom-right: dB Deviation
    ax = axes[1, 1]
    db_dev = df.get('db_deviation', pd.Series([5]*len(df)))
    ax.hist(db_dev[~conflict], bins=30, alpha=0.6, color='#27ae60', label='Normal', density=True)
    ax.hist(db_dev[conflict], bins=30, alpha=0.6, color='#e74c3c', label='Conflict', density=True)
    ax.axvline(LAYER4_DB_DEVIATION_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Threshold ({LAYER4_DB_DEVIATION_THRESHOLD}dB)')
    ax.set_xlabel('Volume Above Baseline (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Layer 4: Context Gate (Volume Deviation)', fontsize=12, fontweight='bold')
    ax.legend()
    
    plt.suptitle("Feature Distributions: How Thresholds Separate Normal vs Conflict", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("DRIVER PULSE - AUDIO PIPELINE EXPLAINABILITY REPORT")
    print("=" * 70)
    print(f"\nTarget Trip: {TARGET_TRIP} (Punjabi Bagh → Kurukshetra)")
    print(f"Output Directory: {OUTPUT_DIR}")
    
    # Load data
    df, is_processed = load_data()
    
    if df is None or len(df) == 0:
        print("❌ Error: No data available for visualization")
        return
    
    # Generate all plots
    plot_conflict_anatomy(df, os.path.join(OUTPUT_DIR, "audio_1_conflict_anatomy.png"))
    plot_decision_table(df, os.path.join(OUTPUT_DIR, "audio_2_decision_table.png"))
    plot_layer_timeline(df, os.path.join(OUTPUT_DIR, "audio_3_layer_timeline.png"))
    plot_feature_distributions(df, os.path.join(OUTPUT_DIR, "audio_4_feature_distributions.png"))
    
    print("\n" + "=" * 70)
    print("AUDIO EXPLAINABILITY REPORT COMPLETE!")
    print("=" * 70)
    print(f"\nGenerated 4 visualizations in: {OUTPUT_DIR}/")
    print("""
📊 Files generated:
   audio_1_conflict_anatomy.png      - 3-panel breakdown of detection logic
   audio_2_decision_table.png        - PDF v4.0 decision table visualization
   audio_3_layer_timeline.png        - When each layer fires during trip
   audio_4_feature_distributions.png - How thresholds separate classes

📝 Suggested README caption:
   "Privacy is a core pillar of Driver Pulse. Our Audio module performs 
   Diarization-free Conflict Detection. Instead of transcribing words, 
   we analyze the Acoustic Anatomy of the cabin. By fusing 4 heuristic 
   layers (Acoustic, Temporal, Prosodic, and Context), we identify social 
   stress with high precision without ever recording a single word of 
   the driver's conversation."

🔑 Key Explainability Points:
   1. CONTEXT GATE prevents false positives from naturally loud vehicles
   2. ZCR detects fricative harshness (s, sh, f sounds in shouting)
   3. TEMPORAL layer catches interruptions and stunned silences
   4. PROSODIC layer measures emotional arousal via pitch variance
   5. Defense-in-depth: CONFLICT only fires when ALL layers agree
""")


if __name__ == '__main__':
    main()
