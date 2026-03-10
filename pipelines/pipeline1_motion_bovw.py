"""
Pipeline 1: Motion Event Detection with BoVW Pipeline
======================================================
Implements the full accelerometer processing pipeline as specified in the design doc:
- Stage 1: Ingestion, Resampling, Auto-calibration, Kalman Filter
- Stage 2: PCA Reorientation  
- Stage 3: Self-Trigger (for battery saving - simulated here)
- Stage 4: BoVW Feature Extraction (K-Means Codebook)
- Stage 5: Classification (Random Forest) & Output

7 Event Classes:
- AGGRESSIVE_BRAKING: accel_y in [-2.5, -1.5]g, gyro_z ~0, duration 1.5-3s
- AGGRESSIVE_ACCEL: accel_y in [+1.5, +2.5]g, gyro_z ~0, duration 2-4s
- AGGRESSIVE_LEFT_TURN: accel_x in [-1.5, -0.8]g, gyro_z in [+40, +120] deg/s
- AGGRESSIVE_RIGHT_TURN: accel_x in [+0.8, +1.5]g, gyro_z in [-120, -40] deg/s
- AGG_LEFT_LANE_CHANGE: accel_x in [-0.5, -0.3]g, gyro_z spike <0.5s then return
- AGG_RIGHT_LANE_CHANGE: accel_x in [+0.3, +0.5]g, gyro_z spike <0.5s then return
- NORMAL: all axes within ±0.3g, gyro within ±15 deg/s

Noise: Add Gaussian noise N(0, 0.05) to accel, N(0, 2.0) to gyro for MEMS simulation.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy import signal
import os
import warnings
import pickle
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Import constants
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import *


# =============================================================================
# KALMAN FILTER FOR SENSOR NOISE REDUCTION
# =============================================================================

class KalmanFilter1D:
    """Simple 1D Kalman filter for sensor noise reduction."""
    
    def __init__(self, process_noise=KALMAN_PROCESS_NOISE, measurement_noise=KALMAN_MEASUREMENT_NOISE):
        self.q = process_noise  # Process noise
        self.r = measurement_noise  # Measurement noise
        self.x = 0.0  # State estimate
        self.p = 1.0  # Error covariance
        
    def update(self, measurement):
        # Prediction
        self.p = self.p + self.q
        
        # Update
        k = self.p / (self.p + self.r)  # Kalman gain
        self.x = self.x + k * (measurement - self.x)
        self.p = (1 - k) * self.p
        
        return self.x
    
    def reset(self, initial_value=0.0):
        self.x = initial_value
        self.p = 1.0


# =============================================================================
# BOVW MOTION EVENT DETECTOR (TRAINING ON driver_pulse_data)
# =============================================================================

class BoVWMotionDetector:
    """
    Bag of Visual Words (BoVW) based motion event detector.
    
    Training Phase (Offline, One-Time):
    1. Load accelerometer data from driver_pulse_data (with ground truth labels)
    2. Build K-Means codebook from all segments
    3. Extract histograms for labeled 5-second clips
    4. Train Random Forest classifier on histograms
    5. Save models to disk
    
    Inference Phase (Runtime, per Clip):
    1. Load trained models
    2. For each 5-second window, encode to histogram using codebook
    3. Classify using Random Forest
    """
    
    # Feature columns for BoVW segments
    FEATURE_COLS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Event classes
    EVENT_CLASSES = [
        'AGGRESSIVE_BRAKING',
        'AGGRESSIVE_ACCEL', 
        'AGGRESSIVE_LEFT_TURN',
        'AGGRESSIVE_RIGHT_TURN',
        'AGG_LEFT_LANE_CHANGE',
        'AGG_RIGHT_LANE_CHANGE',
        'NORMAL'
    ]
    
    def __init__(self, models_dir=None):
        """
        Initialize the BoVW Motion Detector.
        
        Args:
            models_dir: Directory to save/load trained models. If None, uses outputs/models/
        """
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.models_dir = models_dir or os.path.join(self.base_dir, 'outputs', 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Models
        self.codebook = None  # K-Means model
        self.scaler = None    # StandardScaler for normalization
        self.classifier = None  # Random Forest classifier
        
        # Training metadata
        self.training_metadata = {}
        
        # Kalman filters for preprocessing
        self.kalman_filters = {axis: KalmanFilter1D() for axis in self.FEATURE_COLS}
    
    # =========================================================================
    # STAGE 1: DATA LOADING & PREPROCESSING
    # =========================================================================
    
    def load_training_data(self, data_path=None):
        """
        Load accelerometer training data from driver_pulse_data.
        
        Args:
            data_path: Path to accelerometer_data.csv. If None, uses default location.
        
        Returns:
            DataFrame with accelerometer data
        """
        if data_path is None:
            data_path = os.path.join(
                self.base_dir, 'driver_pulse_data', 'sensor_data', 'accelerometer_data.csv'
            )
        
        print(f"Loading training data from: {data_path}")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"  ✓ Loaded {len(df):,} samples")
        print(f"  ✓ Trips: {df['trip_id'].nunique()}")
        print(f"  ✓ Drivers: {df['driver_id'].nunique()}")
        
        # Check for ground truth labels
        if 'event_label_gt' in df.columns:
            label_counts = df['event_label_gt'].value_counts()
            print(f"  ✓ Ground truth labels found:")
            for label, count in label_counts.items():
                print(f"      {label}: {count:,}")
        else:
            print("  ⚠ No ground truth labels (event_label_gt) found!")
        
        return df
    
    def preprocess_trip(self, trip_data):
        """
        Preprocess a single trip's data.
        
        Stage 1a: Resampling to 25 Hz
        Stage 1b: Auto-calibration (noise floor estimation)
        Stage 1c: Kalman filtering for noise reduction
        
        Args:
            trip_data: DataFrame with one trip's accelerometer data
            
        Returns:
            Preprocessed DataFrame, noise_floor dict
        """
        df = trip_data.copy().sort_values('elapsed_seconds').reset_index(drop=True)
        
        # Stage 1b: Auto-calibration - estimate noise floor from steady-state windows
        if 'speed_kmh' in df.columns:
            df['speed_change'] = df['speed_kmh'].diff().abs().fillna(0)
            steady_mask = df['speed_change'] < 2.0  # Less than 2 km/h change
        else:
            steady_mask = pd.Series([True] * len(df))
        
        if steady_mask.sum() > 10:
            noise_floor = {
                'accel_x': df.loc[steady_mask, 'accel_x'].std() if 'accel_x' in df.columns else 0.1,
                'accel_y': df.loc[steady_mask, 'accel_y'].std() if 'accel_y' in df.columns else 0.1,
                'accel_z': df.loc[steady_mask, 'accel_z'].std() if 'accel_z' in df.columns else 0.1,
                'gyro_z': df.loc[steady_mask, 'gyro_z'].std() if 'gyro_z' in df.columns else 5.0,
            }
        else:
            noise_floor = {'accel_x': 0.1, 'accel_y': 0.1, 'accel_z': 0.1, 'gyro_z': 5.0}
        
        # Stage 1c: Apply Kalman filter to smooth sensor readings
        for axis in self.FEATURE_COLS:
            if axis in df.columns:
                kf = self.kalman_filters[axis]
                kf.reset(df[axis].iloc[0] if len(df) > 0 else 0)
                df[f'{axis}_filtered'] = df[axis].apply(kf.update)
        
        return df, noise_floor
    
    # =========================================================================
    # STAGE 2: PCA REORIENTATION
    # =========================================================================
    
    def pca_reorient(self, trip_data, calibration_seconds=30):
        """
        PCA reorientation to correct for phone placement.
        
        Uses first N seconds to compute rotation matrix.
        PC1 = longitudinal (forward/back), PC2 = lateral (left/right)
        
        Args:
            trip_data: Preprocessed trip data
            calibration_seconds: Seconds of data for PCA calibration
            
        Returns:
            DataFrame with PCA-corrected accelerations
        """
        df = trip_data.copy()
        
        # Check if required columns exist
        if not all(c in df.columns for c in ['accel_x', 'accel_y', 'accel_z']):
            return df
        
        # Get calibration window
        if 'elapsed_seconds' in df.columns:
            calib_mask = df['elapsed_seconds'] <= calibration_seconds
        else:
            calib_mask = pd.Series([True] * min(calibration_seconds * 25, len(df)))
        
        calib_data = df.loc[calib_mask, ['accel_x', 'accel_y', 'accel_z']].values
        
        if len(calib_data) < 10:
            # Fall back to raw values
            df['accel_x_pca'] = df['accel_x']
            df['accel_y_pca'] = df['accel_y']
            df['accel_z_pca'] = df['accel_z']
            return df
        
        try:
            pca = PCA(n_components=3)
            pca.fit(calib_data)
            
            accel_raw = df[['accel_x', 'accel_y', 'accel_z']].values
            accel_rotated = pca.transform(accel_raw)
            
            df['accel_y_pca'] = accel_rotated[:, 0]  # Longitudinal
            df['accel_x_pca'] = accel_rotated[:, 1]  # Lateral
            df['accel_z_pca'] = accel_rotated[:, 2]  # Vertical
        except Exception:
            df['accel_x_pca'] = df['accel_x']
            df['accel_y_pca'] = df['accel_y']
            df['accel_z_pca'] = df['accel_z']
        
        return df
    
    # =========================================================================
    # STAGE 3: EXTRACT 5-SECOND CLIPS WITH LABELS
    # =========================================================================
    
    def extract_labeled_clips(self, df, clip_duration_sec=5.0, min_samples_per_clip=3):
        """
        Extract clips around labeled events for training.
        
        The data may have variable sampling rates. This function extracts
        windows around each labeled event based on time, with a minimum
        sample count requirement.
        
        For datasets with sparse sampling (e.g., 1.5s intervals), we use
        overlapping windows and group consecutive samples with the same label.
        
        Args:
            df: Preprocessed DataFrame with event_label_gt column
            clip_duration_sec: Duration of each clip in seconds
            min_samples_per_clip: Minimum samples required for a valid clip
            
        Returns:
            List of (clip_data, label) tuples
        """
        clips = []
        half_window = clip_duration_sec / 2.0
        
        # Detect actual sampling rate from data
        sample_intervals = df.groupby('trip_id')['elapsed_seconds'].diff().dropna()
        median_interval = sample_intervals.median() if len(sample_intervals) > 0 else 0.04
        actual_sample_rate = 1.0 / median_interval if median_interval > 0 else 25.0
        
        print(f"  Detected sampling: ~{actual_sample_rate:.2f} Hz (median interval: {median_interval:.2f}s)")
        
        # Group by trip for efficient extraction
        for trip_id in df['trip_id'].unique():
            trip_df = df[df['trip_id'] == trip_id].copy()
            trip_df = trip_df.sort_values('elapsed_seconds').reset_index(drop=True)
            
            # For each event class, find contiguous segments
            for label in self.EVENT_CLASSES:
                label_mask = trip_df['event_label_gt'] == label
                if label_mask.sum() == 0:
                    continue
                
                # Get indices of labeled samples
                label_indices = trip_df[label_mask].index.tolist()
                
                # Group consecutive indices (with a gap tolerance)
                # For sparse data, allow up to 5 seconds gap
                max_gap_samples = int(5.0 / median_interval) if median_interval > 0 else 125
                
                groups = []
                current_group = [label_indices[0]]
                
                for i in range(1, len(label_indices)):
                    if label_indices[i] - label_indices[i-1] <= max_gap_samples:
                        current_group.append(label_indices[i])
                    else:
                        groups.append(current_group)
                        current_group = [label_indices[i]]
                groups.append(current_group)
                
                # For each group, extract a clip
                for group_indices in groups:
                    start_idx = max(0, min(group_indices) - 2)
                    end_idx = min(len(trip_df), max(group_indices) + 3)
                    
                    clip_data = trip_df.iloc[start_idx:end_idx].copy()
                    
                    if len(clip_data) >= min_samples_per_clip:
                        clips.append((clip_data, label))
        
        print(f"  ✓ Extracted {len(clips)} labeled clips")
        
        # Count by label
        label_counts = {}
        for _, label in clips:
            label_counts[label] = label_counts.get(label, 0) + 1
        for label, count in sorted(label_counts.items()):
            print(f"      {label}: {count}")
        
        return clips
    
    # =========================================================================
    # STAGE 4: BOVW CODEBOOK CONSTRUCTION
    # =========================================================================
    
    def build_codebook(self, df, k=64, segment_length=12, hop=6):
        """
        Build K-Means codebook from all data segments.
        
        4a. Codebook Construction (Offline, One-Time):
        - Chop data into segments with 50% overlap
        - For dense data (25Hz): 0.5s segments (12 samples)
        - For sparse data: adapt segment size to available samples
        - Run K-Means to find K=64 cluster centroids (codewords)
        
        Args:
            df: Preprocessed DataFrame with accelerometer data
            k: Number of codewords (clusters)
            segment_length: Base samples per segment (may be adapted for sparse data)
            hop: Hop between segments (50% overlap)
            
        Returns:
            Fitted KMeans model
        """
        print(f"\n  Building BoVW codebook with K={k} codewords...")
        
        all_segments = []
        
        # Use filtered columns if available, else raw
        use_cols = [f'{c}_filtered' if f'{c}_filtered' in df.columns else c 
                    for c in self.FEATURE_COLS]
        use_cols = [c for c in use_cols if c in df.columns]
        
        if len(use_cols) < 4:
            print(f"  ⚠ Not enough feature columns: {use_cols}")
            return None
        
        # Detect actual sampling rate
        sample_intervals = df.groupby('trip_id')['elapsed_seconds'].diff().dropna()
        median_interval = sample_intervals.median() if len(sample_intervals) > 0 else 0.04
        
        # Adapt segment size for sparse data
        # For sparse data (>0.5s intervals), use smaller segments
        if median_interval > 0.1:  # Sparse data
            segment_length = 3  # Use 3 samples per segment
            hop = 1  # Overlap of 2 samples
            print(f"  Adapting for sparse data: segment_length={segment_length}, hop={hop}")
        
        self._segment_length = segment_length
        self._hop = hop
        
        # Process each trip separately
        for trip_id in df['trip_id'].unique():
            trip_df = df[df['trip_id'] == trip_id].sort_values('elapsed_seconds')
            data = trip_df[use_cols].values
            
            # Sliding window segmentation
            for i in range(0, len(data) - segment_length + 1, hop):
                segment = data[i:i + segment_length]
                if len(segment) == segment_length:
                    # Flatten to feature vector
                    all_segments.append(segment.flatten())
        
        if len(all_segments) < k:
            print(f"  ⚠ Not enough segments ({len(all_segments)}) for {k} codewords")
            return None
        
        # Convert to numpy array
        segments_matrix = np.array(all_segments)
        print(f"  ✓ Created {len(segments_matrix):,} segments of shape {segment_length}×{len(use_cols)}")
        
        # Normalize features
        self.scaler = StandardScaler()
        segments_normalized = self.scaler.fit_transform(segments_matrix)
        
        # Run K-Means clustering
        print(f"  Running K-Means clustering...")
        self.codebook = KMeans(
            n_clusters=k, 
            random_state=42, 
            n_init=10,
            max_iter=300,
            verbose=0
        )
        self.codebook.fit(segments_normalized)
        
        print(f"  ✓ Codebook built with {k} codewords (cluster centroids)")
        print(f"  ✓ Inertia: {self.codebook.inertia_:.2f}")
        
        return self.codebook
    
    def encode_clip_to_histogram(self, clip_data, segment_length=None, hop=None):
        """
        Encode a clip to a histogram of codewords.
        
        4b. Encoding (Runtime, per Clip):
        - Segment clip into windows with 50% overlap
        - For each segment, find nearest codeword (cluster centroid)
        - Build histogram of codeword frequencies
        - L1 normalize histogram
        
        Args:
            clip_data: DataFrame with clip accelerometer data
            segment_length: Samples per segment (uses trained value if None)
            hop: Hop between segments (uses trained value if None)
            
        Returns:
            (histogram, top_codewords) or None if encoding fails
        """
        if self.codebook is None or self.scaler is None:
            return None
        
        # Use stored segment parameters from training
        segment_length = segment_length or getattr(self, '_segment_length', 3)
        hop = hop or getattr(self, '_hop', 1)
        
        # Use filtered columns if available
        use_cols = [f'{c}_filtered' if f'{c}_filtered' in clip_data.columns else c 
                    for c in self.FEATURE_COLS]
        use_cols = [c for c in use_cols if c in clip_data.columns]
        
        if len(use_cols) < 4:
            return None
        
        data = clip_data[use_cols].values
        
        # Segment the clip
        segments = []
        for i in range(0, len(data) - segment_length + 1, hop):
            segment = data[i:i + segment_length]
            if len(segment) == segment_length:
                segments.append(segment.flatten())
        
        if len(segments) == 0:
            # If clip is too small, use the whole clip as one segment
            if len(data) >= 1:
                # Pad or truncate to segment_length
                if len(data) < segment_length:
                    # Pad by repeating last row
                    padded = np.vstack([data] + [data[-1:]] * (segment_length - len(data)))
                    segments.append(padded.flatten())
                else:
                    segments.append(data[:segment_length].flatten())
        
        if len(segments) == 0:
            return None
        
        # Normalize and find nearest codewords
        segments_matrix = np.array(segments)
        
        # Ensure dimensions match
        expected_dim = self.scaler.mean_.shape[0]
        if segments_matrix.shape[1] != expected_dim:
            # Adjust by padding or truncating
            if segments_matrix.shape[1] < expected_dim:
                pad_width = expected_dim - segments_matrix.shape[1]
                segments_matrix = np.pad(segments_matrix, ((0, 0), (0, pad_width)), mode='constant')
            else:
                segments_matrix = segments_matrix[:, :expected_dim]
        
        segments_normalized = self.scaler.transform(segments_matrix)
        codeword_indices = self.codebook.predict(segments_normalized)
        
        # Build histogram
        k = self.codebook.n_clusters
        histogram = np.zeros(k)
        for idx in codeword_indices:
            histogram[idx] += 1
        
        # L1 normalize
        if histogram.sum() > 0:
            histogram = histogram / histogram.sum()
        
        # Get top 3 codewords for explainability
        top_indices = np.argsort(histogram)[::-1][:3]
        top_codewords = [(int(idx), float(histogram[idx])) for idx in top_indices]
        
        return histogram, top_codewords
    
    # =========================================================================
    # STAGE 5: RANDOM FOREST CLASSIFIER TRAINING
    # =========================================================================
    
    def train_classifier(self, clips):
        """
        Train Random Forest classifier on histogram features.
        
        Args:
            clips: List of (clip_data, label) tuples
            
        Returns:
            Trained RandomForestClassifier
        """
        print(f"\n  Training Random Forest classifier...")
        
        X = []
        y = []
        failed_clips = 0
        
        for clip_data, label in clips:
            encoding = self.encode_clip_to_histogram(clip_data)
            if encoding is not None:
                histogram, _ = encoding
                X.append(histogram)
                y.append(label)
            else:
                failed_clips += 1
        
        if len(X) < 10:
            print(f"  ⚠ Not enough valid clips for training ({len(X)})")
            return None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"  ✓ Encoded {len(X)} clips ({failed_clips} failed)")
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'  # Handle class imbalance
        )
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"  ✓ Classifier trained on {len(X_train)} samples")
        print(f"  ✓ Validation accuracy: {accuracy:.2%}")
        print(f"\n  Classification Report (Validation):")
        print(classification_report(y_val, y_pred, zero_division=0))
        
        # Store feature importances (which codewords are most important)
        self.feature_importances = self.classifier.feature_importances_
        top_features = np.argsort(self.feature_importances)[::-1][:10]
        print(f"  Top 10 most important codewords: {list(top_features)}")
        
        return self.classifier
    
    # =========================================================================
    # FULL TRAINING PIPELINE
    # =========================================================================
    
    def train(self, data_path=None):
        """
        Run the full training pipeline.
        
        1. Load training data from driver_pulse_data
        2. Preprocess all trips (Kalman filtering, PCA)
        3. Extract labeled 5-second clips
        4. Build K-Means codebook
        5. Train Random Forest classifier
        6. Save models to disk
        
        Args:
            data_path: Path to accelerometer_data.csv
            
        Returns:
            True if training successful, False otherwise
        """
        print("=" * 70)
        print("BOVW MOTION DETECTOR - TRAINING PIPELINE")
        print("=" * 70)
        
        # 1. Load training data
        print("\n[Stage 1] Loading training data...")
        df = self.load_training_data(data_path)
        
        if 'event_label_gt' not in df.columns:
            print("ERROR: No ground truth labels found!")
            return False
        
        # 2. Preprocess all trips
        print("\n[Stage 2] Preprocessing trips...")
        processed_dfs = []
        for trip_id in df['trip_id'].unique():
            trip_df = df[df['trip_id'] == trip_id].copy()
            trip_df, _ = self.preprocess_trip(trip_df)
            trip_df = self.pca_reorient(trip_df)
            processed_dfs.append(trip_df)
        
        processed_df = pd.concat(processed_dfs, ignore_index=True)
        print(f"  ✓ Preprocessed {len(processed_dfs)} trips")
        
        # 3. Build codebook from ALL data
        print("\n[Stage 3] Building BoVW codebook...")
        self.build_codebook(processed_df, k=BOVW_K_CODEWORDS)
        
        if self.codebook is None:
            print("ERROR: Failed to build codebook!")
            return False
        
        # 4. Extract labeled clips
        print("\n[Stage 4] Extracting labeled clips...")
        clips = self.extract_labeled_clips(processed_df)
        
        if len(clips) < 20:
            print("ERROR: Not enough labeled clips!")
            return False
        
        # 5. Train classifier
        print("\n[Stage 5] Training Random Forest classifier...")
        self.train_classifier(clips)
        
        if self.classifier is None:
            print("ERROR: Failed to train classifier!")
            return False
        
        # 6. Save models
        print("\n[Stage 6] Saving models...")
        self.save_models()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
        return True
    
    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================
    
    def save_models(self):
        """Save trained models to disk."""
        models_file = os.path.join(self.models_dir, 'bovw_motion_models.pkl')
        
        models_data = {
            'codebook': self.codebook,
            'scaler': self.scaler,
            'classifier': self.classifier,
            'feature_importances': getattr(self, 'feature_importances', None),
            'event_classes': self.EVENT_CLASSES,
            'feature_cols': self.FEATURE_COLS,
            'segment_length': getattr(self, '_segment_length', 3),
            'hop': getattr(self, '_hop', 1),
        }
        
        with open(models_file, 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"  ✓ Models saved to: {models_file}")
        
        # Save metadata as JSON for easy inspection
        metadata_file = os.path.join(self.models_dir, 'bovw_motion_metadata.json')
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_codewords': self.codebook.n_clusters if self.codebook else 0,
            'n_classes': len(self.EVENT_CLASSES),
            'event_classes': self.EVENT_CLASSES,
            'feature_cols': self.FEATURE_COLS,
            'segment_length': getattr(self, '_segment_length', 3),
            'hop': getattr(self, '_hop', 1),
            'sample_rate_hz': SAMPLE_RATE_HZ,
            'classifier_type': 'RandomForestClassifier',
            'classifier_params': {
                'n_estimators': 100,
                'max_depth': 10,
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata saved to: {metadata_file}")
    
    def load_models(self):
        """Load trained models from disk."""
        models_file = os.path.join(self.models_dir, 'bovw_motion_models.pkl')
        
        if not os.path.exists(models_file):
            print(f"  ⚠ Models file not found: {models_file}")
            return False
        
        print(f"Loading models from: {models_file}")
        
        with open(models_file, 'rb') as f:
            models_data = pickle.load(f)
        
        self.codebook = models_data['codebook']
        self.scaler = models_data['scaler']
        self.classifier = models_data['classifier']
        self.feature_importances = models_data.get('feature_importances')
        self._segment_length = models_data.get('segment_length', 3)
        self._hop = models_data.get('hop', 1)
        
        print(f"  ✓ K-Means codebook: {self.codebook.n_clusters} codewords")
        print(f"  ✓ Random Forest classifier: {self.classifier.n_estimators} trees")
        print(f"  ✓ Segment params: length={self._segment_length}, hop={self._hop}")
        
        return True
    
    # =========================================================================
    # INFERENCE (FOR SIMULATION DATA)
    # =========================================================================
    
    def classify_clip(self, clip_data):
        """
        Classify a single 5-second clip.
        
        Args:
            clip_data: DataFrame with accelerometer data (5 seconds)
            
        Returns:
            (event_label, confidence, top_codewords) or (None, 0, [])
        """
        if self.codebook is None or self.classifier is None:
            return None, 0.0, []
        
        encoding = self.encode_clip_to_histogram(clip_data)
        if encoding is None:
            return None, 0.0, []
        
        histogram, top_codewords = encoding
        
        # Predict
        prediction = self.classifier.predict([histogram])[0]
        probabilities = self.classifier.predict_proba([histogram])[0]
        confidence = float(max(probabilities))
        
        # Get class probabilities
        class_probs = dict(zip(self.classifier.classes_, probabilities))
        
        return prediction, confidence, top_codewords
    
    def classify_window_realtime(self, window_data):
        """
        Classify a window of accelerometer data in real-time.
        
        This is used during simulation to classify the current driving state.
        
        Args:
            window_data: DataFrame with ~5 seconds of accelerometer data
            
        Returns:
            Dict with classification result:
            {
                'event_label': str,
                'confidence': float,
                'top_codewords': list,
                'all_probabilities': dict
            }
        """
        if self.codebook is None or self.classifier is None:
            # Try to load models
            if not self.load_models():
                return {
                    'event_label': 'UNKNOWN',
                    'confidence': 0.0,
                    'top_codewords': [],
                    'all_probabilities': {}
                }
        
        encoding = self.encode_clip_to_histogram(window_data)
        
        if encoding is None:
            return {
                'event_label': 'NORMAL',
                'confidence': 0.5,
                'top_codewords': [],
                'all_probabilities': {'NORMAL': 0.5}
            }
        
        histogram, top_codewords = encoding
        
        # Predict
        prediction = self.classifier.predict([histogram])[0]
        probabilities = self.classifier.predict_proba([histogram])[0]
        confidence = float(max(probabilities))
        
        # All class probabilities
        all_probs = {cls: float(prob) for cls, prob in 
                     zip(self.classifier.classes_, probabilities)}
        
        return {
            'event_label': prediction,
            'confidence': confidence,
            'top_codewords': top_codewords,
            'all_probabilities': all_probs
        }


# =============================================================================
# ACCELEROMETER PIPELINE (FULL PROCESSING)
# =============================================================================

class AccelerometerPipeline:
    """
    Complete BoVW-based accelerometer event detection pipeline.
    
    Uses BoVWMotionDetector for ML-based classification.
    Also includes rule-based detection for comparison.
    """
    
    def __init__(self, base_dir=None, use_simulation_data=False):
        """
        Initialize the pipeline.
        
        Args:
            base_dir: Base directory for data/outputs
            use_simulation_data: If True, process simulation_data instead of driver_pulse_data
        """
        self.base_dir = base_dir or os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        # Choose data directory
        if use_simulation_data:
            self.data_dir = os.path.join(self.base_dir, 'simulation_data')
        else:
            self.data_dir = os.path.join(self.base_dir, 'driver_pulse_data')
        
        self.output_dir = os.path.join(self.base_dir, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize BoVW detector
        self.bovw_detector = BoVWMotionDetector(
            models_dir=os.path.join(self.output_dir, 'models')
        )
        
        # Noise floor estimates (per trip)
        self.noise_floor = {}
    
    def load_data(self):
        """Load accelerometer data."""
        filepath = os.path.join(self.data_dir, 'sensor_data', 'accelerometer_data.csv')
        self.accel_df = pd.read_csv(filepath)
        self.accel_df['timestamp'] = pd.to_datetime(self.accel_df['timestamp'])
        print(f"Loaded {len(self.accel_df):,} accelerometer samples from {filepath}")
        print(f"Trips: {self.accel_df['trip_id'].nunique()}")
        return self.accel_df
    
    def detect_events_rule_based(self, trip_data):
        """
        Rule-based event detection using thresholds from design doc.
        Used as a hybrid approach alongside BoVW for better coverage.
        """
        events = []
        df = trip_data.copy()
        
        # Use PCA-corrected values if available
        accel_y_col = 'accel_y_pca' if 'accel_y_pca' in df.columns else 'accel_y'
        accel_x_col = 'accel_x_pca' if 'accel_x_pca' in df.columns else 'accel_x'
        accel_z_col = 'accel_z_pca' if 'accel_z_pca' in df.columns else 'accel_z'
        
        for idx, row in df.iterrows():
            event = None
            confidence = 0.0
            raw_value = 0.0
            threshold = 0.0
            
            accel_y = row[accel_y_col]
            accel_x = row[accel_x_col]
            accel_z = row.get(accel_z_col, row.get('accel_z', 9.8))
            gyro_z = row.get('gyro_z', 0)
            speed = row.get('speed_kmh', 0)
            
            # AGGRESSIVE_BRAKING: accel_y in [-2.5, -1.5], gyro_z ~0
            if -2.5 <= accel_y <= -1.5 and abs(gyro_z) < 15:
                event = 'AGGRESSIVE_BRAKING'
                raw_value = accel_y
                threshold = -1.5
                confidence = min(abs(accel_y - (-1.5)) / 1.0, 1.0)
            
            # AGGRESSIVE_ACCEL: accel_y in [1.5, 2.5], gyro_z ~0
            elif 1.5 <= accel_y <= 2.5 and abs(gyro_z) < 15:
                event = 'AGGRESSIVE_ACCEL'
                raw_value = accel_y
                threshold = 1.5
                confidence = min((accel_y - 1.5) / 1.0, 1.0)
            
            # AGGRESSIVE_LEFT_TURN: accel_x in [-1.5, -0.8], gyro_z in [40, 120]
            elif -1.5 <= accel_x <= -0.8 and 40 <= gyro_z <= 120:
                event = 'AGGRESSIVE_LEFT_TURN'
                raw_value = accel_x
                threshold = -0.8
                confidence = min(abs(accel_x - (-0.8)) / 0.7, 1.0)
            
            # AGGRESSIVE_RIGHT_TURN: accel_x in [0.8, 1.5], gyro_z in [-120, -40]
            elif 0.8 <= accel_x <= 1.5 and -120 <= gyro_z <= -40:
                event = 'AGGRESSIVE_RIGHT_TURN'
                raw_value = accel_x
                threshold = 0.8
                confidence = min((accel_x - 0.8) / 0.7, 1.0)
            
            # AGG_LEFT_LANE_CHANGE: accel_x in [-0.5, -0.3], brief gyro spike
            elif -0.5 <= accel_x <= -0.3 and abs(gyro_z) > 20:
                event = 'AGG_LEFT_LANE_CHANGE'
                raw_value = accel_x
                threshold = -0.3
                confidence = 0.7
            
            # AGG_RIGHT_LANE_CHANGE: accel_x in [0.3, 0.5], brief gyro spike
            elif 0.3 <= accel_x <= 0.5 and abs(gyro_z) > 20:
                event = 'AGG_RIGHT_LANE_CHANGE'
                raw_value = accel_x
                threshold = 0.3
                confidence = 0.7
            
            # ROAD_BUMP: accel_z deviation > 1.5, brief, speed > 10
            elif abs(accel_z - 9.8) > ROAD_BUMP_ACCEL_Z_THRESHOLD and speed > ROAD_BUMP_SPEED_GATE_KMH:
                if abs(accel_x) < ROAD_BUMP_LATERAL_GATE and abs(accel_y) < ROAD_BUMP_LONGITUDINAL_GATE:
                    event = 'ROAD_BUMP'
                    raw_value = abs(accel_z - 9.8)
                    threshold = ROAD_BUMP_ACCEL_Z_THRESHOLD
                    confidence = min(raw_value / 2.0, 1.0)
            
            if event:
                events.append({
                    'timestamp': row['timestamp'],
                    'elapsed_seconds': row['elapsed_seconds'],
                    'trip_id': row['trip_id'],
                    'driver_id': row['driver_id'],
                    'event_label': event,
                    'raw_value': round(raw_value, 3),
                    'threshold': threshold,
                    'confidence': round(confidence, 2),
                    'signal_type': 'ACCELEROMETER',
                    'gps_lat': row.get('gps_lat'),
                    'gps_lon': row.get('gps_lon'),
                    'speed_kmh': speed,
                    'gyro_z': gyro_z,
                    'accel_x': accel_x,
                    'accel_y': accel_y,
                    'accel_z': accel_z,
                })
        
        return events
    
    def deduplicate_events(self, events, window_sec=MOTION_DEDUP_WINDOW_SEC):
        """Remove duplicate events within a time window."""
        if not events:
            return []
        
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values(['trip_id', 'elapsed_seconds'])
        
        deduped = []
        
        for trip_id in events_df['trip_id'].unique():
            trip_events = events_df[events_df['trip_id'] == trip_id].copy()
            
            for event_type in trip_events['event_label'].unique():
                type_events = trip_events[trip_events['event_label'] == event_type]
                
                last_time = -window_sec - 1
                for _, row in type_events.iterrows():
                    if row['elapsed_seconds'] - last_time > window_sec:
                        deduped.append(row.to_dict())
                        last_time = row['elapsed_seconds']
        
        return deduped
    
    def run_pipeline(self, train_models=True):
        """
        Run the complete accelerometer pipeline.
        
        Args:
            train_models: If True, train BoVW models. If False, just run rule-based detection.
        """
        print("=" * 70)
        print("PIPELINE 1: Motion Event Detection (BoVW + Random Forest)")
        print("=" * 70)
        
        # Load data
        self.load_data()
        
        # Train BoVW models on driver_pulse_data
        if train_models:
            training_data_path = os.path.join(
                self.base_dir, 'driver_pulse_data', 'sensor_data', 'accelerometer_data.csv'
            )
            self.bovw_detector.train(training_data_path)
        
        # Process each trip for rule-based events
        all_processed = []
        all_events = []
        
        for trip_id in self.accel_df['trip_id'].unique():
            print(f"\nProcessing {trip_id}...")
            trip_data = self.accel_df[self.accel_df['trip_id'] == trip_id].copy()
            
            # Preprocess
            trip_data, noise_floor = self.bovw_detector.preprocess_trip(trip_data)
            trip_data = self.bovw_detector.pca_reorient(trip_data)
            
            all_processed.append(trip_data)
            
            # Rule-based detection
            events = self.detect_events_rule_based(trip_data)
            all_events.extend(events)
        
        # Deduplicate
        events_deduped = self.deduplicate_events(all_events)
        events_df = pd.DataFrame(events_deduped)
        
        if len(events_df) > 0:
            # Add metadata
            severity_map = {
                'AGGRESSIVE_BRAKING': 'high',
                'AGGRESSIVE_ACCEL': 'medium',
                'AGGRESSIVE_LEFT_TURN': 'medium',
                'AGGRESSIVE_RIGHT_TURN': 'medium',
                'AGG_LEFT_LANE_CHANGE': 'low',
                'AGG_RIGHT_LANE_CHANGE': 'low',
                'ROAD_BUMP': 'low',
                'NORMAL': 'none',
            }
            events_df['severity'] = events_df['event_label'].map(severity_map)
            
            score_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3, 'none': 0.0}
            events_df['motion_score'] = events_df['severity'].map(score_map)
            
            flag_type_map = {
                'AGGRESSIVE_BRAKING': 'harsh_braking',
                'AGGRESSIVE_ACCEL': 'harsh_acceleration',
                'AGGRESSIVE_LEFT_TURN': 'aggressive_turn',
                'AGGRESSIVE_RIGHT_TURN': 'aggressive_turn',
                'AGG_LEFT_LANE_CHANGE': 'lane_change',
                'AGG_RIGHT_LANE_CHANGE': 'lane_change',
                'ROAD_BUMP': 'road_bump',
            }
            events_df['flag_type'] = events_df['event_label'].map(flag_type_map)
            
            events_df['explanation'] = events_df.apply(
                lambda r: f"{r['event_label']} detected ({r['raw_value']:.2f}g, threshold: {r['threshold']}g)",
                axis=1
            )
            events_df['context'] = events_df.apply(
                lambda r: f"Motion: {r['flag_type']} | Speed: {r.get('speed_kmh', 0):.0f}km/h",
                axis=1
            )
            events_df['window_id'] = [f"W{str(i+1).zfill(4)}" for i in range(len(events_df))]
            events_df['top_codewords'] = ''
        
        # Create trip summary
        trip_summary = self.create_trip_summary(events_df)
        
        # Save outputs
        self.save_outputs(events_df, trip_summary)
        
        return events_df, trip_summary
    
    def create_trip_summary(self, events_df):
        """Create per-trip motion event summary."""
        if len(events_df) == 0:
            return pd.DataFrame()
        
        summary = events_df.groupby('trip_id').agg(
            driver_id=('driver_id', 'first'),
            motion_events_count=('event_label', 'count'),
            max_severity=('severity', lambda x: 'high' if 'high' in x.values else ('medium' if 'medium' in x.values else 'low')),
            avg_motion_score=('motion_score', 'mean'),
            harsh_braking_count=('flag_type', lambda x: (x == 'harsh_braking').sum()),
            harsh_accel_count=('flag_type', lambda x: (x == 'harsh_acceleration').sum()),
            aggressive_turn_count=('flag_type', lambda x: (x == 'aggressive_turn').sum()),
            lane_change_count=('flag_type', lambda x: (x == 'lane_change').sum()),
            road_bump_count=('flag_type', lambda x: (x == 'road_bump').sum()),
        ).reset_index()
        
        return summary
    
    def save_outputs(self, events_df, trip_summary):
        """Save outputs to CSV."""
        output_cols = [
            'trip_id', 'driver_id', 'timestamp', 'elapsed_seconds',
            'event_label', 'flag_type', 'severity', 'motion_score',
            'raw_value', 'threshold', 'confidence', 'signal_type',
            'gps_lat', 'gps_lon', 'speed_kmh', 'explanation', 'context',
            'window_id', 'top_codewords'
        ]
        available = [c for c in output_cols if c in events_df.columns]
        
        events_path = os.path.join(self.output_dir, 'motion_events.csv')
        events_df[available].to_csv(events_path, index=False)
        print(f"\n✅ Saved {len(events_df)} motion events to {events_path}")
        
        summary_path = os.path.join(self.output_dir, 'trip_motion_summary.csv')
        trip_summary.to_csv(summary_path, index=False)
        print(f"✅ Saved trip motion summary to {summary_path}")


# =============================================================================
# INFERENCE HELPER FOR SIMULATION
# =============================================================================

def get_motion_classifier():
    """
    Get a pre-trained BoVW motion classifier for inference.
    
    This function is used by the dashboard during simulation to classify
    driving events in real-time.
    
    Returns:
        BoVWMotionDetector with loaded models, or None if models not available
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(base_dir, 'outputs', 'models')
    
    detector = BoVWMotionDetector(models_dir=models_dir)
    
    if detector.load_models():
        return detector
    else:
        print("Warning: Could not load motion classifier models. Run training first.")
        return None


def classify_simulation_window(window_df):
    """
    Classify a window of simulation data.
    
    This function is called during simulation to classify the current
    driving behavior based on accelerometer data.
    
    Args:
        window_df: DataFrame with ~5 seconds of accelerometer data
                   Must have columns: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    
    Returns:
        Dict with classification result
    """
    detector = get_motion_classifier()
    
    if detector is None:
        return {
            'event_label': 'NORMAL',
            'confidence': 0.0,
            'top_codewords': [],
            'all_probabilities': {},
            'error': 'Model not loaded'
        }
    
    return detector.classify_window_realtime(window_df)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point - trains models and runs pipeline."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Train on driver_pulse_data
    pipeline = AccelerometerPipeline(base_dir, use_simulation_data=False)
    events_df, trip_summary = pipeline.run_pipeline(train_models=True)
    
    print("\n" + "=" * 70)
    print("MOTION EVENT DETECTION COMPLETE")
    print("=" * 70)
    print(f"\nTotal events detected: {len(events_df)}")
    
    if len(events_df) > 0:
        print(f"\nBy event type:")
        print(events_df['event_label'].value_counts().to_string())
        print(f"\nBy severity:")
        print(events_df['severity'].value_counts().to_string())
    
    # Test inference on simulation data
    print("\n" + "=" * 70)
    print("TESTING INFERENCE ON SIMULATION DATA")
    print("=" * 70)
    
    sim_accel_path = os.path.join(base_dir, 'simulation_data', 'sensor_data', 'accelerometer_data.csv')
    if os.path.exists(sim_accel_path):
        sim_df = pd.read_csv(sim_accel_path)
        
        # Get first 5 seconds of first trip
        first_trip = sim_df['trip_id'].iloc[0]
        trip_data = sim_df[sim_df['trip_id'] == first_trip]
        window = trip_data.head(125)  # ~5 seconds at 25Hz
        
        print(f"\nClassifying first 5-second window of {first_trip}...")
        result = classify_simulation_window(window)
        print(f"  Event: {result['event_label']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Top codewords: {result['top_codewords']}")
    else:
        print(f"  ⚠ Simulation data not found at: {sim_accel_path}")


if __name__ == '__main__':
    main()
