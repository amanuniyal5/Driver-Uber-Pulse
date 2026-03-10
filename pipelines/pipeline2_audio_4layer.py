"""
Pipeline 2: Audio-Based Conflict/Stress Detection (4-Layer System)
===================================================================
Implements the complete audio processing pipeline as specified:

Layer 1 (Acoustic Features):
- ZCR (Zero-Crossing Rate) > 0.55
- Spectral Centroid > 2000 Hz
- Spectral Flux > 2× baseline

Layer 2 (Temporal Dynamics):
- turn_gap_sec < 1.5s (rushed turn-taking)
- gap_ratio < 0.6 (imbalanced conversation)
- energy_slope > 0.3 dB/sec (escalation)

Layer 3 (Prosodic Markers):
- f0_std > 40 Hz (pitch variability)
- speech_rate outside [2.5, 5.5] syl/sec

Layer 4 (Context Gate):
- db_deviation > 12 dB above baseline (conversation happening)

Decision Rules:
- Path A (Conflict): 2+ consecutive windows with L1+L2+L4, OR 1 window + motion event
- Path B (Acute Safety): accel > 5.0 m/s² AND db > 20 dB deviation within 2s

Output:
- conflict_detected (boolean)
- stress_event (boolean)  
- acute_safety_event (boolean)
- stress_score (0-100)
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import constants - add parent dir to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from constants import (
    ACOUSTIC_ZCR_THRESHOLD, ACOUSTIC_CENTROID_THRESHOLD, ACOUSTIC_FLUX_MULTIPLIER,
    TEMPORAL_GAP_MIN, TEMPORAL_GAP_MAX, TEMPORAL_GAP_RATIO_THRESHOLD, TEMPORAL_ENERGY_SLOPE_THRESHOLD,
    PROSODIC_F0_STD_THRESHOLD, PROSODIC_SPEECH_RATE_MIN, PROSODIC_SPEECH_RATE_MAX,
    CONTEXT_DB_DEVIATION_THRESHOLD, ACUTE_DB_DEVIATION_THRESHOLD,
    CONFLICT_CONSECUTIVE_WINDOWS, AUDIO_DEDUP_WINDOW_SEC
)

# Alias for pipeline code
LAYER1_ZCR_THRESHOLD = ACOUSTIC_ZCR_THRESHOLD
LAYER1_CENTROID_THRESHOLD = ACOUSTIC_CENTROID_THRESHOLD
LAYER1_FLUX_MULTIPLIER = ACOUSTIC_FLUX_MULTIPLIER
FLUX_BASELINE = 0.02  # Baseline for normal speech (conflict ~0.06-0.10)

# Layer 2 Thresholds (PDF v4.0 compliant)
# Fragmented = gap < 0.3s OR gap > 3.0s, gap_ratio > 0.85, slope > 50 dB/s
LAYER2_TURN_GAP_MIN = TEMPORAL_GAP_MIN      # 0.3s - rushed turn-taking
LAYER2_TURN_GAP_MAX = TEMPORAL_GAP_MAX      # 3.0s - stunned silence
LAYER2_GAP_RATIO_THRESHOLD = TEMPORAL_GAP_RATIO_THRESHOLD  # 0.85 - one party dominating
LAYER2_ENERGY_SLOPE_THRESHOLD = TEMPORAL_ENERGY_SLOPE_THRESHOLD  # 50 dB/s - escalation

LAYER3_F0_STD_THRESHOLD = PROSODIC_F0_STD_THRESHOLD
LAYER3_SPEECH_RATE_LOW = PROSODIC_SPEECH_RATE_MIN
LAYER3_SPEECH_RATE_HIGH = PROSODIC_SPEECH_RATE_MAX
LAYER4_DB_DEVIATION_THRESHOLD = CONTEXT_DB_DEVIATION_THRESHOLD
LAYER4_FLIP_BACK_WINDOWS = 12  # 60 seconds at 5s windows - prevents singing false positives
PATH_B_DB_THRESHOLD = ACUTE_DB_DEVIATION_THRESHOLD
PATH_A_PERSISTENCE_WINDOWS = CONFLICT_CONSECUTIVE_WINDOWS
MOTION_AUDIO_FUSION_WINDOW_SEC = 2


class AudioPipeline:
    """
    4-Layer Audio Detection Pipeline for conflict and stress detection.
    """
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'driver_pulse_data')
        self.output_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data
        self.audio_df = None
        self.motion_events_df = None
        
    def load_data(self):
        """Load audio features and motion events."""
        # Audio features
        audio_path = os.path.join(self.data_dir, 'sensor_data', 'audio_features.csv')
        self.audio_df = pd.read_csv(audio_path)
        # Use window_start as timestamp if timestamp column doesn't exist
        if 'timestamp' not in self.audio_df.columns:
            self.audio_df['timestamp'] = pd.to_datetime(self.audio_df['window_start'])
        else:
            self.audio_df['timestamp'] = pd.to_datetime(self.audio_df['timestamp'])
        
        print(f"Loaded {len(self.audio_df)} audio samples")
        print(f"Columns: {list(self.audio_df.columns)}")
        
        # Try to load motion events from Pipeline 1
        motion_path = os.path.join(self.output_dir, 'motion_events.csv')
        if os.path.exists(motion_path):
            self.motion_events_df = pd.read_csv(motion_path)
            self.motion_events_df['timestamp'] = pd.to_datetime(self.motion_events_df['timestamp'])
            print(f"Loaded {len(self.motion_events_df)} motion events")
        else:
            print("No motion events found (Pipeline 1 not run yet)")
            self.motion_events_df = pd.DataFrame()
        
        return self.audio_df
    
    # =========================================================================
    # LAYER 1: Acoustic Features
    # =========================================================================
    
    def layer1_acoustic(self, row):
        """
        Layer 1: Acoustic Feature Detection
        - ZCR > 0.55 (raised voice signature)
        - Spectral Centroid > 2000 Hz (harsh tones)
        - Spectral Flux > 2× baseline (rapid changes)
        """
        zcr = row.get('zcr', 0)
        centroid = row.get('spectral_centroid', 0)
        flux = row.get('spectral_flux', 0)
        
        # ZCR check
        zcr_flag = zcr > LAYER1_ZCR_THRESHOLD
        
        # Centroid check
        centroid_flag = centroid > LAYER1_CENTROID_THRESHOLD
        
        # Flux check (> 2× baseline, baseline ~ 0.5)
        flux_flag = flux > LAYER1_FLUX_MULTIPLIER * FLUX_BASELINE
        
        # Layer 1 fires if ALL conditions met
        layer1_active = zcr_flag and centroid_flag and flux_flag
        
        return {
            'layer1_active': layer1_active,
            'layer1_zcr': zcr_flag,
            'layer1_centroid': centroid_flag,
            'layer1_flux': flux_flag,
            'zcr': zcr,
            'centroid': centroid,
            'flux': flux,
        }
    
    # =========================================================================
    # LAYER 2: Temporal Dynamics
    # =========================================================================
    
    def layer2_temporal(self, row):
        """
        Layer 2: Temporal Dynamics Detection (PDF v4.0 Compliant)
        "Fragmented" state indicates abnormal turn-taking:
        - turn_gap_sec < 0.3s (rushed/interrupted) OR > 3.0s (stunned silence)
        - gap_ratio > 0.85 (one party dominating conversation)
        - energy_slope > 50 dB/s (rapid escalation)
        """
        turn_gap = row.get('turn_gap_sec', 1.5)  # Default to normal mid-range
        gap_ratio = row.get('gap_ratio', 0.5)    # Default to balanced
        energy_slope = row.get('energy_slope', 0)
        
        # Turn gap check: fragmented if too fast (<0.3s) OR too slow (>3.0s)
        turn_gap_flag = (turn_gap < LAYER2_TURN_GAP_MIN) or (turn_gap > LAYER2_TURN_GAP_MAX)
        
        # Gap ratio check: one party dominating (>85%)
        gap_ratio_flag = gap_ratio > LAYER2_GAP_RATIO_THRESHOLD
        
        # Energy slope check: rapid escalation (>50 dB/s)
        energy_slope_flag = energy_slope > LAYER2_ENERGY_SLOPE_THRESHOLD
        
        # Layer 2 (Fragmented) fires if ANY 2 conditions met
        active_count = sum([turn_gap_flag, gap_ratio_flag, energy_slope_flag])
        layer2_active = active_count >= 2
        
        return {
            'layer2_active': layer2_active,
            'layer2_turn_gap': turn_gap_flag,
            'layer2_gap_ratio': gap_ratio_flag,
            'layer2_energy_slope': energy_slope_flag,
            'turn_gap': turn_gap,
            'gap_ratio': gap_ratio,
            'energy_slope': energy_slope,
        }
    
    # =========================================================================
    # LAYER 3: Prosodic Markers
    # =========================================================================
    
    def layer3_prosodic(self, row):
        """
        Layer 3: Prosodic Markers Detection
        - f0_std > 40 Hz (high pitch variability indicates stress)
        - speech_rate outside [2.5, 5.5] syl/sec (abnormal pace)
        """
        f0_std = row.get('f0_std', 20)      # Default to normal
        speech_rate = row.get('speech_rate', 4.0)  # Default to normal
        
        # Pitch variability check
        f0_flag = f0_std > LAYER3_F0_STD_THRESHOLD
        
        # Speech rate check (too fast or too slow)
        speech_rate_flag = (speech_rate < LAYER3_SPEECH_RATE_LOW or 
                          speech_rate > LAYER3_SPEECH_RATE_HIGH)
        
        # Layer 3 fires if ANY condition met
        layer3_active = f0_flag or speech_rate_flag
        
        return {
            'layer3_active': layer3_active,
            'layer3_f0_std': f0_flag,
            'layer3_speech_rate': speech_rate_flag,
            'f0_std': f0_std,
            'speech_rate': speech_rate,
        }
    
    # =========================================================================
    # LAYER 4: Context Gate
    # =========================================================================
    
    def layer4_context(self, row, consecutive_high_windows=0):
        """
        Layer 4: Context Gate (PDF v4.0 Compliant with Flip-Back Rule)
        - db_deviation > 12 dB above baseline
        - Indicates actual conversation is happening
        - FLIP-BACK RULE: If audio stays high for >12 windows (60s), 
          reset to False to prevent sustained excitement (singing) false positives
        """
        db_level = row.get('db_level', 50)
        baseline_db = row.get('baseline_db', 55)
        db_deviation = row.get('db_deviation', db_level - baseline_db)
        
        # Context gate check
        raw_gate_open = db_deviation > LAYER4_DB_DEVIATION_THRESHOLD
        
        # Apply flip-back rule: if open for >12 consecutive windows, force close
        if raw_gate_open and consecutive_high_windows >= LAYER4_FLIP_BACK_WINDOWS:
            layer4_active = False  # Flip back - sustained excitement (singing, podcast, etc.)
            flip_back_triggered = True
        else:
            layer4_active = raw_gate_open
            flip_back_triggered = False
        
        return {
            'layer4_active': layer4_active,
            'layer4_raw': raw_gate_open,
            'layer4_flip_back': flip_back_triggered,
            'db_deviation': db_deviation,
            'db_level': db_level,
            'baseline_db': baseline_db,
        }
    
    # =========================================================================
    # Detection Paths
    # =========================================================================
    
    def check_path_a_conflict(self, window_results, motion_in_window):
        """
        Path A: Conflict Detection
        - 2+ consecutive windows with L1+L2+L4 active
        - OR: 1 window with L1+L2+L4 + motion event within 2s
        """
        # Check if current window has L1+L2+L4
        l1 = window_results.get('layer1_active', False)
        l2 = window_results.get('layer2_active', False)
        l4 = window_results.get('layer4_active', False)
        
        audio_positive = l1 and l2 and l4
        
        # Path A triggers with audio + motion
        if audio_positive and motion_in_window:
            return True, 'conflict_with_motion'
        
        return audio_positive, 'audio_positive' if audio_positive else None
    
    def check_path_b_acute(self, row, motion_in_window):
        """
        Path B: Acute Safety Event (bypasses persistence)
        - Acceleration > 5.0 m/s²
        - db_deviation > 20 dB within 2s
        """
        db_deviation = row.get('db_deviation', 0)
        
        # Check acute conditions
        high_db = db_deviation > PATH_B_DB_THRESHOLD
        
        if high_db and motion_in_window:
            return True
        
        return False
    
    # =========================================================================
    # Stress Score Computation
    # =========================================================================
    
    def compute_stress_score(self, window_results):
        """
        Compute stress score (0-100) based on layer activations.
        
        Weights:
        - Layer 1 (Acoustic): 30%
        - Layer 2 (Temporal): 25%
        - Layer 3 (Prosodic): 25%
        - Layer 4 (Context): 20%
        """
        score = 0.0
        
        # Layer 1 contribution
        if window_results.get('layer1_active', False):
            score += 30
        elif window_results.get('layer1_zcr', False) or window_results.get('layer1_centroid', False):
            score += 15
        
        # Layer 2 contribution
        if window_results.get('layer2_active', False):
            score += 25
        else:
            l2_count = sum([
                window_results.get('layer2_turn_gap', False),
                window_results.get('layer2_gap_ratio', False),
                window_results.get('layer2_energy_slope', False),
            ])
            score += l2_count * 8
        
        # Layer 3 contribution
        if window_results.get('layer3_active', False):
            score += 25
        elif window_results.get('layer3_f0_std', False):
            score += 15
        elif window_results.get('layer3_speech_rate', False):
            score += 10
        
        # Layer 4 contribution (context)
        if window_results.get('layer4_active', False):
            score += 20
        
        return min(score, 100)
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    def process_trip(self, trip_data, trip_motion_events):
        """
        Process a single trip through all 4 layers with PDF v4.0 Decision Table.
        
        Decision Table (Page 6):
        ========================
        | L4(Context) | L1(Harsh) | L2(Frag) | L3(Aroused) | → Label      |
        |-------------|-----------|----------|-------------|--------------|
        | False       | *         | *        | *           | NORMAL       |
        | True        | True      | True     | *           | CONFLICT     |
        | True        | True      | False    | True        | AMBIGUOUS    |
        | True        | False     | True     | True        | AMBIGUOUS    |
        | True        | *         | *        | False       | NORMAL_LOUD  |
        
        Path A triggers on: CONFLICT label with 2-window persistence OR motion fusion
        """
        results = []
        
        # Track consecutive positive windows for Path A
        consecutive_positives = 0
        
        # Track consecutive high-volume windows for L4 flip-back rule
        consecutive_high_windows = 0
        
        for idx, row in trip_data.iterrows():
            window_results = {}
            
            # Run all 4 layers
            layer1 = self.layer1_acoustic(row)
            layer2 = self.layer2_temporal(row)
            layer3 = self.layer3_prosodic(row)
            layer4 = self.layer4_context(row, consecutive_high_windows)
            
            window_results.update(layer1)
            window_results.update(layer2)
            window_results.update(layer3)
            window_results.update(layer4)
            
            # Update flip-back counter
            if layer4.get('layer4_raw', False):
                consecutive_high_windows += 1
            else:
                consecutive_high_windows = 0
            
            # Check for motion events within 2s window
            motion_in_window = False
            if len(trip_motion_events) > 0 and 'elapsed_seconds' in row:
                elapsed = row['elapsed_seconds']
                motion_mask = (
                    (trip_motion_events['elapsed_seconds'] >= elapsed - MOTION_AUDIO_FUSION_WINDOW_SEC) &
                    (trip_motion_events['elapsed_seconds'] <= elapsed + MOTION_AUDIO_FUSION_WINDOW_SEC)
                )
                motion_in_window = motion_mask.any()
            
            # ================================================================
            # DECISION TABLE LOGIC (PDF v4.0, Page 6)
            # ================================================================
            is_harsh = window_results.get('layer1_active', False)       # L1: Acoustic
            is_fragmented = window_results.get('layer2_active', False)  # L2: Temporal
            is_aroused = window_results.get('layer3_active', False)     # L3: Prosodic
            is_deviated = window_results.get('layer4_active', False)    # L4: Context Gate
            
            # Context Gate acts as primary filter
            if not is_deviated:
                window_label = "NORMAL"
            elif is_harsh and is_fragmented:
                # Harsh + Fragmented (+ any L3 state) = CONFLICT
                window_label = "CONFLICT"
            elif (is_harsh and is_aroused) or (is_fragmented and is_aroused):
                # Only one of L1/L2 active, but with arousal = AMBIGUOUS
                window_label = "AMBIGUOUS"
            else:
                # Deviated but no conflict indicators
                window_label = "NORMAL_LOUD"
            
            # ================================================================
            # PATH A: Conflict Detection with Persistence
            # ================================================================
            is_conflict_window = (window_label == "CONFLICT")
            
            if is_conflict_window:
                consecutive_positives += 1
            else:
                consecutive_positives = 0
            
            # Conflict confirmed with 2-window persistence OR motion override
            conflict_detected = (
                consecutive_positives >= PATH_A_PERSISTENCE_WINDOWS or 
                (is_conflict_window and motion_in_window)
            )
            
            # ================================================================
            # PATH B: Acute Safety Event (bypasses all)
            # ================================================================
            acute_safety = self.check_path_b_acute(row, motion_in_window)
            
            # Stress event = conflict, acute, or high arousal
            stress_event = conflict_detected or acute_safety or (is_aroused and is_deviated)
            
            # Compute stress score
            stress_score = self.compute_stress_score(window_results)
            
            # Build layers_fired string for explainability
            layers_fired = []
            if is_harsh:
                layers_fired.append('A')  # Acoustic
            if is_fragmented:
                layers_fired.append('T')  # Temporal
            if is_aroused:
                layers_fired.append('P')  # Prosodic
            if is_deviated:
                layers_fired.append('C')  # Context
            layers_fired_str = ','.join(layers_fired) if layers_fired else 'None'
            
            results.append({
                'timestamp': row['timestamp'],
                'elapsed_seconds': row.get('elapsed_seconds', 0),
                'trip_id': row['trip_id'],
                'driver_id': row['driver_id'],
                
                # Layer activations
                'layer1_active': window_results['layer1_active'],
                'layer2_active': window_results['layer2_active'],
                'layer3_active': window_results['layer3_active'],
                'layer4_active': window_results['layer4_active'],
                
                # Raw values
                'zcr': window_results.get('zcr', 0),
                'spectral_centroid': window_results.get('centroid', 0),
                'spectral_flux': window_results.get('flux', 0),
                'f0_std': window_results.get('f0_std', 0),
                'speech_rate': window_results.get('speech_rate', 0),
                'db_deviation': window_results.get('db_deviation', 0),
                
                # Decision table label
                'window_label': window_label,
                'layers_fired': layers_fired_str,
                
                # Outputs
                'conflict_detected': conflict_detected,
                'stress_event': stress_event,
                'acute_safety_event': acute_safety,
                'stress_score': stress_score,
                
                # Additional context
                'motion_in_window': motion_in_window,
                'consecutive_positives': consecutive_positives,
                'flip_back_triggered': window_results.get('layer4_flip_back', False),
            })
        
        return pd.DataFrame(results)
    
    def generate_flagged_moments(self, processed_df):
        """Generate flagged moments from audio detections with layers_fired explainability."""
        flagged = []
        
        # Filter to actual detections
        detections = processed_df[
            processed_df['conflict_detected'] | 
            processed_df['stress_event'] | 
            processed_df['acute_safety_event']
        ].copy()
        
        for idx, row in detections.iterrows():
            # Determine flag type and severity based on Decision Table label
            window_label = row.get('window_label', 'UNKNOWN')
            
            if row['acute_safety_event']:
                flag_type = 'acute_safety_event'
                severity = 'critical'
                event_label = 'ACUTE_SAFETY'
            elif row['conflict_detected']:
                flag_type = 'passenger_conflict'
                severity = 'high'
                event_label = 'CONFLICT_DETECTED'
            elif window_label == 'AMBIGUOUS':
                flag_type = 'stress_event'
                severity = 'medium'
                event_label = 'STRESS_AMBIGUOUS'
            else:
                flag_type = 'stress_event'
                severity = 'medium'
                event_label = 'STRESS_DETECTED'
            
            # Generate explanation with layers_fired
            explanation = self._generate_explanation(row)
            layers_fired = row.get('layers_fired', '')
            
            flagged.append({
                'trip_id': row['trip_id'],
                'driver_id': row['driver_id'],
                'timestamp': row['timestamp'],
                'elapsed_seconds': row['elapsed_seconds'],
                'signal_type': 'AUDIO',
                'event_label': event_label,
                'window_label': window_label,
                'flag_type': flag_type,
                'severity': severity,
                'stress_score': row['stress_score'],
                'explanation': explanation,
                'layers_fired': layers_fired,  # e.g. "A,T,P,C"
                'layer1_active': row['layer1_active'],
                'layer2_active': row['layer2_active'],
                'layer3_active': row['layer3_active'],
                'layer4_active': row['layer4_active'],
                'zcr': row['zcr'],
                'spectral_centroid': row['spectral_centroid'],
                'f0_std': row['f0_std'],
                'db_deviation': row['db_deviation'],
            })
        
        return pd.DataFrame(flagged)
    
    def _generate_explanation(self, row):
        """Generate human-readable explanation for detection."""
        parts = []
        
        if row['layer1_active']:
            parts.append(f"Raised voice detected (ZCR={row['zcr']:.2f})")
        
        if row['layer2_active']:
            parts.append("Rushed conversation pattern")
        
        if row['layer3_active']:
            parts.append(f"Stress markers (pitch variability={row['f0_std']:.1f}Hz)")
        
        if row['layer4_active']:
            parts.append(f"Elevated audio ({row['db_deviation']:.1f}dB above baseline)")
        
        if row.get('acute_safety_event', False):
            parts.append("⚠️ ACUTE SAFETY EVENT")
        
        return " | ".join(parts) if parts else "Audio anomaly detected"
    
    def deduplicate_flags(self, flagged_df, window_sec=AUDIO_DEDUP_WINDOW_SEC):
        """Remove duplicate flags within time window.
        
        Keep only the HIGHEST severity event per time window:
        - acute_safety_event > stress_event
        """
        if len(flagged_df) == 0:
            return flagged_df
        
        # Severity ranking (higher = more severe, kept first)
        severity_rank = {'acute_safety_event': 3, 'passenger_conflict': 2, 'stress_event': 1}
        
        deduped = []
        
        for trip_id in flagged_df['trip_id'].unique():
            trip_flags = flagged_df[flagged_df['trip_id'] == trip_id].copy()
            trip_flags = trip_flags.sort_values('elapsed_seconds')
            
            # Track last flagged time across ALL event types
            last_time = -window_sec - 1
            
            # Sort by elapsed_seconds then by severity (most severe first)
            trip_flags['_severity_rank'] = trip_flags['flag_type'].map(severity_rank).fillna(0)
            trip_flags = trip_flags.sort_values(['elapsed_seconds', '_severity_rank'], ascending=[True, False])
            
            for _, row in trip_flags.iterrows():
                if row['elapsed_seconds'] - last_time > window_sec:
                    deduped.append(row.to_dict())
                    last_time = row['elapsed_seconds']
        
        result = pd.DataFrame(deduped)
        if '_severity_rank' in result.columns:
            result = result.drop(columns=['_severity_rank'])
        return result
    
    def create_trip_summary(self, flagged_df, processed_df):
        """Create per-trip audio summary."""
        summaries = []
        
        for trip_id in processed_df['trip_id'].unique():
            trip_processed = processed_df[processed_df['trip_id'] == trip_id]
            trip_flags = flagged_df[flagged_df['trip_id'] == trip_id] if len(flagged_df) > 0 else pd.DataFrame()
            
            summary = {
                'trip_id': trip_id,
                'driver_id': trip_processed['driver_id'].iloc[0],
                'avg_stress_score': trip_processed['stress_score'].mean(),
                'max_stress_score': trip_processed['stress_score'].max(),
                'conflict_count': len(trip_flags[trip_flags['flag_type'] == 'passenger_conflict']) if len(trip_flags) > 0 else 0,
                'stress_event_count': len(trip_flags[trip_flags['flag_type'] == 'stress_event']) if len(trip_flags) > 0 else 0,
                'acute_safety_count': len(trip_flags[trip_flags['flag_type'] == 'acute_safety_event']) if len(trip_flags) > 0 else 0,
                'layer1_triggers': trip_processed['layer1_active'].sum(),
                'layer2_triggers': trip_processed['layer2_active'].sum(),
                'layer3_triggers': trip_processed['layer3_active'].sum(),
                'layer4_triggers': trip_processed['layer4_active'].sum(),
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def run_pipeline(self):
        """Run the complete audio pipeline."""
        print("=" * 60)
        print("PIPELINE 2: Audio-Based Conflict/Stress Detection")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        all_processed = []
        
        # Process each trip
        for trip_id in self.audio_df['trip_id'].unique():
            print(f"\nProcessing {trip_id}...")
            trip_data = self.audio_df[self.audio_df['trip_id'] == trip_id].copy()
            
            # Get motion events for this trip
            if self.motion_events_df is not None and len(self.motion_events_df) > 0:
                trip_motion = self.motion_events_df[self.motion_events_df['trip_id'] == trip_id]
            else:
                trip_motion = pd.DataFrame()
            
            # Process through all layers
            processed = self.process_trip(trip_data, trip_motion)
            all_processed.append(processed)
        
        # Combine all processed data
        processed_df = pd.concat(all_processed, ignore_index=True)
        
        # Generate flagged moments
        flagged_df = self.generate_flagged_moments(processed_df)
        
        # Deduplicate
        flagged_deduped = self.deduplicate_flags(flagged_df)
        
        # Create trip summary
        trip_summary = self.create_trip_summary(flagged_deduped, processed_df)
        
        # Save outputs
        self.save_outputs(processed_df, flagged_deduped, trip_summary)
        
        return processed_df, flagged_deduped, trip_summary
    
    def save_outputs(self, processed_df, flagged_df, trip_summary):
        """Save outputs to CSV."""
        # Save full processed data
        processed_path = os.path.join(self.output_dir, 'audio_processed.csv')
        processed_df.to_csv(processed_path, index=False)
        print(f"\n✅ Saved {len(processed_df)} processed audio windows to {processed_path}")
        
        # Save flagged moments
        flagged_path = os.path.join(self.output_dir, 'audio_flagged_moments.csv')
        flagged_df.to_csv(flagged_path, index=False)
        print(f"✅ Saved {len(flagged_df)} audio flags to {flagged_path}")
        
        # Save trip summary
        summary_path = os.path.join(self.output_dir, 'trip_audio_summary.csv')
        trip_summary.to_csv(summary_path, index=False)
        print(f"✅ Saved trip audio summary to {summary_path}")


def main():
    """Main entry point."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pipeline = AudioPipeline(base_dir)
    processed_df, flagged_df, trip_summary = pipeline.run_pipeline()
    
    print("\n" + "=" * 60)
    print("AUDIO DETECTION COMPLETE")
    print("=" * 60)
    print(f"\nTotal audio windows: {len(processed_df)}")
    print(f"Flagged moments: {len(flagged_df)}")
    
    if len(flagged_df) > 0:
        print(f"\nBy flag type:")
        print(flagged_df['flag_type'].value_counts().to_string())
        print(f"\nBy severity:")
        print(flagged_df['severity'].value_counts().to_string())
    
    print(f"\nTrip summary:")
    print(trip_summary.to_string())


if __name__ == '__main__':
    main()
