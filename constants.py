"""
Driver Pulse — Global Constants
All threshold values and configuration parameters in one place.
"""

# =============================================================================
# ACCELEROMETER PIPELINE CONSTANTS (BoVW)
# =============================================================================

# Resampling
SAMPLE_RATE_HZ = 25  # Target sample rate
WINDOW_SIZE_SAMPLES = 12  # 0.5 seconds at 25 Hz (for BoVW segments)
WINDOW_HOP_SAMPLES = 6  # 50% overlap

# K-Means Codebook
BOVW_K_CODEWORDS = 64  # Number of codewords in codebook
BOVW_SEGMENT_LENGTH = 12  # Samples per segment (0.5s at 25Hz)
BOVW_CLIP_DURATION_SEC = 5  # Duration of clips for classification

# Self-Trigger Thresholds
TRIGGER_MULTIPLIER = 3.0  # Multiplier for noise floor
CONSECUTIVE_SAMPLES_TRIGGER = 3  # Samples above threshold to trigger

# PCA Reorientation
PCA_CALIBRATION_SECONDS = 30  # Seconds of data for PCA calibration

# Kalman Filter
KALMAN_PROCESS_NOISE = 0.05
KALMAN_MEASUREMENT_NOISE = 0.1

# Event Detection Thresholds (from design doc)
ACCEL_THRESHOLDS = {
    'AGGRESSIVE_BRAKING': {'axis': 'accel_y', 'min': -2.5, 'max': -1.5, 'gyro_z_max': 15},
    'AGGRESSIVE_ACCEL': {'axis': 'accel_y', 'min': 1.5, 'max': 2.5, 'gyro_z_max': 15},
    'AGGRESSIVE_LEFT_TURN': {'axis': 'accel_x', 'min': -1.5, 'max': -0.8, 'gyro_z_min': 40, 'gyro_z_max': 120},
    'AGGRESSIVE_RIGHT_TURN': {'axis': 'accel_x', 'min': 0.8, 'max': 1.5, 'gyro_z_min': -120, 'gyro_z_max': -40},
    'AGG_LEFT_LANE_CHANGE': {'axis': 'accel_x', 'min': -0.5, 'max': -0.3, 'gyro_z_spike': True, 'duration_max': 0.5},
    'AGG_RIGHT_LANE_CHANGE': {'axis': 'accel_x', 'min': 0.3, 'max': 0.5, 'gyro_z_spike': True, 'duration_max': 0.5},
    'NORMAL': {'accel_max': 0.3, 'gyro_max': 15},
}

# Road Bump Detection
ROAD_BUMP_ACCEL_Z_THRESHOLD = 1.5  # g deviation from gravity (~9.8)
ROAD_BUMP_MAX_DURATION_SAMPLES = 8  # at 25 Hz = 0.32s
ROAD_BUMP_SPEED_GATE_KMH = 10
ROAD_BUMP_LATERAL_GATE = 0.5  # |accel_x| < this
ROAD_BUMP_LONGITUDINAL_GATE = 0.5  # |accel_y| < this

# Deduplication
MOTION_DEDUP_WINDOW_SEC = 30

# Simulation Parameters (slow automatic simulation)
SIMULATION_STEP = 1.0  # 1% per update = 100 updates for full trip
SIMULATION_REFRESH_RATE = 1.5  # 1.5 seconds between updates (slow and visible)

# =============================================================================
# AUDIO PIPELINE CONSTANTS (4-Layer Detection)
# =============================================================================

# Layer 1 - Acoustic State (harsh/clean)
ACOUSTIC_ZCR_THRESHOLD = 0.55
ACOUSTIC_CENTROID_THRESHOLD = 2000  # Hz
ACOUSTIC_FLUX_MULTIPLIER = 2.0  # × baseline

# Layer 2 - Temporal State (fragmented/flowing)
TEMPORAL_GAP_MIN = 0.3  # seconds
TEMPORAL_GAP_MAX = 3.0  # seconds
TEMPORAL_GAP_RATIO_THRESHOLD = 0.85
TEMPORAL_ENERGY_SLOPE_THRESHOLD = 50  # dB/s

# Layer 3 - Prosodic State (aroused/relaxed)
PROSODIC_F0_STD_THRESHOLD = 40  # Hz
PROSODIC_SPEECH_RATE_MIN = 2.5  # syllables/sec
PROSODIC_SPEECH_RATE_MAX = 5.5  # syllables/sec

# Layer 4 - Context State (deviated/baseline)
CONTEXT_DB_DEVIATION_THRESHOLD = 12  # dB
CONTEXT_MAX_CONSECUTIVE_WINDOWS = 12  # 60 seconds at 5s windows

# Path A - Conflict Detection
CONFLICT_CONSECUTIVE_WINDOWS = 2  # 10 seconds sustained
CONFLICT_ACCEL_DELTA_THRESHOLD = 2.0  # m/s²
CONFLICT_ACCEL_WINDOW_SEC = 10

# Path B - Acute Safety Event
ACUTE_ACCEL_DELTA_THRESHOLD = 5.0  # m/s² (hard brake/impact)
ACUTE_DB_DEVIATION_THRESHOLD = 20  # dB above baseline
ACUTE_COINCIDENCE_WINDOW_SEC = 2

# Audio Deduplication
AUDIO_DEDUP_WINDOW_SEC = 180

# =============================================================================
# SIGNAL FUSION CONSTANTS
# =============================================================================

COMBINED_SCORE_MOTION_WEIGHT = 0.55
COMBINED_SCORE_AUDIO_WEIGHT = 0.45

COMPOUND_EVENT_WINDOW_SEC = 120  # 2 minutes

# Stress Score
STRESS_WEIGHT_ACCEL = 0.4
STRESS_WEIGHT_AUDIO = 0.3
STRESS_WEIGHT_COMPOUND = 0.3

STRESS_MODERATE_THRESHOLD = 0.40
STRESS_HIGH_THRESHOLD = 0.70

# Trip Quality Rating
STRESS_EXCELLENT_MAX = 0.10
STRESS_GOOD_MAX = 0.30
STRESS_AVERAGE_MAX = 0.50
STRESS_POOR_MAX = 0.70

# Hard-Brake Zone Detection
HARD_BRAKE_ZONE_MIN_EVENTS = 3
HARD_BRAKE_ZONE_RADIUS_M = 500
HARD_BRAKE_ZONE_GRID_PRECISION = 0.01  # ~1km grid cells

# Road Quality
ROAD_BUMP_SEGMENT_LENGTH_M = 200
ROAD_BUMP_SEGMENT_MIN_COUNT = 3

# =============================================================================
# EARNINGS FORECAST CONSTANTS (19-Step Pipeline)
# =============================================================================

# Experience Factor
EXPERIENCE_TRIPS_MAX = 500  # At 500+ trips, exp_factor = 1.0

# V_expected Weights (Step 13)
V_EXPECTED_DRIVER_WEIGHT = 0.4
V_EXPECTED_LOCATION_WEIGHT = 0.2
V_EXPECTED_OPPORTUNITY_WEIGHT = 0.2
V_EXPECTED_AREA_WEIGHT = 0.2

# Distance Factor (Step 11)
D_SCALE_KM = 10.0  # Reference distance at which value halves

# Goal Score Classification (Step 18)
GOAL_AHEAD_THRESHOLD = 1.1
GOAL_ON_TRACK_MIN = 0.9

# Velocity Smoothing
V_RECENT_WINDOW = 3  # Number of trips for rolling mean

# =============================================================================
# DASHBOARD CONSTANTS
# =============================================================================

COLD_START_MIN_ELAPSED_MIN = 15
BEST_ZONE_TOP_N = 3
RECOMMENDATIONS_MAX_PER_TRIP = 3

# H3 Grid Resolutions
H3_RESOLUTION_BRAKE_ZONES = 9  # ~0.1 km²
H3_RESOLUTION_MARKET = 7  # ~5 km²

# Simulation
SIMULATION_SPEED_MULTIPLIER = 1  # How fast to simulate (10x = 1 min trip in 6 sec)
TRIP_EVENT_INTERVAL_SEC = 1  # How often to update during simulation

# Add these to constants.py
SIMULATION_STEP = 2.0  # Percentage to move every refresh (e.g., 2%)
SIMULATION_REFRESH_RATE = 0.1  # Seconds between UI refreshes (0.1 = 10fps)

# =============================================================================
# UI THEME COLORS (Uber-style dark theme)
# =============================================================================

THEME = {
    'BG': '#0a0a0a',
    'CARD': '#1c1c1e',
    'CARD2': '#2c2c2e',
    'WHITE': '#ffffff',
    'GRAY': '#8e8e93',
    'GREEN': '#30d158',  # ahead
    'AMBER': '#ff9f0a',  # on_track
    'RED': '#ff453a',    # at_risk
    'BLUE': '#0a84ff',
    'TEAL': '#5ac8fa',
    'PURPLE': '#bf5af2',
}

# Severity Colors
SEVERITY_COLORS = {
    'high': '#ff453a',
    'medium': '#ff9f0a',
    'low': '#5ac8fa',
    'none': '#8e8e93',
}

# Event Type Colors for Map
EVENT_COLORS = {
    'AGGRESSIVE_BRAKING': '#ff453a',
    'HARSH_BRAKING': '#ff453a',
    'harsh_braking': '#ff453a',
    'AGGRESSIVE_ACCEL': '#ff9f0a',
    'HARSH_ACCELERATION': '#ff9f0a',
    'harsh_acceleration': '#ff9f0a',
    'AGGRESSIVE_LEFT_TURN': '#bf5af2',
    'AGGRESSIVE_RIGHT_TURN': '#bf5af2',
    'aggressive_turn': '#bf5af2',
    'AGG_LEFT_LANE_CHANGE': '#bf5af2',
    'AGG_RIGHT_LANE_CHANGE': '#bf5af2',
    'lane_change': '#bf5af2',
    'ROAD_BUMP': '#ffcc00',
    'road_bump': '#ffcc00',
    'NOISE_SPIKE': '#5ac8fa',
    'noise_spike': '#5ac8fa',
    'CONFLICT': '#ff453a',
    'conflict_moment': '#ff453a',
    'NORMAL': '#30d158',
}
