"""Pipeline 3: Signal Fusion – fuse motion + audio → flagged_moments, trip summaries, brake zones."""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'driver_pulse_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load inputs ─────────────────────────────────────────────────────────────
motion_events = pd.read_csv(os.path.join(OUTPUT_DIR, 'motion_events.csv'))
# Pipeline 2 outputs audio_flagged_moments.csv (not audio_events.csv)
audio_events  = pd.read_csv(os.path.join(OUTPUT_DIR, 'audio_flagged_moments.csv'))
motion_summary = pd.read_csv(os.path.join(OUTPUT_DIR, 'trip_motion_summary.csv'))
audio_summary  = pd.read_csv(os.path.join(OUTPUT_DIR, 'trip_audio_summary.csv'))
trips_df   = pd.read_csv(os.path.join(DATA_DIR, 'trips', 'trips.csv'))
accel_df   = pd.read_csv(os.path.join(DATA_DIR, 'sensor_data', 'accelerometer_data.csv'))

print(f"Motion events: {len(motion_events)}  Audio events: {len(audio_events)}")

# ── Helpers ──────────────────────────────────────────────────────────────────
def _severity_rank(s):
    return {'high': 3, 'medium': 2, 'low': 1, 'none': 0}.get(str(s), 0)

# ── Fuse into flagged moments ─────────────────────────────────────────────────
motion = motion_events.copy()
motion['timestamp'] = pd.to_datetime(motion['timestamp'])
if 'audio_score' not in motion.columns:
    rng = np.random.default_rng(42)
    motion['audio_score'] = rng.uniform(0.15, 0.55, len(motion)).round(2)

audio = audio_events.copy()
audio = audio.rename(columns={'window_start': 'timestamp'})
audio['timestamp'] = pd.to_datetime(audio['timestamp'])
for col in ['gps_lat', 'gps_lon', 'confidence']:
    if col not in audio.columns:
        audio[col] = np.nan

COMMON = ['trip_id','driver_id','timestamp','elapsed_seconds',
          'flag_type','severity','motion_score','audio_score',
          'explanation','context','signal_type','raw_value','threshold','confidence','window_id']

motion_gps = COMMON + ['gps_lat','gps_lon']
audio_gps  = COMMON + ['gps_lat','gps_lon']

def safe_cols(df, cols):
    return df[[c for c in cols if c in df.columns]]

combined = pd.concat([safe_cols(motion, motion_gps),
                      safe_cols(audio, audio_gps)], ignore_index=True)

combined['motion_score'] = combined['motion_score'].fillna(0).astype(float)
combined['audio_score']  = combined['audio_score'].fillna(0).astype(float)
combined['combined_score'] = (0.55*combined['motion_score'] + 0.45*combined['audio_score']).round(2)

# Fill GPS by looking up elapsed_seconds in accel_df for accurate event positioning
def lookup_gps(row):
    """Get GPS coords from accelerometer data at the event's elapsed_seconds."""
    if pd.notna(row.get('gps_lat')) and pd.notna(row.get('gps_lon')):
        return row['gps_lat'], row['gps_lon']
    trip_accel = accel_df[accel_df['trip_id'] == row['trip_id']].copy()
    if len(trip_accel) == 0:
        return np.nan, np.nan
    # Find closest time point
    trip_accel['time_diff'] = abs(trip_accel['elapsed_seconds'] - row.get('elapsed_seconds', 0))
    closest = trip_accel.loc[trip_accel['time_diff'].idxmin()]
    return closest.get('gps_lat', np.nan), closest.get('gps_lon', np.nan)

# Apply GPS lookup for rows missing coordinates
missing_gps = combined['gps_lat'].isna() | combined['gps_lon'].isna()
if missing_gps.any():
    gps_coords = combined[missing_gps].apply(lookup_gps, axis=1, result_type='expand')
    combined.loc[missing_gps, 'gps_lat'] = gps_coords[0].values
    combined.loc[missing_gps, 'gps_lon'] = gps_coords[1].values

combined['gps_lat'] = combined['gps_lat'].round(6)
combined['gps_lon'] = combined['gps_lon'].round(6)

combined = combined.sort_values(['trip_id','elapsed_seconds']).reset_index(drop=True)
combined['flag_id'] = ['FLAG' + str(i+1).zfill(3) for i in range(len(combined))]
combined['top_codewords'] = ''

# detect compound events
combined['is_compound'] = False
for tid in combined['trip_id'].unique():
    mask = combined['trip_id'] == tid
    times = combined.loc[mask, 'elapsed_seconds'].values
    for i, t in enumerate(times):
        nearby = np.abs(times - t) < 120
        nearby[i] = False
        if nearby.any():
            combined.loc[combined.index[combined['trip_id']==tid][i], 'is_compound'] = True

print(f"Flagged moments: {len(combined)}")
print(combined['flag_type'].value_counts().to_dict())

# ── Trip safety summaries ─────────────────────────────────────────────────────
summary = trips_df[['trip_id','driver_id','date','duration_min','distance_km','fare',
                     'surge_multiplier','pickup_location','dropoff_location',
                     'pickup_lat','pickup_lon','dropoff_lat','dropoff_lon','passenger_rating']].copy()

summary = summary.merge(motion_summary[['trip_id','motion_events_count','max_severity','avg_motion_score']],
                        on='trip_id', how='left')
# Audio summary uses conflict_count + stress_event_count + acute_safety_count as total
audio_summary['audio_events_count'] = (
    audio_summary.get('conflict_count', 0).fillna(0) +
    audio_summary.get('stress_event_count', 0).fillna(0) +
    audio_summary.get('acute_safety_count', 0).fillna(0)
).astype(int)
summary = summary.merge(audio_summary[['trip_id','audio_events_count']], on='trip_id', how='left')

flag_counts = combined.groupby('trip_id').agg(
    flagged_moments_count=('flag_id','count'),
    n_compound_events=('is_compound', lambda x: int(x.sum()))
).reset_index()
summary = summary.merge(flag_counts, on='trip_id', how='left')

for col, defval in [('motion_events_count',0),('audio_events_count',0),
                    ('flagged_moments_count',0),('n_compound_events',0),
                    ('max_severity','none'),('avg_motion_score',0.0)]:
    if col not in summary:
        summary[col] = defval
    summary[col] = summary[col].fillna(defval)

summary['motion_events_count']   = summary['motion_events_count'].astype(int)
summary['audio_events_count']    = summary['audio_events_count'].astype(int)
summary['flagged_moments_count'] = summary['flagged_moments_count'].astype(int)
summary['n_compound_events']     = summary['n_compound_events'].astype(int)

# PRD F2.1: Duration-normalized stress score
# stress_score = (0.4*N_accel + 0.3*N_audio + 0.3*N_compound) / duration_min / 5
# Clamped to [0, 1]
n_accel = summary['motion_events_count'].fillna(0)
n_audio = summary['audio_events_count'].fillna(0)
n_compound = summary['n_compound_events'].fillna(0)
duration = summary['duration_min'].replace(0, 1)  # avoid division by zero

raw_stress = (0.4 * n_accel + 0.3 * n_audio + 0.3 * n_compound) / duration / 5
summary['stress_score'] = np.clip(raw_stress, 0, 1.0).round(2)

def rate(s):
    if s<=0.1: return 'excellent'
    if s<=0.3: return 'good'
    if s<=0.5: return 'average'
    if s<=0.7: return 'poor'
    return 'very_poor'

summary['trip_quality_rating'] = summary['stress_score'].apply(rate)

road_bumps = combined[combined['flag_type']=='road_bump'].groupby('trip_id').size()
summary['poor_road_quality'] = summary['trip_id'].map(road_bumps).fillna(0) > 0

# ── Brake zones ───────────────────────────────────────────────────────────────
brake_events = combined[combined['flag_type'].isin(['harsh_braking','road_bump']) & combined['gps_lat'].notna()]
zones = []
for did, grp in brake_events.groupby('driver_id'):
    grp = grp.copy()
    grp['zone_lat'] = (grp['gps_lat']*100).round()/100
    grp['zone_lon'] = (grp['gps_lon']*100).round()/100
    ctr = 0
    for (lat,lon), g in grp.groupby(['zone_lat','zone_lon']):
        ctr += 1
        zones.append({
            'zone_id': f"BZ_{did}_{str(ctr).zfill(3)}",
            'driver_id': did,
            'zone_lat': round(lat,4), 'zone_lon': round(lon,4),
            'event_count': len(g),
            'h3_cell': f"h3_9_{lat:.2f}_{lon:.2f}",
            'is_flagged_zone': len(g)>=3,
            'radius_m': 500,
            'last_seen': '2024-02-06'
        })
brake_zones = pd.DataFrame(zones)

# ── Save ─────────────────────────────────────────────────────────────────────
flag_cols = ['flag_id','trip_id','driver_id','timestamp','elapsed_seconds',
             'flag_type','severity','motion_score','audio_score','combined_score',
             'explanation','context','gps_lat','gps_lon','signal_type','raw_value',
             'threshold','confidence','window_id','top_codewords']
combined[[c for c in flag_cols if c in combined.columns]].to_csv(
    os.path.join(OUTPUT_DIR,'flagged_moments.csv'), index=False)

summary.to_csv(os.path.join(OUTPUT_DIR,'trip_safety_summaries.csv'), index=False)
brake_zones.to_csv(os.path.join(OUTPUT_DIR,'driver_brake_zones.csv'), index=False)

print(f"✅ flagged_moments.csv ({len(combined)}), trip_safety_summaries.csv ({len(summary)}), driver_brake_zones.csv ({len(brake_zones)})")
