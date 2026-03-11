"""
Simulation Bridge - Dynamic Pipeline Trigger
=============================================
Connects the live Streamlit simulation to the backend ML pipelines.

When a driver completes a trip in the UI, this bridge:
1. Takes the live-detected events (MOTION from BoVW + AUDIO from 4-Layer)
2. ACCUMULATES motion_events.csv and audio_flagged_moments.csv
3. Runs Pipeline 3 (Signal Fusion) ONLY for completed trip(s)
4. Runs Pipeline 4 (Earnings Forecast) ONLY for completed trip(s)
5. All outputs → simulation_data/processed_outputs/

IMPORTANT: Both motion and audio events are detected LIVE during the
simulation (BoVW + Audio 4-Layer run in real-time). The bridge does
NOT re-run Pipeline 2 — it saves the live audio events directly.
"""

import pandas as pd
import numpy as np
import os
import sys
import shutil
import traceback
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Ensure project root is in path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def process_completed_trip(base_dir, trip_id, driver_id, live_events,
                           completed_trip_ids=None, trip_start_time=None):
    """
    Bridge between the live Streamlit simulation and the backend pipelines.

    Args:
        base_dir:            Project root directory
        trip_id:             Current trip ID (e.g. 'TRIP_DRV003_01')
        driver_id:           Driver ID (e.g. 'DRV003')
        live_events:         List of event dicts from st.session_state.detected_events
                             (contains BOTH MOTION and AUDIO events detected live)
        completed_trip_ids:  ALL trip IDs completed so far (including current).
        trip_start_time:     Trip start datetime (for timestamp construction)

    Returns:
        True if all pipelines ran successfully
    """
    sim_data_dir = os.path.join(base_dir, 'simulation_data')
    sim_out_dir = os.path.join(sim_data_dir, 'processed_outputs')
    os.makedirs(sim_out_dir, exist_ok=True)

    if completed_trip_ids is None:
        completed_trip_ids = [trip_id]

    # Split live events by signal type
    motion_events = [e for e in (live_events or [])
                     if e.get('signal_type') == 'MOTION']
    audio_events = [e for e in (live_events or [])
                    if e.get('signal_type') == 'AUDIO']

    print(f"\n{'='*60}")
    print(f"  PROCESSING TRIP: {trip_id}")
    print(f"  Live events: {len(motion_events)} motion, {len(audio_events)} audio")
    print(f"  All completed trips: {completed_trip_ids}")
    print(f"{'='*60}")

    success = True

    # ─── Step 1: Save live MOTION events (accumulate) ─────────────────
    print("\n[Step 1/4] Saving live ML motion events...")
    _save_motion_events(sim_out_dir, sim_data_dir, trip_id,
                        driver_id, motion_events, trip_start_time)

    # ─── Step 2: Save live AUDIO events (accumulate) ──────────────────
    print("\n[Step 2/4] Saving live audio events...")
    _save_audio_events(sim_out_dir, trip_id, driver_id,
                       audio_events, trip_start_time)

    # ─── Step 3: Signal Fusion (only completed trips) ─────────────────
    print("\n[Step 3/4] Running Signal Fusion Pipeline...")
    success &= _run_fusion_pipeline(base_dir, sim_data_dir, sim_out_dir,
                                     completed_trip_ids)

    # ─── Step 4: Earnings Forecast (only completed trips) ─────────────
    print("\n[Step 4/4] Running Earnings Forecast Pipeline...")
    success &= _run_earnings_pipeline(base_dir, sim_data_dir, sim_out_dir,
                                       completed_trip_ids)

    print(f"\n{'='*60}")
    status = "✅ COMPLETE" if success else "⚠️ FINISHED WITH WARNINGS"
    print(f"  {status} — {trip_id}")
    print(f"{'='*60}\n")

    return success


# =============================================================================
# Step 1: Format Live MOTION Events → motion_events.csv (ACCUMULATE)
# =============================================================================

def _save_motion_events(out_dir, data_dir, trip_id, driver_id,
                        motion_events, trip_start_time):
    """Save live BoVW motion events. Accumulates across completed trips."""
    # Load accelerometer data for GPS fallback
    accel_path = os.path.join(data_dir, 'sensor_data', 'accelerometer_data.csv')
    try:
        accel_df = pd.read_csv(accel_path)
    except Exception:
        accel_df = pd.DataFrame()

    new_records = []
    for evt in motion_events:
        elapsed = evt.get('elapsed_seconds', 0)
        ts = (trip_start_time + timedelta(seconds=elapsed)
              if trip_start_time else datetime.now())

        gps_lat = evt.get('gps_lat', None)
        gps_lon = evt.get('gps_lon', None)
        if (gps_lat is None or gps_lon is None) and len(accel_df) > 0:
            gps_lat, gps_lon = _lookup_gps(accel_df, trip_id, elapsed)

        new_records.append({
            'trip_id': trip_id,
            'driver_id': driver_id,
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'event_label': evt.get('event_label', 'UNKNOWN'),
            'flag_type': _event_to_flag_type(evt.get('event_label', '')),
            'severity': evt.get('severity', 'medium'),
            'motion_score': round(evt.get('confidence', 0.5), 2),
            'raw_value': 0,
            'threshold': 0,
            'confidence': round(evt.get('confidence', 0.5), 2),
            'signal_type': 'MOTION',
            'gps_lat': gps_lat,
            'gps_lon': gps_lon,
            'speed_kmh': evt.get('speed_kmh', 0),
            'explanation': evt.get('explanation', ''),
            'context': (f"Motion: {evt.get('event_label', '')} | "
                        f"Speed: {evt.get('speed_kmh', 0):.0f}km/h"),
            'window_id': f"W{int(elapsed):04d}",
            'top_codewords': '',
        })

    new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

    motion_cols = [
        'trip_id', 'driver_id', 'timestamp', 'elapsed_seconds',
        'event_label', 'flag_type', 'severity', 'motion_score',
        'raw_value', 'threshold', 'confidence', 'signal_type',
        'gps_lat', 'gps_lon', 'speed_kmh', 'explanation',
        'context', 'window_id', 'top_codewords',
    ]
    path = os.path.join(out_dir, 'motion_events.csv')
    existing = _load_existing_except_trip(path, trip_id)

    parts = [df for df in [existing, new_df] if len(df) > 0]
    motion_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=motion_cols)
    motion_df.to_csv(path, index=False)
    print(f"  ✅ {len(new_records)} motion events for {trip_id} "
          f"→ motion_events.csv ({len(motion_df)} total)")

    # Also build trip_motion_summary.csv
    _build_trip_summary(motion_df, out_dir, 'trip_motion_summary.csv',
                        'motion_score', 'avg_motion_score')


# =============================================================================
# Step 2: Format Live AUDIO Events → audio_flagged_moments.csv (ACCUMULATE)
# =============================================================================

def _save_audio_events(out_dir, trip_id, driver_id,
                       audio_events, trip_start_time):
    """Save live 4-Layer audio events. Accumulates across completed trips."""
    new_records = []
    for evt in audio_events:
        elapsed = evt.get('elapsed_seconds', 0)
        ts = (trip_start_time + timedelta(seconds=elapsed)
              if trip_start_time else datetime.now())

        new_records.append({
            'trip_id': trip_id,
            'driver_id': driver_id,
            'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed,
            'event_label': evt.get('event_label', 'STRESS_EVENT'),
            'flag_type': 'stress_event',
            'severity': evt.get('severity', 'medium'),
            'signal_type': 'AUDIO',
            'explanation': evt.get('explanation', ''),
            'stress_score': round(evt.get('confidence', 0.5) * 100, 1),
            'zcr': evt.get('zcr', 0),
            'f0_std': evt.get('f0_std', 0),
            'db_deviation': evt.get('db_deviation', 0),
            'layers_fired': evt.get('layers_fired', ''),
            'window_label': evt.get('event_label', 'STRESS_EVENT'),
        })

    new_df = pd.DataFrame(new_records) if new_records else pd.DataFrame()

    audio_cols = [
        'trip_id', 'driver_id', 'timestamp', 'elapsed_seconds',
        'event_label', 'flag_type', 'severity', 'signal_type',
        'explanation', 'stress_score', 'zcr', 'f0_std', 'db_deviation',
        'layers_fired', 'window_label',
    ]
    path = os.path.join(out_dir, 'audio_flagged_moments.csv')
    existing = _load_existing_except_trip(path, trip_id)

    parts = [df for df in [existing, new_df] if len(df) > 0]
    audio_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=audio_cols)
    audio_df.to_csv(path, index=False)
    print(f"  ✅ {len(new_records)} audio events for {trip_id} "
          f"→ audio_flagged_moments.csv ({len(audio_df)} total)")

    # Also build trip_audio_summary.csv
    if len(audio_df) > 0 and 'trip_id' in audio_df.columns:
        rows = []
        for tid in audio_df['trip_id'].unique():
            t = audio_df[audio_df['trip_id'] == tid]
            rows.append({
                'trip_id': tid,
                'stress_event_count': len(t),
                'conflict_count': len(t[t['event_label'].str.contains('CONFLICT', case=False, na=False)]),
                'acute_safety_count': len(t[t['event_label'].str.contains('ACUTE', case=False, na=False)]),
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, 'trip_audio_summary.csv'), index=False)
    else:
        pd.DataFrame(columns=[
            'trip_id', 'stress_event_count', 'conflict_count', 'acute_safety_count'
        ]).to_csv(os.path.join(out_dir, 'trip_audio_summary.csv'), index=False)

    # Create empty audio_processed.csv (not needed but prevents errors)
    pd.DataFrame(columns=['trip_id', 'driver_id', 'timestamp', 'elapsed_seconds']
                 ).to_csv(os.path.join(out_dir, 'audio_processed.csv'), index=False)


# =============================================================================
# Helpers
# =============================================================================

def _load_existing_except_trip(path, trip_id):
    """Load an existing CSV, removing rows for the given trip_id (idempotent re-run)."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if 'trip_id' in df.columns:
            df = df[df['trip_id'] != trip_id]
        return df
    except Exception:
        return pd.DataFrame()


def _build_trip_summary(events_df, out_dir, filename, score_col, avg_col):
    """Build a per-trip summary CSV from accumulated events."""
    if len(events_df) > 0 and 'trip_id' in events_df.columns:
        rows = []
        for tid in events_df['trip_id'].unique():
            t = events_df[events_df['trip_id'] == tid]
            sevs = t['severity'].values
            max_sev = ('high' if 'high' in sevs
                       else ('medium' if 'medium' in sevs else 'low'))
            rows.append({
                'trip_id': tid,
                'motion_events_count': len(t),
                'max_severity': max_sev,
                avg_col: round(t[score_col].mean(), 2) if score_col in t.columns else 0,
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, filename), index=False)
    else:
        pd.DataFrame(columns=[
            'trip_id', 'motion_events_count', 'max_severity', avg_col
        ]).to_csv(os.path.join(out_dir, filename), index=False)


def _lookup_gps(accel_df, trip_id, elapsed_seconds):
    """Look up GPS coordinates from accelerometer data for a given elapsed time."""
    trip_accel = accel_df[accel_df['trip_id'] == trip_id]
    if len(trip_accel) == 0:
        return None, None
    trip_accel = trip_accel.copy()
    trip_accel['time_diff'] = abs(trip_accel['elapsed_seconds'] - elapsed_seconds)
    closest = trip_accel.loc[trip_accel['time_diff'].idxmin()]
    return closest.get('gps_lat', None), closest.get('gps_lon', None)


def _event_to_flag_type(event_label):
    """Map BoVW event labels to flag_type strings used by downstream pipelines."""
    mapping = {
        'AGGRESSIVE_BRAKING': 'harsh_braking',
        'AGGRESSIVE_ACCEL': 'rapid_acceleration',
        'AGGRESSIVE_LEFT_TURN': 'harsh_turn',
        'AGGRESSIVE_RIGHT_TURN': 'harsh_turn',
        'AGG_LEFT_LANE_CHANGE': 'lane_change',
        'AGG_RIGHT_LANE_CHANGE': 'lane_change',
        'POTHOLE': 'road_bump',
        'SPEED_BUMP': 'road_bump',
        'ROAD_BUMP': 'road_bump',
    }
    return mapping.get(event_label, 'driving_event')


# =============================================================================
# Step 3: Signal Fusion — FILTERED to completed trips
# =============================================================================

def _run_fusion_pipeline(base_dir, data_dir, out_dir, completed_trip_ids):
    """Run Pipeline 3 (Signal Fusion) ONLY for completed trips."""
    try:
        from pipelines.pipeline3_signal_fusion import SignalFusionPipeline

        pipe = SignalFusionPipeline(base_dir)
        pipe.data_dir = data_dir
        pipe.output_dir = out_dir

        pipe.load_data()

        before_trips = len(pipe.trips_df)
        pipe.trips_df = pipe.trips_df[
            pipe.trips_df['trip_id'].isin(completed_trip_ids)
        ].copy()
        print(f"  ℹ️  Trips filtered: {before_trips} → {len(pipe.trips_df)} "
              f"(completed: {completed_trip_ids})")

        if len(pipe.motion_events) > 0 and 'trip_id' in pipe.motion_events.columns:
            pipe.motion_events = pipe.motion_events[
                pipe.motion_events['trip_id'].isin(completed_trip_ids)
            ].copy()

        if len(pipe.audio_flags) > 0 and 'trip_id' in pipe.audio_flags.columns:
            pipe.audio_flags = pipe.audio_flags[
                pipe.audio_flags['trip_id'].isin(completed_trip_ids)
            ].copy()

        print(f"  ℹ️  Events: {len(pipe.motion_events)} motion, "
              f"{len(pipe.audio_flags)} audio")

        fused_events = pipe.fuse_events()
        hard_brake_zones = pipe.detect_hard_brake_zones()
        road_quality_zones = pipe.detect_road_quality_issues()
        trip_summary = pipe.compute_trip_scores(fused_events)
        recommendations = pipe.generate_recommendations(
            trip_summary, hard_brake_zones, road_quality_zones)
        pipe.save_outputs(fused_events, hard_brake_zones, road_quality_zones,
                          trip_summary, recommendations)

        print(f"  ✅ Fusion Pipeline: {len(fused_events)} fused events, "
              f"{len(hard_brake_zones)} brake zones, "
              f"{len(trip_summary)} trip summaries")

        hbz = os.path.join(out_dir, 'hard_brake_zones.csv')
        dbz = os.path.join(out_dir, 'driver_brake_zones.csv')
        if os.path.exists(hbz):
            shutil.copy2(hbz, dbz)

        return True

    except Exception as e:
        print(f"  ⚠️ Fusion pipeline error: {e}")
        traceback.print_exc()
        return False


# =============================================================================
# Step 4: Earnings Forecast — FILTERED to completed trips
# =============================================================================

_ZONE_CENTROIDS = {
    'South Delhi':   (28.5500, 77.2200),
    'Central Delhi': (28.6400, 77.2200),
    'North Delhi':   (28.7100, 77.2000),
    'Noida':         (28.5700, 77.3500),
    'Gurgaon':       (28.4600, 77.0300),
}


def _create_compatible_market_context(sim_market_df):
    """Transform simulation market_context schema to Pipeline 4 format."""
    records = []
    demand_to_trips = {'low': 1.2, 'medium': 2.5, 'high': 4.0}

    for _, row in sim_market_df.iterrows():
        zone = row.get('zone', 'Central Delhi')
        lat, lon = _ZONE_CENTROIDS.get(zone, (28.6400, 77.2200))
        demand = str(row.get('demand_level', 'medium')).lower()
        surge = float(row.get('surge_multiplier', 1.0))
        supply = int(row.get('competitor_supply', 100))

        trips_per_hour = demand_to_trips.get(demand, 2.0) * surge
        mean_fare = 120 + (surge - 1.0) * 80
        peer_velocity = 280 * surge

        records.append({
            'h3_cell': f"h3_9_{lat:.2f}_{lon:.2f}",
            'city': 'Delhi',
            'area_centroid_lat': lat,
            'area_centroid_lon': lon,
            'trips_per_hour': round(trips_per_hour, 2),
            'mean_fare_inr': round(mean_fare, 2),
            'peer_velocity_mean': round(peer_velocity, 2),
            'peer_driver_count': supply,
            'snapshot_time': row.get('timestamp', '2024-02-06 08:00:00'),
        })

    return pd.DataFrame(records)


def _run_earnings_pipeline(base_dir, data_dir, out_dir, completed_trip_ids):
    """Run Pipeline 4 (Earnings Forecast) with trips FILTERED to completed only."""
    try:
        from pipelines.pipeline4_earnings_forecast import EarningsForecastPipeline

        pipe = EarningsForecastPipeline(base_dir)
        pipe.data_dir = data_dir
        pipe.output_dir = out_dir

        pipe.load_data()

        before = len(pipe.trips_df)
        pipe.trips_df = pipe.trips_df[
            pipe.trips_df['trip_id'].isin(completed_trip_ids)
        ].copy()
        print(f"  ℹ️  Trips filtered: {before} → {len(pipe.trips_df)} "
              f"(completed: {completed_trip_ids})")

        if (pipe.market_context is not None
                and 'city' not in pipe.market_context.columns):
            print("  ℹ️  Adapting market_context schema for Pipeline 4...")
            pipe.market_context = _create_compatible_market_context(
                pipe.market_context)
            pipe.market_context['timestamp'] = pd.to_datetime(
                pipe.market_context['snapshot_time'])

        forecasts = []
        for did in pipe.drivers_df['driver_id'].unique():
            driver_trips = pipe.trips_df[pipe.trips_df['driver_id'] == did]
            if len(driver_trips) > 0:
                current_time = driver_trips['start_time'].max()
            else:
                current_time = datetime.now()

            forecast = pipe.compute_forecast_for_driver(did, current_time)
            if forecast:
                forecasts.append(forecast)

        forecasts_df = pd.DataFrame(forecasts)
        pipe.save_outputs(forecasts_df)
        print(f"  ✅ Earnings Pipeline: Forecast for {len(forecasts_df)} driver(s)")
        return True

    except Exception as e:
        print(f"  ⚠️ Earnings pipeline error: {e}")
        traceback.print_exc()
        return False
