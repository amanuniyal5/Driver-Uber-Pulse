"""
Pipeline 3: Signal Fusion - Combining Motion + Audio Events
============================================================
Fuses outputs from Pipeline 1 (Motion) and Pipeline 2 (Audio) to:
- Detect correlated events (motion + audio within 2s)
- Identify hard-brake zones
- Detect road quality issues (potholes)
- Generate unified flagged moments
- Compute trip-level stress scores
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
    HARD_BRAKE_ZONE_MIN_EVENTS, HARD_BRAKE_ZONE_GRID_PRECISION
)

# Local constants
FUSION_CORRELATION_WINDOW_SEC = 2
HARD_BRAKE_ZONE_PRECISION = 3  # decimal places for GPS rounding
ROAD_QUALITY_ZONE_PRECISION = 3  # decimal places
SAFETY_SCORE_CRITICAL_PENALTY = 15
SAFETY_SCORE_MOTION_PENALTY = 5
SAFETY_SCORE_AUDIO_PENALTY = 3


class SignalFusionPipeline:
    """
    Fuses motion and audio signals for comprehensive event detection.
    """
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'driver_pulse_data')
        self.output_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data
        self.motion_events = None
        self.audio_flags = None
        self.trips_df = None
        self.accel_df = None
        
    def load_data(self):
        """Load outputs from Pipeline 1 & 2 and raw data."""
        # Motion events from Pipeline 1
        motion_path = os.path.join(self.output_dir, 'motion_events.csv')
        if os.path.exists(motion_path):
            self.motion_events = pd.read_csv(motion_path)
            self.motion_events['timestamp'] = pd.to_datetime(self.motion_events['timestamp'])
            print(f"Loaded {len(self.motion_events)} motion events")
        else:
            print("⚠️ No motion events found - run Pipeline 1 first")
            self.motion_events = pd.DataFrame()
        
        # Audio flags from Pipeline 2
        audio_path = os.path.join(self.output_dir, 'audio_flagged_moments.csv')
        if os.path.exists(audio_path):
            self.audio_flags = pd.read_csv(audio_path)
            self.audio_flags['timestamp'] = pd.to_datetime(self.audio_flags['timestamp'])
            print(f"Loaded {len(self.audio_flags)} audio flags")
        else:
            print("⚠️ No audio flags found - run Pipeline 2 first")
            self.audio_flags = pd.DataFrame()
        
        # Raw trip data
        trips_path = os.path.join(self.data_dir, 'trips', 'trips.csv')
        self.trips_df = pd.read_csv(trips_path)
        print(f"Loaded {len(self.trips_df)} trips")
        
        # Raw accelerometer data for GPS coordinates
        accel_path = os.path.join(self.data_dir, 'sensor_data', 'accelerometer_data.csv')
        self.accel_df = pd.read_csv(accel_path)
        self.accel_df['timestamp'] = pd.to_datetime(self.accel_df['timestamp'])
        
        return self.motion_events, self.audio_flags
    
    # =========================================================================
    # Event Fusion
    # =========================================================================
    
    def fuse_events(self):
        """
        Fuse motion and audio events within 2-second windows.
        Creates correlated events with higher severity.
        """
        if len(self.motion_events) == 0 and len(self.audio_flags) == 0:
            print("No events to fuse")
            return pd.DataFrame()
        
        fused_events = []
        
        # Process each trip
        all_trips = set()
        if len(self.motion_events) > 0:
            all_trips.update(self.motion_events['trip_id'].unique())
        if len(self.audio_flags) > 0:
            all_trips.update(self.audio_flags['trip_id'].unique())
        
        for trip_id in all_trips:
            # Get motion events for trip
            trip_motion = self.motion_events[self.motion_events['trip_id'] == trip_id] if len(self.motion_events) > 0 else pd.DataFrame()
            # Get audio flags for trip
            trip_audio = self.audio_flags[self.audio_flags['trip_id'] == trip_id] if len(self.audio_flags) > 0 else pd.DataFrame()
            
            # Track which events are correlated
            correlated_motion_idx = set()
            correlated_audio_idx = set()
            
            # Find correlations
            for m_idx, m_row in trip_motion.iterrows():
                m_time = m_row['elapsed_seconds']
                
                for a_idx, a_row in trip_audio.iterrows():
                    a_time = a_row['elapsed_seconds']
                    
                    if abs(m_time - a_time) <= FUSION_CORRELATION_WINDOW_SEC:
                        # Found correlation
                        correlated_motion_idx.add(m_idx)
                        correlated_audio_idx.add(a_idx)
                        
                        # Create fused event
                        fused_events.append({
                            'trip_id': trip_id,
                            'driver_id': m_row.get('driver_id', a_row.get('driver_id')),
                            'timestamp': m_row['timestamp'],
                            'elapsed_seconds': m_time,
                            'signal_type': 'FUSED',
                            'event_label': 'CORRELATED_EVENT',
                            'flag_type': 'motion_audio_correlation',
                            'severity': 'critical',
                            'motion_event': m_row.get('event_label'),
                            'audio_event': a_row.get('event_label'),
                            'motion_score': m_row.get('motion_score', 0),
                            'stress_score': a_row.get('stress_score', 0),
                            'gps_lat': m_row.get('gps_lat'),
                            'gps_lon': m_row.get('gps_lon'),
                            'speed_kmh': m_row.get('speed_kmh'),
                            'explanation': f"Correlated: {m_row.get('event_label')} + {a_row.get('event_label')}",
                        })
            
            # Add uncorrelated motion events
            for m_idx, m_row in trip_motion.iterrows():
                if m_idx not in correlated_motion_idx:
                    fused_events.append({
                        'trip_id': trip_id,
                        'driver_id': m_row.get('driver_id'),
                        'timestamp': m_row['timestamp'],
                        'elapsed_seconds': m_row['elapsed_seconds'],
                        'signal_type': 'MOTION',
                        'event_label': m_row.get('event_label'),
                        'flag_type': m_row.get('flag_type'),
                        'severity': m_row.get('severity', 'medium'),
                        'motion_event': m_row.get('event_label'),
                        'audio_event': None,
                        'motion_score': m_row.get('motion_score', 0),
                        'stress_score': 0,
                        'gps_lat': m_row.get('gps_lat'),
                        'gps_lon': m_row.get('gps_lon'),
                        'speed_kmh': m_row.get('speed_kmh'),
                        'explanation': m_row.get('explanation'),
                    })
            
            # Add uncorrelated audio events
            for a_idx, a_row in trip_audio.iterrows():
                if a_idx not in correlated_audio_idx:
                    fused_events.append({
                        'trip_id': trip_id,
                        'driver_id': a_row.get('driver_id'),
                        'timestamp': a_row['timestamp'],
                        'elapsed_seconds': a_row['elapsed_seconds'],
                        'signal_type': 'AUDIO',
                        'event_label': a_row.get('event_label'),
                        'flag_type': a_row.get('flag_type'),
                        'severity': a_row.get('severity', 'medium'),
                        'motion_event': None,
                        'audio_event': a_row.get('event_label'),
                        'motion_score': 0,
                        'stress_score': a_row.get('stress_score', 0),
                        'gps_lat': None,
                        'gps_lon': None,
                        'speed_kmh': None,
                        'explanation': a_row.get('explanation'),
                    })
        
        return pd.DataFrame(fused_events)
    
    # =========================================================================
    # Hard-Brake Zone Detection
    # =========================================================================
    
    def detect_hard_brake_zones(self):
        """
        Identify locations where multiple harsh braking events occur.
        Uses H3 Resolution-9 hexagons (~0.1 km²) per PRD F3.1.
        """
        if len(self.motion_events) == 0:
            return pd.DataFrame()
        
        # Filter to harsh braking events
        brake_events = self.motion_events[
            self.motion_events['flag_type'] == 'harsh_braking'
        ].copy()
        
        if len(brake_events) == 0:
            return pd.DataFrame()
        
        # Try to use H3 hexagons (PRD F3.1 compliant)
        try:
            import h3
            H3_AVAILABLE = True
        except ImportError:
            H3_AVAILABLE = False
            print("  ⚠️ h3 library not available, falling back to coordinate rounding")
        
        if H3_AVAILABLE:
            # Use H3 Resolution-9 cells (~0.1 km² hexagons)
            brake_events['h3_cell'] = brake_events.apply(
                lambda x: h3.latlng_to_cell(x['gps_lat'], x['gps_lon'], 9) 
                if pd.notna(x['gps_lat']) and pd.notna(x['gps_lon']) else None,
                axis=1
            )
            
            # Remove events without valid H3 cells
            brake_events = brake_events[brake_events['h3_cell'].notna()]
            
            if len(brake_events) == 0:
                return pd.DataFrame()
            
            # Aggregate by H3 cell
            zones = brake_events.groupby('h3_cell').agg(
                event_count=('trip_id', 'count'),
                trips=('trip_id', 'nunique'),
                drivers=('driver_id', 'nunique'),
                avg_speed=('speed_kmh', 'mean'),
                max_decel=('raw_value', 'min'),
                lat_zone=('gps_lat', 'mean'),  # Centroid for display
                lon_zone=('gps_lon', 'mean'),
            ).reset_index()
        else:
            # Fallback: Round GPS coordinates to create zones
            brake_events['lat_zone'] = brake_events['gps_lat'].round(HARD_BRAKE_ZONE_PRECISION)
            brake_events['lon_zone'] = brake_events['gps_lon'].round(HARD_BRAKE_ZONE_PRECISION)
            
            zones = brake_events.groupby(['lat_zone', 'lon_zone']).agg(
                event_count=('trip_id', 'count'),
                trips=('trip_id', 'nunique'),
                drivers=('driver_id', 'nunique'),
                avg_speed=('speed_kmh', 'mean'),
                max_decel=('raw_value', 'min'),
            ).reset_index()
        
        # Filter to zones with multiple events
        hard_brake_zones = zones[zones['event_count'] >= HARD_BRAKE_ZONE_MIN_EVENTS].copy()
        
        # Classify zone severity
        def classify_severity(row):
            if row['event_count'] >= 5 or row['max_decel'] < -2.0:
                return 'critical'
            elif row['event_count'] >= 3:
                return 'high'
            else:
                return 'medium'
        
        hard_brake_zones['severity'] = hard_brake_zones.apply(classify_severity, axis=1)
        hard_brake_zones['zone_id'] = [f"HBZ{str(i+1).zfill(3)}" for i in range(len(hard_brake_zones))]
        
        return hard_brake_zones
    
    # =========================================================================
    # Road Quality Detection
    # =========================================================================
    
    def detect_road_quality_issues(self):
        """
        Identify road bumps/potholes from motion data.
        Uses H3 Resolution-9 hexagons per PRD F3.1.
        """
        if len(self.motion_events) == 0:
            return pd.DataFrame()
        
        # Filter to road bump events
        road_bumps = self.motion_events[
            self.motion_events['flag_type'] == 'road_bump'
        ].copy()
        
        if len(road_bumps) == 0:
            return pd.DataFrame()
        
        # Try to use H3 hexagons
        try:
            import h3
            H3_AVAILABLE = True
        except ImportError:
            H3_AVAILABLE = False
        
        if H3_AVAILABLE:
            # Use H3 Resolution-9 cells
            road_bumps['h3_cell'] = road_bumps.apply(
                lambda x: h3.latlng_to_cell(x['gps_lat'], x['gps_lon'], 9)
                if pd.notna(x['gps_lat']) and pd.notna(x['gps_lon']) else None,
                axis=1
            )
            road_bumps = road_bumps[road_bumps['h3_cell'].notna()]
            
            if len(road_bumps) == 0:
                return pd.DataFrame()
            
            zones = road_bumps.groupby('h3_cell').agg(
                bump_count=('trip_id', 'count'),
                trips=('trip_id', 'nunique'),
                avg_severity=('raw_value', 'mean'),
                lat_zone=('gps_lat', 'mean'),
                lon_zone=('gps_lon', 'mean'),
            ).reset_index()
        else:
            # Fallback: Round GPS to create zones
            road_bumps['lat_zone'] = road_bumps['gps_lat'].round(ROAD_QUALITY_ZONE_PRECISION)
            road_bumps['lon_zone'] = road_bumps['gps_lon'].round(ROAD_QUALITY_ZONE_PRECISION)
            
            zones = road_bumps.groupby(['lat_zone', 'lon_zone']).agg(
                bump_count=('trip_id', 'count'),
                trips=('trip_id', 'nunique'),
                avg_severity=('raw_value', 'mean'),
            ).reset_index()
        
        # Score road quality (lower = worse)
        def road_quality_score(row):
            base_score = 100
            base_score -= row['bump_count'] * 10
            base_score -= row['avg_severity'] * 5
            return max(base_score, 0)
        
        zones['road_quality_score'] = zones.apply(road_quality_score, axis=1)
        zones['zone_id'] = [f"RQZ{str(i+1).zfill(3)}" for i in range(len(zones))]
        
        return zones
    
    # =========================================================================
    # Trip-Level Scoring
    # =========================================================================
    
    def compute_trip_scores(self, fused_events):
        """
        Compute comprehensive trip-level scores using PRD F2.1 formula.
        
        PRD Formula F2.1:
        stress_score = (0.4 × N_accel + 0.3 × N_audio + 0.3 × N_compound) / duration_min
        
        This normalizes by trip duration to fairly compare short vs long trips.
        """
        if len(fused_events) == 0:
            # Return empty summary for all trips
            return pd.DataFrame([{
                'trip_id': row['trip_id'],
                'driver_id': row['driver_id'],
                'total_events': 0,
                'critical_events': 0,
                'n_accel_events': 0,
                'n_audio_events': 0,
                'n_compound_events': 0,
                'stress_score': 0.0,
                'overall_safety_score': 100,
                'risk_level': 'low',
                'poor_road_quality': False,
            } for _, row in self.trips_df.iterrows()])
        
        summaries = []
        
        for trip_id in self.trips_df['trip_id'].unique():
            trip_events = fused_events[fused_events['trip_id'] == trip_id]
            trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
            
            # 1. Gather counts for PRD Formula F2.1
            n_accel = len(trip_events[trip_events['signal_type'] == 'MOTION'])
            n_audio = len(trip_events[trip_events['signal_type'] == 'AUDIO'])
            n_compound = len(trip_events[trip_events['signal_type'] == 'FUSED'])
            critical_events = len(trip_events[trip_events['severity'] == 'critical'])
            total_events = len(trip_events)
            
            # Get trip duration in minutes
            duration = trip_info.get('duration_min', 30)  # Default 30 min if missing
            if duration <= 0:
                duration = 30  # Safeguard
            
            # 2. PRD Formula F2.1 Implementation
            # Weights: Accel=0.4, Audio=0.3, Compound=0.3
            # Scale factor of 10 to normalize for typical event counts
            # A trip with 10 accel events + 5 audio events in 30 min = (0.4*10 + 0.3*5) / 30 / 10 = 0.018
            # After min/max clamping, this gives us a 0-1 range where ~0.1 is concerning
            raw_stress = (0.4 * n_accel + 0.3 * n_audio + 0.3 * n_compound) / duration
            # Normalize: typical trips have 5-50 events/hour, so divide by expected max
            # A very bad trip might have 300 events in 30 min = 600/hr -> raw_stress = 4.0
            # We want that to map to ~0.8-1.0
            normalized_stress = raw_stress / 5.0  # Scale so 5 events/min = 1.0 
            stress_score = round(min(max(normalized_stress, 0), 1), 2)
            
            # 3. Derive Safety Score for UI (0-100)
            # Higher stress = lower safety
            safety_score = round((1 - stress_score) * 100, 1)
            
            # 4. Determine Risk Level based on stress_score thresholds
            if stress_score < 0.40:
                risk_level = 'low'
            elif stress_score < 0.70:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            # 5. Check road quality (PRD F3.3 - 3+ bumps = poor road)
            road_bump_count = len(trip_events[trip_events['event_label'] == 'ROAD_BUMP'])
            poor_road_quality = road_bump_count >= 3
            
            summaries.append({
                'trip_id': trip_id,
                'driver_id': trip_info['driver_id'],
                'total_events': total_events,
                'critical_events': critical_events,
                'n_accel_events': n_accel,
                'n_audio_events': n_audio,
                'n_compound_events': n_compound,
                'stress_score': stress_score,  # PRD F2.1 normalized score (0-1)
                'overall_safety_score': safety_score,  # For UI (0-100)
                'risk_level': risk_level,
                'poor_road_quality': poor_road_quality,
            })
        
        return pd.DataFrame(summaries)
    
    # =========================================================================
    # Recommendations
    # =========================================================================
    
    def generate_recommendations(self, trip_summary, hard_brake_zones, road_quality_zones):
        """
        Generate actionable recommendations based on analysis.
        Uses PRD-compliant column names.
        """
        recommendations = []
        
        for _, trip in trip_summary.iterrows():
            trip_recs = []
            
            if trip['risk_level'] in ['high', 'critical']:
                trip_recs.append({
                    'type': 'safety',
                    'priority': 'high',
                    'message': f"Trip {trip['trip_id']} had elevated risk. Consider taking a break.",
                })
            
            # Use new PRD-compliant column names
            n_accel = trip.get('n_accel_events', 0)
            n_audio = trip.get('n_audio_events', 0)
            n_compound = trip.get('n_compound_events', 0)
            
            if n_accel > 5:
                trip_recs.append({
                    'type': 'driving_style',
                    'priority': 'medium',
                    'message': "Multiple harsh driving events detected. Smoother driving recommended.",
                })
            
            if n_audio > 3:
                trip_recs.append({
                    'type': 'wellbeing',
                    'priority': 'medium',
                    'message': "Elevated stress patterns detected. Consider de-escalation techniques.",
                })
            
            if n_compound > 0:
                trip_recs.append({
                    'type': 'correlation',
                    'priority': 'high',
                    'message': f"{n_compound} events where driving and stress coincided. These require attention.",
                })
            
            if trip.get('poor_road_quality', False):
                trip_recs.append({
                    'type': 'route',
                    'priority': 'low',
                    'message': "Road quality issues detected on this route. Consider alternate paths.",
                })
            
            for rec in trip_recs:
                rec['trip_id'] = trip['trip_id']
                rec['driver_id'] = trip['driver_id']
                recommendations.append(rec)
        
        # Add zone-based recommendations
        if len(hard_brake_zones) > 0:
            for _, zone in hard_brake_zones[hard_brake_zones['severity'] == 'critical'].iterrows():
                recommendations.append({
                    'trip_id': 'ALL',
                    'driver_id': 'ALL',
                    'type': 'route',
                    'priority': 'high',
                    'message': f"Hard-brake zone at ({zone['lat_zone']}, {zone['lon_zone']}). Consider alternate route.",
                })
        
        return pd.DataFrame(recommendations)
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def run_pipeline(self):
        """Run the complete signal fusion pipeline."""
        print("=" * 60)
        print("PIPELINE 3: Signal Fusion")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Fuse events
        print("\nFusing motion and audio events...")
        fused_events = self.fuse_events()
        print(f"Created {len(fused_events)} fused events")
        
        # Detect hard-brake zones
        print("\nDetecting hard-brake zones...")
        hard_brake_zones = self.detect_hard_brake_zones()
        print(f"Found {len(hard_brake_zones)} hard-brake zones")
        
        # Detect road quality issues
        print("\nDetecting road quality issues...")
        road_quality_zones = self.detect_road_quality_issues()
        print(f"Found {len(road_quality_zones)} road quality zones")
        
        # Compute trip scores
        print("\nComputing trip scores...")
        trip_summary = self.compute_trip_scores(fused_events)
        
        # Generate recommendations
        print("\nGenerating recommendations...")
        recommendations = self.generate_recommendations(trip_summary, hard_brake_zones, road_quality_zones)
        print(f"Generated {len(recommendations)} recommendations")
        
        # Save outputs
        self.save_outputs(fused_events, hard_brake_zones, road_quality_zones, trip_summary, recommendations)
        
        return fused_events, hard_brake_zones, trip_summary, recommendations
    
    def save_outputs(self, fused_events, hard_brake_zones, road_quality_zones, trip_summary, recommendations):
        """Save all outputs."""
        # Fused events (main flagged moments)
        if len(fused_events) > 0:
            fused_path = os.path.join(self.output_dir, 'flagged_moments.csv')
            fused_events.to_csv(fused_path, index=False)
            print(f"\n✅ Saved {len(fused_events)} flagged moments to {fused_path}")
        
        # Hard-brake zones
        if len(hard_brake_zones) > 0:
            hbz_path = os.path.join(self.output_dir, 'hard_brake_zones.csv')
            hard_brake_zones.to_csv(hbz_path, index=False)
            print(f"✅ Saved {len(hard_brake_zones)} hard-brake zones to {hbz_path}")
        
        # Road quality zones
        if len(road_quality_zones) > 0:
            rq_path = os.path.join(self.output_dir, 'road_quality_zones.csv')
            road_quality_zones.to_csv(rq_path, index=False)
            print(f"✅ Saved {len(road_quality_zones)} road quality zones to {rq_path}")
        
        # Trip summary
        summary_path = os.path.join(self.output_dir, 'trip_summaries.csv')
        trip_summary.to_csv(summary_path, index=False)
        print(f"✅ Saved trip summaries to {summary_path}")
        
        # Recommendations
        if len(recommendations) > 0:
            rec_path = os.path.join(self.output_dir, 'recommendations.csv')
            recommendations.to_csv(rec_path, index=False)
            print(f"✅ Saved {len(recommendations)} recommendations to {rec_path}")


def main():
    """Main entry point."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pipeline = SignalFusionPipeline(base_dir)
    fused_events, hard_brake_zones, trip_summary, recommendations = pipeline.run_pipeline()
    
    print("\n" + "=" * 60)
    print("SIGNAL FUSION COMPLETE")
    print("=" * 60)
    
    print(f"\nTrip Summary:")
    print(trip_summary.to_string())
    
    if len(recommendations) > 0:
        print(f"\n⚠️ High Priority Recommendations:")
        high_priority = recommendations[recommendations['priority'] == 'high']
        for _, rec in high_priority.iterrows():
            print(f"  • {rec['message']}")


if __name__ == '__main__':
    main()
