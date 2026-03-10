"""
Driver Pulse - Driver-Facing Simulation Dashboard
=================================================
Uber-style driver interface with automatic trip simulation,
H3 hexagon zones, Short-Trip Mode, and real-time event detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import folium
import folium.plugins as plugins
import requests
from streamlit_folium import st_folium
from datetime import datetime, timedelta, time as dt_time
import time
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *

# Import BoVW Motion Detector for real-time classification
try:
    from pipelines.pipeline1_motion_bovw import BoVWMotionDetector, classify_simulation_window
    BOVW_AVAILABLE = True
except ImportError:
    BOVW_AVAILABLE = False
    print("Warning: BoVW motion detector not available")

# Try to import h3 for hexagonal grids
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

# Import professional Leaflet map component (replaces Folium for flicker-free updates)
try:
    from dashboard.components.map_component import render_leaflet_map, send_highlight_message, send_reset_message
    LEAFLET_AVAILABLE = True
except ImportError:
    LEAFLET_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Driver Pulse",
    page_icon="●",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Uber-style dark theme CSS with fixed button colors
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        background-color: #000000;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #8E8E93;
        margin-bottom: 1.5rem;
    }
    
    .uber-card {
        background-color: #1C1C1E;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #2C2C2E;
    }
    
    .uber-card-highlight {
        background-color: #1C1C1E;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border: 1px solid #30D158;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #8E8E93;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .trip-card {
        background-color: #1C1C1E;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #2C2C2E;
    }
    
    .alert-item {
        background-color: #2C2C2E;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-left: 3px solid #FF453A;
    }
    
    .alert-item-warning {
        border-left-color: #FF9F0A;
    }
    
    .alert-item-success {
        border-left-color: #30D158;
    }
    
    /* Fixed button styling - force black text on all buttons */
    .stButton button,
    .stButton > button,
    button[kind="secondary"],
    button[kind="primary"],
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
    }
    
    .stButton button:hover,
    .stButton > button:hover {
        background-color: #E5E5E5 !important;
        color: #000000 !important;
    }
    
    .stButton button:active,
    .stButton > button:active {
        background-color: #D0D0D0 !important;
        color: #000000 !important;
    }
    
    /* Primary button - green with black text */
    .stButton button[kind="primary"],
    [data-testid="baseButton-primary"] {
        background-color: #30D158 !important;
        color: #000000 !important;
    }
    
    .stButton button[kind="primary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        background-color: #28B84C !important;
        color: #000000 !important;
    }
    
    /* Force all button text to be black */
    .stButton button p,
    .stButton button span,
    .stButton > button p,
    .stButton > button span {
        color: #000000 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1C1C1E;
        border-right: 1px solid #2C2C2E;
    }
    
    [data-testid="stSidebar"] .stButton button,
    [data-testid="stSidebar"] .stButton > button {
        background-color: #2C2C2E !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover,
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #3C3C3E !important;
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] .stButton button p,
    [data-testid="stSidebar"] .stButton button span {
        color: #FFFFFF !important;
    }
    
    .stProgress > div > div {
        background-color: #30D158;
    }
    
    [data-testid="stMetricValue"] {
        color: #FFFFFF;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #8E8E93;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    hr { border-color: #2C2C2E; }
    
    p, span, label { color: #FFFFFF; }
    
    .green-text { color: #30D158; font-weight: 600; }
    
    .short-trip-alert {
        background: linear-gradient(135deg, #FF9F0A 0%, #FF453A 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class DriverPulseApp:
    """Main application class."""
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'simulation_data')
        self.output_dir = os.path.join(self.base_dir, 'simulation_data', 'processed_outputs')
        self._init_session_state()
        self.load_data()
    
    def _init_session_state(self):
        """Initialize session state variables."""
        defaults = {
            'logged_in': False,
            'driver_id': None,
            'driver_name': None,
            'shift_started': False,
            'target_earnings': 2000,
            'shift_start_time': None,
            'shift_end_time': None,
            'shift_hours': 8,
            'current_trip': None,
            'simulation_running': False,
            'simulation_progress': 0,
            'completed_trips': [],
            'total_earnings': 0,
            'detected_events': [],
            'last_update_time': None,
            'auto_simulate': True
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_data(self):
        """Load all required data."""
        self.drivers_df = pd.read_csv(os.path.join(self.data_dir, 'drivers', 'drivers.csv'))
        
        self.trips_df = pd.read_csv(os.path.join(self.data_dir, 'trips', 'trips.csv'))
        self.trips_df['start_time'] = pd.to_datetime(self.trips_df['start_time'])
        self.trips_df['end_time'] = pd.to_datetime(self.trips_df['end_time'])
        
        self.accel_df = pd.read_csv(os.path.join(self.data_dir, 'sensor_data', 'accelerometer_data.csv'))
        self.accel_df['timestamp'] = pd.to_datetime(self.accel_df['timestamp'])
        
        self._load_pipeline_outputs()
    
    def _load_pipeline_outputs(self):
        """Load outputs from pipelines."""
        # Motion events (from trip summaries)
        motion_path = os.path.join(self.output_dir, 'trip_summaries.csv')
        self.motion_events = pd.read_csv(motion_path) if os.path.exists(motion_path) else pd.DataFrame()
        
        # Load MOTION flagged moments (from simulation - only motion events)
        flagged_path = os.path.join(self.output_dir, 'flagged_moments.csv')
        motion_flagged = pd.DataFrame()
        if os.path.exists(flagged_path):
            motion_flagged = pd.read_csv(flagged_path)
            # Ensure signal_type column exists
            if 'signal_type' not in motion_flagged.columns:
                motion_flagged['signal_type'] = 'MOTION'
        
        # Load AUDIO flagged moments (from pipeline2 - real conflict detection)
        audio_path = os.path.join(self.output_dir, 'audio_flagged_moments.csv')
        audio_flagged = pd.DataFrame()
        if os.path.exists(audio_path):
            audio_flagged = pd.read_csv(audio_path)
        
        # Combine motion and audio flagged moments (keep them separate by signal_type)
        combined_frames = []
        if len(motion_flagged) > 0:
            combined_frames.append(motion_flagged)
        if len(audio_flagged) > 0:
            combined_frames.append(audio_flagged)
        
        if combined_frames:
            self.flagged_moments = pd.concat(combined_frames, ignore_index=True)
        else:
            self.flagged_moments = pd.DataFrame()
            
        if len(self.flagged_moments) > 0 and 'timestamp' in self.flagged_moments.columns:
            self.flagged_moments['timestamp'] = pd.to_datetime(self.flagged_moments['timestamp'])
        
        # Hard brake zones
        zones_path = os.path.join(self.output_dir, 'driver_brake_zones.csv')
        self.brake_zones = pd.read_csv(zones_path) if os.path.exists(zones_path) else pd.DataFrame()
        
        # Earnings forecast
        forecast_path = os.path.join(self.output_dir, 'earnings_forecast.csv')
        self.earnings_forecast = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()
        
        # Driver goals and earnings velocity (from driver_pulse_data)
        goals_path = os.path.join(self.data_dir, 'earnings', 'driver_goals.csv')
        self.driver_goals = pd.read_csv(goals_path) if os.path.exists(goals_path) else pd.DataFrame()
        
        velocity_log_path = os.path.join(self.data_dir, 'earnings', 'earnings_velocity_log.csv')
        self.earnings_velocity_log = pd.read_csv(velocity_log_path) if os.path.exists(velocity_log_path) else pd.DataFrame()

    # =========================================================================
    # VIEW 1: Login Screen
    # =========================================================================
    
    def render_login(self):
        """Render the login screen."""
        st.markdown('<h1 class="main-header">Driver Pulse</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your Smart Driving Companion</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Sign In")
            
            driver_options = self.drivers_df.apply(
                lambda x: f"{x['name']} ({x['driver_id']}) - {x['city']}", axis=1
            ).tolist()
            
            selected = st.selectbox("Select your profile:", driver_options, label_visibility="collapsed")
            
            driver_id = selected.split('(')[1].split(')')[0]
            driver_info = self.drivers_df[self.drivers_df['driver_id'] == driver_id].iloc[0]
            
            st.markdown(f"""
            <div class="uber-card">
                <p style="color: #8E8E93; font-size: 0.75rem;">DRIVER PROFILE</p>
                <p style="font-size: 1.25rem; font-weight: 600;">{driver_info['name']}</p>
                <p style="color: #8E8E93;">{driver_info['city']} | Rating {driver_info['rating']:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            password = st.text_input("Password:", type="password", value="demo123", label_visibility="collapsed", placeholder="Enter password")
            
            if st.button("Sign In", type="primary", use_container_width=True):
                if password == "demo123":
                    st.session_state.logged_in = True
                    st.session_state.driver_id = driver_id
                    st.session_state.driver_name = driver_info['name']
                    st.rerun()
                else:
                    st.error("Invalid password")

    # =========================================================================
    # VIEW 2: Pre-Shift Setup
    # =========================================================================
    
    def render_pre_shift_setup(self):
        """Render the pre-shift setup screen."""
        st.markdown(f'<h1 class="main-header">Welcome, {st.session_state.driver_name}</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Set up your shift</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Target earnings
            st.markdown("**Target Earnings**")
            target_earnings = st.slider(
                "target", min_value=500, max_value=5000,
                value=st.session_state.target_earnings, step=100,
                label_visibility="collapsed", format="₹%d"
            )
            
            st.markdown("---")
            
            # Shift time selection
            st.markdown("**Shift Schedule**")
            time_col1, time_col2 = st.columns(2)
            
            with time_col1:
                st.markdown('<p style="color: #8E8E93; font-size: 0.875rem;">Start Time</p>', unsafe_allow_html=True)
                shift_start = st.time_input("shift_start", value=dt_time(8, 0), label_visibility="collapsed")
            
            with time_col2:
                st.markdown('<p style="color: #8E8E93; font-size: 0.875rem;">End Time</p>', unsafe_allow_html=True)
                shift_end = st.time_input("shift_end", value=dt_time(16, 0), label_visibility="collapsed")
            
            # Calculate shift duration
            start_dt = datetime.combine(datetime.today(), shift_start)
            end_dt = datetime.combine(datetime.today(), shift_end)
            if end_dt <= start_dt:
                end_dt += timedelta(days=1)
            
            shift_duration = (end_dt - start_dt).total_seconds() / 3600
            expected_velocity = target_earnings / shift_duration if shift_duration > 0 else 0
            
            st.markdown("---")
            
            # Show metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div style="text-align:center;"><p class="metric-label">TARGET</p><p class="metric-value green-text">₹{target_earnings:,}</p></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div style="text-align:center;"><p class="metric-label">DURATION</p><p class="metric-value">{shift_duration:.1f}h</p></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div style="text-align:center;"><p class="metric-label">REQUIRED RATE</p><p class="metric-value">₹{expected_velocity:.0f}/hr</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            
            if st.button("Start Shift", type="primary", use_container_width=True):
                st.session_state.target_earnings = target_earnings
                st.session_state.shift_start_time = shift_start
                st.session_state.shift_end_time = shift_end
                st.session_state.shift_hours = shift_duration
                st.session_state.shift_started = True
                st.session_state.shift_actual_start = datetime.now()
                st.rerun()

    # =========================================================================
    # VIEW 3: Trip Selection
    # =========================================================================
    
    def render_trip_selection(self):
        """Render the trip selection screen with smart recommendations."""
        st.markdown('<h1 class="main-header">Available Trips</h1>', unsafe_allow_html=True)
        
        self._render_shift_progress_bar()
        
        # Short-Trip Mode Alert
        self._render_short_trip_mode_alert()
        
        driver_trips = self.trips_df[
            self.trips_df['driver_id'] == st.session_state.driver_id
        ].copy()
        
        completed_ids = st.session_state.completed_trips
        available_trips = driver_trips[~driver_trips['trip_id'].isin(completed_ids)]
        
        if len(available_trips) == 0:
            st.markdown("""
            <div class="uber-card-highlight">
                <p style="color: #30D158; font-weight: 600; font-size: 1.25rem;">All trips completed</p>
                <p style="color: #8E8E93;">Great work today!</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View Shift Summary"):
                st.session_state.view = 'end_of_day'
                st.rerun()
            return
        
        # Calculate current earnings status for recommendations
        total_earnings = st.session_state.total_earnings
        target = st.session_state.target_earnings
        shift_hours = st.session_state.shift_hours
        
        # Time calculations
        shift_start = st.session_state.get('shift_actual_start', datetime.now())
        elapsed_hours = max(0.1, (datetime.now() - shift_start).total_seconds() / 3600)
        remaining_hours = max(0.1, shift_hours - elapsed_hours)
        
        # Velocity calculations
        current_velocity = total_earnings / elapsed_hours if elapsed_hours > 0 else 0
        remaining_to_goal = max(0, target - total_earnings)
        required_velocity = remaining_to_goal / remaining_hours if remaining_hours > 0 else 0
        
        # Get last trip stress if available - only show after first completed trip
        last_trip_stress = st.session_state.get('last_trip_stress', 0)
        has_completed_trips = len(st.session_state.completed_trips) > 0
        
        # Show stress status indicator only after at least one trip is completed
        if has_completed_trips and last_trip_stress > 0:
            if last_trip_stress >= 0.7:
                stress_icon = "😓"
                stress_label = "High Stress"
                stress_color = "#FF453A"
                stress_advice = "Your last trip was intense. Consider a shorter, easier route to recover."
            elif last_trip_stress >= 0.4:
                stress_icon = "😐"
                stress_label = "Moderate Stress"
                stress_color = "#FF9F0A"
                stress_advice = "You're doing fine. Any trip works, but shorter ones might feel easier."
            else:
                stress_icon = "😊"
                stress_label = "Low Stress"
                stress_color = "#30D158"
                stress_advice = "You're fresh and ready! Take on any trip you like."
            
            st.markdown(f"""
            <div style="background: {stress_color}22; border-left: 4px solid {stress_color}; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 1.5rem;">{stress_icon}</span>
                        <span style="color: {stress_color}; font-weight: 700; font-size: 1rem; margin-left: 8px;">Your Status: {stress_label}</span>
                    </div>
                    <span style="color: #8E8E93; font-size: 0.85rem;">Last trip stress: {last_trip_stress*100:.0f}%</span>
                </div>
                <p style="color: #E5E5EA; font-size: 0.9rem; margin: 8px 0 0 0;">{stress_advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Add velocity to each trip for ranking
        available_trips = available_trips.copy()
        available_trips['velocity'] = available_trips['fare'] / (available_trips['duration_min'] / 60)
        
        # Smart trip recommendations
        self._render_smart_trip_recommendations(
            available_trips, current_velocity, required_velocity, 
            remaining_to_goal, remaining_hours, last_trip_stress
        )
        
        st.markdown("### All Available Trips")
        
        # Sort by velocity (best rate first)
        available_trips = available_trips.sort_values('velocity', ascending=False)
        
        # Calculate target velocity for explanation
        target_velocity = st.session_state.target_earnings / st.session_state.shift_hours if st.session_state.shift_hours > 0 else 300
        
        for _, trip in available_trips.iterrows():
            velocity = trip['velocity']
            pickup = trip.get('pickup_location', 'Pickup')
            dropoff = trip.get('dropoff_location', 'Dropoff')
            trip_id = trip['trip_id']
            
            # Estimate trip difficulty based on duration only
            # (We can't know events before the trip - those are discovered during the ride)
            is_long_trip = trip['duration_min'] > 45
            is_short_trip = trip['duration_min'] <= 25
            
            # Simple duration-based badge (only shown if driver is stressed from last trip)
            stress_badge = ""
            if has_completed_trips and last_trip_stress >= 0.5:
                # After a stressful trip, highlight short trips as easier options
                if is_short_trip:
                    stress_badge = '<span style="background: #30D15833; color: #30D158; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-left: 6px;">😌 Quick & Easy</span>'
                elif is_long_trip:
                    stress_badge = '<span style="background: #FF9F0A33; color: #FF9F0A; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; margin-left: 6px;">⏱️ Longer trip</span>'
            
            # Determine if this trip is recommended
            is_short = trip['duration_min'] <= 30
            is_high_velocity = velocity >= required_velocity if required_velocity > 0 else velocity >= target_velocity
            velocity_diff = velocity - target_velocity
            
            # Badge and explanation for trips
            badge = ""
            badge_explanation = ""
            
            # Best choice: after stressful trip, recommend short high-rate trips
            if remaining_to_goal > 0 and is_high_velocity and is_short and last_trip_stress >= 0.5:
                badge = '<span style="background: #30D158; color: #000; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">⭐ BEST CHOICE</span>'
                badge_explanation = f'<p style="color: #30D158; font-size: 0.8rem; margin-top: 4px;">Short trip + high earnings — perfect after a stressful ride</p>'
            elif is_high_velocity:
                badge = '<span style="background: #007AFF; color: #FFF; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">💰 HIGH RATE</span>'
                badge_explanation = f'<p style="color: #007AFF; font-size: 0.8rem; margin-top: 4px;">₹{velocity:.0f}/hr is {velocity_diff:+.0f}/hr above your target rate</p>'
            elif is_short:
                badge = '<span style="background: #8E8E93; color: #FFF; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">⏱️ QUICK</span>'
                badge_explanation = f'<p style="color: #8E8E93; font-size: 0.8rem; margin-top: 4px;">Short {trip["duration_min"]:.0f} min trip — good for a quick earn</p>'
            else:
                badge = '<span style="background: #2C2C2E; color: #FFF; padding: 3px 10px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">Standard</span>'
                badge_explanation = f'<p style="color: #8E8E93; font-size: 0.8rem; margin-top: 4px;">₹{velocity:.0f}/hr — a solid choice</p>'
            
            col1, col2 = st.columns([4, 1])
            
            with col1:
                try:
                    # Escape HTML special characters in location names
                    pickup_safe = str(pickup).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    dropoff_safe = str(dropoff).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    card_html = f"""<div class="trip-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p style="font-weight: 600; font-size: 1.1rem;">{pickup_safe} → {dropoff_safe} {badge}</p>
                                <p style="color: #8E8E93; font-size: 0.875rem;">{trip['distance_km']:.1f} km · {trip['duration_min']:.0f} min {stress_badge}</p>
                                {badge_explanation}
                            </div>
                            <div style="text-align: right;">
                                <p style="font-weight: 700; font-size: 1.25rem; color: #30D158;">₹{trip['fare']:.0f}</p>
                                <p style="color: #8E8E93; font-size: 0.8rem;">₹{velocity:.0f}/hr</p>
                            </div>
                        </div>
                    </div>"""
                    st.markdown(card_html, unsafe_allow_html=True)
                except Exception as e:
                    # Fallback to simple text if HTML rendering fails
                    st.markdown(f"**{pickup} → {dropoff}** | {trip['distance_km']:.1f} km · {trip['duration_min']:.0f} min | ₹{trip['fare']:.0f}")
            
            with col2:
                if st.button("Start", key=f"start_{trip['trip_id']}", use_container_width=True):
                    st.session_state.current_trip = trip['trip_id']
                    st.session_state.simulation_progress = 0
                    st.session_state.detected_events = []
                    st.session_state.last_update_time = time.time()
                    st.session_state.view = 'simulation'
                    st.rerun()
            
            # Show brake zone warnings for this trip
            self._check_brake_zone_warning(trip)
    
    def _render_smart_trip_recommendations(self, available_trips, current_velocity, required_velocity, 
                                           remaining_to_goal, remaining_hours, last_trip_stress):
        """Render smart trip recommendations based on earnings velocity, stress, and goal progress."""
        
        if remaining_to_goal <= 0:
            # Goal already achieved - suggest easy trips
            st.markdown("""
            <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <p style="color: #30D158; font-weight: 700; font-size: 1rem;">🎉 Goal Achieved!</p>
                <p style="color: #E5E5EA; font-size: 0.95rem;">You've hit your target! Take any trip you like — or call it a day with a great shift.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Calculate recommendations
        short_trips = available_trips[available_trips['duration_min'] <= 30]
        high_velocity_trips = available_trips[available_trips['velocity'] >= required_velocity]
        
        # Build recommendation message
        rec_shown = False
        
        # Case 1: High stress from last trip - recommend shorter trips
        if last_trip_stress >= 0.7:
            best_short = short_trips.nlargest(1, 'velocity').iloc[0] if len(short_trips) > 0 else None
            if best_short is not None:
                st.markdown(f"""
                <div style="background: #FF9F0A22; border-left: 4px solid #FF9F0A; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <p style="color: #FF9F0A; font-weight: 700; font-size: 1rem;">😌 Take It Easy</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">Your last trip was stressful. We suggest a shorter trip like 
                    <strong>{best_short.get('pickup_location', 'Pickup')} → {best_short.get('dropoff_location', 'Dropoff')}</strong> 
                    ({best_short['duration_min']:.0f} min, ₹{best_short['velocity']:.0f}/hr) to help you reset.</p>
                </div>
                """, unsafe_allow_html=True)
                rec_shown = True
        
        # Case 2: Behind on goal - recommend high-velocity trips
        elif current_velocity < required_velocity and len(high_velocity_trips) > 0:
            best_velocity = high_velocity_trips.nlargest(1, 'velocity').iloc[0]
            hrs_needed = remaining_to_goal / best_velocity['velocity'] if best_velocity['velocity'] > 0 else 0
            
            st.markdown(f"""
            <div style="background: #007AFF22; border-left: 4px solid #007AFF; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <p style="color: #007AFF; font-weight: 700; font-size: 1rem;">📈 Boost Your Earnings</p>
                <p style="color: #E5E5EA; font-size: 0.95rem;">You need ₹{required_velocity:.0f}/hr to hit your goal. 
                <strong>{best_velocity.get('pickup_location', 'Pickup')} → {best_velocity.get('dropoff_location', 'Dropoff')}</strong> 
                pays ₹{best_velocity['velocity']:.0f}/hr — {hrs_needed:.1f} hours of trips like this will get you there!</p>
            </div>
            """, unsafe_allow_html=True)
            rec_shown = True
        
        # Case 3: Close to goal - suggest quick trips to finish
        elif remaining_to_goal > 0 and remaining_to_goal <= 500:
            # Find a trip that covers the remaining amount
            covering_trips = available_trips[available_trips['fare'] >= remaining_to_goal * 0.8]
            if len(covering_trips) > 0:
                quick_finish = covering_trips.nsmallest(1, 'duration_min').iloc[0]
                st.markdown(f"""
                <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                    <p style="color: #30D158; font-weight: 700; font-size: 1rem;">🏁 Almost There!</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">Just ₹{remaining_to_goal:.0f} to go! 
                    <strong>{quick_finish.get('pickup_location', 'Pickup')} → {quick_finish.get('dropoff_location', 'Dropoff')}</strong> 
                    (₹{quick_finish['fare']:.0f}) could be your last trip of the day.</p>
                </div>
                """, unsafe_allow_html=True)
                rec_shown = True
        
        # Case 4: On track - gentle encouragement
        if not rec_shown and current_velocity >= required_velocity:
            hrs = int(remaining_hours)
            mins = int((remaining_hours - hrs) * 60)
            st.markdown(f"""
            <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1rem; margin-bottom: 1rem;">
                <p style="color: #30D158; font-weight: 700; font-size: 1rem;">✅ You're On Track</p>
                <p style="color: #E5E5EA; font-size: 0.95rem;">At ₹{current_velocity:.0f}/hr, you'll hit your ₹{st.session_state.target_earnings:,.0f} goal 
                with time to spare. Keep up the great pace!</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _check_brake_zone_warning(self, trip):
        """Warn if trip route passes near a known hard brake zone for this driver."""
        if len(self.brake_zones) == 0:
            return
        
        driver_zones = self.brake_zones[
            self.brake_zones['driver_id'] == st.session_state.driver_id
        ] if 'driver_id' in self.brake_zones.columns else self.brake_zones
        
        if len(driver_zones) == 0:
            return
        
        # Check proximity of pickup/dropoff to known brake zones
        warnings = []
        for _, zone in driver_zones.iterrows():
            zone_lat = zone.get('zone_lat', zone.get('center_lat', zone.get('lat')))
            zone_lon = zone.get('zone_lon', zone.get('center_lon', zone.get('lon')))
            
            if pd.isna(zone_lat) or pd.isna(zone_lon):
                continue
            
            for point_lat, point_lon, label in [
                (trip.get('pickup_lat'), trip.get('pickup_lon'), 'near pickup'),
                (trip.get('dropoff_lat'), trip.get('dropoff_lon'), 'near dropoff'),
            ]:
                if pd.isna(point_lat) or pd.isna(point_lon):
                    continue
                
                # Simple distance check (~500m threshold)
                lat_diff = abs(zone_lat - point_lat)
                lon_diff = abs(zone_lon - point_lon)
                approx_dist_km = ((lat_diff**2 + lon_diff**2)**0.5) * 111
                
                if approx_dist_km < 0.5:
                    count = zone.get('event_count', 3)
                    warnings.append(f"⚠️ Hard brake zone {label} ({int(count)} previous events)")
        
        for w in warnings:
            st.markdown(f"""
            <div style="background: #FF453A22; border-left: 3px solid #FF453A;
                        border-radius: 6px; padding: 0.5rem 0.75rem; margin: 0.25rem 0;">
                <p style="color: #FF453A; font-size: 0.8rem; margin: 0;">{w} — drive carefully</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_short_trip_mode_alert(self):
        """Render Short-Trip Mode recommendation if applicable."""
        if not st.session_state.shift_started:
            return
        
        # Calculate time remaining in shift
        if st.session_state.shift_actual_start:
            elapsed = (datetime.now() - st.session_state.shift_actual_start).total_seconds() / 3600
            remaining_hours = st.session_state.shift_hours - elapsed
            
            # Check if at risk (less than target progress) and less than 2 hours left
            target = st.session_state.target_earnings
            earned = st.session_state.total_earnings
            progress_pct = (earned / target * 100) if target > 0 else 0
            expected_pct = (elapsed / st.session_state.shift_hours * 100) if st.session_state.shift_hours > 0 else 0
            
            if remaining_hours < 2 and progress_pct < expected_pct - 10:
                # At risk with < 2 hours left - show Short-Trip Mode
                shortfall = target - earned
                st.markdown(f"""
                <div class="short-trip-alert">
                    <p style="color: #000; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                        Short-Trip Mode Recommended
                    </p>
                    <p style="color: #000; font-size: 0.875rem;">
                        You have {remaining_hours:.1f}h left and ₹{shortfall:,.0f} to reach your goal.
                        Focus on quick, nearby trips to maximize earnings.
                    </p>
                </div>
                """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW 4: Trip Simulation (Automatic)
    # =========================================================================
    
    def render_trip_simulation(self):
        """Automatic trip simulation with slow, visible progress."""
        trip_id = st.session_state.current_trip
        trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
        
        # Get route from actual GPS data
        cache_key = f"route_{trip_id}"
        if cache_key not in st.session_state:
            route = self._get_gps_route(trip_id)
            st.session_state[cache_key] = route
        route = st.session_state[cache_key]
        
        pickup = trip_info.get('pickup_location', 'Pickup')
        dropoff = trip_info.get('dropoff_location', 'Dropoff')
        
        st.markdown(f'<h1 class="main-header">{pickup} → {dropoff}</h1>', unsafe_allow_html=True)
        
        # Trip metrics row with LARGER font for FARE
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">FARE</p>
                <p style="font-size: 2.8rem; font-weight: 700; color: #30D158;">₹{trip_info["fare"]:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">DISTANCE</p>
                <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF;">{trip_info["distance_km"]:.1f}<span style="font-size: 1rem; color: #8E8E93;"> km</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">DURATION</p>
                <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF;">{trip_info["duration_min"]:.0f}<span style="font-size: 1rem; color: #8E8E93;"> min</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            velocity = trip_info['fare'] / (trip_info['duration_min'] / 60)
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">RATE</p>
                <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF;">₹{velocity:.0f}<span style="font-size: 1rem; color: #8E8E93;">/hr</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Current progress
        progress = st.session_state.simulation_progress
        
        # Calculate distance and time remaining
        distance_total = trip_info['distance_km']
        duration_total = trip_info['duration_min']
        distance_covered = distance_total * (progress / 100)
        distance_left = distance_total - distance_covered
        time_left = duration_total * (1 - progress / 100)
        
        # Layout
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            # Progress bar with distance and time left
            prog_col1, prog_col2, prog_col3 = st.columns([2, 1, 1])
            with prog_col1:
                st.markdown(f"**Progress: {progress:.0f}%**")
            with prog_col2:
                st.markdown(f"<p style='color: #8E8E93; font-size: 0.9rem; margin: 0;'>📍 <b style='color: #FFFFFF;'>{distance_left:.1f} km</b> left</p>", unsafe_allow_html=True)
            with prog_col3:
                time_mins = int(time_left)
                st.markdown(f"<p style='color: #8E8E93; font-size: 0.9rem; margin: 0;'>⏱️ <b style='color: #FFFFFF;'>{time_mins} min</b> left</p>", unsafe_allow_html=True)
            
            st.progress(progress / 100)
            
            # THE MAP
            self._render_clean_map(trip_id, progress / 100, route)
        
        with col_right:
            # Control Buttons
            if not st.session_state.simulation_running and progress < 100:
                if st.button("▶ START DRIVING", type="primary", use_container_width=True):
                    st.session_state.simulation_running = True
                    st.rerun()
            
            if st.session_state.simulation_running and progress < 100:
                if st.button("⏸ PAUSE", use_container_width=True):
                    st.session_state.simulation_running = False
                    st.rerun()
            
            st.markdown("### 🔔 Live Safety Feed")
            self._render_live_events(trip_id)
        
        # Trip Completion
        if progress >= 100:
            st.success("🎉 Arrived at Destination!")
            if st.button("COMPLETE TRIP & VIEW INSIGHTS", type="primary", use_container_width=True):
                st.session_state.completed_trips.append(trip_id)
                st.session_state.total_earnings += trip_info['fare']
                st.session_state.simulation_running = False
                st.session_state.view = 'post_trip'
                st.rerun()
        
        # AUTO-ADVANCE - slow and steady
        if st.session_state.simulation_running and progress < 100:
            new_progress = min(100, progress + SIMULATION_STEP)
            st.session_state.simulation_progress = new_progress
            self._detect_events_at_progress(trip_id, new_progress)
            time.sleep(SIMULATION_REFRESH_RATE)
            st.rerun()
    
    def _get_gps_route(self, trip_id: str):
        """Get route from actual GPS data in accelerometer CSV."""
        # Cache route in session state to avoid recalculating
        cache_key = f'route_{trip_id}'
        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        trip_accel = self.accel_df[self.accel_df['trip_id'] == trip_id].copy()
        
        if len(trip_accel) == 0:
            # Fallback to pickup/dropoff straight line
            trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
            route = [
                [trip_info['pickup_lat'], trip_info['pickup_lon']],
                [trip_info['dropoff_lat'], trip_info['dropoff_lon']]
            ]
            st.session_state[cache_key] = route
            return route
        
        # Sort by time and clean GPS data
        trip_accel = trip_accel.sort_values('elapsed_seconds')
        trip_accel = trip_accel.dropna(subset=['gps_lat', 'gps_lon'])
        
        if len(trip_accel) < 2:
            trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
            route = [
                [trip_info['pickup_lat'], trip_info['pickup_lon']],
                [trip_info['dropoff_lat'], trip_info['dropoff_lon']]
            ]
            st.session_state[cache_key] = route
            return route
        
        # Use ALL GPS points for smoother animation (no sampling)
        # This gives us maximum smoothness for the car movement
        route = trip_accel[['gps_lat', 'gps_lon']].values.tolist()
        
        # Cache the route
        st.session_state[cache_key] = route
        
        return route
    
    def _render_clean_map(self, trip_id: str, progress: float, route: list):
        """Render professional Uber-style map with actual GPS route - now flicker-free!"""
        trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
        
        if len(route) < 2:
            st.info("Route unavailable")
            return
        
        # Use new Leaflet component if available (flicker-free)
        if LEAFLET_AVAILABLE:
            # Get flagged moments for this trip with GPS coordinates
            trip_moments = self.flagged_moments[self.flagged_moments['trip_id'] == trip_id].copy()
            trip_accel = self.accel_df[self.accel_df['trip_id'] == trip_id]
            total_time = trip_accel['elapsed_seconds'].max() if len(trip_accel) > 0 and 'elapsed_seconds' in trip_accel.columns else 100
            
            # Prepare events with GPS coordinates and progress
            flagged_events = []
            for _, evt in trip_moments.iterrows():
                evt_time = evt.get('elapsed_seconds', 0)
                evt_progress = evt_time / total_time if total_time > 0 else 0
                
                # Calculate GPS position from route based on progress
                event_route_idx = int(evt_progress * len(route))
                event_route_idx = max(0, min(event_route_idx, len(route) - 1))
                event_pos = route[event_route_idx]
                
                flagged_events.append({
                    'flag_id': evt.get('flag_id', ''),
                    'gps_lat': event_pos[0],
                    'gps_lon': event_pos[1],
                    'event_label': evt.get('event_label', 'Event'),
                    'severity': str(evt.get('severity', 'medium')).lower(),
                    'signal_type': evt.get('signal_type', 'MOTION'),
                    'explanation': evt.get('explanation', ''),
                    'progress': evt_progress
                })
            
            # Get brake zones if available
            brake_zones_list = []
            if H3_AVAILABLE and len(self.brake_zones) > 0:
                for _, zone in self.brake_zones.iterrows():
                    lat = zone.get('center_lat', zone.get('lat'))
                    lon = zone.get('center_lon', zone.get('lon'))
                    if lat and lon and not pd.isna(lat) and not pd.isna(lon):
                        brake_zones_list.append({
                            'center_lat': float(lat),
                            'center_lon': float(lon),
                            'event_count': zone.get('event_count', 'N/A')
                        })
            
            # Render the flicker-free Leaflet map
            render_leaflet_map(
                route=route,
                flagged_moments=flagged_events,
                progress=progress,
                trip_info={'pickup_location': trip_info.get('pickup_location', 'Pickup'),
                          'dropoff_location': trip_info.get('dropoff_location', 'Dropoff')},
                brake_zones=brake_zones_list,
                height=500,
                map_key=f"leaflet_map_{trip_id}"
            )
            return
        
        # Fallback to Folium if Leaflet component not available
        # Split route into driven vs remaining
        split_idx = max(1, int(len(route) * progress))
        driven = route[:split_idx]
        
        # Car position = last driven point
        car_pos = driven[-1] if driven else route[0]
        
        # Calculate bounds for the full route
        lats = [p[0] for p in route]
        lons = [p[1] for p in route]
        
        # Create map centered on full route
        m = folium.Map(tiles='CartoDB dark_matter')
        
        # Fit to full route bounds with padding
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=[30, 30])
        
        # Ghost route (gray, full path)
        folium.PolyLine(route, weight=4, color='#444444', opacity=0.6).add_to(m)
        
        # Driven route with animated AntPath (green with moving dashes)
        if len(driven) > 1:
            # Static green line as base
            folium.PolyLine(driven, weight=6, color='#30D158', opacity=0.8).add_to(m)
            # Animated ant path overlay showing movement direction
            plugins.AntPath(
                driven,
                weight=4,
                color='#30D158',
                pulse_color='#FFFFFF',
                delay=800,
                dash_array=[20, 30],
                opacity=0.9
            ).add_to(m)
        
        # Start marker
        pickup_loc = trip_info.get('pickup_location', 'Pickup')
        folium.CircleMarker(
            location=route[0], radius=8,
            color='#FFFFFF', fill=True, fillColor='#FFFFFF', fillOpacity=1,
            popup=f"Pickup: {pickup_loc}"
        ).add_to(m)
        
        # End marker
        dropoff_loc = trip_info.get('dropoff_location', 'Dropoff')
        folium.CircleMarker(
            location=route[-1], radius=8,
            color='#30D158', fill=True, fillColor='#30D158', fillOpacity=1,
            popup=f"Dropoff: {dropoff_loc}"
        ).add_to(m)
        
        # Moving car icon with pulsing effect
        car_html = '''
        <div style="
            background-color: #007AFF;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            border: 3px solid white;
            box-shadow: 0 0 10px rgba(0,122,255,0.8), 0 0 20px rgba(0,122,255,0.5);
            animation: pulse 1.5s ease-in-out infinite;
        ">
        <style>
            @keyframes pulse {
                0% { box-shadow: 0 0 10px rgba(0,122,255,0.8), 0 0 20px rgba(0,122,255,0.5); }
                50% { box-shadow: 0 0 20px rgba(0,122,255,1), 0 0 40px rgba(0,122,255,0.8); }
                100% { box-shadow: 0 0 10px rgba(0,122,255,0.8), 0 0 20px rgba(0,122,255,0.5); }
            }
        </style>
        </div>
        '''
        folium.Marker(
            location=car_pos,
            icon=folium.DivIcon(
                html=car_html,
                icon_size=(26, 26),
                icon_anchor=(13, 13)
            ),
            tooltip=f"Your Vehicle ({progress*100:.0f}%)"
        ).add_to(m)
        
        # Flagged events - only show if car has passed that point
        trip_moments = self.flagged_moments[self.flagged_moments['trip_id'] == trip_id]
        trip_accel = self.accel_df[self.accel_df['trip_id'] == trip_id]
        total_time = trip_accel['elapsed_seconds'].max() if len(trip_accel) > 0 and 'elapsed_seconds' in trip_accel.columns else 100
        
        for _, event in trip_moments.iterrows():
            evt_time = event.get('elapsed_seconds', 0)
            if total_time and total_time > 0:
                event_progress = evt_time / total_time
            else:
                event_progress = 0
            
            if event_progress > progress:
                continue
            
            event_route_idx = int(event_progress * len(route))
            event_route_idx = max(0, min(event_route_idx, len(route) - 1))
            event_pos = route[event_route_idx]
            
            # Determine icon color based on severity
            severity = str(event.get('severity', 'medium')).lower()
            signal_type = str(event.get('signal_type', 'MOTION'))
            
            # Red for critical/high, orange for medium, gray for low
            if severity in ['critical', 'high'] or signal_type == 'COMPOUND':
                icon_color = 'red'
            elif severity == 'medium':
                icon_color = 'orange'
            else:
                icon_color = 'gray'
            
            # Choose icon based on signal_type
            if signal_type == 'AUDIO':
                icon_type = 'volume-up'
            elif signal_type == 'COMPOUND':
                icon_type = 'bolt'
            else:  # MOTION
                icon_type = 'exclamation-triangle'
            
            event_label = str(event.get('event_label', 'Event'))
            
            folium.Marker(
                location=event_pos,
                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa'),
                popup=f"<b>{event_label}</b><br>{event.get('explanation', '')}"
            ).add_to(m)
        
        # Add H3 hard-brake zones if available
        if H3_AVAILABLE and len(self.brake_zones) > 0:
            self._add_h3_brake_zones(m)
        
        # Use unique key based on progress to force map refresh - use full container width
        st_folium(m, width=None, height=500, key=f"sim_map_{int(progress * 1000)}", use_container_width=True)
    
    def _add_h3_brake_zones(self, m):
        """Add H3 hexagonal hard-brake zones to map."""
        for _, zone in self.brake_zones.iterrows():
            try:
                lat = zone.get('center_lat', zone.get('lat'))
                lon = zone.get('center_lon', zone.get('lon'))
                
                if pd.isna(lat) or pd.isna(lon):
                    continue
                
                # Get H3 index
                h3_index = h3.geo_to_h3(lat, lon, H3_RESOLUTION_BRAKE_ZONES)
                
                # Get hexagon boundary
                boundary = h3.h3_to_geo_boundary(h3_index, geo_json=True)
                
                # Draw hexagon
                folium.Polygon(
                    locations=[(coord[1], coord[0]) for coord in boundary],
                    color='#FF453A',
                    fill=True,
                    fillColor='#FF453A',
                    fillOpacity=0.3,
                    weight=2,
                    popup=f"Hard Brake Zone ({zone.get('event_count', 'N/A')} events)"
                ).add_to(m)
            except Exception:
                pass
    
    def _render_live_events(self, trip_id: str):
        """Render live event alerts showing flagged moments based on simulation progress."""
        progress = st.session_state.get('simulation_progress', 0) / 100.0
        
        # Get flagged moments from CSV for this trip
        trip_moments = self.flagged_moments[self.flagged_moments['trip_id'] == trip_id].copy()
        
        # Get trip total duration to calculate event progress
        trip_accel = self.accel_df[self.accel_df['trip_id'] == trip_id]
        total_time = trip_accel['elapsed_seconds'].max() if len(trip_accel) > 0 and 'elapsed_seconds' in trip_accel.columns else 100
        
        # Filter to events that car has already passed
        visible_events = []
        if len(trip_moments) > 0 and total_time > 0:
            for _, evt in trip_moments.iterrows():
                evt_time = evt.get('elapsed_seconds', 0)
                evt_progress = evt_time / total_time if total_time > 0 else 0
                if evt_progress <= progress:
                    visible_events.append(evt.to_dict())
        
        # Also include any real-time detected events
        realtime_events = st.session_state.detected_events
        
        # Combine (flagged moments take precedence for display)
        all_events = visible_events  # Show CSV flagged moments as primary
        
        if not all_events and not realtime_events:
            st.markdown("""
            <div class="alert-item alert-item-success">
                <p style="color: #30D158; font-weight: 600; font-size: 1rem;">✓ All Clear</p>
                <p style="color: #8E8E93; font-size: 0.9rem;">Smooth driving — keep it up!</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Show event count summary with clear legend
        total_count = len(all_events) + len(realtime_events)
        st.markdown(f"""
        <div style="background: rgba(255,69,58,0.1); padding: 10px 14px; border-radius: 8px; margin-bottom: 12px;">
            <span style="color: #FF453A; font-weight: 600; font-size: 1rem;">⚠️ {total_count} Alert{'s' if total_count > 1 else ''} Detected</span>
        </div>
        <div style="display: flex; gap: 16px; margin-bottom: 12px; flex-wrap: wrap;">
            <span style="color: #8E8E93; font-size: 0.85rem;">🔴 High Risk</span>
            <span style="color: #8E8E93; font-size: 0.85rem;">🟠 Caution</span>
            <span style="color: #8E8E93; font-size: 0.85rem;">🟢 Minor</span>
            <span style="color: #8E8E93; font-size: 0.85rem;">⚡ Multi-Signal</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Prepare events for display (convert CSV format to display format)
        display_events = []
        for idx, evt in enumerate(all_events):
            # Get event_label from CSV (this is the correct column name)
            event_label_raw = evt.get('event_label', 'Event')
            # Handle NaN/float values
            if not isinstance(event_label_raw, str) or pd.isna(event_label_raw):
                event_label_raw = 'Event'
            display_events.append({
                'flag_id': evt.get('flag_id', f'evt_{idx}'),  # For hover sync
                'event_label': event_label_raw,  # Keep original format for matching
                'severity': evt.get('severity', 'medium'),
                'signal_type': evt.get('signal_type', 'MOTION'),
                'elapsed_seconds': evt.get('elapsed_seconds', 0),
                'explanation': evt.get('explanation', ''),
                'layers_fired': evt.get('layers_fired', ''),
                'confidence': evt.get('confidence', 0.5),
                'motion_score': evt.get('motion_score', 0),
                'audio_score': evt.get('audio_score', 0),
                'combined_score': evt.get('combined_score', 0),
                # Audio metrics for nuanced categorization
                'zcr': evt.get('zcr', 0),
                'f0_std': evt.get('f0_std', 0),
                'db_deviation': evt.get('db_deviation', 0)
            })
        
        # Add realtime events
        display_events.extend(realtime_events)
        
        # Sort by elapsed time (most recent last)
        display_events.sort(key=lambda x: x.get('elapsed_seconds', 0))
        
        # Create scrollable container for all events with hover sync support
        st.markdown("""
        <style>
        .scrollable-feed {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 8px;
        }
        .scrollable-feed::-webkit-scrollbar {
            width: 6px;
        }
        .scrollable-feed::-webkit-scrollbar-track {
            background: #1C1C1E;
            border-radius: 3px;
        }
        .scrollable-feed::-webkit-scrollbar-thumb {
            background: #3C3C3E;
            border-radius: 3px;
        }
        .scrollable-feed::-webkit-scrollbar-thumb:hover {
            background: #5C5C5E;
        }
        /* Hover sync - highlight effect */
        .event-card {
            transition: all 0.2s ease;
            cursor: pointer;
        }
        .event-card:hover {
            transform: translateX(4px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        </style>
        <div class="scrollable-feed">
        """, unsafe_allow_html=True)
        
        # Show ALL events (reversed so newest is at top)
        for event in reversed(display_events):
            severity = event.get('severity', 'medium')
            if severity == 'critical' or severity == 'high':
                alert_class = 'alert-item'
                color = '#FF453A'
                border_color = '#FF453A'
                icon = '🔴'
            elif severity == 'medium':
                alert_class = 'alert-item alert-item-warning'
                color = '#FF9F0A'
                border_color = '#FF9F0A'
                icon = '🟠'
            else:
                alert_class = 'alert-item alert-item-success'
                color = '#30D158'
                border_color = '#30D158'
                icon = '🟢'
            
            # Get event details
            event_label = event.get('event_label', 'Event Detected')
            signal_type = event.get('signal_type', 'MOTION')
            confidence = event.get('confidence', 0)
            elapsed = event.get('elapsed_seconds', 0)
            explanation = event.get('explanation', '')
            layers_fired = event.get('layers_fired', '')
            
            # Clean up NaN values
            explanation = str(explanation) if explanation and not (isinstance(explanation, float) and pd.isna(explanation)) else ''
            layers_fired = str(layers_fired) if layers_fired and not (isinstance(layers_fired, float) and pd.isna(layers_fired)) else ''
            
            # Format time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            time_str = f"{mins}:{secs:02d}"
            
            # Build user-friendly description based on signal type
            description = ""
            event_type_label = ""
            advice = ""
            
            if signal_type == 'COMPOUND':
                icon = '⚡'
                # Get specific compound event type
                compound_evt = event.get('event_label', '').upper()
                if 'BRAKE' in compound_evt:
                    event_type_label = "Hard Brake + Cabin Stress"
                    description = "You braked suddenly AND cabin tension was detected — passengers felt both the stop and the stress."
                    advice = "💡 Stay calm. Hard situations happen. Take a breath and focus on smooth driving ahead."
                elif 'TURN' in compound_evt:
                    event_type_label = "Sharp Turn + Cabin Stress"
                    description = "A sharp turn coincided with elevated cabin tension — multiple stress signals at once."
                    advice = "💡 If passengers are tense, slow down your turns even more. Smooth driving calms everyone."
                else:
                    event_type_label = "Multi-Signal Alert"
                    description = "Both driving stress and cabin tension detected at the same time."
                    advice = "💡 Pull over safely if needed. Take 3 deep breaths. A calm driver is a safe driver."
            elif signal_type == 'MOTION' or signal_type == 'ACCELEROMETER':
                motion_evt = event.get('event_label', event.get('flag_type', event_label))
                if isinstance(motion_evt, str) and motion_evt:
                    motion_evt_upper = motion_evt.upper().replace(' ', '_')
                    # Detailed explanations and advice for each event type from actual data
                    event_details = {
                        # From flagged_moments.csv event_label values
                        'AGGRESSIVE_BRAKING': {
                            'label': '🛑 Hard Brake',
                            'desc': 'You braked harder than usual — this can startle passengers and wear your brakes faster.',
                            'advice': '💡 Try to anticipate stops earlier. Look 2-3 vehicles ahead to spot slowing traffic.'
                        },
                        'RAPID_ACCELERATION': {
                            'label': '🚀 Quick Acceleration',
                            'desc': 'Sudden acceleration detected — this uses more fuel and can push passengers back.',
                            'advice': '💡 Ease onto the accelerator gently. Smooth starts save fuel and improve ratings.'
                        },
                        'HARSH_TURN': {
                            'label': '↩️ Sharp Turn',
                            'desc': 'You took that turn faster than ideal — passengers may have felt the lateral force.',
                            'advice': '💡 Slow down before turns, not during. Enter turns at a comfortable speed.'
                        },
                        'POTHOLE': {
                            'label': '🕳️ Pothole / Bump',
                            'desc': 'Road irregularity detected — your vehicle hit a pothole or speed bump.',
                            'advice': '💡 Watch for road damage ahead. Slow down over bumps to protect your suspension.'
                        },
                        # Legacy support for other formats
                        'AGGRESSIVE BRAKING': {
                            'label': '🛑 Hard Brake',
                            'desc': 'You braked harder than usual — this can startle passengers and wear your brakes faster.',
                            'advice': '💡 Try to anticipate stops earlier. Look 2-3 vehicles ahead to spot slowing traffic.'
                        },
                        'AGGRESSIVE ACCEL': {
                            'label': '🚀 Quick Acceleration',
                            'desc': 'Sudden acceleration detected — this uses more fuel and can push passengers back.',
                            'advice': '💡 Ease onto the accelerator gently. Smooth starts save fuel and improve ratings.'
                        },
                        'AGGRESSIVE LEFT TURN': {
                            'label': '↩️ Sharp Left Turn',
                            'desc': 'You took that left turn faster than ideal — passengers may have felt the lateral force.',
                            'advice': '💡 Slow down before turns, not during. Enter turns at a comfortable speed.'
                        },
                        'AGGRESSIVE RIGHT TURN': {
                            'label': '↪️ Sharp Right Turn',
                            'desc': 'You took that right turn faster than ideal — passengers may have felt the lateral force.',
                            'advice': '💡 Slow down before turns, not during. Enter turns at a comfortable speed.'
                        },
                        'AGG LEFT LANE CHANGE': {
                            'label': '⬅️ Quick Lane Change',
                            'desc': 'Rapid lane change detected — sudden lateral movement can unsettle passengers.',
                            'advice': '💡 Signal earlier and change lanes gradually. Check mirrors twice before moving.'
                        },
                        'AGG RIGHT LANE CHANGE': {
                            'label': '➡️ Quick Lane Change',
                            'desc': 'Rapid lane change detected — sudden lateral movement can unsettle passengers.',
                            'advice': '💡 Signal earlier and change lanes gradually. Check mirrors twice before moving.'
                        },
                    }
                    details = event_details.get(motion_evt_upper, event_details.get(motion_evt.upper(), {
                        'label': f'⚠️ {motion_evt.replace("_", " ").title()}',
                        'desc': f'Driving event detected: {motion_evt.replace("_", " ").lower()}.',
                        'advice': '💡 Maintain steady, predictable driving for passenger comfort.'
                    }))
                    event_type_label = details['label']
                    description = details['desc']
                    advice = details['advice']
            elif signal_type == 'AUDIO':
                # Get actual audio metrics for more nuanced categorization
                # Since all flagged events have all 4 layers, use the actual values
                zcr_val = event.get('zcr', 0)
                f0_std_val = event.get('f0_std', 0)
                db_val = event.get('db_deviation', 0)
                event_label_audio = str(event.get('event_label', '')).upper()
                severity_audio = str(event.get('severity', 'medium')).lower()
                
                # Convert to float if string
                try:
                    zcr_val = float(zcr_val) if zcr_val else 0
                    f0_std_val = float(f0_std_val) if f0_std_val else 0
                    db_val = float(db_val) if db_val else 0
                except (ValueError, TypeError):
                    zcr_val, f0_std_val, db_val = 0, 0, 0
                
                # Categorize based on actual values and severity
                if severity_audio == 'critical' or 'ACUTE' in event_label_audio:
                    # Critical/Acute safety events
                    event_type_label = "🚨 Acute Safety Alert"
                    description = "Significant cabin disturbance detected — this may require your attention for safety."
                    advice = "💡 Stay calm and assess the situation. Pull over safely if needed."
                elif zcr_val > 0.75 and db_val > 24:
                    # Very high voice activity + loud = heated argument
                    event_type_label = "🔊 Heated Argument"
                    description = "Loud, rapid speech patterns detected — passengers appear to be in a heated discussion."
                    advice = "💡 Stay neutral and professional. Don't engage — focus on the destination."
                elif zcr_val > 0.70 and f0_std_val > 65:
                    # High voice activity + pitch variation = tense conversation
                    event_type_label = "😤 Tense Conversation"
                    description = "Signs of frustration detected — voices raised with emotional undertones."
                    advice = "💡 Keep your cool. A calm, quiet response often de-escalates the situation."
                elif db_val > 25:
                    # Very loud cabin
                    event_type_label = "🗣️ Raised Voices"
                    description = "Elevated volume in the cabin — could be excitement or frustration."
                    advice = "💡 If it's tension, stay focused on driving. If it's excitement, that's okay!"
                elif f0_std_val > 70:
                    # High pitch variability = emotional tone
                    event_type_label = "😟 Emotional Tone"
                    description = "Stress markers detected in speech — passenger may be upset or anxious."
                    advice = "💡 A smooth, quiet ride helps anxious passengers relax. Drive gently."
                elif zcr_val > 0.68:
                    # Moderate voice activity = rushed speech
                    event_type_label = "💬 Rushed Speech"
                    description = "Rapid speech patterns detected — could indicate urgency or mild stress."
                    advice = "💡 If passenger seems rushed, confirm their destination and focus on efficient routing."
                else:
                    # General cabin stress
                    event_type_label = "🎤 Cabin Tension"
                    description = "Elevated stress levels detected in the cabin environment."
                    advice = "💡 Keep driving smoothly. A calm ride often calms passengers too."
            else:
                event_type_label = event_label
                advice = ""
            
            # Get flag_id for hover sync
            flag_id = event.get('flag_id', '')
            
            # Event card with hover sync - sends message to map on hover
            st.markdown(f"""
            <div class="event-card" style="background: rgba(30,30,30,0.9); border-left: 4px solid {border_color}; padding: 14px 16px; border-radius: 0 10px 10px 0; margin-bottom: 12px;"
                 onmouseenter="window.parent.postMessage({{type:'HIGHLIGHT_MARKER', marker_id:'{flag_id}'}}, '*')"
                 onmouseleave="window.parent.postMessage({{type:'RESET_MARKER', marker_id:'{flag_id}'}}, '*')">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: {color}; font-weight: 700; font-size: 1.05rem;">{icon} {event_type_label}</span>
                    <span style="color: #8E8E93; font-size: 0.9rem;">{time_str}</span>
                </div>
                <p style="color: #E5E5EA; font-size: 0.95rem; margin: 10px 0 8px 0; line-height: 1.4;">{description}</p>
                <p style="color: #30D158; font-size: 0.9rem; margin: 0; line-height: 1.4;">{advice}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Close scrollable container
        st.markdown("</div>", unsafe_allow_html=True)
    
    def _detect_events_at_progress(self, trip_id: str, progress: float):
        """
        Detect events using trained BoVW + Random Forest model.
        
        At each progress step:
        1. Get the accelerometer window for current position
        2. Run through K-Means to get codeword histogram
        3. Classify using Random Forest
        4. Add detected event to session state
        """
        # Initialize model if needed (singleton pattern)
        if 'motion_detector' not in st.session_state:
            if BOVW_AVAILABLE:
                detector = BoVWMotionDetector()
                if detector.load_models():
                    st.session_state.motion_detector = detector
                else:
                    st.session_state.motion_detector = None
            else:
                st.session_state.motion_detector = None
        
        detector = st.session_state.motion_detector
        
        # Get trip accelerometer data
        trip_accel = self.accel_df[self.accel_df['trip_id'] == trip_id].copy()
        if len(trip_accel) == 0:
            return
        
        trip_accel = trip_accel.sort_values('elapsed_seconds')
        total_time = trip_accel['elapsed_seconds'].max()
        
        if total_time <= 0:
            return
        
        # Calculate current time position in trip
        current_time = (progress / 100.0) * total_time
        
        # Get last checked time (to avoid re-processing)
        last_checked_key = f'last_checked_{trip_id}'
        last_checked = st.session_state.get(last_checked_key, -5.0)
        
        # Only classify if we've moved at least 3 seconds forward
        if current_time - last_checked < 3.0:
            return
        
        st.session_state[last_checked_key] = current_time
        
        # Get window of data around current position (~5 seconds)
        window_start = max(0, current_time - 2.5)
        window_end = min(total_time, current_time + 2.5)
        
        window_data = trip_accel[
            (trip_accel['elapsed_seconds'] >= window_start) &
            (trip_accel['elapsed_seconds'] <= window_end)
        ]
        
        if len(window_data) < 2:
            return
        
        # Use BoVW model if available
        if detector is not None:
            result = detector.classify_window_realtime(window_data)
            event_label = result['event_label']
            confidence = result['confidence']
            top_codewords = result.get('top_codewords', [])
            
            # Only report non-NORMAL events with decent confidence
            if event_label != 'NORMAL' and confidence > 0.4:
                # Check if we already detected this event type recently
                recent_events = st.session_state.detected_events[-3:] if st.session_state.detected_events else []
                already_detected = any(e.get('event_label') == event_label for e in recent_events)
                
                if not already_detected:
                    # Get current GPS position
                    current_pos = window_data.iloc[len(window_data)//2]
                    
                    # Determine severity based on event type
                    severity_map = {
                        'AGGRESSIVE_BRAKING': 'high',
                        'AGGRESSIVE_ACCEL': 'medium',
                        'AGGRESSIVE_LEFT_TURN': 'medium',
                        'AGGRESSIVE_RIGHT_TURN': 'medium',
                        'AGG_LEFT_LANE_CHANGE': 'low',
                        'AGG_RIGHT_LANE_CHANGE': 'low',
                    }
                    severity = severity_map.get(event_label, 'medium')
                    
                    # Build explanation with codeword info
                    codeword_str = ', '.join([f'CW{cw[0]}:{cw[1]:.2f}' for cw in top_codewords[:3]])
                    explanation = f"ML detected: {event_label} (confidence: {confidence:.0%})"
                    if codeword_str:
                        explanation += f" | Top codewords: [{codeword_str}]"
                    
                    st.session_state.detected_events.append({
                        'event_label': event_label,
                        'severity': severity,
                        'signal_type': 'MOTION',
                        'motion_event': event_label,
                        'audio_event': '',
                        'explanation': explanation,
                        'elapsed_seconds': current_time,
                        'speed_kmh': current_pos.get('speed_kmh', 0),
                        'confidence': confidence,
                        'source': 'bovw_model'
                    })
        else:
            # Fallback: Use pre-computed flagged moments from CSV
            if len(self.flagged_moments) == 0:
                return
            
            trip_events = self.flagged_moments[self.flagged_moments['trip_id'] == trip_id]
            if len(trip_events) == 0:
                return
            
            # Calculate how many events should be detected by now
            n_total = len(trip_events)
            n_detected = int(n_total * progress / 100)
            
            current_count = len(st.session_state.detected_events)
            
            if n_detected > current_count:
                # Add new events with full details from CSV
                new_events = trip_events.iloc[current_count:n_detected]
                
                for _, evt in new_events.iterrows():
                    event_label = evt.get('event_label', evt.get('event_type', 'DRIVING_EVENT'))
                    severity = evt.get('severity', 'medium')
                    signal_type = evt.get('signal_type', 'MOTION')
                    motion_event = evt.get('motion_event', '')
                    audio_event = evt.get('audio_event', '')
                    explanation = evt.get('explanation', '')
                    elapsed_sec = evt.get('elapsed_seconds', 0)
                    speed = evt.get('speed_kmh', 0)
                    layers_fired = evt.get('layers_fired', '')
                    
                    st.session_state.detected_events.append({
                        'event_label': event_label,
                        'severity': severity,
                        'signal_type': signal_type,
                        'motion_event': motion_event,
                        'audio_event': audio_event,
                        'explanation': explanation,
                        'elapsed_seconds': elapsed_sec,
                        'layers_fired': layers_fired,
                        'speed_kmh': speed,
                        'source': 'csv_fallback'
                    })

    # =========================================================================
    # VIEW 5: Post-Trip Insights
    # =========================================================================
    
    def render_post_trip_insights(self):
        """Render post-trip insights with PRD-compliant stress/safety scoring."""
        trip_id = st.session_state.current_trip
        trip_info = self.trips_df[self.trips_df['trip_id'] == trip_id].iloc[0]
        
        # Get flagged events for this trip
        if len(self.flagged_moments) > 0:
            trip_flags = self.flagged_moments[self.flagged_moments['trip_id'] == trip_id]
        else:
            trip_flags = pd.DataFrame()
        
        # Count event types using signal_type column
        n_accel = 0
        n_audio = 0
        n_compound = 0
        
        if len(trip_flags) > 0:
            # Count motion events (signal_type == 'MOTION')
            if 'signal_type' in trip_flags.columns:
                n_accel = len(trip_flags[trip_flags['signal_type'] == 'MOTION'])
                n_audio = len(trip_flags[trip_flags['signal_type'] == 'AUDIO'])
                # Count compound events (COMPOUND signal type)
                n_compound = len(trip_flags[trip_flags['signal_type'] == 'COMPOUND'])
        
        # Stress score based on event counts and severity
        # Formula: weighted sum of events, normalized to 0-1 scale
        # More events = higher stress, compound events weighted highest
        duration = trip_info['duration_min']
        total_events = n_accel + n_audio + n_compound
        
        if total_events > 0:
            # Calculate stress based on event density and severity weighting
            # Compound events are most severe (weight 3), motion (weight 2), audio (weight 1.5)
            weighted_events = (2.0 * n_accel + 1.5 * n_audio + 3.0 * n_compound)
            # Normalize: expect ~5-10 weighted events per hour for moderate stress
            events_per_hour = weighted_events / (duration / 60) if duration > 0 else 0
            # Scale to 0-1: 0 events = 0, 10+ events/hr = 1.0
            stress_score = round(min(events_per_hour / 10, 1.0), 2)
        else:
            stress_score = 0.0
        
        # Safety score (inverse of stress, 0-100)
        safety_score = round((1 - stress_score) * 100)
        
        # Stress band
        if stress_score < 0.40:
            stress_band = 'LOW'
            stress_color = '#30D158'
            stress_label = 'Good Trip'
        elif stress_score < 0.70:
            stress_band = 'MODERATE'
            stress_color = '#FF9F0A'
            stress_label = 'Moderate Stress'
        else:
            stress_band = 'HIGH'
            stress_color = '#FF453A'
            stress_label = 'High Stress'
        
        # Calculate earnings velocity and goal progress
        # Note: total_earnings was already updated when trip completed (in render_trip_simulation)
        total_earnings = st.session_state.total_earnings
        target = st.session_state.target_earnings
        shift_hours = st.session_state.shift_hours
        
        # Time elapsed in shift - use trip duration as minimum elapsed time
        # Since we're simulating, elapsed time = sum of completed trip durations
        shift_start = st.session_state.get('shift_actual_start', datetime.now())
        
        # Calculate actual trip time elapsed (sum of all completed trip durations)
        completed_trip_ids = st.session_state.completed_trips
        total_trip_minutes = 0
        for tid in completed_trip_ids:
            trip_data = self.trips_df[self.trips_df['trip_id'] == tid]
            if len(trip_data) > 0:
                total_trip_minutes += trip_data.iloc[0]['duration_min']
        
        # Use trip durations as elapsed time (more realistic for simulation)
        elapsed_hours = max(0.1, total_trip_minutes / 60)
        
        # Current earning velocity (₹/hr)
        current_velocity = total_earnings / elapsed_hours if elapsed_hours > 0 else 0
        
        # Target velocity needed to reach goal
        remaining_hours = max(0.1, shift_hours - elapsed_hours)
        remaining_to_goal = max(0, target - total_earnings)
        required_velocity = remaining_to_goal / remaining_hours if remaining_hours > 0 else 0
        
        # Goal forecast
        projected_earnings = total_earnings + (current_velocity * remaining_hours)
        on_track = projected_earnings >= target
        goal_percent = (total_earnings / target * 100) if target > 0 else 0
        
        # Header
        st.markdown(f'<h1 class="main-header">Trip Complete</h1>', unsafe_allow_html=True)
        
        pickup = trip_info.get('pickup_location', 'Pickup')
        dropoff = trip_info.get('dropoff_location', 'Dropoff')
        st.markdown(f'<p class="sub-header">{pickup} → {dropoff}</p>', unsafe_allow_html=True)
        
        # Top metrics with LARGER fonts
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">EARNED</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #30D158;">₹{trip_info["fare"]:.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">DURATION</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">{trip_info["duration_min"]:.0f}<span style="font-size: 1.2rem; color: #8E8E93;"> min</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">DISTANCE</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">{trip_info["distance_km"]:.1f}<span style="font-size: 1.2rem; color: #8E8E93;"> km</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            velocity = trip_info['fare'] / (trip_info['duration_min'] / 60)
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">TRIP RATE</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">₹{velocity:.0f}<span style="font-size: 1.2rem; color: #8E8E93;">/hr</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Earnings Velocity & Goal Progress Section
        st.markdown("### 📊 Today's Progress")
        
        vel_c1, vel_c2, vel_c3 = st.columns(3)
        
        with vel_c1:
            vel_color = '#30D158' if current_velocity >= required_velocity else '#FF9F0A'
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">YOUR EARNING RATE</p>
                <p style="font-size: 2rem; font-weight: 700; color: {vel_color};">₹{current_velocity:.0f}/hr</p>
                <p style="color: #8E8E93; font-size: 0.85rem;">Current velocity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with vel_c2:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">TOTAL EARNED TODAY</p>
                <p style="font-size: 2rem; font-weight: 700; color: #FFFFFF;">₹{total_earnings:,.0f}</p>
                <p style="color: #8E8E93; font-size: 0.85rem;">{goal_percent:.0f}% of ₹{target:,.0f} goal</p>
            </div>
            """, unsafe_allow_html=True)
        
        with vel_c3:
            if on_track:
                track_color = '#30D158'
                track_icon = '✅'
                track_text = 'On Track!'
            else:
                track_color = '#FF9F0A'
                track_icon = '⚡'
                track_text = 'Push Needed'
            
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">GOAL STATUS</p>
                <p style="font-size: 2rem; font-weight: 700; color: {track_color};">{track_icon} {track_text}</p>
                <p style="color: #8E8E93; font-size: 0.85rem;">₹{remaining_to_goal:,.0f} remaining</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Goal prediction message
        if remaining_to_goal > 0:
            time_to_goal = remaining_to_goal / current_velocity if current_velocity > 0 else float('inf')
            if time_to_goal <= remaining_hours:
                hrs = int(time_to_goal)
                mins = int((time_to_goal - hrs) * 60)
                st.markdown(f"""
                <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <p style="color: #30D158; font-weight: 700; font-size: 1rem;">🎯 You're on pace to hit your goal!</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">At ₹{current_velocity:.0f}/hr, you'll reach ₹{target:,.0f} in about <strong>{hrs}h {mins}m</strong>.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                needed_velocity = remaining_to_goal / remaining_hours if remaining_hours > 0 else 0
                st.markdown(f"""
                <div style="background: #FF9F0A22; border-left: 4px solid #FF9F0A; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <p style="color: #FF9F0A; font-weight: 700; font-size: 1rem;">📈 Speed up to hit your goal</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">You need ₹{needed_velocity:.0f}/hr to reach ₹{target:,.0f} before shift ends. 
                    Consider shorter, high-fare trips or surge areas.</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <p style="color: #30D158; font-weight: 700; font-size: 1.1rem;">🎉 Goal Achieved!</p>
                <p style="color: #E5E5EA; font-size: 0.95rem;">Amazing work! You've hit ₹{target:,.0f}. Every trip now is bonus earnings!</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Trip Health")
            # Safety score card
            score_color = '#30D158' if safety_score >= 80 else '#FF9F0A' if safety_score >= 60 else '#FF453A'
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">SAFETY SCORE</p>
                <p style="font-size: 3rem; font-weight: 700; color: {score_color};">{safety_score}</p>
                <p style="color: #8E8E93;">out of 100</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Stress score card
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">STRESS LEVEL</p>
                <p style="font-size: 2rem; font-weight: 700; color: {stress_color};">{stress_label}</p>
                <p style="color: #8E8E93;">{stress_score:.2f} / 1.0</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Event breakdown with cleaner labels
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">WHAT HAPPENED</p>
                <p style="color: #FF453A; font-size: 0.95rem;">🚗 Sudden maneuvers: {n_accel}</p>
                <p style="color: #FF9F0A; font-size: 0.95rem;">🎤 Cabin stress moments: {n_audio}</p>
                <p style="color: #bf5af2; font-size: 0.95rem;">⚡ Combined alerts: {n_compound}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### What We Noticed")
            self._generate_recommendations(
                trip_id, trip_info, n_accel, n_audio,
                n_compound, stress_score, stress_band
            )
        
        st.markdown("---")
        
        # Save stress score for next trip recommendations
        st.session_state['last_trip_stress'] = stress_score
        
        # Break recommendation based on stress band
        if stress_band == 'HIGH':
            st.markdown("""
            <div style="background: #FF453A22; border-left: 4px solid #FF453A; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                <p style="color: #FF453A; font-weight: 700; font-size: 1rem;">⚠️ High Stress Trip</p>
                <p style="color: #E5E5EA; font-size: 0.95rem;">Take a 5–10 minute break before your next trip. A quick stretch or water break can help you reset.</p>
            </div>
            """, unsafe_allow_html=True)
        elif stress_band == 'MODERATE':
            remaining_hours_check = st.session_state.shift_hours - (
                (datetime.now() - st.session_state.get('shift_actual_start', datetime.now())).total_seconds() / 3600
            )
            if remaining_hours_check <= 2:
                st.markdown("""
                <div style="background: #FF9F0A22; border-left: 4px solid #FF9F0A; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
                    <p style="color: #FF9F0A; font-weight: 700; font-size: 1rem;">⏱️ Running Low on Time</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">With limited time left, consider shorter trips nearby to maximize your earnings per hour.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Action buttons with better styling
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🚗 Next Trip", use_container_width=True):
                st.session_state.current_trip = None
                st.session_state.simulation_progress = 0
                st.session_state.detected_events = []
                st.session_state.view = 'trip_selection'
                st.rerun()
        with c2:
            if st.button("📊 Day Summary", use_container_width=True):
                st.session_state.view = 'end_of_day'
                st.rerun()
        with c3:
            if st.button("🏁 End Shift", type="primary", use_container_width=True):
                st.session_state.view = 'end_of_day'
                st.rerun()
    
    def _generate_recommendations(self, trip_id, trip_info, n_accel, n_audio, n_compound, stress_score, stress_band):
        """Generate up to 3 prioritised recommendations with driver-friendly language."""
        recommendations = []
        passenger_rating = trip_info.get('passenger_rating', 5.0)
        
        # Priority 1: Multi-signal events (was "compound")
        if n_compound >= 1:
            recommendations.append({
                'color': '#FF453A',
                'icon': '⚡',
                'title': 'Take a Moment to Reset',
                'body': f'We detected {n_compound} moment{"s" if n_compound > 1 else ""} where driving stress and cabin tension happened together. '
                        'These can be draining — consider a quick break before your next trip.'
            })
        
        # Priority 2: Harsh driving + low rating
        if n_accel >= 2 and passenger_rating <= 4.0:
            recommendations.append({
                'color': '#FF453A',
                'icon': '🚗',
                'title': 'Smoother Rides = Better Ratings',
                'body': f'This trip had {n_accel} sudden maneuvers and a {passenger_rating:.1f}★ rating. '
                        'Try braking a bit earlier at stops and taking turns more gently — passengers notice the difference!'
            })
        elif n_accel >= 2:
            recommendations.append({
                'color': '#FF9F0A',
                'icon': '🚗',
                'title': 'Driving Tip',
                'body': f'We noticed {n_accel} sudden movements this trip. '
                        'Anticipating stops earlier can make the ride smoother and reduce wear on your vehicle too.'
            })
        
        # Priority 3: Audio events
        if n_audio >= 1:
            if passenger_rating <= 4.0:
                recommendations.append({
                    'color': '#FF9F0A',
                    'icon': '🎧',
                    'title': 'Cabin Comfort Matters',
                    'body': 'High cabin stress was detected. A quieter, calmer environment '
                            'often leads to better passenger ratings. Consider soft music or a peaceful ride.'
                })
            else:
                recommendations.append({
                    'color': '#30D158',
                    'icon': '💪',
                    'title': 'You Handled It Well',
                    'body': 'Some cabin stress was detected, but you still got a good rating. '
                            'Nice work staying calm under pressure!'
                })
        
        # Priority 4: Good trip
        if not recommendations:
            recommendations.append({
                'color': '#30D158',
                'icon': '🌟',
                'title': 'Excellent Trip!',
                'body': 'Smooth driving, calm cabin — you nailed it! Keep this up for great ratings and earnings.'
            })
        
        # Show max 3 with improved styling
        for rec in recommendations[:3]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2C2C2E 0%, #1C1C1E 100%); 
                        border-left: 4px solid {rec['color']};
                        border-radius: 10px; padding: 1rem 1.25rem; margin: 0.75rem 0;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.3);">
                <p style="color: {rec['color']}; font-weight: 700; font-size: 1rem; margin-bottom: 8px;">
                    {rec['icon']} {rec['title']}
                </p>
                <p style="color: #E5E5EA; font-size: 0.95rem; line-height: 1.5; margin: 0;">
                    {rec['body']}
                </p>
            </div>
            """, unsafe_allow_html=True)

    # =========================================================================
    # VIEW 6: End-of-Day Summary
    # =========================================================================
    
    def render_end_of_day_summary(self):
        """Render end-of-day summary."""
        st.markdown('<h1 class="main-header">Shift Summary</h1>', unsafe_allow_html=True)
        
        completed = st.session_state.completed_trips
        total_earnings = st.session_state.total_earnings
        target = st.session_state.target_earnings
        
        if completed:
            completed_df = self.trips_df[self.trips_df['trip_id'].isin(completed)]
            total_duration = completed_df['duration_min'].sum()
            total_distance = completed_df['distance_km'].sum()
            avg_rate = total_earnings / (total_duration / 60) if total_duration > 0 else 0
        else:
            total_duration = total_distance = avg_rate = 0
        
        goal_pct = (total_earnings / target * 100) if target > 0 else 0
        
        # Summary cards with LARGE font sizes
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="uber-card-highlight">
                <p class="metric-label">EARNED</p>
                <p style="font-size: 3rem; font-weight: 700; color: #30D158;">₹{total_earnings:,.0f}</p>
                <p style="color:#8E8E93;">{goal_pct:.0f}% of goal</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">TRIPS</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">{len(completed)}</p>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">TIME</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">{total_duration:.0f}<span style="font-size: 1.2rem; color: #8E8E93;"> min</span></p>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">AVG RATE</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #FFFFFF;">₹{avg_rate:.0f}<span style="font-size: 1.2rem; color: #8E8E93;">/hr</span></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Goal progress visualization
        st.markdown("### 🎯 Goal Progress")
        st.progress(min(goal_pct / 100, 1.0))
        
        if total_earnings >= target:
            st.markdown(f"""
            <div style="background: #30D15822; border-left: 4px solid #30D158; border-radius: 8px; padding: 1.25rem; margin: 1rem 0;">
                <p style="color: #30D158; font-weight: 700; font-size: 1.5rem;">🎉 Goal Achieved!</p>
                <p style="color: #E5E5EA; font-size: 1rem;">You earned ₹{total_earnings - target:,.0f} above your target. Outstanding work!</p>
            </div>
            """, unsafe_allow_html=True)
            st.balloons()
        else:
            remaining = target - total_earnings
            # Calculate what it would have taken
            if avg_rate > 0:
                hrs_needed = remaining / avg_rate
                st.markdown(f"""
                <div style="background: #FF9F0A22; border-left: 4px solid #FF9F0A; border-radius: 8px; padding: 1.25rem; margin: 1rem 0;">
                    <p style="color: #FF9F0A; font-weight: 700; font-size: 1.25rem;">₹{remaining:,.0f} short of goal</p>
                    <p style="color: #E5E5EA; font-size: 0.95rem;">At your ₹{avg_rate:.0f}/hr rate, about {hrs_needed:.1f} more hours would have gotten you there. 
                    Tomorrow's a new day!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="uber-card">
                    <p style="color: #FF9F0A;">₹{remaining:,.0f} more needed to reach your goal</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Earnings velocity summary
        st.markdown("### 📈 Earnings Velocity Analysis")
        
        vel_c1, vel_c2 = st.columns(2)
        with vel_c1:
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">YOUR AVERAGE RATE</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: #30D158;">₹{avg_rate:.0f}/hr</p>
                <p style="color: #8E8E93; font-size: 0.85rem;">Across {len(completed)} trip{"s" if len(completed) != 1 else ""}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with vel_c2:
            # Target velocity that was needed
            target_velocity = target / (st.session_state.shift_hours) if st.session_state.shift_hours > 0 else 0
            vel_diff = avg_rate - target_velocity
            vel_color = '#30D158' if vel_diff >= 0 else '#FF9F0A'
            vel_sign = '+' if vel_diff >= 0 else ''
            
            st.markdown(f"""
            <div class="uber-card">
                <p class="metric-label">VS TARGET RATE</p>
                <p style="font-size: 2.5rem; font-weight: 700; color: {vel_color};">{vel_sign}₹{vel_diff:.0f}/hr</p>
                <p style="color: #8E8E93; font-size: 0.85rem;">Target was ₹{target_velocity:.0f}/hr</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Trip-by-trip breakdown
        if completed:
            st.markdown("### 🚗 Trip Breakdown")
            for trip_id in completed:
                trip_data = self.trips_df[self.trips_df['trip_id'] == trip_id]
                if len(trip_data) > 0:
                    trip = trip_data.iloc[0]
                    trip_velocity = trip['fare'] / (trip['duration_min'] / 60) if trip['duration_min'] > 0 else 0
                    pickup = trip.get('pickup_location', 'Pickup')
                    dropoff = trip.get('dropoff_location', 'Dropoff')
                    
                    st.markdown(f"""
                    <div style="background: #2C2C2E; border-radius: 8px; padding: 0.75rem 1rem; margin: 0.5rem 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <p style="color: #FFFFFF; font-weight: 500; margin: 0;">{pickup} → {dropoff}</p>
                                <p style="color: #8E8E93; font-size: 0.8rem; margin: 0;">{trip['duration_min']:.0f} min · {trip['distance_km']:.1f} km</p>
                            </div>
                            <div style="text-align: right;">
                                <p style="color: #30D158; font-weight: 600; font-size: 1.1rem; margin: 0;">₹{trip['fare']:.0f}</p>
                                <p style="color: #8E8E93; font-size: 0.8rem; margin: 0;">₹{trip_velocity:.0f}/hr</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🚗 Back to Trips", use_container_width=True):
                st.session_state.view = 'trip_selection'
                st.rerun()
        with c2:
            if st.button("🚪 End Shift & Sign Out", type="primary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _render_shift_progress_bar(self):
        """Render enhanced shift progress bar with visual indicators."""
        if not st.session_state.shift_started:
            return
        
        total = st.session_state.total_earnings
        target = st.session_state.target_earnings
        progress = min(total / target, 1.0) if target > 0 else 0
        remaining = max(0, target - total)
        percent = progress * 100
        
        # Color based on progress
        if percent >= 100:
            bar_color = '#30D158'  # Green - goal achieved
            status_text = '🎉 Goal Achieved!'
            status_color = '#30D158'
        elif percent >= 75:
            bar_color = '#30D158'  # Green - on track
            status_text = '🔥 Almost there!'
            status_color = '#30D158'
        elif percent >= 50:
            bar_color = '#007AFF'  # Blue - halfway
            status_text = '💪 Keep going!'
            status_color = '#007AFF'
        elif percent >= 25:
            bar_color = '#FF9F0A'  # Orange - need to pick up pace
            status_text = '⏱️ Pick up the pace'
            status_color = '#FF9F0A'
        else:
            bar_color = '#8E8E93'  # Gray - just started
            status_text = '🚀 Let\'s go!'
            status_color = '#8E8E93'
        
        st.markdown(f"""
        <div style="background: #1C1C1E; border-radius: 12px; padding: 16px 20px; margin-bottom: 16px; border: 1px solid #2C2C2E;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <div>
                    <p style="color: #8E8E93; font-size: 0.8rem; margin: 0;">YOUR PROGRESS</p>
                    <p style="color: {status_color}; font-size: 1rem; font-weight: 600; margin: 4px 0 0 0;">{status_text}</p>
                </div>
                <div style="text-align: right;">
                    <p style="color: #30D158; font-size: 1.8rem; font-weight: 700; margin: 0;">₹{total:,.0f}</p>
                    <p style="color: #8E8E93; font-size: 0.85rem; margin: 0;">of ₹{target:,} target</p>
                </div>
            </div>
            <div style="background: #2C2C2E; border-radius: 8px; height: 12px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {bar_color}, {bar_color}dd); width: {percent:.1f}%; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 8px;">
                <span style="color: #8E8E93; font-size: 0.8rem;">{percent:.0f}% complete</span>
                <span style="color: #8E8E93; font-size: 0.8rem;">₹{remaining:,.0f} remaining</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================================================================
    # Main Router
    # =========================================================================
    
    def run(self):
        """Main application router."""
        # Sidebar
        with st.sidebar:
            st.markdown('<h2 style="color:#FFFFFF; font-weight:700;">Driver Pulse</h2>', unsafe_allow_html=True)
            st.markdown("---")
            
            if st.session_state.logged_in:
                st.markdown(f"""
                <div>
                    <p style="color: #8E8E93; font-size: 0.75rem;">DRIVER</p>
                    <p style="color: #FFFFFF; font-weight: 500;">{st.session_state.driver_name}</p>
                    <p style="color: #8E8E93;">{st.session_state.driver_id}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.session_state.shift_started:
                    st.markdown("---")
                    st.markdown(f"""
                    <div>
                        <p style="color: #8E8E93; font-size: 0.75rem;">SHIFT STATS</p>
                        <p>Target: <span style="color: #30D158;">₹{st.session_state.target_earnings:,}</span></p>
                        <p>Earned: <span style="font-weight: 600;">₹{st.session_state.total_earnings:,.0f}</span></p>
                        <p>Trips: {len(st.session_state.completed_trips)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                if st.button("Sign Out", use_container_width=True):
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
        
        # Route to correct view
        if not st.session_state.logged_in:
            self.render_login()
        elif not st.session_state.shift_started:
            self.render_pre_shift_setup()
        elif st.session_state.get('view') == 'simulation':
            self.render_trip_simulation()
        elif st.session_state.get('view') == 'post_trip':
            self.render_post_trip_insights()
        elif st.session_state.get('view') == 'end_of_day':
            self.render_end_of_day_summary()
        else:
            self.render_trip_selection()


def main():
    app = DriverPulseApp()
    app.run()


if __name__ == '__main__':
    main()
