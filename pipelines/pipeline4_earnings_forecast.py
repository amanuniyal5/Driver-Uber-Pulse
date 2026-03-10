"""
Pipeline 4: Earnings Velocity Forecast (19-Step Deterministic Pipeline)
=======================================================================
Implements the complete earnings forecast pipeline as per PDF v1.0:

Velocity Signals:
1. V_driver: Past 7-day rolling average (driver's historical rate) - ₹/hr
2. V_location: Area-specific benchmark (what drivers earn in this zone) - ₹/hr
3. V_opportunity: λ × P_avg (trips_per_hour × mean_fare) - ₹/hr [Step 10]
4. V_area: V_opportunity × distance_factor - ₹/hr [Step 12]
5. V_recent: Last 2-hour moving average (session momentum) - ₹/hr

Composite Formula (Step 13):
V_expected = 0.4 × V_driver + 0.2 × V_location + 0.2 × V_opportunity + 0.2 × V_area

Forecast Labels:
- "ahead": V_recent > V_expected × 1.1
- "on_track": V_expected × 0.9 ≤ V_recent ≤ V_expected × 1.1
- "at_risk": V_recent < V_expected × 0.9

Cold-Start Detection:
- is_cold_start = True if total_trips < 3 OR shift_time < 15 min
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import constants - add parent dir to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from constants import (
    V_EXPECTED_DRIVER_WEIGHT, V_EXPECTED_LOCATION_WEIGHT, V_EXPECTED_OPPORTUNITY_WEIGHT,
    V_EXPECTED_AREA_WEIGHT, GOAL_AHEAD_THRESHOLD, GOAL_ON_TRACK_MIN
)

# Local constants with sensible defaults
V_DRIVER_LOOKBACK_DAYS = 7
DEFAULT_V_LOCATION = {
    'Mumbai': 300,
    'Bangalore': 280,
    'Delhi': 320,
    'Hyderabad': 270,
    'Chennai': 260,
}
DEFAULT_V_DRIVER = 280  # ₹/hour
DEFAULT_V_OPPORTUNITY = 1.0  # Multiplier
PEAK_HOUR_BONUS_MULTIPLIER = 1.15
V_RECENT_LOOKBACK_HOURS = 2
WEIGHT_V_DRIVER = V_EXPECTED_DRIVER_WEIGHT
WEIGHT_V_LOCATION = V_EXPECTED_LOCATION_WEIGHT
WEIGHT_V_OPPORTUNITY = V_EXPECTED_OPPORTUNITY_WEIGHT
WEIGHT_V_AREA = V_EXPECTED_AREA_WEIGHT
FORECAST_AHEAD_THRESHOLD = GOAL_AHEAD_THRESHOLD
FORECAST_AT_RISK_THRESHOLD = GOAL_ON_TRACK_MIN
DEFAULT_DAILY_GOAL = 2000  # ₹
DEFAULT_WEEKLY_GOAL = 12000  # ₹


class EarningsForecastPipeline:
    """
    19-Step Deterministic Earnings Forecast Pipeline.
    """
    
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'driver_pulse_data')
        self.output_dir = os.path.join(base_dir, 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Data
        self.drivers_df = None
        self.trips_df = None
        self.earnings_log = None
        self.driver_goals = None
        self.market_context = None
        
    def load_data(self):
        """Load all earnings-related data."""
        # Drivers
        drivers_path = os.path.join(self.data_dir, 'drivers', 'drivers.csv')
        self.drivers_df = pd.read_csv(drivers_path)
        print(f"Loaded {len(self.drivers_df)} drivers")
        
        # Trips
        trips_path = os.path.join(self.data_dir, 'trips', 'trips.csv')
        self.trips_df = pd.read_csv(trips_path)
        self.trips_df['start_time'] = pd.to_datetime(self.trips_df['start_time'])
        self.trips_df['end_time'] = pd.to_datetime(self.trips_df['end_time'])
        print(f"Loaded {len(self.trips_df)} trips")
        
        # Earnings velocity log
        earnings_path = os.path.join(self.data_dir, 'earnings', 'earnings_velocity_log.csv')
        if os.path.exists(earnings_path):
            self.earnings_log = pd.read_csv(earnings_path)
            self.earnings_log['timestamp'] = pd.to_datetime(self.earnings_log['timestamp'])
            print(f"Loaded {len(self.earnings_log)} earnings log entries")
        else:
            print("No earnings log found, will compute from trips")
            self.earnings_log = pd.DataFrame()
        
        # Driver goals
        goals_path = os.path.join(self.data_dir, 'earnings', 'driver_goals.csv')
        if os.path.exists(goals_path):
            self.driver_goals = pd.read_csv(goals_path)
            print(f"Loaded goals for {len(self.driver_goals)} drivers")
        else:
            print("No driver goals found, will use defaults")
            self.driver_goals = pd.DataFrame()
        
        # Market context
        market_path = os.path.join(self.data_dir, 'market', 'market_context.csv')
        if os.path.exists(market_path):
            self.market_context = pd.read_csv(market_path)
            # Use snapshot_time if timestamp doesn't exist
            if 'timestamp' not in self.market_context.columns and 'snapshot_time' in self.market_context.columns:
                self.market_context['timestamp'] = pd.to_datetime(self.market_context['snapshot_time'])
            elif 'timestamp' in self.market_context.columns:
                self.market_context['timestamp'] = pd.to_datetime(self.market_context['timestamp'])
            print(f"Loaded {len(self.market_context)} market context entries")
        else:
            print("No market context found, will use defaults")
            self.market_context = pd.DataFrame()
        
        return self.drivers_df, self.trips_df
    
    # =========================================================================
    # STEP 1-5: Base Velocity Computation
    # =========================================================================
    
    def compute_v_driver(self, driver_id, current_time=None):
        """
        Step 1: V_driver - Past 7-day rolling average velocity.
        Units: ₹/hour
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get driver's trips in last 7 days
        seven_days_ago = current_time - timedelta(days=V_DRIVER_LOOKBACK_DAYS)
        
        driver_trips = self.trips_df[
            (self.trips_df['driver_id'] == driver_id) &
            (self.trips_df['start_time'] >= seven_days_ago) &
            (self.trips_df['start_time'] <= current_time)
        ]
        
        if len(driver_trips) == 0:
            # Default to city average
            city = self.drivers_df[self.drivers_df['driver_id'] == driver_id]['city'].iloc[0]
            return DEFAULT_V_LOCATION.get(city, DEFAULT_V_DRIVER)
        
        # Calculate earnings velocity (₹/hour)
        total_earnings = driver_trips['fare'].sum()
        total_hours = driver_trips['duration_min'].sum() / 60
        
        if total_hours > 0:
            v_driver = total_earnings / total_hours
        else:
            v_driver = DEFAULT_V_DRIVER
        
        return round(v_driver, 2)
    
    def compute_v_location(self, city, current_time=None):
        """
        Step 2: V_location - Area-specific benchmark velocity.
        Based on what drivers typically earn in this city/zone.
        """
        # Use city-specific benchmarks from constants
        v_location = DEFAULT_V_LOCATION.get(city, DEFAULT_V_DRIVER)
        
        # Adjust based on market context if available
        if self.market_context is not None and len(self.market_context) > 0:
            city_market = self.market_context[self.market_context['city'] == city]
            if len(city_market) > 0:
                latest = city_market.iloc[-1]
                v_location *= (1 + latest.get('demand_multiplier', 0) * 0.1)
        
        return round(v_location, 2)
    
    def compute_v_opportunity(self, city, current_time=None):
        """
        Step 10: V_opportunity - λ × P_avg (trips_per_hour × mean_fare).
        
        This represents the earning opportunity rate based on market conditions.
        Units: ₹/hour
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Default values
        lam = 2.0  # trips per hour
        p_avg = 150.0  # mean fare in ₹
        
        if self.market_context is not None and len(self.market_context) > 0:
            # Filter by city and time
            city_market = self.market_context[
                (self.market_context['city'] == city) &
                (self.market_context['timestamp'] <= current_time)
            ]
            if len(city_market) > 0:
                latest = city_market.iloc[-1]
                lam = latest.get('trips_per_hour', 2.0)
                p_avg = latest.get('mean_fare_inr', 150.0)
        
        # V_opp = λ * P_avg
        v_opportunity = lam * p_avg
        
        return round(v_opportunity, 2)
    
    def compute_v_area(self, v_opportunity, distance_km=5.0):
        """
        Step 12: V_area - Area-adjusted opportunity velocity.
        
        V_area = V_opportunity × distance_factor
        where distance_factor = 1 / (1 + d / d_scale)
        
        This reduces opportunity based on distance from hotspots.
        Units: ₹/hour
        """
        d_scale = 10.0  # scaling factor in km
        dist_factor = 1 / (1 + distance_km / d_scale)
        v_area = v_opportunity * dist_factor
        
        return round(v_area, 2)
    
    def compute_v_recent(self, driver_id, current_time=None):
        """
        Step 5: V_recent - Last 2-hour moving average.
        Represents current session momentum.
        """
        if current_time is None:
            current_time = datetime.now()
        
        two_hours_ago = current_time - timedelta(hours=V_RECENT_LOOKBACK_HOURS)
        
        # Get recent trips
        recent_trips = self.trips_df[
            (self.trips_df['driver_id'] == driver_id) &
            (self.trips_df['start_time'] >= two_hours_ago) &
            (self.trips_df['start_time'] <= current_time)
        ]
        
        if len(recent_trips) == 0:
            return 0  # No recent activity
        
        total_earnings = recent_trips['fare'].sum()
        total_hours = recent_trips['duration_min'].sum() / 60
        
        if total_hours > 0:
            v_recent = total_earnings / total_hours
        else:
            v_recent = 0
        
        return round(v_recent, 2)
    
    # =========================================================================
    # STEP 6: Composite V_expected
    # =========================================================================
    
    def compute_v_expected(self, v_driver, v_location, v_opportunity, v_area):
        """
        Step 13: Compute composite V_expected using simple weighted sum.
        
        V_expected = 0.4 × V_driver + 0.2 × V_location + 0.2 × V_opportunity + 0.2 × V_area
        
        All velocity signals are now in ₹/hour units, so we use direct weighted sum.
        """
        v_expected = (
            WEIGHT_V_DRIVER * v_driver +
            WEIGHT_V_LOCATION * v_location +
            WEIGHT_V_OPPORTUNITY * v_opportunity +
            WEIGHT_V_AREA * v_area
        )
        
        return round(v_expected, 2)
    
    # =========================================================================
    # STEP 7-10: Forecast Label Generation
    # =========================================================================
    
    def generate_forecast_label(self, v_recent, v_expected):
        """
        Steps 7-10: Generate forecast label based on comparison.
        """
        if v_expected <= 0:
            return 'no_data', 0, 0
        
        ratio = v_recent / v_expected
        
        # Thresholds from constants
        if ratio > FORECAST_AHEAD_THRESHOLD:
            label = 'ahead'
        elif ratio >= FORECAST_AT_RISK_THRESHOLD:
            label = 'on_track'
        else:
            label = 'at_risk'
        
        return label, round(ratio, 2), round((ratio - 1) * 100, 1)  # label, ratio, percentage deviation
    
    # =========================================================================
    # STEP 11-14: Goal Tracking
    # =========================================================================
    
    def compute_goal_progress(self, driver_id, current_time=None):
        """
        Steps 11-14: Compute progress toward daily/weekly goals.
        """
        if current_time is None:
            current_time = datetime.now()
        
        # Get driver's goal
        if self.driver_goals is not None and len(self.driver_goals) > 0:
            driver_goal = self.driver_goals[self.driver_goals['driver_id'] == driver_id]
            if len(driver_goal) > 0:
                daily_goal = driver_goal.iloc[0].get('daily_target', DEFAULT_DAILY_GOAL)
                weekly_goal = driver_goal.iloc[0].get('weekly_target', DEFAULT_WEEKLY_GOAL)
            else:
                daily_goal = DEFAULT_DAILY_GOAL
                weekly_goal = DEFAULT_WEEKLY_GOAL
        else:
            daily_goal = DEFAULT_DAILY_GOAL
            weekly_goal = DEFAULT_WEEKLY_GOAL
        
        # Calculate today's earnings
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_trips = self.trips_df[
            (self.trips_df['driver_id'] == driver_id) &
            (self.trips_df['start_time'] >= today_start) &
            (self.trips_df['start_time'] <= current_time)
        ]
        today_earnings = today_trips['fare'].sum() if len(today_trips) > 0 else 0
        
        # Calculate this week's earnings
        week_start = today_start - timedelta(days=today_start.weekday())
        week_trips = self.trips_df[
            (self.trips_df['driver_id'] == driver_id) &
            (self.trips_df['start_time'] >= week_start) &
            (self.trips_df['start_time'] <= current_time)
        ]
        week_earnings = week_trips['fare'].sum() if len(week_trips) > 0 else 0
        
        # Progress percentages
        daily_progress = round((today_earnings / daily_goal) * 100, 1) if daily_goal > 0 else 0
        weekly_progress = round((week_earnings / weekly_goal) * 100, 1) if weekly_goal > 0 else 0
        
        return {
            'daily_goal': daily_goal,
            'daily_earnings': round(today_earnings, 2),
            'daily_progress': daily_progress,
            'daily_remaining': round(max(daily_goal - today_earnings, 0), 2),
            'weekly_goal': weekly_goal,
            'weekly_earnings': round(week_earnings, 2),
            'weekly_progress': weekly_progress,
            'weekly_remaining': round(max(weekly_goal - week_earnings, 0), 2),
        }
    
    # =========================================================================
    # STEP 15-17: Time to Goal Estimation
    # =========================================================================
    
    def estimate_time_to_goal(self, v_expected, goal_remaining):
        """
        Steps 15-17: Estimate time to reach goal at current velocity.
        """
        if v_expected <= 0 or goal_remaining <= 0:
            return 0, 'goal_reached'
        
        hours_needed = goal_remaining / v_expected
        
        if hours_needed <= 2:
            difficulty = 'easy'
        elif hours_needed <= 4:
            difficulty = 'moderate'
        elif hours_needed <= 6:
            difficulty = 'challenging'
        else:
            difficulty = 'unlikely_today'
        
        return round(hours_needed, 1), difficulty
    
    # =========================================================================
    # STEP 18-19: Recommendations
    # =========================================================================
    
    def generate_earnings_recommendations(self, forecast_data):
        """
        Steps 18-19: Generate actionable earnings recommendations.
        """
        recommendations = []
        
        label = forecast_data.get('forecast_label', 'on_track')
        v_recent = forecast_data.get('v_recent', 0)
        v_expected = forecast_data.get('v_expected', 0)
        daily_progress = forecast_data.get('daily_progress', 0)
        
        if label == 'ahead':
            recommendations.append({
                'type': 'positive',
                'message': f"Great momentum! You're earning {forecast_data.get('deviation_percent', 0):.0f}% above expected.",
                'action': 'Keep going at this pace!'
            })
        
        elif label == 'at_risk':
            recommendations.append({
                'type': 'warning',
                'message': f"Earnings velocity below target. Current: ₹{v_recent:.0f}/hr vs Expected: ₹{v_expected:.0f}/hr",
                'action': 'Consider high-demand zones or peak hour driving.'
            })
        
        if daily_progress < 50 and forecast_data.get('hours_worked', 0) > 4:
            recommendations.append({
                'type': 'goal',
                'message': f"Daily goal is {daily_progress:.0f}% complete with {forecast_data.get('hours_remaining', 8):.1f} hours typical remaining.",
                'action': f"Need ₹{forecast_data.get('daily_remaining', 0):.0f} more to hit target."
            })
        
        # High opportunity recommendation (V_opp is now ₹/hr)
        v_opportunity = forecast_data.get('v_opportunity', 300.0)
        v_location = forecast_data.get('v_location', 280.0)
        if v_opportunity > v_location * 1.2:  # Opportunity 20% above baseline
            recommendations.append({
                'type': 'surge',
                'message': f"High demand area! Opportunity rate: ₹{v_opportunity:.0f}/hr vs baseline ₹{v_location:.0f}/hr.",
                'action': 'Great time to take trips!'
            })
        
        # Cold-start warning
        if forecast_data.get('is_cold_start', False):
            recommendations.append({
                'type': 'info',
                'message': f"Limited data: {forecast_data.get('total_trips_7d', 0)} trips in 7 days, {forecast_data.get('shift_time_min', 0):.0f} min in shift.",
                'action': 'Forecast may be less accurate. Complete more trips for better predictions.'
            })
        
        return recommendations
    
    # =========================================================================
    # Main Pipeline
    # =========================================================================
    
    def compute_forecast_for_driver(self, driver_id, current_time=None):
        """Compute complete forecast for a single driver."""
        if current_time is None:
            current_time = datetime.now()
        
        # Get driver info
        driver_info = self.drivers_df[self.drivers_df['driver_id'] == driver_id]
        if len(driver_info) == 0:
            return None
        driver_info = driver_info.iloc[0]
        city = driver_info['city']
        
        # Step 1-5: Compute velocity signals
        v_driver = self.compute_v_driver(driver_id, current_time)
        v_location = self.compute_v_location(city, current_time)
        v_opportunity = self.compute_v_opportunity(city, current_time)  # Step 10: λ × P_avg
        v_area = self.compute_v_area(v_opportunity)  # Step 12: V_opp × dist_factor
        v_recent = self.compute_v_recent(driver_id, current_time)
        
        # Cold-start detection: flag if insufficient data
        # Total trips in last 7 days
        seven_days_ago = current_time - timedelta(days=7)
        driver_trips = self.trips_df[
            (self.trips_df['driver_id'] == driver_id) &
            (self.trips_df['start_time'] >= seven_days_ago) &
            (self.trips_df['start_time'] <= current_time)
        ]
        total_trips = len(driver_trips)
        
        # Shift time: time since first trip today
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_trips = driver_trips[driver_trips['start_time'] >= today_start]
        if len(today_trips) > 0:
            shift_start = today_trips['start_time'].min()
            shift_time_min = (current_time - shift_start).total_seconds() / 60
        else:
            shift_time_min = 0
        
        # is_cold_start: True if total_trips < 3 OR shift_time < 15 min
        is_cold_start = (total_trips < 3) or (shift_time_min < 15)
        
        # Step 6: Composite V_expected
        v_expected = self.compute_v_expected(v_driver, v_location, v_opportunity, v_area)
        
        # Steps 7-10: Forecast label
        forecast_label, ratio, deviation_percent = self.generate_forecast_label(v_recent, v_expected)
        
        # Steps 11-14: Goal progress
        goal_progress = self.compute_goal_progress(driver_id, current_time)
        
        # Steps 15-17: Time to goal
        hours_to_daily, daily_difficulty = self.estimate_time_to_goal(
            v_expected, goal_progress['daily_remaining']
        )
        
        # Compile forecast
        forecast = {
            'driver_id': driver_id,
            'driver_name': driver_info['name'],
            'city': city,
            'timestamp': current_time,
            
            # Velocity signals
            'v_driver': v_driver,
            'v_location': v_location,
            'v_opportunity': v_opportunity,
            'v_area': v_area,
            'v_recent': v_recent,
            'v_expected': v_expected,
            
            # Cold-start flag
            'is_cold_start': is_cold_start,
            'total_trips_7d': total_trips,
            'shift_time_min': round(shift_time_min, 1),
            
            # Forecast
            'forecast_label': forecast_label,
            'velocity_ratio': ratio,
            'deviation_percent': deviation_percent,
            
            # Goals
            **goal_progress,
            
            # Time estimates
            'hours_to_daily_goal': hours_to_daily,
            'daily_goal_difficulty': daily_difficulty,
        }
        
        # Steps 18-19: Recommendations
        forecast['recommendations'] = self.generate_earnings_recommendations(forecast)
        
        return forecast
    
    def run_pipeline(self):
        """Run the complete earnings forecast pipeline."""
        print("=" * 60)
        print("PIPELINE 4: Earnings Velocity Forecast")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Compute forecast for each driver
        forecasts = []
        
        for driver_id in self.drivers_df['driver_id'].unique():
            print(f"\nComputing forecast for {driver_id}...")
            
            # Use the latest trip time as "current time" for simulation
            driver_trips = self.trips_df[self.trips_df['driver_id'] == driver_id]
            if len(driver_trips) > 0:
                current_time = driver_trips['start_time'].max()
            else:
                current_time = datetime.now()
            
            forecast = self.compute_forecast_for_driver(driver_id, current_time)
            if forecast:
                forecasts.append(forecast)
        
        # Create forecast DataFrame
        forecasts_df = pd.DataFrame(forecasts)
        
        # Save outputs
        self.save_outputs(forecasts_df)
        
        return forecasts_df
    
    def save_outputs(self, forecasts_df):
        """Save forecast outputs."""
        # Main forecast output
        output_path = os.path.join(self.output_dir, 'earnings_forecast.csv')
        
        # Drop recommendations column for CSV (it's a list)
        save_df = forecasts_df.drop(columns=['recommendations'], errors='ignore')
        save_df.to_csv(output_path, index=False)
        print(f"\n✅ Saved earnings forecasts to {output_path}")
        
        # Save recommendations separately
        all_recs = []
        for _, row in forecasts_df.iterrows():
            for rec in row.get('recommendations', []):
                rec['driver_id'] = row['driver_id']
                all_recs.append(rec)
        
        if all_recs:
            recs_df = pd.DataFrame(all_recs)
            recs_path = os.path.join(self.output_dir, 'earnings_recommendations.csv')
            recs_df.to_csv(recs_path, index=False)
            print(f"✅ Saved {len(all_recs)} earnings recommendations to {recs_path}")


def main():
    """Main entry point."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pipeline = EarningsForecastPipeline(base_dir)
    forecasts_df = pipeline.run_pipeline()
    
    print("\n" + "=" * 60)
    print("EARNINGS FORECAST COMPLETE")
    print("=" * 60)
    
    print("\nForecast Summary:")
    for _, row in forecasts_df.iterrows():
        print(f"\n{row['driver_name']} ({row['driver_id']}) - {row['city']}:")
        print(f"  V_expected: ₹{row['v_expected']:.0f}/hr | V_recent: ₹{row['v_recent']:.0f}/hr")
        print(f"  Status: {row['forecast_label'].upper()} ({row['deviation_percent']:+.0f}%)")
        print(f"  Daily: {row['daily_progress']:.0f}% complete (₹{row['daily_earnings']:.0f} / ₹{row['daily_goal']:.0f})")
        print(f"  Time to goal: {row['hours_to_daily_goal']:.1f}h ({row['daily_goal_difficulty']})")


if __name__ == '__main__':
    main()
