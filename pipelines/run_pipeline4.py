"""Pipeline 4: 19-Step Earnings Forecast → trip_summaries.csv."""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings('ignore')

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR  = os.path.join(BASE_DIR, 'driver_pulse_data')
OUTPUT_DIR= os.path.join(BASE_DIR, 'outputs')

trips_df  = pd.read_csv(os.path.join(DATA_DIR,'trips','trips.csv'))
drivers_df= pd.read_csv(os.path.join(DATA_DIR,'drivers','drivers.csv'))
goals_df  = pd.read_csv(os.path.join(DATA_DIR,'earnings','driver_goals.csv'))
market_df = pd.read_csv(os.path.join(DATA_DIR,'market','market_context.csv'))

trips_df['start_time'] = pd.to_datetime(trips_df['start_time'])
trips_df['end_time']   = pd.to_datetime(trips_df['end_time'])

try:
    safety_df = pd.read_csv(os.path.join(OUTPUT_DIR,'trip_safety_summaries.csv'))
    print(f"Safety loaded: {len(safety_df)}")
except:
    safety_df = None

# ── Steps 1-3: Earnings state ─────────────────────────────────────────────────
records = []
for did in trips_df['driver_id'].unique():
    dtrips = trips_df[trips_df['driver_id']==did].sort_values('start_time').copy()
    goal   = goals_df[goals_df['driver_id']==did].iloc[0]
    drv    = drivers_df[drivers_df['driver_id']==did].iloc[0]

    t_start = pd.to_datetime(f"{goal['date']} {goal['shift_start_time']}")
    t_end   = pd.to_datetime(f"{goal['date']} {goal['shift_end_time']}")
    total_hr= (t_end-t_start).total_seconds()/3600

    # S1 – cumulative earnings
    dtrips['e_current'] = dtrips['fare'].cumsum().round(2)
    dtrips['duration_hr'] = dtrips['duration_min']/60.0
    dtrips['t_elapsed_hr'] = ((dtrips['end_time']-t_start).dt.total_seconds()/3600).round(4)
    dtrips['t_remaining_hr'] = np.clip(total_hr - dtrips['t_elapsed_hr'], 0, total_hr).round(4)

    # S2 – trip velocity (₹/hr)
    dtrips['v_trip'] = (dtrips['fare']/dtrips['duration_hr']).round(2)

    # S3 – V_recent: rolling 3-trip mean
    dtrips['v_recent'] = dtrips['v_trip'].rolling(3, min_periods=1).mean().round(2)

    # S4-6 – Driver ability
    v_hist    = drv['total_earnings'] / max(drv['total_drive_hours'],1)
    exp_factor= min(drv['total_trips']/500.0, 1.0)
    dtrips['v_hist']      = round(v_hist, 2)
    dtrips['exp_factor']  = round(exp_factor, 3)
    dtrips['v_driver']    = ((1-exp_factor)*dtrips['v_recent'] + exp_factor*v_hist).round(2)

    # Market conditions (steps 7-12) per trip
    market_df['snapshot_time'] = pd.to_datetime(market_df['snapshot_time'])

    v_locations, v_opps, v_areas = [], [], []
    D_SCALE = 10.0
    for _, row in dtrips.iterrows():
        dists = np.sqrt(
            (market_df['area_centroid_lat']-row['pickup_lat'])**2 +
            (market_df['area_centroid_lon']-row['pickup_lon'])**2
        )*111.0
        ni = dists.idxmin()
        mrow = market_df.loc[ni]
        v_loc  = mrow['peer_velocity_mean']
        lam    = mrow['trips_per_hour']
        p_avg  = mrow['mean_fare_inr']
        v_opp  = lam * p_avg
        dfact  = 1.0/(1.0+dists[ni]/D_SCALE)
        v_area = v_opp * dfact
        v_locations.append(round(v_loc,2))
        v_opps.append(round(v_opp,2))
        v_areas.append(round(v_area,2))

    dtrips['v_location']   = v_locations
    dtrips['v_opportunity']= v_opps
    dtrips['v_area']       = v_areas

    # S13 – V_expected
    dtrips['v_expected'] = (
        0.4*dtrips['v_driver'] + 0.2*dtrips['v_location'] +
        0.2*dtrips['v_opportunity'] + 0.2*dtrips['v_area']
    ).round(2)

    # S15-16 – Forecast earnings
    dtrips['e_future_predicted'] = (dtrips['v_expected']*dtrips['t_remaining_hr']).round(2)
    dtrips['e_predicted_total']  = (dtrips['e_current']+dtrips['e_future_predicted']).round(2)

    dtrips['earnings_target']  = goal['target_earnings']
    dtrips['required_velocity']= np.where(
        dtrips['t_remaining_hr']>0,
        ((goal['target_earnings']-dtrips['e_current'])/dtrips['t_remaining_hr']).round(2), 0)

    # S17-18
    dtrips['goal_score'] = (dtrips['e_predicted_total']/goal['target_earnings']).round(3)
    dtrips['forecast_status'] = dtrips['goal_score'].apply(
        lambda s: 'ahead' if s>1.1 else ('on_track' if s>=0.9 else 'at_risk'))

    # S19 – confidence
    rolling_std = dtrips['v_trip'].expanding(min_periods=1).std().fillna(0)
    dtrips['velocity_std'] = rolling_std.round(2)
    dtrips['confidence']   = (1.0/(1.0+rolling_std/100.0)).round(3)

    # earnings velocity
    dtrips['earnings_velocity'] = np.where(
        dtrips['t_elapsed_hr']>0,
        (dtrips['e_current']/dtrips['t_elapsed_hr']).round(2), 0)

    records.append(dtrips)

forecast_df = pd.concat(records, ignore_index=True)
print(f"Forecast computed for {len(forecast_df)} trips")
print(forecast_df['forecast_status'].value_counts().to_dict())

# ── Merge with safety ─────────────────────────────────────────────────────────
SAFETY_COLS = ['trip_id','motion_events_count','audio_events_count',
               'flagged_moments_count','n_compound_events','max_severity',
               'stress_score','trip_quality_rating','poor_road_quality']

if safety_df is not None:
    avail = [c for c in SAFETY_COLS if c in safety_df.columns]
    forecast_df = forecast_df.merge(safety_df[avail], on='trip_id', how='left')
else:
    for col,val in [('motion_events_count',0),('audio_events_count',0),
                    ('flagged_moments_count',0),('n_compound_events',0),
                    ('max_severity','none'),('stress_score',0.0),
                    ('trip_quality_rating','excellent'),('poor_road_quality',False)]:
        forecast_df[col] = val

for col,defval in [('motion_events_count',0),('audio_events_count',0),
                   ('flagged_moments_count',0),('n_compound_events',0),
                   ('max_severity','none'),('stress_score',0.0),
                   ('trip_quality_rating','excellent'),('poor_road_quality',False)]:
    forecast_df[col] = forecast_df[col].fillna(defval)

# ── Final column order (matching reference) ───────────────────────────────────
FINAL_COLS = [
    'trip_id','driver_id','date','duration_min','distance_km','fare',
    'earnings_velocity','motion_events_count','audio_events_count',
    'flagged_moments_count','n_compound_events','max_severity',
    'stress_score','trip_quality_rating','passenger_rating',
    'poor_road_quality','surge_multiplier',
    'pickup_location','dropoff_location','pickup_lat','pickup_lon','dropoff_lat','dropoff_lon',
    'v_recent','v_hist','v_driver','v_location','v_opportunity','v_area','v_expected',
    't_elapsed_hr','t_remaining_hr','e_current','earnings_target','required_velocity',
    'e_future_predicted','e_predicted_total','goal_score','forecast_status','confidence','velocity_std'
]
avail = [c for c in FINAL_COLS if c in forecast_df.columns]
forecast_df[avail].to_csv(os.path.join(OUTPUT_DIR,'trip_summaries.csv'), index=False)
print(f"✅ trip_summaries.csv saved ({len(forecast_df)} rows)")
