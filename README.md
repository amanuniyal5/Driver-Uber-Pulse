Driver Pulse: Team 18
• Demo Video: https://drive.google.com/file/d/19h9dp1o0-V_gJu7G0i_lymJM4l8sBseG/view?usp=sharing  
• Live Application: https://amanuniyal5-driver-uber-pulse-dashboarddriver-app-d8lpxi.streamlit.app/

# 🚘 Driver Pulse: Engineering Handoff

## 📖 Project Overview
Driver Pulse is an experimental intelligence layer designed to give Uber drivers a clearer picture of their shift in real-time. By fusing noisy, pre-existing device signals (accelerometer, gyroscope, microphone magnitude/frequency, and trip earnings), the system transforms raw data into actionable insights:
1. **Safety & Stress Tracking:** Identifies harsh driving, road quality issues, and in-cabin social conflicts without ever recording raw audio conversations.
2. **Earnings Velocity:** Predicts whether a driver will hit their daily financial goal using a deterministic, explainable forecast model.
3. **Actionable Coaching:** Provides post-trip, glanceable recommendations (e.g., "Short-Trip Mode") to optimize the remainder of the shift.

---

## 🏗️ System Architecture

Our solution is divided into a lightweight **Edge Compute** (on-device) layer for privacy and battery management, and a **Cloud Compute** (server-side) layer for authoritative classification and market aggregation.

### 1. Motion Event Detection (Bag of Visual Words)
* **Ingestion & Prep (Edge):** Resamples raw accel/gyro data to 25 Hz. Uses a 1D Kalman Filter to smooth sensor noise and performs PCA reorientation using the first 30 seconds of a trip to normalize phone mounting angles (cupholder vs. dashboard).
* **Quantization & ML (Cloud/Edge):** Chops windows into overlapping segments, mapping them to a K-Means Codebook (K=64) to create "Visual Fingerprints" (Histograms) of motion. A Random Forest classifier then identifies 7 distinct driving maneuvers (e.g., `AGGRESSIVE_BRAKING`, `AGG_LEFT_LANE_CHANGE`).

### 2. Audio Conflict Detection (Privacy-First 4-Layer Heuristic)
Processes continuous audio into purely mathematical features (RMS energy & FFT bins) every 25ms. **No raw audio is ever stored or transcribed.**
* **Layer 1 (Acoustic):** High Zero-Crossing Rate (ZCR) & Centroid detect harsh fricatives/shouting.
* **Layer 2 (Temporal):** Gap ratios and turn-taking latency detect fragmented, interrupted speech.
* **Layer 3 (Prosodic):** F0 (Pitch) variance detects emotional arousal.
* **Layer 4 (Context):** Dynamic dB deviation filters out baseline road noise, with a 60-second "flip-back" rule to prevent sustained singing/music from causing false positives.

### 3. Signal Fusion & Stress Scoring
Fuses the outputs of the Motion and Audio pipelines. 
* Detects **Compound Events** (e.g., Hard brake + Audio spike within 2 seconds = `ACUTE_SAFETY_EVENT`).
* Computes an overall PRD-compliant **Stress Score** normalized by trip duration.
* Aggregates historical GPS flags into H3 Hexagonal **Hard-Brake Zones** and **Road Quality (Pothole)** segments.

### 4. Earnings Velocity Forecast
A 19-step deterministic pipeline that calculates expected earnings (`V_expected`) by blending:
1. `V_driver`: The driver's historical average.
2. `V_recent`: The driver's momentum over the last 3 trips.
3. `V_opportunity`: Local demand based on $\lambda$ (trip requests/hr) and average fare.
4. `V_area`: Distance-penalized market rates.

---

## Architecture

```
Raw Sensor Data (Accelerometer + Audio Features)
        │
        ▼
┌───────────────────┐    ┌───────────────────────┐
│  Pipeline 1       │    │  Pipeline 2            │
│  Motion Detection │    │  Audio 4-Layer         │
│  BoVW + RF        │    │  Conflict Detection    │
└────────┬──────────┘    └────────────┬──────────┘
         │                            │
         └──────────┬─────────────────┘
                    ▼
         ┌──────────────────┐
         │  Pipeline 3      │
         │  Signal Fusion   │
         │  (Motion+Audio)  │
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │  Pipeline 4      │
         │  Earnings        │
         │  Velocity Model  │
         └────────┬─────────┘
                  │
         ┌────────▼─────────┐
         │  Streamlit       │
         │  Dashboard       │
         │  (driver_app.py) │
         └──────────────────┘
```

---

## Project Structure

```
source code/
├── constants.py                    # All threshold values & config
├── requirements.txt
├── run_all_pipelines.py            # Run all 4 pipelines in sequence
├── generate_simulation_data.py     # Generate DRV003 simulation dataset
│
├── pipelines/
│   ├── pipeline1_motion_bovw.py    # Motion event detection (BoVW + RF)
│   ├── pipeline2_audio_4layer.py   # Audio conflict detection (4-layer)
│   ├── pipeline3_signal_fusion.py  # Signal fusion & stress scoring
│   ├── pipeline4_earnings_forecast.py  # Earnings velocity forecast
│   ├── run_pipeline3.py            # Standalone runner for P3
│   └── run_pipeline4.py            # Standalone runner for P4
│
├── dashboard/
│   ├── driver_app.py               # Main Streamlit app (~2500 lines)
│   ├── simulation_bridge.py        # Dynamic pipeline trigger on trip completion
│   └── components/
│       └── map_component.py        # Interactive Leaflet map (flicker-free)
│
├── scripts/
│   ├── explain_bovw_logic.py       # BoVW model explainability
│   └── explain_audio_logic.py      # Audio pipeline explainability
│
├── driver_pulse_data/              # Training data (DRV001, DRV002, DRV003)
│   ├── drivers/drivers.csv
│   ├── trips/trips.csv
│   ├── sensor_data/
│   │   ├── accelerometer_data.csv
│   │   └── audio_features.csv
│   ├── earnings/
│   │   ├── driver_goals.csv
│   │   └── earnings_velocity_log.csv
│   └── market/market_context.csv
│
├── simulation_data/                # DRV003-only data for the dashboard demo
│   ├── drivers/
│   ├── trips/                      # 5 Delhi trips (GPS-traced)
│   ├── sensor_data/
│   ├── earnings/
│   ├── market/
│   └── processed_outputs/          # Generated dynamically by simulation_bridge
│       ├── flagged_moments.csv
│       ├── audio_flagged_moments.csv
│       ├── motion_events.csv
│       ├── trip_summaries.csv
│       ├── trip_motion_summary.csv
│       ├── trip_audio_summary.csv
│       ├── earnings_forecast.csv
│       ├── earnings_recommendations.csv
│       └── audio_processed.csv
│
├── outputs/                        # Full pipeline outputs (all 3 drivers)
│   ├── models/
│   │   ├── bovw_motion_models.pkl  # Trained BoVW + Random Forest model
│   │   └── bovw_motion_metadata.json
│   └── explainability/             # Pipeline visualisation charts
│       ├── 1_pca_reorientation.png
│       ├── 2_bovw_histogram.png
│       ├── 3_feature_importance.png
│       ├── 4_cluster_visualization.png
│       ├── 5_pipeline_summary.png
│       ├── audio_1_conflict_anatomy.png
│       ├── audio_2_decision_table.png
│       ├── audio_3_layer_timeline.png
│       └── audio_4_feature_distributions.png
│
└── .streamlit/                     # Streamlit theme config
```

---

## Data

### Training Data (`driver_pulse_data/`)
Contains data for **3 drivers** (DRV001, DRV002, DRV003) used for training the BoVW + Random Forest model.

| File | Description |
|------|-------------|
| `drivers/drivers.csv` | Driver profiles (city, rating, experience) |
| `trips/trips.csv` | Trip metadata (route, fare, duration, GPS) |
| `sensor_data/accelerometer_data.csv` | Raw accelerometer readings at 25 Hz: `accel_x/y/z`, `gyro_z`, `speed_kmh`, GPS |
| `sensor_data/audio_features.csv` | Pre-extracted audio features: ZCR, spectral centroid, f0_std, db_deviation |
| `earnings/driver_goals.csv` | Per-driver daily/weekly earnings targets |
| `earnings/earnings_velocity_log.csv` | Historical earnings velocity per shift |
| `market/market_context.csv` | Demand level, surge multiplier, weather per zone/time |

### Simulation Data (`simulation_data/`)
**DRV003 (Rajesh Patel, Delhi) only.** 5 real GPS-traced Delhi routes:

| Trip | Route | Distance | Duration | Fare |
|------|-------|----------|----------|------|
| TRIP_DRV003_01 | Dwarka → Meerut | 88.6 km | 80 min | ₹500 |
| TRIP_DRV003_02 | Karol Bagh → Greater Noida | 55.2 km | 57 min | ₹875 |
| TRIP_DRV003_03 | Punjabi Bagh → Faridabad | 47.2 km | 47 min | ₹1,017 |
| TRIP_DRV003_04 | Rohini → Gurugram | 51.3 km | 51 min | ₹637 |
| TRIP_DRV003_05 | Lajpat Nagar → Hapur | 71.8 km | 79 min | ₹369 |

---

## Pipelines

### Pipeline 1 — Motion Event Detection (BoVW)

**File:** `pipelines/pipeline1_motion_bovw.py`

Detects 6 classes of aggressive driving events from raw accelerometer data using a **Bag-of-Visual-Words (BoVW)** approach with a **Random Forest** classifier.

#### Signal Processing Stages
| Stage | Description |
|-------|-------------|
| 1a | Resample to 25 Hz |
| 1b | Auto-calibration (noise floor estimation from steady-state windows) |
| 1c | Kalman filtering — reduces MEMS sensor noise |
| 2 | PCA reorientation — aligns axes to vehicle frame |
| 3 | Self-trigger gate — only processes windows above noise threshold |
| 4 | BoVW feature extraction — K-Means codebook (64 codewords) |
| 5 | Random Forest classification (100 trees) |

#### Event Classes
| Event | Trigger |
|-------|---------|
| `AGGRESSIVE_BRAKING` | `accel_y` ∈ [−2.5, −1.5] g, `gyro_z` < 15°/s |
| `AGGRESSIVE_ACCEL` | `accel_y` ∈ [+1.5, +2.5] g, `gyro_z` < 15°/s |
| `AGGRESSIVE_LEFT_TURN` | `accel_x` ∈ [−1.5, −0.8] g, `gyro_z` ∈ [+40, +120]°/s |
| `AGGRESSIVE_RIGHT_TURN` | `accel_x` ∈ [+0.8, +1.5] g, `gyro_z` ∈ [−120, −40]°/s |
| `AGG_LEFT_LANE_CHANGE` | `accel_x` ∈ [−0.5, −0.3] g, gyro spike < 0.5s |
| `AGG_RIGHT_LANE_CHANGE` | `accel_x` ∈ [+0.3, +0.5] g, gyro spike < 0.5s |
| `ROAD_BUMP` | `accel_z` deviation > 1.5g, duration < 8 samples |

Also detects road bumps and hard-brake zones using GPS clustering.

In a real-world driving environment, simple **rule-based detection** (e.g., `if accel_y > 2.0g: flag_harsh_braking`) fails because it is too rigid and cannot distinguish between high-intensity noise and actual intentional maneuvers.

Our **Bag of Visual Words (BoVW) + Random Forest** approach is superior for the following five reasons:

### 1. Recognition of "Temporal Signatures" (Shape vs. Peak)
*   **Rule-Based Problem:** A rule-based system only looks for a single "peak" value. If a driver hits a pothole or drops their phone, the accelerometer might spike to 3.0g for a split second. A rule-based system would incorrectly flag this as "Harsh Braking."
*   **Our Advantage:** BoVW looks at the **entire shape** of the 0.5s window. It recognizes that "Harsh Braking" has a specific sustained curve (a ramp-up and a plateau), whereas a pothole is an instantaneous "jitter." Our model identifies the **signature**, not just the maximum value.

### 2. Disambiguation of Maneuvers (Lane Change vs. Turn)
*   **Rule-Based Problem:** A sharp lane change and a hard left turn often produce similar lateral acceleration ($accel\_x$) magnitudes. Simple thresholds cannot tell them apart.
*   **Our Advantage:** By using a K-Means Codebook that includes **Gyroscope ($gyro\_z$) data**, our model learns distinct "codewords" for rotation. It recognizes that a lane change involves a quick "S-curve" rotation, while a turn involves a sustained yaw rate. The ML model correlates these axes automatically, whereas writing rules for every possible "if-this-then-that" combination of accel and gyro is nearly impossible.

### 3. Robustness to Phone Placement (PCA Reorientation)
*   **Rule-Based Problem:** Most rule-based algorithms assume the phone is perfectly mounted in a cradle. If the phone is tilted in a cupholder or lying flat in a pocket, the "Forward" ($accel\_y$) axis and "Vertical" ($accel\_z$) axis get swapped or mixed. The rules break instantly.
*   **Our Advantage:** We use **Principal Component Analysis (PCA)** to dynamically reorient the axes to the vehicle's frame of reference every trip. Our ML model then classifies the data based on these **vehicle-aligned axes**, making the system work regardless of whether the phone is in a pocket, a bag, or a mount.

### 4. Suppression of False Positives (Environmental Noise)
*   **Rule-Based Problem:** Real-world driving is full of "noise"—engine vibrations, speed bumps, and closing car doors. These all create high-frequency spikes that trip simple threshold rules.
*   **Our Advantage:** Our model uses a **Kalman Filter** to smooth sensor noise and **K-Means Quantization** to ignore irrelevant micro-vibrations. Because the Random Forest is trained on a "Normal Driving" class, it learns to ignore high-magnitude events that don't match the "fingerprint" of dangerous driving.

### 5. High Explainability (The "Why" Factor)
*   **Rule-Based Problem:** When a driver asks "Why was I flagged?", a rule-based system can only say "Because you hit 2.1g." This feels arbitrary to a driver.
*   **Our Advantage:** Because we use **Random Forest Feature Importance**, we can pinpoint exactly which **codewords** triggered the event. We can tell a reviewer: *"This was flagged because the system detected a sequence of 'Rapid Longitudinal Decompression' codewords characteristic of a hard brake, which were absent in the preceding 5 minutes of normal driving."*


**Outputs:** `outputs/motion_events.csv`, `outputs/flagged_moments.csv`, `outputs/models/`

---

### Pipeline 2 — Audio Conflict Detection (4-Layer)

**File:** `pipelines/pipeline2_audio_4layer.py`

A **4-layer hierarchical detection** system that identifies passenger conflict and acute safety events from in-cabin audio features (no raw audio stored — only pre-extracted features).

#### Detection Layers

| Layer | Name | Features | Threshold |
|-------|------|----------|-----------|
| L1 | Acoustic State | ZCR, Spectral Centroid | ZCR > 0.55 AND Centroid > 2000 Hz |
| L2 | Temporal Dynamics | turn_gap, gap_ratio, energy_slope | gap < 1.5s OR energy_slope > 0.3 dB/s |
| L3 | Prosodic Markers | f0_std, speech_rate | f0_std > 40 Hz |
| L4 | Context Gate | db_deviation | > 12 dB above baseline |

#### Decision Paths
- **Path A (Conflict):** 2+ consecutive windows triggering L1 + L2 + L4
- **Path B (Acute Safety):** `accel` > 5.0 m/s² AND `db_deviation` > 20 dB within 2s window

**Outputs:** `outputs/audio_flagged_moments.csv`, `outputs/trip_audio_summary.csv`

---

### Pipeline 3 — Signal Fusion

**File:** `pipelines/pipeline3_signal_fusion.py`

Combines motion and audio events into a **unified safety picture** per trip.

#### What it does
- **Compound event detection** — correlates motion + audio events within a 2-second window → labels as `COMPOUND` (highest severity)
- **Hard-brake zone clustering** — groups repeated braking events by GPS grid cell (precision: 0.01°, ~1 km)
- **Road quality scoring** — flags trips with ≥ 3 road bumps per 200m segment
- **Trip-level stress score** — weighted formula:

$$\text{stress} = 0.4 \times w_{accel} + 0.3 \times w_{audio} + 0.3 \times w_{compound}$$

- **Safety score** (0–100) — inverse of stress, penalised per critical/motion/audio event

**Outputs:** `outputs/trip_summaries.csv`, `outputs/flagged_moments.csv` (fused), `outputs/driver_brake_zones.csv`

---

### Pipeline 4 — Earnings Velocity Forecast

**File:** `pipelines/pipeline4_earnings_forecast.py`

A **19-step deterministic pipeline** that forecasts whether the driver will hit their daily earnings goal.

#### Velocity Signals (₹/hr)

| Signal | Description | Weight |
|--------|-------------|--------|
| $V_{driver}$ | 7-day rolling average (driver's historical rate) | 0.40 |
| $V_{location}$ | Zone-specific benchmark | 0.20 |
| $V_{opportunity}$ | λ × P_avg (trips/hr × mean fare) | 0.20 |
| $V_{area}$ | V_opportunity × distance factor | 0.20 |

$$V_{expected} = 0.4 \cdot V_{driver} + 0.2 \cdot V_{location} + 0.2 \cdot V_{opportunity} + 0.2 \cdot V_{area}$$

#### Forecast Labels
| Label | Condition |
|-------|-----------|
| `ahead` | $V_{recent} > V_{expected} \times 1.1$ |
| `on_track` | $V_{expected} \times 0.9 \leq V_{recent} \leq V_{expected} \times 1.1$ |
| `at_risk` | $V_{recent} < V_{expected} \times 0.9$ |

**Cold-start detection:** `is_cold_start = True` if total trips < 3 OR shift elapsed < 15 min.

**Outputs:** `outputs/earnings_forecast.csv`, `outputs/earnings_recommendations.csv`

---

## Dashboard

**File:** `dashboard/driver_app.py`

A **Streamlit-based driver companion app** styled as a dark Uber-style UI. Simulates a full shift for Rajesh Patel across 5 Delhi trips.

### App Flow

```
Login → Pre-Shift Setup → Trip Selection → Trip Simulation → Post-Trip Insights
                                ↑                                    │
                                └────────── (next trip) ─────────────┘
                                                                     ↓
                                                            End-of-Day Summary
```

### Views

| View | Description |
|------|-------------|
| **Login** | Driver login (DRV003 / `demo123`) |
| **Pre-Shift Setup** | Set earnings target (₹500–₹5,000), shift hours (4–12h), start time |
| **Trip Selection** | Smart trip recommendations based on earnings velocity and stress history. Shows shift progress bar, distance, fare, and stress hint |
| **Trip Simulation** | Live GPS map (Folium), real-time safety feed, progress bar with km/min remaining, auto-advance at 1% per 1.5s |
| **Post-Trip Insights** | Stress score, safety score (0–100), event breakdown (motion/audio/compound), earnings velocity analysis, goal forecast, next-trip recommendation |
| **End-of-Day Summary** | Total earnings, safety score trend across all trips, shift summary |

### Map Component

**File:** `dashboard/components/map_component.py`

- GPS route tracing from accelerometer data
- Event markers (colour-coded by severity and type)
- H3 hexagonal brake zone overlays (resolution 9, ~0.1 km²)
- Live car position marker advancing with simulation progress

### Key Features
- **Stress badge** — Only appears after the first completed trip, uses event density and severity
- **Short-trip mode** — Surfaces easier trips when driver is fatigued/behind target
- **Cold-start handling** — Different recommendations for first 15 min of shift
- **Live safety feed** — Shows events as they are "detected" during simulation, proportional to trip progress

---



1. Ensure you have Python 3.9+ installed.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Data Pipelines:** Generate the processed analytical data from raw sensor logs. This runs the entire pipeline for all the drivers and for all the trips for our synthetic dataset.
   ```bash
   python run_all_pipelines.py
   ```
5. **Start the Dashboard:**
   ```bash
   streamlit run dashboard/driver_app.py
   ```
After running the simulation, the output files are stored in simulation_data/processed_outputs.
---

## ⚖️ Trade-offs & Assumptions

Engineering a real-world system from messy telemetry required several deliberate compromises:

1. **Interpretability over Deep Learning (Motion):** We sacrificed marginal accuracy gains from using a heavy CNN/LSTM model in favor of the Bag of Visual Words (BoVW) + Random Forest approach. This allows our system to be highly interpretable—we can pinpoint exactly which "micro-movement" (codeword) triggered a hard brake, which is crucial for explaining flags to drivers without it feeling like a black box.
2. **False Negative vs. False Positive Balancing (Audio):** A driver getting a "Conflict Warning" because they were singing along to the radio is a terrible UX. We implemented a strict **4-layer gating system** and a 60-second flip-back rule. We deliberately accept some *false negatives* (missing a mild argument) to guarantee a near-zero *false positive* rate.
3. **Deterministic Math over ML for Earnings:** We chose a 19-step mathematical formula with fixed weights (0.4 Driver, 0.2 Market, etc.) rather than a Gradient Boosted Model for the earnings forecast. This ensures the output is 100% transparent. A driver is told exactly *why* they are falling behind (e.g., "Market opportunity dropped," not just "The ML model said so").
4. **25Hz Resampling:** We downsampled all erratic device telemetry to a fixed 25Hz grid. While we lose sub-25Hz micro-vibrations (making subtle pothole detection slightly harder), it halves CPU/battery overhead compared to 50Hz, fitting within our strict <3% battery per hour constraint.
5. **No Local Tunnels:** Per hackathon rules, the app relies purely on state-managed, post-trip UI rendering rather than establishing ephemeral tunnels for live streaming from a physical device.

---