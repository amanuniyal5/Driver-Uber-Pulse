"""
Generate simulation dataset for Delhi driver with 5 real trips.
Uses actual route data from GPS coordinates.
"""

import os
import csv
import random
from datetime import datetime, timedelta
import math

# Base path
BASE_PATH = "/home/aman-uniyal/Desktop/hackathon/source code"
OUTPUT_PATH = os.path.join(BASE_PATH, "simulation_data")

# Create folder structure
os.makedirs(os.path.join(OUTPUT_PATH, "drivers"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "trips"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "sensor_data"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "earnings"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "market"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "processed_outputs"), exist_ok=True)

# 5 Trip route data extracted from the CSV files
# Format: Longitude (scaled by 1e6), Latitude (scaled by 1e6), Time (seconds)

# Trip 1: Dwarka, South West Delhi -> Meerut (88.6km)
TRIP_1_RAW = [
    # From South West Delhi to Meerut route data
    (77189274, 28653008, 0),
    (77189574, 28653415, 9),
    (77194683, 28650607, 86),
    (77194746, 28650568, 86),
    (77198732, 28648020, 113),
    (77199338, 28648833, 132),
    (77199421, 28648976, 140),
    (77201806, 28647702, 171),
    (77205173, 28646768, 194),
    (77205272, 28646755, 200),
    (77205529, 28646617, 209),
    (77209619, 28645866, 249),
    (77209726, 28645845, 255),
    (77211074, 28645587, 279),
    (77217717, 28644275, 326),
    (77223496, 28645237, 353),
    (77223764, 28645215, 363),
    (77223811, 28645219, 364),
    (77226144, 28644571, 401),
    (77227620, 28643859, 414),
    (77227958, 28643593, 431),
    (77238952, 28640684, 521),
    (77240554, 28640214, 541),
    (77240719, 28640231, 542),
    (77240982, 28640258, 552),
    (77241027, 28640065, 561),
    (77241050, 28639931, 568),
    (77241394, 28628249, 649),
    (77241638, 28628011, 657),
    (77241920, 28627899, 661),
    (77247084, 28628245, 694),
    (77247787, 28628248, 697),
    (77249689, 28628191, 704),
    (77251598, 28628123, 712),
    (77252147, 28628133, 714),
    (77253378, 28628139, 719),
    (77260554, 28628221, 747),
    (77266805, 28628224, 771),
    (77270979, 28628234, 788),
    (77271931, 28628236, 791),
    (77274909, 28628901, 808),
    (77275004, 28628779, 816),
    (77274546, 28628451, 829),
    (77275023, 28625634, 867),
    (77276319, 28623732, 881),
    (77279545, 28618772, 908),
    (77279750, 28618456, 910),
    (77281369, 28616042, 922),
    (77282382, 28614501, 932),
    (77287390, 28606733, 986),
    (77297471, 28587751, 1105),
    (77299341, 28585613, 1122),
    (77299907, 28585315, 1127),
    (77301177, 28584103, 1138),
    (77302780, 28582023, 1153),
    (77308254, 28574582, 1203),
    (77311163, 28570687, 1230),
    (77320386, 28559428, 1300),
    (77452891, 28978532, 4800),  # Meerut endpoint
]

# Trip 2: Karol Bagh -> Greater Noida (55.2km)
TRIP_2_RAW = [
    (77189274, 28653008, 0),
    (77189574, 28653415, 21),
    (77194683, 28650607, 86),
    (77194746, 28650568, 86),
    (77198732, 28648020, 113),
    (77199338, 28648833, 132),
    (77199421, 28648976, 140),
    (77201806, 28647702, 171),
    (77205173, 28646768, 194),
    (77205272, 28646755, 200),
    (77205529, 28646617, 209),
    (77209619, 28645866, 249),
    (77209726, 28645845, 255),
    (77211074, 28645587, 279),
    (77217717, 28644275, 326),
    (77223496, 28645237, 353),
    (77223764, 28645215, 363),
    (77223811, 28645219, 364),
    (77226144, 28644571, 401),
    (77227620, 28643859, 414),
    (77227958, 28643593, 431),
    (77238952, 28640684, 521),
    (77240554, 28640214, 541),
    (77240719, 28640231, 542),
    (77240982, 28640258, 552),
    (77241027, 28640065, 561),
    (77241050, 28639931, 568),
    (77241394, 28628249, 649),
    (77241638, 28628011, 657),
    (77241920, 28627899, 661),
    (77247084, 28628245, 694),
    (77247787, 28628248, 697),
    (77249689, 28628191, 704),
    (77251598, 28628123, 712),
    (77252147, 28628133, 714),
    (77253378, 28628139, 719),
    (77260554, 28628221, 747),
    (77266805, 28628224, 771),
    (77270979, 28628234, 788),
    (77271931, 28628236, 791),
    (77274909, 28628901, 808),
    (77275004, 28628779, 816),
    (77274546, 28628451, 829),
    (77275023, 28625634, 867),
    (77276319, 28623732, 881),
    (77279545, 28618772, 908),
    (77279750, 28618456, 910),
    (77281369, 28616042, 922),
    (77282382, 28614501, 932),
    (77287390, 28606733, 986),
    (77297471, 28587751, 1105),
    (77299341, 28585613, 1122),
    (77299907, 28585315, 1127),
    (77301177, 28584103, 1138),
    (77302780, 28582023, 1153),
    (77308254, 28574582, 1203),
    (77311163, 28570687, 1230),
    (77320386, 28559428, 1300),
    (77321392, 28558893, 1309),
    (77325739, 28555864, 1336),
    (77326331, 28555289, 1339),
    (77354157, 28535011, 1481),
    (77361923, 28529668, 1524),
    (77383654, 28514163, 1635),
    (77391019, 28506958, 1679),
    (77423647, 28492253, 1840),
    (77437222, 28482033, 1919),
    (77488612, 28447352, 2216),
    (77491412, 28449700, 2233),
    (77498852, 28455527, 2272),
    (77499300, 28455920, 2275),
    (77501005, 28457210, 2284),
    (77505831, 28461014, 2312),
    (77509688, 28464124, 2333),
    (77510913, 28465218, 2343),
    (77511685, 28465296, 2365),
    (77520474, 28458427, 2434),
    (77520851, 28458076, 2445),
    (77544642, 28436729, 2647),
    (77545562, 28436555, 2657),
    (77545579, 28422903, 2776),
    (77548987, 28423164, 2841),
    (77558977, 28416247, 2958),
    (77557082, 28413432, 2998),
    (77559180, 28408676, 3060),
    (77559535, 28407618, 3066),
    (77557098, 28405764, 3100),
    (77560343, 28403938, 3171),
    (77563419, 28397457, 3340),
    (77565099, 28396533, 3358),
    (77566429, 28393940, 3420),
]

# Trip 3: Punjabi Bagh -> Kurukshetra (158.3km) 
TRIP_3_RAW = [
    (77138837, 28671190, 0),
    (77139764, 28673475, 62),
    (77140002, 28674060, 78),
    (77141097, 28677179, 113),
    (77173293, 28707018, 429),
    (77173940, 28707815, 435),
    (77174071, 28707968, 436),
    (77175641, 28710023, 454),
    (77175509, 28710649, 466),
    (77174100, 28712245, 487),
    (77173092, 28713428, 499),
    (77171308, 28715474, 535),
    (77170042, 28716889, 549),
    (77168672, 28718453, 565),
    (77165174, 28722397, 606),
    (77161916, 28726724, 663),
    (77158718, 28731077, 721),
    (77158380, 28731540, 725),
    (77156216, 28734338, 768),
    (77153493, 28740894, 817),
    (77136479, 28810425, 1263),
    (77129540, 28832690, 1396),
    (77127560, 28841245, 1448),
    (77125650, 28851335, 1528),
    (77125503, 28852180, 1535),
    (77124398, 28858285, 1574),
    (77103504, 28921237, 1952),
    (77059387, 29075615, 2672),
    (77059358, 29075711, 2673),
    (76979053, 29355486, 3977),
    (76974146, 29376635, 4073),
    (76968949, 29408801, 4222),
    (76969365, 29418683, 4266),
    (76969673, 29425395, 4295),
    (76969937, 29433259, 4330),
    (76969932, 29433364, 4331),
    (76972744, 29530566, 4794),
    (76972747, 29530737, 4795),
    (76973874, 29545900, 4862),
    (76977379, 29572569, 4981),
    (76978373, 29580659, 5017),
    (76978421, 29581054, 5019),
    (76980440, 29597279, 5095),
    (76982553, 29621568, 5212),
    (76989297, 29664047, 5437),
    (76990875, 29668038, 5459),
    (76991479, 29668625, 5466),
    (76995152, 29670592, 5493),
    (76995693, 29670865, 5498),
    (76997156, 29671612, 5505),
    (76997935, 29672009, 5511),
    (77005869, 29690525, 5613),
    (77006623, 29696969, 5642),
    (77002257, 29709309, 5717),
    (76982847, 29717185, 5807),
    (76944230, 29819165, 6313),
    (76944064, 29819627, 6315),
    (76899361, 29942449, 6921),
    (76898918, 29943512, 6928),
    (76897948, 29947407, 6952),
    (76892532, 29976240, 7116),
    (76856294, 29973372, 7344),
    (76856073, 29973343, 7351),
    (76849205, 29972895, 7403),
    (76849224, 29971014, 7433),
    (76848201, 29970256, 7451),
    (76848070, 29970180, 7461),
    (76844525, 29969466, 7491),
]

# Trip 4: Lajpat Nagar -> Roorkee (173.4km)
TRIP_4_RAW = [
    (77244232, 28564861, 0),
    (77249493, 28565334, 89),
    (77249364, 28565417, 95),
    (77249210, 28565407, 116),
    (77249182, 28565709, 125),
    (77249843, 28565764, 138),
    (77251202, 28565905, 161),
    (77253838, 28568761, 196),
    (77256996, 28571619, 238),
    (77260389, 28573936, 276),
    (77266436, 28578032, 316),
    (77266240, 28581132, 350),
    (77265972, 28581475, 354),
    (77263174, 28583404, 378),
    (77259299, 28586901, 413),
    (77256257, 28595689, 475),
    (77256174, 28596476, 480),
    (77267719, 28603494, 565),
    (77269796, 28604192, 580),
    (77281068, 28611398, 637),
    (77320965, 28628531, 848),
    (77331117, 28631820, 891),
    (77339867, 28632484, 935),
    (77342149, 28632580, 947),
    (77342157, 28632681, 950),
    (77342259, 28632682, 954),
    (77343995, 28632652, 976),
    (77347358, 28632698, 997),
    (77355714, 28632611, 1034),
    (77363973, 28632563, 1075),
    (77371160, 28632397, 1109),
    (77374313, 28632365, 1137),
    (77384058, 28633021, 1186),
    (77388631, 28634355, 1207),
    (77396714, 28636888, 1244),
    (77406312, 28639991, 1284),
    (77408471, 28652596, 1384),
    (77408487, 28652692, 1390),
    (77408513, 28652816, 1397),
    (77411256, 28671430, 1538),
    (77412007, 28671767, 1550),
    (77414931, 28673455, 1579),
    (77444108, 28697939, 1839),
    (77444279, 28698118, 1840),
    (77475909, 28731864, 2129),
    (77478301, 28735129, 2151),
    (77518209, 28778696, 2544),
    (77519171, 28779433, 2554),
    (77519185, 28779444, 2554),
    (77550065, 28963217, 3573),
    (77687061, 29237832, 5117),
    (77730906, 29289686, 5434),
    (77731046, 29289731, 5437),
    (77851744, 29423439, 6395),
    (77852200, 29423909, 6411),
    (77870756, 29662179, 7795),
    (77882467, 29857361, 9029),
    (77881451, 29857649, 9048),
    (77881515, 29857728, 9052),
    (77886397, 29872300, 9174),
    (77885014, 29873309, 9199),
    (77886092, 29874979, 9227),
]

# Trip 5: Indirapuram -> Palwal (67km)
TRIP_5_RAW = [
    (77369412, 28645919, 0),
    (77357416, 28647021, 172),
    (77339929, 28633526, 360),
    (77339404, 28633477, 372),
    (77332396, 28625255, 456),
    (77332296, 28625153, 456),
    (77332019, 28625327, 465),
    (77322004, 28613710, 573),
    (77321574, 28614012, 584),
    (77321286, 28614183, 589),
    (77320909, 28614439, 597),
    (77320302, 28614845, 606),
    (77299089, 28590468, 785),
    (77297267, 28588764, 822),
    (77297125, 28588706, 828),
    (77298829, 28586026, 865),
    (77299341, 28585613, 871),
    (77299907, 28585315, 876),
    (77301177, 28584103, 888),
    (77302780, 28582023, 903),
    (77308254, 28574582, 953),
    (77311163, 28570687, 980),
    (77320386, 28559428, 1050),
    (77321392, 28558893, 1059),
    (77325739, 28555864, 1085),
    (77327155, 28555773, 1105),
    (77327195, 28555844, 1108),
    (77326125, 28556633, 1138),
    (77310815, 28543445, 1278),
    (77310088, 28543518, 1295),
    (77308874, 28542583, 1316),
    (77308726, 28542425, 1318),
    (77308310, 28541942, 1323),
    (77308209, 28541832, 1324),
    (77308091, 28541678, 1326),
    (77306276, 28527004, 1430),
    (77313597, 28511854, 1512),
    (77313741, 28511406, 1515),
    (77316120, 28500003, 1573),
    (77315606, 28499975, 1586),
    (77302633, 28497018, 1717),
    (77303780, 28492928, 1755),
    (77304191, 28490776, 1776),
    (77304446, 28489674, 1783),
    (77304615, 28488197, 1795),
    (77304700, 28487113, 1801),
    (77306072, 28471751, 1900),
    (77306139, 28470947, 1909),
    (77306071, 28470476, 1916),
    (77309139, 28318478, 2627),
    (77290827, 28249107, 3009),
    (77290378, 28246471, 3031),
    (77309992, 28197844, 3346),
    (77318318, 28175300, 3464),
    (77318280, 28174802, 3470),
    (77328206, 28155251, 3601),
    (77334568, 28151275, 3659),
    (77333669, 28137998, 3748),
    (77336302, 28136120, 3786),
    (77336862, 28137226, 3829),
    (77337745, 28137221, 3834),
]

# Trip metadata
TRIPS_META = [
    {
        "id": "TRIP_DRV003_01",
        "name": "Dwarka → Meerut",
        "pickup": "Dwarka, South West Delhi",
        "dropoff": "Meerut",
        "distance_km": 88.6,
        "data": TRIP_1_RAW,
    },
    {
        "id": "TRIP_DRV003_02", 
        "name": "Karol Bagh → Greater Noida",
        "pickup": "Karol Bagh",
        "dropoff": "Greater Noida",
        "distance_km": 55.2,
        "data": TRIP_2_RAW,
    },
    {
        "id": "TRIP_DRV003_03",
        "name": "Punjabi Bagh → Kurukshetra",
        "pickup": "Punjabi Bagh",
        "dropoff": "Kurukshetra",
        "distance_km": 158.3,
        "data": TRIP_3_RAW,
    },
    {
        "id": "TRIP_DRV003_04",
        "name": "Lajpat Nagar → Roorkee",
        "pickup": "Lajpat Nagar",
        "dropoff": "Roorkee",
        "distance_km": 173.4,
        "data": TRIP_4_RAW,
    },
    {
        "id": "TRIP_DRV003_05",
        "name": "Indirapuram → Palwal",
        "pickup": "Indirapuram",
        "dropoff": "Palwal",
        "distance_km": 67.0,
        "data": TRIP_5_RAW,
    },
]


def interpolate_points(raw_data, target_points=500):
    """Interpolate between GPS points to create smoother route with more data points."""
    if len(raw_data) < 2:
        return raw_data
    
    result = []
    for i in range(len(raw_data) - 1):
        lon1, lat1, t1 = raw_data[i]
        lon2, lat2, t2 = raw_data[i + 1]
        
        # Number of points between these two depends on time gap
        time_gap = t2 - t1
        if time_gap <= 0:
            time_gap = 1
        
        # Generate approximately 1 point every 1.5 seconds
        num_points = max(1, int(time_gap / 1.5))
        
        for j in range(num_points):
            ratio = j / num_points
            lon = lon1 + (lon2 - lon1) * ratio
            lat = lat1 + (lat2 - lat1) * ratio
            t = t1 + (t2 - t1) * ratio
            
            # Add small random noise to make it realistic (very small - within 10-50 meters)
            noise_scale = 0.00002  # ~2 meters
            lon += random.uniform(-noise_scale * 1e6, noise_scale * 1e6)
            lat += random.uniform(-noise_scale * 1e6, noise_scale * 1e6)
            
            result.append((lon, lat, t))
    
    # Add the last point
    result.append(raw_data[-1])
    
    return result


def generate_event_labels():
    """Generate realistic driving event labels."""
    events = [
        ("NORMAL", 0.70),
        ("AGGRESSIVE_BRAKING", 0.05),
        ("AGGRESSIVE_ACCEL", 0.05),
        ("AGGRESSIVE_RIGHT_TURN", 0.04),
        ("AGGRESSIVE_LEFT_TURN", 0.04),
        ("AGG_RIGHT_LANE_CHANGE", 0.04),
        ("AGG_LEFT_LANE_CHANGE", 0.04),
        ("POTHOLE", 0.02),
        ("SPEED_BUMP", 0.02),
    ]
    r = random.random()
    cumulative = 0
    for event, prob in events:
        cumulative += prob
        if r < cumulative:
            return event
    return "NORMAL"


def generate_accel_values(event_label):
    """Generate accelerometer values based on event type."""
    base_z = 9.8 + random.uniform(-0.2, 0.2)
    
    if event_label == "NORMAL":
        return (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            base_z,
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-2.0, 2.0),
        )
    elif event_label == "AGGRESSIVE_BRAKING":
        return (
            random.uniform(-0.2, 0.2),
            random.uniform(-2.5, -1.5),
            base_z,
            random.uniform(-0.5, 0.5),
            random.uniform(-2.0, -1.0),
            random.uniform(-2.0, 2.0),
        )
    elif event_label == "AGGRESSIVE_ACCEL":
        return (
            random.uniform(-0.2, 0.2),
            random.uniform(1.5, 2.5),
            base_z,
            random.uniform(-0.5, 0.5),
            random.uniform(0.5, 1.5),
            random.uniform(-2.0, 2.0),
        )
    elif "RIGHT_TURN" in event_label or "RIGHT_LANE" in event_label:
        return (
            random.uniform(0.8, 1.5),
            random.uniform(-0.5, 0.5),
            base_z,
            random.uniform(0.5, 1.5),
            random.uniform(-0.8, 0.8),
            random.uniform(-90, -40),
        )
    elif "LEFT_TURN" in event_label or "LEFT_LANE" in event_label:
        return (
            random.uniform(-1.5, -0.8),
            random.uniform(-0.5, 0.5),
            base_z,
            random.uniform(-1.5, -0.5),
            random.uniform(-0.8, 0.8),
            random.uniform(40, 90),
        )
    elif event_label in ("POTHOLE", "SPEED_BUMP"):
        return (
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            base_z + random.uniform(1.0, 3.0),
            random.uniform(-2.0, 2.0),
            random.uniform(-2.0, 2.0),
            random.uniform(-5.0, 5.0),
        )
    else:
        return (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            base_z,
            random.uniform(-1.0, 1.0),
            random.uniform(-1.0, 1.0),
            random.uniform(-2.0, 2.0),
        )


def generate_drivers_csv():
    """Generate drivers.csv with one Delhi driver."""
    filepath = os.path.join(OUTPUT_PATH, "drivers", "drivers.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "driver_id", "name", "city", "shift_preference", "avg_hours_per_day",
            "avg_earnings_per_hour", "experience_months", "rating", "total_trips",
            "total_earnings", "total_drive_hours"
        ])
        writer.writerow([
            "DRV003", "Rajesh Patel", "Delhi", "full_day", 9.5, 175, 36, 4.7,
            62, 27000, 82
        ])
    print(f"Created: {filepath}")


def generate_trips_csv():
    """Generate trips.csv with 5 Delhi trips."""
    filepath = os.path.join(OUTPUT_PATH, "trips", "trips.csv")
    base_date = datetime(2024, 2, 6, 6, 30, 0)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "trip_id", "driver_id", "date", "start_time", "end_time", "duration_min",
            "distance_km", "fare", "surge_multiplier", "pickup_location", "dropoff_location",
            "pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon", "trip_status",
            "passenger_rating"
        ])
        
        current_time = base_date
        
        for trip in TRIPS_META:
            data = trip["data"]
            # Get first and last GPS points
            first_lon, first_lat, _ = data[0]
            last_lon, last_lat, last_time = data[-1]
            
            # Convert scaled values to actual lat/lon
            pickup_lat = first_lat / 1e6
            pickup_lon = first_lon / 1e6
            dropoff_lat = last_lat / 1e6
            dropoff_lon = last_lon / 1e6
            
            # Calculate duration from time data
            duration_min = last_time / 60
            
            # Calculate fare (₹15/km base)
            fare = round(trip["distance_km"] * 15 * random.uniform(0.9, 1.2), 2)
            
            start_time = current_time
            end_time = current_time + timedelta(seconds=last_time)
            
            writer.writerow([
                trip["id"],
                "DRV003",
                start_time.strftime("%Y-%m-%d"),
                start_time.strftime("%Y-%m-%d %H:%M:%S"),
                end_time.strftime("%Y-%m-%d %H:%M:%S"),
                round(duration_min, 1),
                trip["distance_km"],
                fare,
                random.choice([1.0, 1.0, 1.0, 1.2, 1.5]),
                trip["pickup"],
                trip["dropoff"],
                round(pickup_lat, 6),
                round(pickup_lon, 6),
                round(dropoff_lat, 6),
                round(dropoff_lon, 6),
                "completed",
                round(random.uniform(3.5, 5.0), 1)
            ])
            
            # Add gap between trips (15-30 min)
            current_time = end_time + timedelta(minutes=random.randint(15, 30))
    
    print(f"Created: {filepath}")


def generate_accelerometer_csv():
    """Generate accelerometer_data.csv with interpolated GPS data."""
    filepath = os.path.join(OUTPUT_PATH, "sensor_data", "accelerometer_data.csv")
    base_date = datetime(2024, 2, 6, 6, 30, 0)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "sensor_id", "trip_id", "driver_id", "timestamp", "elapsed_seconds",
            "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
            "speed_kmh", "gps_lat", "gps_lon", "sample_rate_hz", "phone_orientation",
            "event_label_gt"
        ])
        
        sensor_id = 1
        current_time = base_date
        orientations = ["PORTRAIT", "LANDSCAPE", "FLAT"]
        
        for trip in TRIPS_META:
            trip_start = current_time
            
            # Interpolate points for smoother route
            interpolated = interpolate_points(trip["data"])
            
            for lon, lat, elapsed in interpolated:
                # Convert to actual lat/lon
                gps_lat = lat / 1e6
                gps_lon = lon / 1e6
                
                timestamp = trip_start + timedelta(seconds=elapsed)
                event_label = generate_event_labels()
                accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z = generate_accel_values(event_label)
                
                # Speed based on distance (rough estimate)
                speed = random.uniform(30, 80) if elapsed > 0 else 0
                
                writer.writerow([
                    f"ACC{sensor_id:05d}",
                    trip["id"],
                    "DRV003",
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    round(elapsed, 1),
                    round(accel_x, 4),
                    round(accel_y, 4),
                    round(accel_z, 4),
                    round(gyro_x, 3),
                    round(gyro_y, 3),
                    round(gyro_z, 3),
                    round(speed, 1),
                    round(gps_lat, 6),
                    round(gps_lon, 6),
                    25,
                    random.choice(orientations),
                    event_label
                ])
                
                sensor_id += 1
            
            # Get last time from trip data
            _, _, last_time = trip["data"][-1]
            current_time = trip_start + timedelta(seconds=last_time) + timedelta(minutes=random.randint(15, 30))
    
    print(f"Created: {filepath}")
    print(f"  Total sensor records: {sensor_id - 1}")


def generate_audio_features_csv():
    """Generate audio_features.csv for all trips with realistic conflict events."""
    filepath = os.path.join(OUTPUT_PATH, "sensor_data", "audio_features.csv")
    base_date = datetime(2024, 2, 6, 6, 30, 0)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "audio_feat_id", "trip_id", "driver_id", "window_start", "window_end",
            "window_size_seconds", "hop_size_seconds", "elapsed_seconds", "db_level",
            "baseline_db", "db_deviation", "energy_slope", "turn_gap_sec", "gap_ratio",
            "zcr", "spectral_centroid", "spectral_flux", "f0_mean", "f0_std",
            "speech_rate", "is_valid"
        ])
        
        audio_id = 1
        current_time = base_date
        
        # Define conflict injection points per trip with RANDOMIZED positions
        # Trip durations: T1=4800s, T2=3400s, T3=7500s, T4=9200s, T5=3800s
        # Each trip gets 2-3 SHORT conflicts at random times + some near-miss elevated moments
        conflict_times = {
            "TRIP_DRV003_01": [720, 2100, 3900],       # ~12min, 35min, 65min (randomized)
            "TRIP_DRV003_02": [600, 1850, 2900],       # ~10min, 31min, 48min
            "TRIP_DRV003_03": [1200, 2800, 4500, 5800, 6800],  # Punjabi Bagh - MORE EVENTS
            "TRIP_DRV003_04": [1800, 5100, 7500],      # ~30min, 85min, 125min
            "TRIP_DRV003_05": [700, 2200, 3200],       # ~12min, 37min, 53min
        }
        
        # Add some "elevated but not conflict" moments (partial layer activation)
        elevated_times = {
            "TRIP_DRV003_01": [1500, 3200],            # Some loud but not conflict
            "TRIP_DRV003_02": [1200, 2400],
            "TRIP_DRV003_03": [1800, 3500, 5200, 6200],  # Punjabi Bagh - MORE elevated
            "TRIP_DRV003_04": [3500, 6000, 8500],
            "TRIP_DRV003_05": [1400, 2700],
        }
        
        # Partial layer activations for Punjabi Bagh (TRIP_DRV003_03) - creates variety in visualizations
        # These trigger specific layers but not all 4
        # Trip duration ~125min (7500s), conflicts at ~20, 47, 75, 97, 113 min
        # Place partials at DIFFERENT times so they're visible as separate events
        partial_layer_times = {
            "TRIP_DRV003_03": {
                # Acoustic only (Layer 1): high ZCR + centroid - at ~8min, ~40min, ~55min
                "acoustic_only": [480, 2400, 3300],
                # Temporal only (Layer 2): speech disruption - at ~30min, ~85min
                "temporal_only": [1800, 5100],
                # Context gate only (Layer 4): loud volume - at ~12min, ~52min, ~105min
                "context_only": [720, 3100, 6300],
            }
        }
        
        for trip in TRIPS_META:
            trip_start = current_time
            _, _, last_time = trip["data"][-1]
            trip_id = trip["id"]
            
            # Get conflict times for this trip (with small random offset)
            trip_conflicts = [t + random.randint(-60, 60) for t in conflict_times.get(trip_id, [])]
            trip_elevated = elevated_times.get(trip_id, [])
            trip_partials = partial_layer_times.get(trip_id, {})
            
            # Generate audio features every 2 seconds
            elapsed = 0
            baseline_db = random.uniform(45, 52)
            
            while elapsed < last_time:
                window_start = trip_start + timedelta(seconds=elapsed)
                window_end = window_start + timedelta(seconds=5)
                
                # Check if this window should be a CONFLICT event (SHORT 20-second window)
                is_conflict = any(abs(elapsed - ct) <= 20 for ct in trip_conflicts)
                
                # Check if this is an "elevated" moment (loud but not full conflict)
                is_elevated = any(abs(elapsed - et) <= 30 for et in trip_elevated)
                
                # Check partial layer activations (for Punjabi Bagh variety)
                # Use 60-second windows so they're clearly visible in timeline plots
                is_acoustic_only = any(abs(elapsed - t) <= 60 for t in trip_partials.get("acoustic_only", []))
                is_temporal_only = any(abs(elapsed - t) <= 60 for t in trip_partials.get("temporal_only", []))
                is_context_only = any(abs(elapsed - t) <= 60 for t in trip_partials.get("context_only", []))
                
                # Don't let partial activations overlap with full conflicts
                if is_conflict:
                    is_acoustic_only = is_temporal_only = is_context_only = False
                
                if is_conflict:
                    # CONFLICT VALUES - all thresholds exceeded simultaneously
                    # MUST exceed: ZCR>0.55, Centroid>2000, f0_std>40, db_deviation>12, flux>0.05
                    db_level = baseline_db + random.uniform(18, 28)
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.70, 0.90), 4)  # Well above 0.55
                    spectral_centroid = round(random.uniform(2500, 3500), 1)  # Well above 2000
                    f0_std = round(random.uniform(55, 80), 2)  # Well above 40
                    f0_mean = round(random.uniform(180, 280), 1)
                    spectral_flux = round(random.uniform(0.06, 0.10), 5)  # Well above 0.05 threshold
                    energy_slope = round(random.uniform(55, 80), 2)
                    turn_gap_sec = round(random.uniform(0.1, 0.25), 3)
                    gap_ratio = round(random.uniform(0.88, 0.98), 3)
                    speech_rate = round(random.uniform(7, 12), 2)
                elif is_elevated:
                    # ELEVATED but not conflict - visual variety only, NO layer triggers
                    # Keep ALL values below thresholds so no events fire
                    db_level = baseline_db + random.uniform(6, 11)   # BELOW 12dB threshold
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.40, 0.52), 4)       # Below 0.55 threshold
                    spectral_centroid = round(random.uniform(1500, 1900), 1)  # Below 2000 threshold
                    f0_std = round(random.uniform(28, 38), 2)        # Below 40Hz threshold
                    f0_mean = round(random.uniform(130, 170), 1)
                    spectral_flux = round(random.uniform(0.02, 0.035), 5)
                    energy_slope = round(random.uniform(20, 40), 2)
                    turn_gap_sec = round(random.uniform(0.35, 0.8), 3)
                    gap_ratio = round(random.uniform(0.4, 0.7), 3)
                    speech_rate = round(random.uniform(5.0, 6.5), 2)
                elif is_acoustic_only:
                    # ACOUSTIC LAYER ONLY (Layer 1): High ZCR + Centroid, but not others
                    db_level = baseline_db + random.uniform(5, 10)   # Below 12dB
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.60, 0.75), 4)       # ABOVE 0.55 - triggers L1
                    spectral_centroid = round(random.uniform(2200, 2800), 1)  # ABOVE 2000 - triggers L1
                    f0_std = round(random.uniform(20, 35), 2)        # BELOW 40 - no L3
                    spectral_flux = round(random.uniform(0.06, 0.09), 5)  # Above threshold for L1
                    f0_mean = round(random.uniform(130, 170), 1)
                    energy_slope = round(random.uniform(20, 40), 2)
                    turn_gap_sec = round(random.uniform(0.4, 0.8), 3)
                    gap_ratio = round(random.uniform(0.4, 0.7), 3)
                    speech_rate = round(random.uniform(4.0, 6.0), 2)
                elif is_temporal_only:
                    # TEMPORAL LAYER ONLY (Layer 2): Disrupted speech patterns
                    db_level = baseline_db + random.uniform(5, 10)
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.25, 0.45), 4)       # Below 0.55 - no L1
                    spectral_centroid = round(random.uniform(1200, 1800), 1)  # Below 2000
                    f0_std = round(random.uniform(20, 35), 2)        # Below 40 - no L3
                    f0_mean = round(random.uniform(130, 170), 1)
                    spectral_flux = round(random.uniform(0.01, 0.03), 5)
                    energy_slope = round(random.uniform(55, 75), 2)  # ABOVE 50 - triggers L2
                    turn_gap_sec = round(random.uniform(0.05, 0.2), 3)  # BELOW 0.3 - triggers L2
                    gap_ratio = round(random.uniform(0.88, 0.95), 3)  # ABOVE 0.85 - triggers L2
                    speech_rate = round(random.uniform(4.0, 6.0), 2)
                elif is_context_only:
                    # CONTEXT GATE ONLY (Layer 4): Loud volume spike
                    db_level = baseline_db + random.uniform(15, 22)  # ABOVE 12dB - triggers L4
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.25, 0.45), 4)       # Below 0.55 - no L1
                    spectral_centroid = round(random.uniform(1200, 1800), 1)  # Below 2000
                    f0_std = round(random.uniform(20, 35), 2)        # Below 40 - no L3
                    f0_mean = round(random.uniform(130, 170), 1)
                    spectral_flux = round(random.uniform(0.01, 0.03), 5)
                    energy_slope = round(random.uniform(20, 40), 2)
                    turn_gap_sec = round(random.uniform(0.4, 0.8), 3)
                    gap_ratio = round(random.uniform(0.4, 0.7), 3)
                    speech_rate = round(random.uniform(4.0, 6.0), 2)
                else:
                    # NORMAL VALUES - below thresholds
                    db_level = baseline_db + random.uniform(-5, 10)
                    db_deviation = db_level - baseline_db
                    zcr = round(random.uniform(0.15, 0.45), 4)
                    spectral_centroid = round(random.uniform(800, 1800), 1)
                    f0_std = round(random.uniform(8, 35), 2)
                    f0_mean = round(random.uniform(110, 160), 1)
                    spectral_flux = round(random.uniform(0.005, 0.025), 5)
                    energy_slope = round(random.uniform(3, 20), 2)
                    turn_gap_sec = round(random.uniform(0.4, 1.5), 3)
                    gap_ratio = round(random.uniform(0.15, 0.6), 3)
                    speech_rate = round(random.uniform(2.5, 5.5), 2)
                
                writer.writerow([
                    f"AF{audio_id:05d}",
                    trip_id,
                    "DRV003",
                    window_start.strftime("%Y-%m-%d %H:%M:%S"),
                    window_end.strftime("%Y-%m-%d %H:%M:%S"),
                    5,
                    2,
                    elapsed,
                    round(db_level, 2),
                    round(baseline_db, 1),
                    round(db_deviation, 2),
                    energy_slope,
                    turn_gap_sec,
                    gap_ratio,
                    zcr,
                    spectral_centroid,
                    spectral_flux,
                    f0_mean,
                    f0_std,
                    speech_rate,
                    True  # All windows are valid
                ])
                
                audio_id += 1
                elapsed += 2
            
            current_time = trip_start + timedelta(seconds=last_time) + timedelta(minutes=random.randint(15, 30))
    
    print(f"Created: {filepath}")
    print(f"  Total audio records: {audio_id - 1}")
    print(f"  Conflict events injected: ~{sum(len(v) for v in conflict_times.values()) * 5} windows")


def generate_earnings_csv():
    """Generate earnings-related CSVs."""
    # driver_goals.csv
    filepath = os.path.join(OUTPUT_PATH, "earnings", "driver_goals.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["driver_id", "goal_type", "target_value", "current_value", "period", "start_date", "end_date"])
        writer.writerow(["DRV003", "daily_earnings", 5000, 0, "daily", "2024-02-06", "2024-02-06"])
        writer.writerow(["DRV003", "weekly_trips", 50, 5, "weekly", "2024-02-05", "2024-02-11"])
    print(f"Created: {filepath}")
    
    # earnings_velocity_log.csv
    filepath = os.path.join(OUTPUT_PATH, "earnings", "earnings_velocity_log.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["driver_id", "timestamp", "cumulative_earnings", "hourly_rate", "trips_completed", "hours_worked"])
        base_time = datetime(2024, 2, 6, 6, 30, 0)
        earnings = 0
        for i, trip in enumerate(TRIPS_META):
            earnings += trip["distance_km"] * 15
            writer.writerow([
                "DRV003",
                (base_time + timedelta(hours=i+1)).strftime("%Y-%m-%d %H:%M:%S"),
                round(earnings, 2),
                round(earnings / (i + 1), 2),
                i + 1,
                round((i + 1) * 1.2, 1)
            ])
    print(f"Created: {filepath}")


def generate_market_csv():
    """Generate market context CSV."""
    filepath = os.path.join(OUTPUT_PATH, "market", "market_context.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "zone", "demand_level", "surge_multiplier", "weather", "event_nearby", "competitor_supply"])
        
        zones = ["South Delhi", "Central Delhi", "Noida", "Gurgaon", "North Delhi"]
        for hour in range(6, 22):
            for zone in zones:
                demand = random.choice(["low", "medium", "high"])
                surge = 1.0 if demand == "low" else (1.2 if demand == "medium" else 1.5)
                writer.writerow([
                    f"2024-02-06 {hour:02d}:00:00",
                    zone,
                    demand,
                    surge,
                    "clear",
                    random.choice([True, False, False, False]),
                    random.randint(50, 200)
                ])
    print(f"Created: {filepath}")


def generate_processed_outputs():
    """Generate processed output CSVs."""
    # trip_summaries.csv
    filepath = os.path.join(OUTPUT_PATH, "processed_outputs", "trip_summaries.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "trip_id", "driver_id", "total_events", "aggressive_braking_count",
            "aggressive_accel_count", "harsh_turn_count", "lane_change_count",
            "avg_speed", "max_speed", "safety_score"
        ])
        
        for trip in TRIPS_META:
            writer.writerow([
                trip["id"],
                "DRV003",
                random.randint(5, 20),
                random.randint(1, 5),
                random.randint(1, 4),
                random.randint(2, 6),
                random.randint(3, 8),
                round(random.uniform(40, 60), 1),
                round(random.uniform(70, 100), 1),
                round(random.uniform(60, 95), 1)
            ])
    print(f"Created: {filepath}")
    
    # flagged_moments.csv - MOTION and COMPOUND events (audio events come from pipeline2)
    filepath = os.path.join(OUTPUT_PATH, "processed_outputs", "flagged_moments.csv")
    base_date = datetime(2024, 2, 6, 6, 30, 0)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "flag_id", "trip_id", "driver_id", "timestamp", "elapsed_seconds",
            "signal_type", "event_label", "severity", "explanation"
        ])
        
        flag_id = 1
        current_time = base_date
        
        # Motion events only - audio comes from pipeline2
        motion_events = [
            ("AGGRESSIVE_BRAKING", "high", "Sudden braking detected - possible traffic or obstacle"),
            ("HARSH_TURN", "medium", "Sharp turn taken at higher speed than recommended"),
            ("POTHOLE", "low", "Road irregularity detected - pothole or speed bump"),
            ("RAPID_ACCELERATION", "medium", "Quick acceleration detected - possibly unsafe"),
        ]
        
        # Compound events (motion + audio coinciding) - most severe
        compound_events = [
            ("COMPOUND_BRAKE_CONFLICT", "critical", "Harsh braking during passenger conflict - high stress moment"),
            ("COMPOUND_TURN_CONFLICT", "high", "Sharp maneuver during raised voices - distracted driving risk"),
        ]
        
        # Define which trips get compound events and when
        compound_times = {
            "TRIP_DRV003_03": [2800, 5800],  # Punjabi Bagh - 2 compound events
            "TRIP_DRV003_04": [5100],         # Lajpat Nagar - 1 compound event
        }
        
        for trip in TRIPS_META:
            trip_start = current_time
            _, _, last_time = trip["data"][-1]
            trip_id = trip["id"]
            
            # Generate motion flagged moments per trip
            # More events for Punjabi Bagh (TRIP_DRV003_03)
            if trip_id == "TRIP_DRV003_03":
                num_flags = random.randint(3, 4)  # More motion events for Punjabi Bagh
            else:
                num_flags = random.randint(1, 2)
            
            flag_times = sorted(random.sample(range(300, int(last_time) - 300), min(num_flags, max(1, int(last_time/400)))))
            
            for elapsed in flag_times:
                timestamp = trip_start + timedelta(seconds=elapsed)
                motion_type, severity, explanation = random.choice(motion_events)
                
                writer.writerow([
                    f"FLAG{flag_id:04d}",
                    trip_id,
                    "DRV003",
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    elapsed,
                    "MOTION",
                    motion_type,
                    severity,
                    explanation
                ])
                flag_id += 1
            
            # Add compound events for specific trips
            if trip_id in compound_times:
                for elapsed in compound_times[trip_id]:
                    timestamp = trip_start + timedelta(seconds=elapsed)
                    compound_type, severity, explanation = random.choice(compound_events)
                    
                    writer.writerow([
                        f"FLAG{flag_id:04d}",
                        trip_id,
                        "DRV003",
                        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        elapsed,
                        "COMPOUND",
                        compound_type,
                        severity,
                        explanation
                    ])
                    flag_id += 1
            
            current_time = trip_start + timedelta(seconds=last_time) + timedelta(minutes=random.randint(15, 30))
    
    print(f"Created: {filepath}")
    
    # driver_brake_zones.csv
    filepath = os.path.join(OUTPUT_PATH, "processed_outputs", "driver_brake_zones.csv")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["h3_cell", "driver_id", "brake_count", "avg_decel", "location_name"])
        
        # Generate some brake zone cells based on trip routes
        zones_data = [
            ("8928308280fffff", "DRV003", 5, 4.2, "Dwarka Sector 21"),
            ("8928308281fffff", "DRV003", 3, 3.8, "Karol Bagh Market"),
            ("8928308282fffff", "DRV003", 7, 5.1, "Punjabi Bagh Chowk"),
            ("8928308283fffff", "DRV003", 4, 4.5, "Lajpat Nagar Metro"),
            ("8928308284fffff", "DRV003", 6, 4.8, "Indirapuram Crossing"),
            ("8928308285fffff", "DRV003", 8, 5.5, "NH-24 Toll Plaza"),
        ]
        
        for row in zones_data:
            writer.writerow(row)
    
    print(f"Created: {filepath}")


def main():
    """Generate all simulation data."""
    print("=" * 60)
    print("Generating Simulation Dataset for Delhi Driver")
    print("=" * 60)
    print()
    
    generate_drivers_csv()
    generate_trips_csv()
    generate_accelerometer_csv()
    generate_audio_features_csv()
    generate_earnings_csv()
    generate_market_csv()
    generate_processed_outputs()
    
    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print(f"Output folder: {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
