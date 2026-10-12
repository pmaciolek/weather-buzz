"""Shared configuration: locations, scoring thresholds, time ranges."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache"
WEB_DATA = ROOT / "web" / "data"

# Climatology baseline: last 10 complete years.
BASELINE_YEARS = list(range(2016, 2026))
# The global grid uses the most recent 5 years: Open-Meteo's free quota (~10k
# weighted calls/day) makes 10 years x 1000+ points take ~2 weeks.
GRID_YEARS = list(range(2021, 2026))

# Key locations (name, lat, lon). "Sicily" is covered by two stations because
# the north (Palermo) and east (Catania) coasts differ noticeably.
CITIES = [
    ("Kraków", 50.0614, 19.9366),
    ("Brussels", 50.8503, 4.3517),
    ("Alicante", 38.3452, -0.4810),
    ("Málaga", 36.7213, -4.4214),
    ("Palermo (Sicily)", 38.1157, 13.3615),
    ("Catania (Sicily)", 37.5079, 15.0830),
    ("New York", 40.7128, -74.0060),
    ("San Francisco", 37.7749, -122.4194),
    ("Mountain View", 37.3861, -122.0839),
    ("London", 51.5074, -0.1278),
    ("Rome", 41.9028, 12.4964),
    ("Split", 43.5081, 16.4402),
    ("Larnaca", 34.9003, 33.6232),
    ("Nice", 43.7102, 7.2620),
    ("Santa Cruz de Tenerife", 28.4636, -16.2518),
    ("Bilbao", 43.2630, -2.9350),
    ("Amsterdam", 52.3676, 4.9041),
    ("Berlin", 52.5200, 13.4050),
    ("Graz", 47.0707, 15.4395),
    ("Gdańsk", 54.3520, 18.6466),
    ("Wrocław", 51.1079, 17.0385),
    ("Warsaw", 52.2297, 21.0122),
    ("Seattle", 47.6062, -122.3321),
    ("Tucson", 32.2226, -110.9747),
    ("Austin", 30.2672, -97.7431),
    ("Miami", 25.7617, -80.1918),
    ("Bogotá", 4.7110, -74.0721),
    ("São Paulo", -23.5505, -46.6333),
    ("Buenos Aires", -34.6037, -58.3816),
    ("Cape Town", -33.9249, 18.4241),
    ("Tokyo", 35.6762, 139.6503),
    ("Seoul", 37.5665, 126.9780),
    ("Sydney", -33.8688, 151.2093),
    ("Melbourne", -37.8136, 144.9631),
]

# --- Scoring rules -----------------------------------------------------------
# Riding window: local clock hours [RIDE_START, RIDE_END) that are also in daylight.
RIDE_START, RIDE_END = 6, 22
MIN_GOOD_HOURS = 6

IDEAL_T = (18.0, 30.0)          # °C
OK_T_LOW = (12.0, 18.0)         # °C, "medium" band below ideal
OK_T_HIGH = (30.0, 33.0)        # °C, "medium" band above ideal

LOW_T = (4.0, 12.0)             # °C, "low" band (dry days only)

DAY_RAIN_MAX_MM = 1.0           # rain during riding window must stay below this
# "Low" warm-and-showery day: some rain, but enough warm dry hours to ride.
WET_WARM_T = 24.0               # °C, hour must be above this
WET_WARM_MIN_DRY_HOURS = 3
WET_WARM_RAIN_MAX_MM = 10.0     # small/medium rain over the riding window
DRY_HOUR_MM = 0.1               # an hour with less precipitation counts as dry
WIND_NOT_IDEAL = 25.0           # km/h: hour downgraded from ideal to medium
WIND_ZERO = 40.0                # km/h: any riding hour above this -> day score 0

# Heat and humidity, as the humidex (from temperature and dew point): caps how good an hour can be.
HUMIDEX_IDEAL_MAX = 25.0        # up to 25: no limit
HUMIDEX_MEDIUM_MAX = 30.0       # 25-30: at best medium
HUMIDEX_LOW_MAX = 35.0          # 30-35: at best low; above 35: not rideable

# Air quality veto (European AQI from CAMS): daytime mean at/above this is "poor".
AQI_VETO = 60

SCORE_IDEAL = 1.0
SCORE_MEDIUM = 0.5
SCORE_LOW = 0.25

WEATHER_VARS = ["temperature_2m", "precipitation", "wind_speed_10m"]
DEW_VARS = ["dew_point_2m"]     # fetched separately, so the existing weather cache stays valid

# --- Projection: the next 10 years ------------------------------------------
# Delta change: the observed 2016-2025 hours are shifted by the CMIP6 multi-model
# change between the 2016-2025 and 2026-2035 means (SSP2-4.5), per calendar month.
PROJ_YEARS = (2026, 2035)
PROJ_SCENARIO = "ssp2_4_5"
PROJ_FIT_YEARS = (2015, 2100)   # span for the pattern-scaling regression (per model, month and cell)
PROJ_MODELS = ["mpi_esm1_2_lr", "miroc6", "mri_esm2_0", "noresm2_mm", "cnrm_cm6_1", "access_cm2",
               "ipsl_cm6a_lr", "ec_earth3_cc"]
PROJ_RAIN_RATIO = (0.8, 1.25)   # clamp on the precipitation scaling
AQ_VARS = ["european_aqi", "pm2_5"]
