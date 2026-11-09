"""
config.py: one place for every fixed project decision.

Every other script imports from here, so the sites, variables, dates,
split years and seeds can never drift apart between scripts.
"""

from pathlib import Path

# ---------- Folders ----------
ROOT = Path(__file__).resolve().parent.parent          # the repo folder
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HOURLY_DIR = RAW_DIR / "hourly"                         # hourly chunks, one file per site per decade
DAILY_EDA_DIR = RAW_DIR / "daily_ub"                    # UB daily chunks for EDA, one file per decade
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
STATIC_DATA_DIR = ROOT / "static" / "data"              # chart JSON for the website

# ---------- Reproducibility ----------
SEED = 42

# ---------- Sites (latitude, longitude) ----------
# Ulaanbaatar is the main site. The other four are upwind: weather in
# Mongolia mostly moves from west/northwest to east, so cold air often
# reaches these places one to three days before it reaches UB.
MAIN_SITE = "ulaanbaatar"
SITES = {
    "ulaanbaatar": (47.92, 106.92),
    "uliastai":    (47.74, 96.84),
    "khovd":       (48.01, 91.64),
    "moron":       (49.64, 100.16),   # Mörön (ASCII name for file names)
    "irkutsk":     (52.29, 104.29),
}
TIMEZONE = "Asia/Ulaanbaatar"

# ---------- Open-Meteo API ----------
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"   # used in Phase 4 (live data)
# Which reanalysis to ask for (always set explicitly, never the API's default):
#  * Daily EDA data, 1940+: pure "era5". ERA5-Land only starts in 1950,
#    so a blend would create a jump in the 1940s.
#  * Hourly model data, 1980+: "era5_seamless" = ERA5-Land (finer ~9 km grid)
#    for land variables plus ERA5 for the rest. Needed because pure ERA5
#    on Open-Meteo returns snow_depth as empty.
ARCHIVE_MODEL_DAILY = "era5"
ARCHIVE_MODEL_HOURLY = "era5_seamless"
USER_AGENT = "ub-forecast-student-project/1.0"

# ---------- What to download ----------
# EDA: Ulaanbaatar daily data, 1940 to today.
EDA_START_YEAR = 1940
DAILY_EDA_VARS = [
    "temperature_2m_mean",
    "temperature_2m_min",
    "temperature_2m_max",
    "precipitation_sum",
    "snowfall_sum",
]

# Modeling: hourly data for all 5 sites, 1980 to the end of the test period.
HOURLY_START_YEAR = 1980
HOURLY_END_YEAR = 2025
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation",
    "snow_depth",
    "shortwave_radiation",
]

# ---------- Time split (never random) ----------
TRAIN_YEARS = (1980, 2019)
VAL_YEARS = (2020, 2022)
TEST_YEARS = (2023, 2025)
GAP_DAYS = 3               # days dropped at each border so day+3 targets cannot leak
HORIZONS = [1, 2, 3]       # forecast day+1, day+2, day+3
