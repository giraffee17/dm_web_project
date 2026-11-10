# Forecasting the Coldest Capital on Earth

Forecasting Ulaanbaatar's daily mean temperature for the next 3 days with gradient boosting,
deployed behind a minimal website. (Full README with results comes at the end of the project.)

## Setup (Windows, Python 3.14, from this folder)

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` is a full lock file: every package and sub-package has a fixed version,
so everyone gets exactly the same environment. To change packages, edit `requirements.in`
and regenerate the lock (command at the top of `requirements.in`).

## Phase 1: download and check the data

```
.venv\Scripts\python -m src.download      # cached: finished chunks are skipped
.venv\Scripts\python -m src.check_data    # should end with ALL CHECKS PASSED
```

How the raw data is protected:

1. **Validated before saving.** Every chunk must have a complete time axis, no missing values
   and values inside physical limits (`VALID_RANGES` in `src/config.py`). A failing chunk is never saved.
2. **Manifest.** `data/raw/manifest.json` records each file's source, model, variables, location,
   dates, row count, ERA5 grid cell and a SHA-256 fingerprint.
3. **Settings-aware cache.** A cached file is reused only if its manifest settings match
   `src/config.py` and its fingerprint still matches; otherwise it is downloaded again.

## Data notes

1. Hourly model data (1980 to 2025, 5 sites) uses `era5_seamless` (ERA5-Land + ERA5), because pure
   ERA5 on Open-Meteo has no snow depth. UB daily EDA data (1940+) uses pure `era5`, because ERA5-Land starts in 1950.
2. Times are local Ulaanbaatar time with a fixed UTC+8 offset (every day has exactly 24 hours).
3. 1940-01-01 is incomplete (ERA5 starts at 00:00 UTC = 08:00 in UB), so the EDA series starts on 1940-01-02.
