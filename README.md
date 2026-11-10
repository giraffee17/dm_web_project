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

## Lessons learned and safeguards

Phase 1 was built with AI assistance (Claude). Several problems came up along the way.
Each one was fixed, and each one led to a permanent safeguard in the code.

### Problems found in Phase 1

| # | Problem found | What we changed | Why it matters |
|---|---|---|---|
| 1 | Pure ERA5 (`models=era5`) returned **snow depth as empty** for every hour. It was only noticed after a full download. | Switched the hourly data to `era5_seamless`, and every source is now tested on a tiny sample before a full download. | A 2 second test request catches a broken variable before an hour of downloading. |
| 2 | The downloader **saved whatever the API sent**. Checks only ran at the very end, so a broken chunk was cached as if it were fine. | Every chunk is validated **before** it is saved: complete time axis, no missing values, values inside physical limits (`VALID_RANGES`). A failing chunk is never saved. | Bad data can never enter the project silently. |
| 3 | After changing the model setting, the old files had to be **deleted by hand**. Otherwise the cache would have reused them. | The cache compares each file's recorded settings with `config.py` and re-downloads automatically when they differ. | Changing a setting can never leave stale data behind. |
| 4 | There was no record of **exactly which data** was downloaded. | `data/raw/manifest.json` stores source, model, variables, location, dates, row count, ERA5 grid cell and a SHA-256 fingerprint for every file. | Proves which data the model was trained on, and detects edited or damaged files. |
| 5 | A file copied between machines silently arrived as an **older version**. | Files are compared byte for byte after every copy. | "I sent the fix" is not the same as "the fix is there". |
| 6 | The free API's **hourly and daily limits** stopped the download, and once a cut-off reply crashed it. | The downloader waits on rate limits, retries broken replies, writes files safely (temporary name, then rename) and resumes where it stopped. | Long downloads survive interruptions without losing work. |
| 7 | Code was tested on Python 3.13, but the project runs on **Python 3.14**. Only the main packages were pinned. | Python 3.14 is pinned (`.python-version`), and `requirements.txt` is a full lock file with every sub-package fixed, for Windows and Linux. | The same code gives the same results on every machine, including the Docker container. |

These safeguards were tested by breaking the data on purpose in 19 ways (empty column, missing hour,
duplicate hour, Kelvin instead of Celsius, edited file, changed model, failed download, and more).
All 19 were caught.

### Risks planned for later phases

| Risk | Plan |
|---|---|
| **Train/serve mismatch:** the model learns from ERA5, but the live site uses the Open-Meteo forecast API, a different weather model. | Phase 4: compare both sources on overlapping days and show a note about it on the site. |
| **"Is this error good?"** An RMSE means little without a comparison. | Compute two simple baselines, **persistence** (tomorrow's anomaly = today's) and **climatology** (tomorrow = the normal), and report them in this README. |
| **Wasted data:** the final model would only learn from 1980 to 2019. | After tuning, retrain the final 3 models on train + validation (1980 to 2022) with the tuned settings, then evaluate once on the locked test set (2023 to 2025). |
| **Data leakage:** features must only use information available at the end of day t. | Checked at the end of every phase before moving on. |
