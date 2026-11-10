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

## Phase 2: features, anomaly target and split

```
.venv\Scripts\python -m src.features    # builds the dataset in a few seconds
.venv\Scripts\python -m tests.run       # should end with ALL TESTS PASSED
```

`src/features.py` is the **only** feature code in the project. Training and the live website both
call `build_features()`, so the model always sees features built the same way.

**Outputs:** `data/processed/dataset.parquet` (16,764 days x 158 features + 3 targets + split label),
`data/processed/daily.parquet` (daily values of all 5 sites, also used for the EDA charts) and
`models/normals.json` (the normal curves, needed when serving).

### Design decisions

1. **Daily features per site (17):** mean, min, max and range of temperature; temperature and dew point at
   23:00 (the newest state when the forecast is made); mean dew point and humidity; mean pressure and its
   24 hour change; wind speed and wind split into a north and an east component (a direction like 359 deg
   cannot be averaged, its components can); cloud cover; precipitation; snow depth at 23:00; solar energy.
   Only complete days (24 hours) are used, and every calendar day gets a row, so "lag 1" always means yesterday.
2. **Normals:** a smooth curve of 3 yearly waves fitted to 1980 to 2019 only. Going from 1 to 3 waves improves the fit a lot;
   more waves only chase noise. Feb 29 needs no special case: the curve uses the position in the year (day / 365 or 366).
3. **Anomaly target:** the models predict "how much warmer or colder than normal" on day+1, +2, +3, then add the
   normal back. The seasonal cycle is known, so the models can spend all their effort on the weather itself.
   Upwind sites are also compared by anomaly, because their elevations differ (Irkutsk 434 m, UB 1,302 m).
4. **Feature engineering (158 features):** UB lags (1, 2, 3, 7, 14 days), UB 3/7/30 day rolling mean and spread,
   today's values at all 5 sites, upwind lags (1, 2 days), upwind minus UB differences, season as sin/cos.
5. **No scaling or log transforms:** tree models (LightGBM, XGBoost, CatBoost) only ask "is x above or below a
   threshold?", so rescaling a feature does not change them.
6. **Split by time with 3 day gaps:** the last 3 days of train and validation are dropped, so no day+3 target
   crosses into the next set (e.g. 2019-12-30 + 3 days = 2020-01-02 would be in validation).

### Evidence for the upwind idea

Correlation between today's temperature anomaly at each site and UB's anomaly 1, 2 and 3 days later
(all 16,764 dataset days, 1980 to 2025): for day+2 and day+3, **Irkutsk and Moron predict UB better than UB itself**.

| Site today | UB day+1 | UB day+2 | UB day+3 |
|---|---|---|---|
| Ulaanbaatar | 0.77 | 0.51 | 0.38 |
| Irkutsk | 0.73 | **0.56** | **0.42** |
| Moron | 0.77 | **0.56** | **0.41** |
| Khovd | 0.63 | 0.50 | 0.37 |
| Uliastai | 0.64 | 0.44 | 0.32 |

### Tests (`python -m tests.run`, 35 checks)

1. **No leakage:** all weather after day t is replaced with random numbers; every feature of day t and
   earlier must stay exactly the same (checked at 4 dates, including the split borders). A deliberately
   planted leak was caught at all 4 dates, so the test really works.
2. **Train/serve match:** features built from only the last 60 days of data (like the live site) equal the
   features built from the full 46 years; a partly finished day is ignored.
3. **Normals** use training years only and handle Feb 29 and New Year. **Split** sets never overlap and no
   target crosses a border. **Targets** are exactly the UB anomaly 1, 2, 3 days later. **Daily aggregation** gives
   the right answers on hand-made data. **Reproducible:** two builds give the same fingerprint.

## Phase 3: feature selection, model choice, tuning and evaluation

```
.venv\Scripts\python -m src.train                  # all training stages (about 25 min on 2 CPU cores)
.venv\Scripts\python -m src.evaluate --unlock-test # the one-time test evaluation (reproduces the same numbers)
.venv\Scripts\python -m src.export                 # chart data for the website + models/meta.json
.venv\Scripts\python -m tests.run                  # 43 checks, should end with ALL TESTS PASSED
```

### Results (test set 2023 to 2025, evaluated once)

Errors are in degC of daily mean temperature. (The forecast is anomaly + normal, and the normal is known
exactly, so the error of the temperature equals the error of the anomaly.)

| Horizon | Model RMSE | Model MAE | Persistence RMSE | Climatology RMSE | Better than persistence by |
|---|---|---|---|---|---|
| day+1 | **1.44** | 1.09 | 3.06 | 4.38 | 53% |
| day+2 | **2.70** | 2.07 | 4.42 | 4.38 | 39% |
| day+3 | **3.39** | 2.60 | 4.99 | 4.38 | 32% |

*Persistence:* "the anomaly tomorrow equals today's". *Climatology:* "tomorrow is a normal day".
These baselines are shown here only (not on the website). Day+1 may be up to about 0.17 degC optimistic
(see the known limitation in Data notes).

### 1. Feature selection: 158 -> 126 -> 40 features

1. **Correlation filter (train data):** 32 near-duplicates (|r| > 0.95) dropped, e.g. Moron's mean and max
   temperature (r = 0.993). Of each pair, the feature more related to the targets is kept.
2. **Permutation importance (validation):** shuffle one feature, measure how much the error grows.
   Top 5: UB anomaly, Moron anomaly, Irkutsk northerly wind, Irkutsk anomaly, UB snow depth.
3. **Top N curve:** mean validation RMSE falls from 3.50 (1 feature) to 2.52 (25), then flattens
   (2.50 at 40, 2.51 at all 126). The smallest N within 0.01 degC of the best is **N = 40**.

### 2. Bake-off (validation RMSE, mean of the 3 horizons)

| Library | Defaults | Fairness check* |
|---|---|---|
| **CatBoost** | **2.436** | **2.405** |
| LightGBM | 2.504 | 2.468 |
| XGBoost | 2.586 | 2.538 |

\*Library defaults are not a level field: CatBoost builds 1,000 trees by default, the others 100.
The fairness check lets every library grow as many trees as it needs (early stopping on validation).
**CatBoost wins both ways.** (Feature selection used LightGBM, which if anything favours LightGBM.)

### 3. Optuna tuning (CatBoost, 50 trials per horizon)

TPE sampler (seed 42), median pruner (97 of 150 trials, about two thirds, stopped early), early stopping inside every trial,
studies stored in `models/optuna.sqlite`. Searched: learning rate, depth, L2 regularization, random strength,
subsample. **Trial 0 is always CatBoost's own default**, so tuning can never end up worse than the defaults.

| Horizon | Defaults (trial 0) | Tuned | Best found at trial |
|---|---|---|---|
| day+1 | 1.396 | 1.370 | 9 |
| day+2 | 2.549 | 2.528 | 31 |
| day+3 | 3.299 | 3.285 | 34 |

Tuning gains are small (0.01 to 0.03 degC): CatBoost's defaults were already close to the best settings.

### 4. Overfitting check

| Horizon | Train RMSE | Validation RMSE | Test RMSE (same model) | Test RMSE (final model) |
|---|---|---|---|---|
| day+1 | 0.48 | 1.37 | 1.45 | 1.44 |
| day+2 | 1.78 | 2.53 | 2.70 | 2.70 |
| day+3 | 2.47 | 3.29 | 3.36 | 3.39 |

1. **Validation -> test gap is small** (+0.08, +0.17, +0.07 for the same model). Persistence also got worse on
   test (2.87 -> 3.06 at day+1), so 2023 to 2025 was a slightly harder period, not a sign of overfitting.
2. **Train error is much lower than validation**, mostly at day+1. Boosting keeps memorizing the training
   years; the learning curves show the validation error is flat after the early stopping point while the training
   error keeps falling. Early stopping picks the point where validation is best, so the extra memorizing costs nothing.
3. Day+1 reached the 8,000 round cap while validation was still improving by less than 0.001 degC.
4. **Final model** (retrained on 1980 to 2022, decided before testing): on test it is equal to the train-only model
   (1.44 vs 1.45, 2.70 vs 2.70, 3.39 vs 3.36). It is the deployed model because it has seen the most recent climate.

### 5. What drives the forecast (SHAP, final models, 2023 to 2025)

Share of total feature importance (mean |SHAP|) by group:

| Group | day+1 | day+2 | day+3 |
|---|---|---|---|
| Upwind sites | 59% | 69% | 68% |
| UB today | 36% | 26% | 24% |
| UB recent days | 5% | 5% | 8% |

UB's own anomaly dominates day+1. From day+2 on, **upwind sites carry about two thirds of the importance**,
and UB's snow depth becomes a top feature (snow cover makes calm, clear nights much colder).

### 6. The test lock

`src/evaluate.py` is the only code that reads the test rows. The first `--unlock-test` run saves fingerprints of the
models in `models/reports/test_unlock_log.json`; running it again with the same models reproduces the same numbers,
but testing *different* models is refused. Retraining on another computer may give slightly different models,
which the lock would then refuse: the models in this repo are the evaluated ones.

## Data notes

1. Hourly model data (1980 to 2025, 5 sites) uses `era5_seamless` (ERA5-Land + ERA5), because pure
   ERA5 on Open-Meteo has no snow depth. UB daily EDA data (1940+) uses pure `era5`, because ERA5-Land starts in 1950.
2. Times are local Ulaanbaatar time with a fixed UTC+8 offset (every day has exactly 24 hours).
3. 1940-01-01 is incomplete (ERA5 starts at 00:00 UTC = 08:00 in UB), so the EDA series starts on 1940-01-02.
4. **Known limitation: reanalysis "sees" a few hours ahead.** ERA5 is produced in 12 hour batches
   (4D-Var windows 09:00 to 21:00 and 21:00 to 09:00 UTC). Every hour in a batch is adjusted using all
   observations in that batch, so the values for 17:00 to 23:00 local time on day t already include
   observations up to 05:00 on day t+1. A real forecast made at the end of day t would not have them.
   We measured the effect: building the features from 00:00 to 16:00 only (strictly clean) raises the
   validation RMSE by at most 0.17 degC at day+1, 0.13 at day+2 and 0.05 at day+3, and part of that is real
   information lost from the evening hours. We kept full days and report this as an upper bound on how
   optimistic the day+1 score may be. The live site uses forecast-API data, which does not have this effect.

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

### Problems found in Phase 3

| # | Problem found | What we changed | Why it matters |
|---|---|---|---|
| 8 | The bake-off with pure defaults compared **1,000 CatBoost trees with 100 LightGBM/XGBoost trees**. | Added a fairness check where every library grows as many trees as it needs. CatBoost wins both ways. | A winner should not depend on an unfair default. |
| 9 | Tuning was slower than planned, and the first speed fix **cut CatBoost's own default learning rate (0.029) out of the search range**. Tuned day+2 was then no better than the defaults. | Learning rate range widened to include it, and **trial 0 of every study is now the library default**. | Tuning can never end up worse than doing nothing. |
| 10 | One tuned setting (`bagging_temperature`) **does nothing** with CatBoost's default sampling method (MVS). | Replaced with `subsample`, which MVS does use. | A search over a setting that is ignored wastes trials and misleads. |

### Risks planned for later phases

| Risk | Plan |
|---|---|
| **Train/serve mismatch:** the model learns from ERA5, but the live site uses the Open-Meteo forecast API, a different weather model. | Phase 4: compare both sources on overlapping days and show a note about it on the site. |
| **"Is this error good?"** An RMSE means little without a comparison. | Done in Phase 3: persistence and climatology baselines are in the results table above. |
| **Wasted data:** the final model would only learn from 1980 to 2019. | Done in Phase 3: the deployed models are retrained on 1980 to 2022 and were evaluated once on the locked test set. |
| **Data leakage:** features must only use information available at the end of day t. | Checked at the end of every phase before moving on. |
