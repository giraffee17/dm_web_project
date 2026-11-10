"""
features.py: THE one feature builder, shared by training (Phase 3) and the
live website (Phase 4). Because both use exactly this code, the model sees
features built the same way in training and in production.

Pipeline (all functions are pure: same input, same output)
  1. hourly_to_daily   hourly weather of one site -> one row per complete day
  2. fit_normals       the "normal" temperature for each day of the year,
                       fitted on TRAINING years only (a smooth 3-wave curve)
  3. add_anomalies     daily mean temperature minus its normal
  4. make_features     lags, rolling stats, upwind sites, differences, season
  5. make_targets      UB anomaly on day t+1, t+2, t+3 (training only)
  6. assign_split      train / validation / test by date, with 3 day gaps

The golden rule: a feature for day t may only use data up to the end of
day t (23:00 local time). tests/ checks this automatically.

Run from the repo folder to build the dataset:
  python -m src.features
"""

import json

import numpy as np
import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# 1. Hourly -> daily (one site)
# ---------------------------------------------------------------------------
DAILY_COLUMNS = [
    "t_mean", "t_min", "t_max", "t_range", "t_end",   # temperature (degC), t_end = value at 23:00
    "dew_mean", "dew_end",                             # dew point (degC)
    "rh_mean",                                         # relative humidity (%)
    "p_mean", "p_change_24h",                          # sea-level pressure (hPa) and its 24 h change
    "wind_speed", "wind_from_n", "wind_from_e",        # km/h; components explained below
    "cloud_mean",                                      # cloud cover (%)
    "precip_sum",                                      # precipitation (mm/day)
    "snow_depth_end",                                  # snow depth at 23:00 (cm)
    "radiation_sum",                                   # solar energy (MJ/m2/day)
]


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one site's hourly data (local time) to daily values.

    * Only COMPLETE days (all 24 hours) are kept. On the live site the
      current day is still running, so it is dropped automatically.
    * The result has one row for EVERY calendar day between the first and
      last complete day (a missing day would become an empty row), so that
      "lag 1" always means "yesterday", never "the previous row".
    """
    h = hourly.sort_values("time").copy()
    # Same number type in training and live serving (the archive stores
    # humidity and cloud cover as whole numbers, other sources may not).
    h[config.HOURLY_VARS] = h[config.HOURLY_VARS].astype("float64")
    h["date"] = h["time"].dt.floor("D")

    # Wind direction is a circle (359 deg and 1 deg are neighbours), so it
    # cannot be averaged directly. Split the wind into two components:
    #   wind_from_n = speed * cos(direction): + means wind FROM the north (cold in winter)
    #   wind_from_e = speed * sin(direction): + means wind FROM the east
    rad = np.deg2rad(h["wind_direction_10m"])
    h["wind_from_n"] = h["wind_speed_10m"] * np.cos(rad)
    h["wind_from_e"] = h["wind_speed_10m"] * np.sin(rad)

    g = h.groupby("date")
    daily = pd.DataFrame({
        "hours": g.size(),
        "t_mean": g["temperature_2m"].mean(),
        "t_min": g["temperature_2m"].min(),
        "t_max": g["temperature_2m"].max(),
        "t_end": g["temperature_2m"].last(),
        "dew_mean": g["dew_point_2m"].mean(),
        "dew_end": g["dew_point_2m"].last(),
        "rh_mean": g["relative_humidity_2m"].mean(),
        "p_mean": g["pressure_msl"].mean(),
        "p_end": g["pressure_msl"].last(),
        "wind_speed": g["wind_speed_10m"].mean(),
        "wind_from_n": g["wind_from_n"].mean(),
        "wind_from_e": g["wind_from_e"].mean(),
        "cloud_mean": g["cloud_cover"].mean(),
        "precip_sum": g["precipitation"].sum(),
        "snow_depth_end": g["snow_depth"].last() * 100,              # m -> cm
        "radiation_sum": g["shortwave_radiation"].sum() * 3600 / 1e6,  # W/m2 hours -> MJ/m2
    })
    daily = daily[daily["hours"] == 24].drop(columns="hours")
    if daily.empty:
        raise ValueError("No complete days in the hourly data.")

    # Calendar-complete index (see docstring).
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", name="date")
    daily = daily.reindex(full)

    daily["t_range"] = daily["t_max"] - daily["t_min"]
    # 24 h pressure change: 23:00 today minus 23:00 yesterday. Falling
    # pressure often means a front is coming; rising often means cold air.
    daily["p_change_24h"] = daily["p_end"] - daily["p_end"].shift(1)
    return daily[DAILY_COLUMNS]


# ---------------------------------------------------------------------------
# 2. Normals: the smooth "usual temperature" for each day of the year
# ---------------------------------------------------------------------------
def year_fraction(dates: pd.DatetimeIndex) -> np.ndarray:
    """Position in the year from 0 (Jan 1) to just below 1 (Dec 31).
    Dividing by the real year length (365 or 366) handles Feb 29 for free:
    it is simply a point between Feb 28 and Mar 1 on the same curve."""
    dates = pd.DatetimeIndex(dates)
    return (dates.dayofyear.to_numpy() - 1) / np.where(dates.is_leap_year, 366, 365)


def _harmonic_matrix(frac: np.ndarray, k_max: int) -> np.ndarray:
    cols = [np.ones_like(frac)]
    for k in range(1, k_max + 1):
        cols += [np.cos(2 * np.pi * k * frac), np.sin(2 * np.pi * k * frac)]
    return np.column_stack(cols)


def fit_normals(daily_by_site: dict, years=config.TRAIN_YEARS,
                k_max=config.NORMAL_HARMONICS) -> dict:
    """For each site, fit  normal(day) = a0 + sum over k of [ak cos + bk sin](2 pi k * year_fraction)
    to the daily mean temperature of the TRAINING years only (least squares).
    Returns {site: [a0, a1, b1, a2, b2, a3, b3]}."""
    normals = {}
    for site, daily in daily_by_site.items():
        d = daily["t_mean"].dropna()
        d = d[(d.index.year >= years[0]) & (d.index.year <= years[1])]
        coefs, *_ = np.linalg.lstsq(_harmonic_matrix(year_fraction(d.index), k_max),
                                    d.to_numpy(), rcond=None)
        normals[site] = [float(c) for c in coefs]
    return normals


def normal_for(dates, coefs) -> np.ndarray:
    """The normal temperature (degC) of a site on the given dates."""
    k_max = (len(coefs) - 1) // 2
    return _harmonic_matrix(year_fraction(dates), k_max) @ np.asarray(coefs)


def save_normals(normals: dict, path=config.NORMALS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"harmonics": config.NORMAL_HARMONICS, "train_years": list(config.TRAIN_YEARS),
               "coefficients": normals}
    path.write_text(json.dumps(payload, indent=2))


def load_normals(path=config.NORMALS_PATH) -> dict:
    return json.loads(path.read_text())["coefficients"]


# ---------------------------------------------------------------------------
# 3. + 4. Anomalies and features
# ---------------------------------------------------------------------------
def add_anomalies(daily_by_site: dict, normals: dict) -> dict:
    """Add t_normal and t_anom (= t_mean - t_normal) to every site.
    Comparing sites by ANOMALY removes their different elevations and
    climates: Irkutsk (434 m) and UB (1302 m) have different normal
    temperatures, but "5 degC colder than usual" means the same everywhere."""
    out = {}
    for site, daily in daily_by_site.items():
        d = daily.copy()
        d["t_normal"] = normal_for(d.index, normals[site])
        d["t_anom"] = d["t_mean"] - d["t_normal"]
        out[site] = d
    return out


def make_features(daily_by_site: dict) -> pd.DataFrame:
    """Build all model features. Row t only uses days <= t (shift(+k) looks
    BACK k days; rolling windows end at day t). Never shift(-k) here."""
    P = config.SITE_PREFIX
    ub = daily_by_site[config.MAIN_SITE]
    feats = {}

    # a) Today at every site: all daily values + today's anomaly.
    for site, d in daily_by_site.items():
        for col in DAILY_COLUMNS + ["t_anom"]:
            feats[f"{P[site]}_{col}"] = d[col]

    # b) UB lags: the same value 1, 2, 3, 7 and 14 days ago.
    for col in config.UB_LAG_VARS:
        for k in config.LAGS:
            feats[f"ub_{col}_lag{k}"] = ub[col].shift(k)

    # c) UB rolling mean and spread over the last 3, 7 and 30 days (including today).
    for col in config.UB_ROLL_VARS:
        for w in config.ROLL_WINDOWS:
            r = ub[col].rolling(w, min_periods=w)
            feats[f"ub_{col}_roll{w}_mean"] = r.mean()
            feats[f"ub_{col}_roll{w}_std"] = r.std()

    # d) Upwind lags and upwind minus UB differences.
    for site in config.UPWIND_SITES:
        d = daily_by_site[site]
        for col in config.UPWIND_LAG_VARS:
            for k in config.UPWIND_LAGS:
                feats[f"{P[site]}_{col}_lag{k}"] = d[col].shift(k)
        for col in config.DIFF_VARS:
            feats[f"{P[site]}_minus_ub_{col}"] = d[col] - ub[col]

    # e) Season as a point on a circle, so Dec 31 and Jan 1 are neighbours.
    frac = year_fraction(ub.index)
    feats["doy_sin"] = pd.Series(np.sin(2 * np.pi * frac), index=ub.index)
    feats["doy_cos"] = pd.Series(np.cos(2 * np.pi * frac), index=ub.index)

    return pd.DataFrame(feats)


def build_features(hourly_by_site: dict, normals: dict) -> tuple[pd.DataFrame, dict]:
    """Entry point used by BOTH training and the live API:
    hourly data per site -> (features per day, daily values per site)."""
    daily = {site: hourly_to_daily(h) for site, h in hourly_by_site.items()}
    # All sites on the same calendar (the days every site has).
    start = max(d.index.min() for d in daily.values())
    end = min(d.index.max() for d in daily.values())
    daily = {s: d.loc[start:end] for s, d in daily.items()}
    daily = add_anomalies(daily, normals)
    return make_features(daily), daily


# ---------------------------------------------------------------------------
# 5. + 6. Targets and split (training only)
# ---------------------------------------------------------------------------
def make_targets(ub_daily: pd.DataFrame) -> pd.DataFrame:
    """target_d{h} = UB temperature anomaly h days AFTER day t.
    This is the only place in the project that looks forward (shift(-h)),
    and targets are never used as features."""
    return pd.DataFrame({f"target_d{h}": ub_daily["t_anom"].shift(-h) for h in config.HORIZONS})


def assign_split(dates: pd.DatetimeIndex) -> pd.Series:
    """train / val / test by date of day t, plus 'gap' for the last 3 days of
    train and val: a row on 2019-12-30 would have its day+3 target on
    2020-01-02, inside the validation period, so it is dropped."""
    dates = pd.DatetimeIndex(dates)
    split = pd.Series("none", index=dates, name="split")
    for name, (y0, y1) in [("train", config.TRAIN_YEARS), ("val", config.VAL_YEARS),
                           ("test", config.TEST_YEARS)]:
        first = pd.Timestamp(f"{y0}-01-01")
        last = pd.Timestamp(f"{y1}-12-31") - pd.Timedelta(days=config.GAP_DAYS)
        split[(dates >= first) & (dates <= last)] = name
        gap = (dates > last) & (dates <= pd.Timestamp(f"{y1}-12-31"))
        split[gap] = "gap"
    return split


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("split",) and not c.startswith("target_")]


def build_dataset(hourly_by_site: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full training dataset: features + targets + split label.
    Returns (dataset, daily values of all sites in one wide table, normals)."""
    daily_raw = {site: hourly_to_daily(h) for site, h in hourly_by_site.items()}
    normals = fit_normals(daily_raw)                 # training years only
    feats, daily = build_features(hourly_by_site, normals)

    ds = feats.join(make_targets(daily[config.MAIN_SITE]))
    ds["split"] = assign_split(ds.index)
    # Warm-up: the first ~30 days have no full 30 day history yet.
    complete = ds[feature_columns(ds)].notna().all(axis=1)
    ds = ds[complete & ds["split"].isin(["train", "val", "test"])]
    ds = ds.dropna(subset=[f"target_d{h}" for h in config.HORIZONS])

    wide = pd.concat({config.SITE_PREFIX[s]: d for s, d in daily.items()}, axis=1)
    wide.columns = [f"{p}_{c}" for p, c in wide.columns]
    return ds, wide, normals


def fingerprint(df: pd.DataFrame) -> str:
    """Short fingerprint of a table's values: same data -> same code on any computer."""
    return format(int(pd.util.hash_pandas_object(df.round(4), index=True).sum()) & 0xFFFFFFFFFFFF, "012x")


if __name__ == "__main__":
    from src.download import load_hourly

    print("Loading hourly data for", ", ".join(config.SITES))
    hourly = {site: load_hourly(site) for site in config.SITES}
    ds, wide, normals = build_dataset(hourly)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(config.DATASET_PATH)
    wide.to_parquet(config.DAILY_PATH)
    save_normals(normals)

    print(f"\nDataset: {len(ds):,} days x {len(feature_columns(ds))} features + {len(config.HORIZONS)} targets")
    for name in ("train", "val", "test"):
        part = ds[ds["split"] == name]
        print(f"  {name:5s} {part.index.min().date()} to {part.index.max().date()}  {len(part):,} days")
    print(f"Saved {config.DATASET_PATH.relative_to(config.ROOT)}, "
          f"{config.DAILY_PATH.relative_to(config.ROOT)}, {config.NORMALS_PATH.relative_to(config.ROOT)}")
    print(f"Fingerprint: {fingerprint(ds)}  (same data + same code = same fingerprint on any computer)")
