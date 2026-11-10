"""
live.py: the live 3 day forecast for the website.

Steps (all features come from src/features.py, the same code used in training):
  1. Fetch the last 60 days of hourly weather for all 5 sites from the
     Open-Meteo FORECAST API (the ERA5 archive is ~5 days behind real time).
  2. Keep only COMPLETE PAST days. The API also returns today, but today's
     later hours are the weather model's forecasts, not observations;
     using them would feed forecasts into our model as if they were facts.
  3. Validate the data (no missing values, physical limits).
  4. Build features with build_features() and predict the anomaly for
     day+1, +2, +3 after the last complete day; add the normal back.
  5. Cache the result once per day (Ulaanbaatar date). If the API fails,
     serve the newest cached forecast marked as "stale".

Try it from the repo folder:
  python -m src.live
"""

import datetime as dt
import functools
import json
import os
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src import config
from src.features import build_features, load_normals, normal_for


class LiveDataError(RuntimeError):
    """The live data could not be fetched or failed validation."""


def today_ub() -> dt.date:
    return dt.datetime.now(ZoneInfo(config.TIMEZONE)).date()


# ---------------------------------------------------------------------------
# 1. Fetch
# ---------------------------------------------------------------------------
def fetch_site(site: str, tries: int = 3) -> pd.DataFrame:
    """Recent hourly weather for one site. Fails fast (a web page should not
    hang): 3 short tries, no waiting for rate limits."""
    lat, lon = config.SITES[site]
    params = {"latitude": lat, "longitude": lon, "hourly": ",".join(config.HOURLY_VARS),
              "timezone": config.TIMEZONE, "past_days": config.LIVE_PAST_DAYS, "forecast_days": 1}
    last_error = ""
    for attempt in range(tries):
        try:
            r = requests.get(config.FORECAST_URL, params=params, timeout=config.LIVE_TIMEOUT,
                             headers={"User-Agent": config.USER_AGENT})
            if r.status_code == 200:
                df = pd.DataFrame(r.json()["hourly"])
                df["time"] = pd.to_datetime(df["time"])
                df.insert(0, "site", site)
                return df
            last_error = f"HTTP {r.status_code}: {r.text[:150]}"
            if r.status_code == 429:
                break                      # rate limited: do not hammer the API
        except (requests.RequestException, ValueError, KeyError) as e:
            last_error = f"{e.__class__.__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    raise LiveDataError(f"{site}: could not fetch live data ({last_error})")


def fetch_all() -> dict:
    return {site: fetch_site(site) for site in config.SITES}


# ---------------------------------------------------------------------------
# 2. + 3. Keep complete past days, validate
# ---------------------------------------------------------------------------
def complete_past_days(hourly: pd.DataFrame, today: dt.date) -> pd.DataFrame:
    """Drop today (and anything later): those hours are model forecasts."""
    return hourly[hourly["time"] < pd.Timestamp(today)].copy()


def validate_live(hourly: pd.DataFrame, site: str):
    """Raise LiveDataError if the recent data cannot be trusted."""
    problems = []
    recent = hourly[hourly["time"] >= hourly["time"].max().floor("D")
                    - pd.Timedelta(days=config.MIN_HISTORY_DAYS + 1)]
    missing = [v for v in config.HOURLY_VARS if v not in hourly.columns]
    if missing:
        raise LiveDataError(f"{site}: variables missing: {missing}")
    nan = recent[config.HOURLY_VARS].isna().sum()
    if (nan > 0).any():
        problems.append(f"missing values {nan[nan > 0].to_dict()}")
    for v in config.HOURLY_VARS:
        lo, hi = config.VALID_RANGES[v]
        s = recent[v].dropna()
        if ((s < lo) | (s > hi)).any():
            problems.append(f"{v} outside [{lo}, {hi}]")
    if recent["time"].duplicated().any():
        problems.append("duplicate hours")
    if problems:
        raise LiveDataError(f"{site}: " + "; ".join(problems))


# ---------------------------------------------------------------------------
# 4. Predict
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_assets():
    """Models, feature list and normals, loaded once per server start."""
    from src.train import load_model
    meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
    models = {h: load_model(meta["library"], h, "final") for h in config.HORIZONS}
    return meta, models, load_normals()


def make_forecast(hourly_by_site: dict | None = None, today: dt.date | None = None) -> dict:
    """The 3 day forecast as a JSON-ready dict."""
    today = today or today_ub()
    hourly_by_site = hourly_by_site if hourly_by_site is not None else fetch_all()
    hourly = {s: complete_past_days(h, today) for s, h in hourly_by_site.items()}
    for site, h in hourly.items():
        validate_live(h, site)

    meta, models, normals = load_assets()
    cols = meta["features"]
    feats, daily = build_features(hourly, normals)
    usable = feats[cols].dropna()
    if usable.empty:
        raise LiveDataError("not enough complete days to build features (31 needed)")
    row = usable.iloc[[-1]]
    issued_from = row.index[0]

    days = []
    for h in config.HORIZONS:
        target = issued_from + pd.Timedelta(days=h)
        anomaly = float(models[h].predict(row)[0])
        normal = float(normal_for(pd.DatetimeIndex([target]), normals[config.MAIN_SITE])[0])
        days.append({"date": str(target.date()), "horizon": h,
                     "temperature": round(normal + anomaly, 1),
                     "normal": round(normal, 1), "anomaly": round(anomaly, 1),
                     "typical_error": round(meta["horizons"][f"d{h}"]["test_rmse"], 1)})   # test RMSE, degC
    ub = daily[config.MAIN_SITE].loc[issued_from]
    return {
        "location": "Ulaanbaatar",
        "data_through": str(issued_from.date()),     # last complete day used
        "today": str(today),
        "data_is_current": issued_from.date() == today - dt.timedelta(days=1),
        "last_day_observed": {"temperature": round(float(ub["t_mean"]), 1),
                              "normal": round(float(ub["t_normal"]), 1),
                              "anomaly": round(float(ub["t_anom"]), 1)},
        "days": days,
        "source": "Open-Meteo forecast API (recent weather, 5 sites)",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "stale": False,
    }


# ---------------------------------------------------------------------------
# 5. Once-per-day cache
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_last_failure = {"time": 0.0, "error": ""}


def cache_path(day: dt.date):
    return config.CACHE_DIR / f"forecast_{day}.json"


def newest_cached() -> dict | None:
    files = sorted(config.CACHE_DIR.glob("forecast_*.json"))
    return json.loads(files[-1].read_text()) if files else None


def get_forecast() -> dict:
    """Today's forecast: from the cache if it exists, otherwise computed once.
    On failure: the newest cached forecast with stale=True (or an error if
    there is none). After a failure, wait LIVE_RETRY_MINUTES before retrying."""
    today = today_ub()
    with _lock:
        path = cache_path(today)
        if path.exists():
            return json.loads(path.read_text())
        recently_failed = time.time() - _last_failure["time"] < config.LIVE_RETRY_MINUTES * 60
        if not recently_failed:
            try:
                result = make_forecast(today=today)
                config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(json.dumps(result, indent=1))
                os.replace(tmp, path)          # safe write: never half a file
                return result
            except (LiveDataError, requests.RequestException) as e:
                _last_failure.update(time=time.time(), error=str(e))
        old = newest_cached()
        if old is None:
            raise LiveDataError(f"No forecast available yet: {_last_failure['error']}")
        return {**old, "stale": True, "stale_reason": _last_failure["error"]}


if __name__ == "__main__":
    f = make_forecast()
    print(f"Data through {f['data_through']} (today is {f['today']}, current: {f['data_is_current']})")
    o = f["last_day_observed"]
    print(f"  observed {f['data_through']}: {o['temperature']:+.1f} degC (normal {o['normal']:+.1f}, anomaly {o['anomaly']:+.1f})")
    for d in f["days"]:
        print(f"  day+{d['horizon']} {d['date']}: {d['temperature']:+.1f} degC "
              f"(normal {d['normal']:+.1f}, anomaly {d['anomaly']:+.1f}, typical error +-{d['typical_error']})")
