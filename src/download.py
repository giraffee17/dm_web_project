"""
download.py: download raw weather data from the Open-Meteo archive (ERA5)
and cache it as Parquet, so each piece is downloaded only once.

What it downloads
  1. Ulaanbaatar DAILY data, 1940 to today (for the EDA charts), pure ERA5.
  2. HOURLY data, 1980 to 2025, for all 5 sites (for the model), ERA5 seamless
     (see config.py for why the two use different settings).

How it stays polite to the free API
  * Data is requested in chunks of one decade per site.
  * It waits between requests.
  * If the API says "too many requests" (HTTP 429), it reads the reason
    and waits a minute, or until the next hour, before trying again.
  * Every chunk is saved to its own Parquet file. If a file already
    exists, that chunk is skipped. So you can stop and re-run at any time.

Run from the repo folder:
  python -m src.download            # everything
  python -m src.download --only daily
  python -m src.download --only hourly
"""

import argparse
import datetime as dt
import json
import time

import pandas as pd
import requests

from src import config

PAUSE_BETWEEN_REQUESTS = 30   # seconds; big chunks "cost" many API calls each
MAX_TRIES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def decade_chunks(start_year: int, end_date: dt.date):
    """Split a period into (start_date, end_date) pieces of one decade each.
    Example: 1980 -> 1989, 1990 -> 1999, ... The last piece ends at end_date."""
    chunks = []
    year = start_year
    while dt.date(year, 1, 1) <= end_date:
        last_year = (year // 10) * 10 + 9
        chunk_end = min(dt.date(last_year, 12, 31), end_date)
        chunks.append((dt.date(year, 1, 1), chunk_end))
        year = last_year + 1
    return chunks


def seconds_until_next_hour() -> int:
    now = dt.datetime.now()
    next_hour = (now + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return int((next_hour - now).total_seconds()) + 60   # plus a 1 minute safety margin


def get_json(params: dict) -> dict:
    """Call the archive API with retries and rate limit handling."""
    headers = {"User-Agent": config.USER_AGENT}
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(config.ARCHIVE_URL, params=params, headers=headers, timeout=180)
        except requests.RequestException as e:
            wait = 20 * attempt
            print(f"    network error ({e.__class__.__name__}), retry in {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                # The reply was cut off or empty (unstable connection): try again.
                wait = 20 * attempt
                print(f"    incomplete reply from the API, retry in {wait}s")
                time.sleep(wait)
                continue

        reason = r.text[:200]
        if r.status_code == 429:
            # Open-Meteo says which limit we hit: minutely, hourly or daily.
            if "Daily" in reason:
                raise SystemExit("Daily API limit reached. Re-run tomorrow; finished chunks are kept.")
            wait = seconds_until_next_hour() if "Hourly" in reason else 65
            print(f"    rate limited ({reason.strip()}), waiting {wait}s")
            time.sleep(wait)
            continue
        if r.status_code >= 500:
            wait = 30 * attempt
            print(f"    server error {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue
        # 4xx other than 429 means our request is wrong: stop and show why.
        raise RuntimeError(f"API error {r.status_code}: {reason}")
    raise RuntimeError("Gave up after too many failed tries.")


def to_frame(payload: dict, block: str, site: str) -> pd.DataFrame:
    """Turn the API's JSON block ('hourly' or 'daily') into a DataFrame."""
    df = pd.DataFrame(payload[block])
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "site", site)
    return df


def save_site_meta(site: str, payload: dict, key: str):
    """Remember the real ERA5 grid cell (lat, lon, elevation) the API used.
    Useful to explain that ERA5 is a 0.25 degree grid, not a weather station."""
    path = config.RAW_DIR / "sites_meta.json"
    meta = json.loads(path.read_text()) if path.exists() else {}
    lat, lon = config.SITES[site]
    meta[key] = {
        "requested_lat": lat, "requested_lon": lon,
        "grid_lat": payload.get("latitude"), "grid_lon": payload.get("longitude"),
        "elevation_m": payload.get("elevation"),
        "utc_offset_seconds": payload.get("utc_offset_seconds"),
    }
    path.write_text(json.dumps(meta, indent=2))


def download_chunk(site, start, end, block, variables, model, out_path):
    """Download one chunk and save it, unless the file already exists."""
    if out_path.exists():
        print(f"  [cached]  {out_path.name}")
        return False
    lat, lon = config.SITES[site]
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        block: ",".join(variables),
        "timezone": config.TIMEZONE,
        "models": model,
    }
    print(f"  [download] {out_path.name}  ({start} to {end})")
    payload = get_json(params)
    df = to_frame(payload, block, site)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    # Hourly and daily data use different grids, so keep them as separate entries.
    save_site_meta(site, payload, key=site if block == "hourly" else f"{site}_daily")
    print(f"             saved {len(df):,} rows")
    return True


# ---------------------------------------------------------------------------
# The two downloads
# ---------------------------------------------------------------------------
def download_daily_eda():
    """Ulaanbaatar daily data, 1940 to today, one file per decade."""
    print("UB daily data for EDA")
    for start, end in decade_chunks(config.EDA_START_YEAR, dt.date.today()):
        out = config.DAILY_EDA_DIR / f"{config.MAIN_SITE}_daily_{start.year}s.parquet"
        if download_chunk(config.MAIN_SITE, start, end, "daily", config.DAILY_EDA_VARS,
                          config.ARCHIVE_MODEL_DAILY, out):
            time.sleep(PAUSE_BETWEEN_REQUESTS)


def download_hourly():
    """Hourly data for all 5 sites, 1980 to 2025, one file per site per decade."""
    print("Hourly data for modeling")
    end_date = dt.date(config.HOURLY_END_YEAR, 12, 31)
    for site in config.SITES:
        for start, end in decade_chunks(config.HOURLY_START_YEAR, end_date):
            out = config.HOURLY_DIR / f"{site}_{start.year}s.parquet"
            if download_chunk(site, start, end, "hourly", config.HOURLY_VARS,
                              config.ARCHIVE_MODEL_HOURLY, out):
                time.sleep(PAUSE_BETWEEN_REQUESTS)


# ---------------------------------------------------------------------------
# Loaders used by later phases
# ---------------------------------------------------------------------------
def load_hourly(site: str) -> pd.DataFrame:
    """All cached hourly chunks for one site, joined and sorted by time."""
    files = sorted(config.HOURLY_DIR.glob(f"{site}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No hourly data for {site}. Run: python -m src.download")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("time").reset_index(drop=True)


def load_daily_eda() -> pd.DataFrame:
    """UB daily data for EDA. The archive is a few days behind real time,
    so the newest days come back empty; those rows are dropped here."""
    files = sorted(config.DAILY_EDA_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError("No daily EDA data. Run: python -m src.download --only daily")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("time").reset_index(drop=True)
    return df.dropna(subset=["temperature_2m_mean"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache Open-Meteo data.")
    parser.add_argument("--only", choices=["daily", "hourly"], help="download just one part")
    args = parser.parse_args()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.only in (None, "hourly"):
        download_hourly()
    if args.only in (None, "daily"):
        download_daily_eda()
    print("Done.")
