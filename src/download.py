"""
download.py: download raw weather data from the Open-Meteo archive (ERA5)
and cache it as Parquet, so each piece is downloaded only once.

What it downloads
  1. Ulaanbaatar DAILY data, 1940 to today (for the EDA charts), pure ERA5.
  2. HOURLY data, 1980 to 2025, for all 5 sites (for the model), ERA5 seamless
     (see config.py for why the two use different settings).

Safety rules (so bad data can never sneak into the project)
  * VALIDATE BEFORE SAVING: every chunk is checked (complete time axis,
    no missing values, values inside physical limits). A chunk that fails
    is NOT saved and the script stops with a clear message.
  * MANIFEST: data/raw/manifest.json records, for every file, the exact
    settings it was downloaded with (model, variables, place, dates),
    its row count, its grid cell and a SHA-256 fingerprint of the file.
  * SETTINGS-AWARE CACHE: a file is only reused if its manifest entry
    matches today's settings in config.py AND its fingerprint still
    matches. Change a setting and that chunk is re-downloaded automatically.
  * SAFE WRITES: files are written to a temporary name first and renamed
    at the end, so a crash can never leave a half-written file.

How it stays polite to the free API
  * One request per site per decade, with a pause between requests.
  * On "too many requests" (HTTP 429) it waits a minute or until the next
    hour; on the daily limit it stops (re-run tomorrow, nothing is lost).

Run from the repo folder:
  python -m src.download                     # everything
  python -m src.download --only hourly       # or --only daily
  python -m src.download --rebuild-manifest  # (one time) register files downloaded before the manifest existed
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

from src import config

PAUSE_BETWEEN_REQUESTS = 30   # seconds; big chunks "cost" many API calls each
MAX_TRIES = 6
OLD_SITES_META = config.RAW_DIR / "sites_meta.json"   # older file, only read by --rebuild-manifest


# ---------------------------------------------------------------------------
# 1. Which chunks the project needs (one list, used by download AND checks)
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


def make_spec(site, block, variables, model, start, end, path: Path) -> dict:
    """Everything that defines one chunk. 'open' = the chunk runs up to today,
    so its end date moves every day (only the newest daily EDA chunk)."""
    lat, lon = config.SITES[site]
    return {
        "file": path.relative_to(config.RAW_DIR).as_posix(),
        "site": site, "block": block, "model": model, "variables": list(variables),
        "lat": lat, "lon": lon, "timezone": config.TIMEZONE,
        "start": start.isoformat(), "end": end.isoformat(),
        "open": end >= dt.date.today() - dt.timedelta(days=1),
    }


def planned_chunks() -> list[dict]:
    """The full list of chunks the project needs, in download order."""
    specs = []
    hourly_end = dt.date(config.HOURLY_END_YEAR, 12, 31)
    for site in config.SITES:
        for start, end in decade_chunks(config.HOURLY_START_YEAR, hourly_end):
            path = config.HOURLY_DIR / f"{site}_{start.year}s.parquet"
            specs.append(make_spec(site, "hourly", config.HOURLY_VARS,
                                   config.ARCHIVE_MODEL_HOURLY, start, end, path))
    for start, end in decade_chunks(config.EDA_START_YEAR, dt.date.today()):
        path = config.DAILY_EDA_DIR / f"{config.MAIN_SITE}_daily_{start.year}s.parquet"
        specs.append(make_spec(config.MAIN_SITE, "daily", config.DAILY_EDA_VARS,
                               config.ARCHIVE_MODEL_DAILY, start, end, path))
    return specs


def spec_path(spec: dict) -> Path:
    return config.RAW_DIR / spec["file"]


# ---------------------------------------------------------------------------
# 2. Validation: is this chunk complete and physically sensible?
# ---------------------------------------------------------------------------
def validate_chunk(df: pd.DataFrame, spec: dict) -> list[str]:
    """Return a list of problems. An empty list means the chunk is OK."""
    variables = spec["variables"]
    missing_cols = [v for v in variables if v not in df.columns]
    if missing_cols:
        return [f"columns missing: {missing_cols}"]

    problems = []
    hourly = spec["block"] == "hourly"
    times = df["time"]

    # a) Time axis: every hour (or day) exactly once, in order, no gaps.
    first = pd.Timestamp(spec["start"])
    if spec["open"]:
        last = times.max()                       # open chunk: ends at the newest available day
    else:
        last = pd.Timestamp(spec["end"]) + (pd.Timedelta(hours=23) if hourly else pd.Timedelta(0))
    expected = pd.date_range(first, last, freq="h" if hourly else "D")
    if len(times) != len(expected) or not (times.to_numpy() == expected.to_numpy()).all():
        problems.append(f"time axis wrong: {len(times):,} rows, expected {len(expected):,} "
                        f"({first} to {last}), duplicates={int(times.duplicated().sum())}")

    # b) No missing values. Only exception: ERA5's first local day (1940-01-01)
    #    is incomplete, because ERA5 starts at 00:00 UTC = 08:00 in UB.
    check = df if hourly else df[df["time"] != pd.Timestamp(config.ERA5_FIRST_DAY)]
    nan = check[variables].isna().sum()
    nan = nan[nan > 0]
    if not nan.empty:
        problems.append(f"missing values: {nan.to_dict()}")

    # c) Values inside physical limits (catches wrong units or garbage).
    for v in variables:
        lo, hi = config.VALID_RANGES[v]
        s = check[v].dropna()
        outside = int(((s < lo) | (s > hi)).sum())
        if outside:
            problems.append(f"{v}: {outside} values outside [{lo}, {hi}] (min {s.min()}, max {s.max()})")

    # d) Daily consistency: min <= mean <= max (small tolerance for rounding).
    if not hourly:
        c = check.dropna(subset=["temperature_2m_min", "temperature_2m_mean", "temperature_2m_max"])
        bad = ((c["temperature_2m_min"] > c["temperature_2m_mean"] + 0.1) |
               (c["temperature_2m_mean"] > c["temperature_2m_max"] + 0.1)).sum()
        if bad:
            problems.append(f"{int(bad)} days where min <= mean <= max is broken")
    return problems


# ---------------------------------------------------------------------------
# 3. Manifest + fingerprints
# ---------------------------------------------------------------------------
SETTINGS_KEYS = ["site", "block", "model", "variables", "lat", "lon", "timezone", "start", "end"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict):
    write_safely(config.MANIFEST_PATH,
                 lambda tmp: tmp.write_text(json.dumps(dict(sorted(manifest.items())), indent=2)))


def settings_differences(spec: dict, entry: dict) -> list[str]:
    """Which settings differ between config.py (spec) and a manifest entry.
    The end date of an 'open' chunk is allowed to differ (it moves daily)."""
    diffs = []
    for k in SETTINGS_KEYS:
        if k == "end" and spec["open"]:
            continue
        if entry.get(k) != spec[k]:
            diffs.append(f"{k}: {entry.get(k)} -> {spec[k]}")
    return diffs


def cache_status(spec: dict, manifest: dict) -> tuple[bool, str]:
    """(True, '') if the cached file can be reused, else (False, reason)."""
    path = spec_path(spec)
    if not path.exists():
        return False, ""
    entry = manifest.get(spec["file"])
    if entry is None:
        return False, "file is not in the manifest (unknown settings)"
    diffs = settings_differences(spec, entry)
    if diffs:
        return False, "settings changed: " + "; ".join(diffs)
    if file_sha256(path) != entry["sha256"]:
        return False, "file content changed since download (fingerprint mismatch)"
    return True, ""


def manifest_entry(spec: dict, df: pd.DataFrame, path: Path, payload_meta: dict,
                   downloaded_at: str) -> dict:
    entry = {k: spec[k] for k in SETTINGS_KEYS}
    entry.update({
        "source": config.ARCHIVE_URL,
        "rows": int(len(df)),
        "first_time": str(df["time"].min()),
        "last_time": str(df["time"].max()),
        "grid_lat": payload_meta.get("latitude"),
        "grid_lon": payload_meta.get("longitude"),
        "elevation_m": payload_meta.get("elevation"),
        "utc_offset_seconds": payload_meta.get("utc_offset_seconds"),
        "downloaded_at": downloaded_at,
        "sha256": file_sha256(path),
    })
    return entry


def write_safely(path: Path, write_fn):
    """Write to 'name.tmp' first, then rename: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# 4. Talking to the API
# ---------------------------------------------------------------------------
def seconds_until_next_hour() -> int:
    now = dt.datetime.now()
    next_hour = (now + dt.timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return int((next_hour - now).total_seconds()) + 60   # plus a 1 minute safety margin


def get_json(params: dict, url: str = config.ARCHIVE_URL) -> dict:
    """Call the archive API with retries and rate limit handling."""
    headers = {"User-Agent": config.USER_AGENT}
    for attempt in range(1, MAX_TRIES + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=180)
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


def to_frame(payload: dict, spec: dict) -> pd.DataFrame:
    """Turn the API's JSON block ('hourly' or 'daily') into a DataFrame."""
    df = pd.DataFrame(payload[spec["block"]])
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "site", spec["site"])
    if spec["open"]:
        # The archive is ~5 days behind real time: the newest days come back
        # empty. Cut those trailing empty rows off before validating/saving.
        has_data = df[spec["variables"]].notna().any(axis=1)
        df = df.loc[: has_data[has_data].index.max()].copy()
    return df


# ---------------------------------------------------------------------------
# 5. Download one chunk / all chunks
# ---------------------------------------------------------------------------
def download_chunk(spec: dict, manifest: dict) -> bool:
    """Download, validate and save one chunk. Returns True if it downloaded."""
    name = Path(spec["file"]).name
    reusable, reason = cache_status(spec, manifest)
    if reusable:
        print(f"  [cached]   {name}")
        return False
    if reason:
        print(f"  [stale]    {name}: {reason}")

    params = {
        "latitude": spec["lat"], "longitude": spec["lon"],
        "start_date": spec["start"], "end_date": spec["end"],
        spec["block"]: ",".join(spec["variables"]),
        "timezone": spec["timezone"],
        "models": spec["model"],
    }
    print(f"  [download] {name}  ({spec['start']} to {spec['end']}, model={spec['model']})")
    payload = get_json(params)
    df = to_frame(payload, spec)

    problems = validate_chunk(df, spec)
    if problems:
        raise SystemExit(f"\n{name} FAILED validation and was NOT saved:\n  - "
                         + "\n  - ".join(problems))

    path = spec_path(spec)
    write_safely(path, lambda tmp: df.to_parquet(tmp, index=False))
    downloaded_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    manifest[spec["file"]] = manifest_entry(spec, df, path, payload, downloaded_at)
    save_manifest(manifest)
    print(f"             validated + saved {len(df):,} rows")
    return True


def download(only: str | None = None):
    manifest = load_manifest()
    for spec in planned_chunks():
        if only and spec["block"] != only:
            continue
        if download_chunk(spec, manifest):
            time.sleep(PAUSE_BETWEEN_REQUESTS)


def rebuild_manifest():
    """One-time helper: register files that were downloaded before the
    manifest existed. Each file is validated first; a file that fails is
    left out, so the next download replaces it."""
    old_meta = json.loads(OLD_SITES_META.read_text()) if OLD_SITES_META.exists() else {}
    manifest = load_manifest()
    added, rejected = 0, 0
    for spec in planned_chunks():
        path = spec_path(spec)
        if not path.exists() or spec["file"] in manifest:
            continue
        df = pd.read_parquet(path)
        problems = validate_chunk(df, spec)
        if problems:
            print(f"  [rejected] {spec['file']}: {'; '.join(problems)}")
            rejected += 1
            continue
        key = spec["site"] if spec["block"] == "hourly" else f"{spec['site']}_daily"
        m = old_meta.get(key, {})
        payload_meta = {"latitude": m.get("grid_lat"), "longitude": m.get("grid_lon"),
                        "elevation": m.get("elevation_m"),
                        "utc_offset_seconds": m.get("utc_offset_seconds")}
        mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
        manifest[spec["file"]] = manifest_entry(spec, df, path, payload_meta,
                                                mtime.isoformat(timespec="seconds"))
        print(f"  [added]    {spec['file']}")
        added += 1
    save_manifest(manifest)
    print(f"Manifest rebuilt: {added} added, {rejected} rejected, {len(manifest)} total.")


# ---------------------------------------------------------------------------
# 6. Loaders used by later phases (they read exactly the planned files)
# ---------------------------------------------------------------------------
def _load(specs: list[dict]) -> pd.DataFrame:
    missing = [s["file"] for s in specs if not spec_path(s).exists()]
    if missing:
        raise FileNotFoundError(f"Missing data files {missing}. Run: python -m src.download")
    df = pd.concat([pd.read_parquet(spec_path(s)) for s in specs], ignore_index=True)
    return df.sort_values("time").reset_index(drop=True)


def load_hourly(site: str) -> pd.DataFrame:
    """All hourly chunks for one site (1980 to 2025), joined and sorted by time."""
    return _load([s for s in planned_chunks() if s["block"] == "hourly" and s["site"] == site])


def load_daily_eda() -> pd.DataFrame:
    """UB daily data for EDA (1940 to the newest archived day).
    Drops ERA5's incomplete first day (1940-01-01)."""
    df = _load([s for s in planned_chunks() if s["block"] == "daily"])
    return df.dropna(subset=["temperature_2m_mean"]).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and cache Open-Meteo data.")
    parser.add_argument("--only", choices=["daily", "hourly"], help="download just one part")
    parser.add_argument("--rebuild-manifest", action="store_true",
                        help="register already-downloaded files in the manifest")
    args = parser.parse_args()
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.rebuild_manifest:
        rebuild_manifest()
    else:
        download(args.only)
        print("Done.")
