"""
check_data.py: prove the raw data is complete, clean and exactly what
config.py asks for.

For EVERY planned file it checks:
  1. the file exists
  2. it is registered in data/raw/manifest.json
  3. its settings in the manifest match config.py (model, variables, place, dates)
  4. its SHA-256 fingerprint matches (the file was not changed or damaged)
  5. it passes validation (complete time axis, no missing values,
     values inside physical limits)
Then, per site, it checks the joined 1980 to 2025 series has every hour once.

Run from the repo folder:
  python -m src.check_data
"""

import pandas as pd

from src import config
from src.download import (cache_status, load_daily_eda, load_hourly, load_manifest,
                          planned_chunks, spec_path, validate_chunk)


def check_files() -> list[str]:
    """Checks 1 to 5 for every planned file. Returns a list of problems."""
    manifest = load_manifest()
    problems = []
    specs = planned_chunks()
    for spec in specs:
        path = spec_path(spec)
        if not path.exists():
            problems.append(f"{spec['file']}: not downloaded yet")
            continue
        ok, reason = cache_status(spec, manifest)
        if not ok:
            problems.append(f"{spec['file']}: {reason}")
            continue
        issues = validate_chunk(pd.read_parquet(path), spec)
        problems += [f"{spec['file']}: {i}" for i in issues]
    n_ok = len(specs) - len({p.split(':')[0] for p in problems})
    print(f"Files: {n_ok} of {len(specs)} OK (exist, in manifest, settings match, fingerprint match, valid)")
    return problems


def check_hourly_series(site: str) -> list[str]:
    """The joined 1980 to 2025 hourly series has every hour exactly once."""
    df = load_hourly(site)
    expected = pd.date_range(f"{config.HOURLY_START_YEAR}-01-01 00:00",
                             f"{config.HOURLY_END_YEAR}-12-31 23:00", freq="h")
    ok = len(df) == len(expected) and (df["time"].to_numpy() == expected.to_numpy()).all()
    print(f"  {site:12s} {len(df):,} hours  {'OK' if ok else 'PROBLEM'}")
    return [] if ok else [f"{site}: joined hourly series is not complete/ordered"]


def check_daily_series() -> list[str]:
    df = load_daily_eda()
    gaps = pd.date_range(df["time"].min(), df["time"].max(), freq="D").difference(df["time"])
    print(f"  UB daily EDA {df['time'].min().date()} to {df['time'].max().date()}  "
          f"{len(df):,} days  {'OK' if len(gaps) == 0 else 'PROBLEM'}")
    return [] if len(gaps) == 0 else [f"UB daily EDA: {len(gaps)} missing days"]


if __name__ == "__main__":
    problems = check_files()
    print("Joined series:")
    for site in config.SITES:
        try:
            problems += check_hourly_series(site)
        except FileNotFoundError as e:
            problems.append(str(e))
    try:
        problems += check_daily_series()
    except FileNotFoundError as e:
        problems.append(f"UB daily EDA: {e}")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        print("\nSOME CHECKS FAILED")
    else:
        print("\nALL CHECKS PASSED")
