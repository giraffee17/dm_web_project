"""
check_data.py: confirm the downloaded raw data is complete and clean.

For every site it checks:
  * the date range is exactly 1980-01-01 00:00 to 2025-12-31 23:00
  * every hour is there once (no gaps, no duplicates, 24 rows per day)
  * no missing values in any of the 10 variables
And for the UB daily EDA data: the date range and missing values.

Run from the repo folder:
  python -m src.check_data
"""

import pandas as pd

from src import config
from src.download import load_daily_eda, load_hourly


def check_hourly(site: str) -> bool:
    df = load_hourly(site)
    start = pd.Timestamp(f"{config.HOURLY_START_YEAR}-01-01 00:00")
    end = pd.Timestamp(f"{config.HOURLY_END_YEAR}-12-31 23:00")
    expected = pd.date_range(start, end, freq="h")

    missing_hours = expected.difference(df["time"])
    duplicates = df["time"].duplicated().sum()
    hours_per_day = df.groupby(df["time"].dt.date).size()
    bad_days = (hours_per_day != 24).sum()
    nan_counts = df[config.HOURLY_VARS].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]

    ok = len(missing_hours) == 0 and duplicates == 0 and bad_days == 0 and nan_counts.empty
    print(f"{site:12s} rows={len(df):,}  expected={len(expected):,}  "
          f"missing_hours={len(missing_hours)}  duplicates={duplicates}  "
          f"days_without_24h={bad_days}  {'OK' if ok else 'PROBLEM'}")
    if not nan_counts.empty:
        print("   missing values:", nan_counts.to_dict())
    return ok


def check_daily_eda() -> bool:
    df = load_daily_eda()
    nan_counts = df[config.DAILY_EDA_VARS].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    gaps = pd.date_range(df["time"].min(), df["time"].max(), freq="D").difference(df["time"])
    ok = nan_counts.empty and len(gaps) == 0
    print(f"UB daily EDA: {df['time'].min().date()} to {df['time'].max().date()}  "
          f"rows={len(df):,}  gaps={len(gaps)}  {'OK' if ok else 'PROBLEM'}")
    if not nan_counts.empty:
        print("   missing values:", nan_counts.to_dict())
    return ok


if __name__ == "__main__":
    results = [check_hourly(site) for site in config.SITES]
    results.append(check_daily_eda())
    print("\nALL CHECKS PASSED" if all(results) else "\nSOME CHECKS FAILED (see above)")
