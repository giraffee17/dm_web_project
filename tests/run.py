"""
Automatic checks for the feature pipeline. Run from the repo folder:
  python -m tests.run

Each test prints PASS or FAIL. The command ends with "ALL TESTS PASSED"
(exit code 0) or lists what failed (exit code 1).

What is tested
  1. No leakage: scramble ALL weather after day t -> features of day t
     and earlier must not change by even 0.000000001.
  2. Train/serve match: features built from only the last 60 days (like
     the live website) equal the features built from 46 years of history.
  3. Normals use training years only, and handle Feb 29 and New Year.
  4. Split: train/val/test never overlap, and no target crosses a border.
  5. Targets are exactly the UB anomaly 1, 2 and 3 days later.
  6. hourly_to_daily on tiny hand-made data with known answers.
  7. Reproducible: building twice gives the same fingerprint.
"""

import sys

import numpy as np
import pandas as pd

from src import config
from src import features as fz
from src.download import load_hourly

RESULTS = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def max_abs_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float(np.nanmax(np.abs(a.to_numpy(float) - b.to_numpy(float)))) if len(a) else 0.0


def same_values(a: pd.DataFrame, b: pd.DataFrame, atol=1e-9) -> bool:
    return a.shape == b.shape and np.allclose(a.to_numpy(float), b.to_numpy(float),
                                              atol=atol, rtol=0, equal_nan=True)


# ---------------------------------------------------------------------------
def test_no_leakage(hourly, normals, full_feats, full_daily):
    print("1. No leakage (scramble the future, the past must not change)")
    rng = np.random.default_rng(config.SEED)
    for cut in ["1995-06-15", "2019-12-28", "2022-12-31", "2023-01-01"]:
        end_of_day = pd.Timestamp(cut) + pd.Timedelta(hours=23)
        scrambled = {}
        for site, h in hourly.items():
            h = h.copy()
            h[config.HOURLY_VARS] = h[config.HOURLY_VARS].astype("float64")
            future = h["time"] > end_of_day
            for v in config.HOURLY_VARS:   # replace every future value with random numbers
                lo, hi = config.VALID_RANGES[v]
                h.loc[future, v] = rng.uniform(lo, hi, future.sum())
            scrambled[site] = h
        feats, daily = fz.build_features(scrambled, normals)
        past = full_feats.index <= pd.Timestamp(cut)
        diff = max_abs_diff(feats.loc[past], full_feats.loc[past])
        check(f"cut {cut}: features up to day t unchanged", same_values(feats.loc[past], full_feats.loc[past]),
              f"max difference {diff:.1e}")
        # Sanity: the scramble really changed the future (otherwise the test proves nothing).
        t1 = pd.Timestamp(cut) + pd.Timedelta(days=1)
        changed = abs(daily[config.MAIN_SITE].loc[t1, "t_anom"] - full_daily[config.MAIN_SITE].loc[t1, "t_anom"]) > 0.01
        check(f"cut {cut}: the scramble really changed day t+1", changed)


def test_train_serve(hourly, normals, full_feats):
    print("2. Train/serve match (60 days of data, like the live site)")
    for end in ["2025-12-31", "2021-07-15"]:
        last_hour = pd.Timestamp(end) + pd.Timedelta(hours=23)
        first_hour = last_hour - pd.Timedelta(days=60) + pd.Timedelta(hours=1)
        window = {s: h[(h["time"] >= first_hour) & (h["time"] <= last_hour)] for s, h in hourly.items()}
        live_feats, _ = fz.build_features(window, normals)
        usable = live_feats.dropna()       # days with a full 30 day history inside the window
        ref = full_feats.loc[usable.index, usable.columns]
        # Tolerance 1e-6: rolling statistics run over 46 years pick up computer
        # rounding noise around the 9th decimal (pressure is ~1020 hPa). A real
        # train/serve bug shows up as 0.01 or more; sensors measure to 0.1.
        check(f"window ending {end}: {len(usable)} usable days match full history",
              len(usable) >= 60 - config.MIN_HISTORY_DAYS and same_values(usable, ref, atol=1e-6),
              f"max difference {max_abs_diff(usable, ref):.1e}")
        check(f"window ending {end}: newest usable day is the last complete day",
              usable.index.max() == pd.Timestamp(end))

    # The live site runs while "today" is still in progress: a partial day must be ignored.
    partial = {}
    for s, h in hourly.items():
        h = h[h["time"] >= pd.Timestamp("2025-11-01")]
        extra = h[h["time"] >= pd.Timestamp("2025-12-31")].iloc[:10].copy()
        extra["time"] += pd.Timedelta(days=1)       # 10 hours of "2026-01-01"
        partial[s] = pd.concat([h, extra])
    f, _ = fz.build_features(partial, normals)
    check("a partial (still running) day is dropped", f.index.max() == pd.Timestamp("2025-12-31"))


def test_normals(hourly, normals):
    print("3. Normals")
    rng = np.random.default_rng(config.SEED)
    scrambled = {}
    for site, h in hourly.items():
        h = h.copy()
        after = h["time"] >= pd.Timestamp(f"{config.VAL_YEARS[0]}-01-01")
        h.loc[after, "temperature_2m"] = rng.uniform(-50, 40, after.sum())
        scrambled[site] = h
    refit = fz.fit_normals({s: fz.hourly_to_daily(h) for s, h in scrambled.items()})
    same = all(np.allclose(refit[s], normals[s], atol=1e-12, rtol=0) for s in normals)
    check("normals ignore 2020+ data (training years only)", same)

    c = normals[config.MAIN_SITE]
    d = pd.to_datetime(["2024-02-28", "2024-02-29", "2024-03-01", "2023-12-31", "2024-01-01"])
    n = fz.normal_for(d, c)
    check("Feb 29 normal lies between Feb 28 and Mar 1", min(n[0], n[2]) <= n[1] <= max(n[0], n[2]),
          f"{n[0]:.2f} / {n[1]:.2f} / {n[2]:.2f} degC")
    check("no jump at New Year (Dec 31 vs Jan 1)", abs(n[3] - n[4]) < 0.2, f"{n[3]:.2f} vs {n[4]:.2f} degC")
    coldest = pd.date_range("2023-01-01", "2023-12-31")[np.argmin(fz.normal_for(pd.date_range("2023-01-01", "2023-12-31"), c))]
    check("UB's coldest normal day is in January", coldest.month == 1, f"{coldest.date()}")


def test_split(ds):
    print("4. Split")
    split = ds["split"]
    check("only train/val/test rows in the dataset", set(split) == {"train", "val", "test"})
    for name, (y0, y1) in [("train", config.TRAIN_YEARS), ("val", config.VAL_YEARS), ("test", config.TEST_YEARS)]:
        days = ds.index[split == name]
        last_target = days.max() + pd.Timedelta(days=max(config.HORIZONS))
        check(f"{name}: days within {y0} to {y1}", days.min().year >= y0 and days.max().year <= y1)
        check(f"{name}: last day+3 target stays inside {y1}", last_target <= pd.Timestamp(f"{y1}-12-31"),
              f"last target {last_target.date()}")
    check("train, val and test never share a day", not ds.index.duplicated().any())


def test_targets(ds, full_daily):
    print("5. Targets")
    anom = full_daily[config.MAIN_SITE]["t_anom"]
    for h in config.HORIZONS:
        expected = anom.reindex(ds.index + pd.Timedelta(days=h)).to_numpy()
        check(f"target_d{h} = UB anomaly {h} day(s) later", np.allclose(ds[f"target_d{h}"].to_numpy(), expected))


def test_hourly_to_daily():
    print("6. hourly_to_daily on hand-made data")
    times = pd.date_range("2024-01-01", periods=24 * 3, freq="h")   # 3 full days
    h = pd.DataFrame({"time": times, "temperature_2m": 0.0, "relative_humidity_2m": 50.0,
                      "dew_point_2m": -5.0, "pressure_msl": 1020.0, "wind_speed_10m": 10.0,
                      "wind_direction_10m": 0.0, "cloud_cover": 50.0, "precipitation": 0.5,
                      "snow_depth": 0.1, "shortwave_radiation": 100.0})
    h.loc[h["time"].dt.date == pd.Timestamp("2024-01-02").date(), "wind_direction_10m"] = 90.0
    h.loc[h["time"] == pd.Timestamp("2024-01-01 15:00"), "temperature_2m"] = 24.0
    d = fz.hourly_to_daily(h)
    check("mean / min / max / range", np.isclose(d.loc["2024-01-01", "t_mean"], 1.0)
          and d.loc["2024-01-01", "t_max"] == 24 and d.loc["2024-01-01", "t_range"] == 24)
    check("north wind -> wind_from_n = speed", np.isclose(d.loc["2024-01-01", "wind_from_n"], 10)
          and np.isclose(d.loc["2024-01-01", "wind_from_e"], 0, atol=1e-9))
    check("east wind -> wind_from_e = speed", np.isclose(d.loc["2024-01-02", "wind_from_e"], 10)
          and np.isclose(d.loc["2024-01-02", "wind_from_n"], 0, atol=1e-9))
    check("sums and units (12 mm rain, 10 cm snow, 8.64 MJ/m2)",
          np.isclose(d.loc["2024-01-01", "precip_sum"], 12) and np.isclose(d.loc["2024-01-01", "snow_depth_end"], 10)
          and np.isclose(d.loc["2024-01-01", "radiation_sum"], 8.64))
    gap = h[h["time"].dt.date != pd.Timestamp("2024-01-02").date()]   # remove the middle day
    d2 = fz.hourly_to_daily(gap)
    check("a missing day becomes an empty row (so lag 1 = yesterday)",
          len(d2) == 3 and d2.loc["2024-01-02"].isna().all())
    check("an incomplete day is dropped", len(fz.hourly_to_daily(h.iloc[:-1])) == 2)


def test_reproducible(hourly, ds):
    print("7. Reproducible")
    ds2, _, _ = fz.build_dataset(hourly)
    check("building twice gives the same fingerprint", fz.fingerprint(ds) == fz.fingerprint(ds2),
          fz.fingerprint(ds))


if __name__ == "__main__":
    print("Loading data...")
    hourly = {site: load_hourly(site) for site in config.SITES}
    ds, _, normals = fz.build_dataset(hourly)
    full_feats, full_daily = fz.build_features(hourly, normals)

    test_no_leakage(hourly, normals, full_feats, full_daily)
    test_train_serve(hourly, normals, full_feats)
    test_normals(hourly, normals)
    test_split(ds)
    test_targets(ds, full_daily)
    test_hourly_to_daily()
    test_reproducible(hourly, ds)

    passed = sum(RESULTS)
    print(f"\n{passed} / {len(RESULTS)} checks passed")
    if passed == len(RESULTS):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
