"""
check_sources.py: do the training data (ERA5 archive) and the live data
(Open-Meteo forecast API) agree on the days both of them cover?

The model was trained on the archive, but the website feeds it data from
the forecast API (the archive is ~5 days behind). If the two disagree, the
live forecasts get worse even with a perfect model. This script measures it
on the overlapping days (~55 recent days):

  1. Daily weather per site: average difference (bias), typical difference
     (MAE) and correlation, for each daily value.
  2. The model's 40 input features: which ones differ the most.
  3. THE KEY NUMBER: the same model fed with archive data vs live data.
     How much does the forecast itself change?

Writes static/data/source_check.json (a short summary is shown on the website
and in /api/meta). Run from the repo folder:
  python -m src.check_sources
"""

import datetime as dt
import json

import numpy as np
import pandas as pd

from src import config
from src.download import get_json
from src.features import build_features, hourly_to_daily, load_normals
from src.live import complete_past_days, fetch_site, load_assets, today_ub

COMPARE = ["t_mean", "t_min", "t_max", "dew_mean", "rh_mean", "p_mean", "wind_speed",
           "cloud_mean", "precip_sum", "snow_depth_end", "radiation_sum"]


def fetch_archive(site: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Same request as the training download (same model and variables)."""
    lat, lon = config.SITES[site]
    payload = get_json({"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                        "end_date": end.isoformat(), "hourly": ",".join(config.HOURLY_VARS),
                        "timezone": config.TIMEZONE, "models": config.ARCHIVE_MODEL_HOURLY})
    df = pd.DataFrame(payload["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "site", site)
    return df


def compare_series(live: pd.Series, archive: pd.Series) -> dict:
    both = pd.concat([live, archive], axis=1, keys=["live", "archive"]).dropna()
    d = both["live"] - both["archive"]
    corr = both["live"].corr(both["archive"]) if both["archive"].std() > 0 else float("nan")
    return {"bias": round(float(d.mean()), 3), "mae": round(float(d.abs().mean()), 3),
            "corr": None if np.isnan(corr) else round(float(corr), 3), "days": int(len(both))}


def run():
    today = today_ub()
    print("Fetching live data (forecast API) and archive data (ERA5) for 5 sites...")
    live = {s: complete_past_days(fetch_site(s), today) for s in config.SITES}
    start = min(h["time"].min() for h in live.values()).date()
    end = today - dt.timedelta(days=1)
    archive = {s: fetch_archive(s, start, end) for s in config.SITES}

    # 1. Daily weather per site.
    per_site = {}
    for s in config.SITES:
        dl, da = hourly_to_daily(live[s]), hourly_to_daily(archive[s].dropna(subset=config.HOURLY_VARS))
        per_site[s] = {v: compare_series(dl[v], da[v]) for v in COMPARE}
    ub_t = per_site[config.MAIN_SITE]["t_mean"]
    print(f"UB daily mean temperature, {ub_t['days']} days: live minus archive = {ub_t['bias']:+.2f} degC "
          f"on average, typical difference {ub_t['mae']:.2f} degC, correlation {ub_t['corr']}")

    # 2. + 3. Model inputs and the forecast itself.
    meta, models, normals = load_assets()
    cols = meta["features"]
    f_live, _ = build_features(live, normals)
    f_arch, _ = build_features({s: a.dropna(subset=config.HOURLY_VARS) for s, a in archive.items()}, normals)
    common = f_live[cols].dropna().index.intersection(f_arch[cols].dropna().index)
    if len(common) < 5:
        raise SystemExit(f"Only {len(common)} overlapping days with complete features; try again later.")
    fl, fa = f_live.loc[common, cols], f_arch.loc[common, cols]

    train = pd.read_parquet(config.DATASET_PATH, columns=cols + ["split"])
    train_std = train[train["split"] == "train"][cols].std()
    feat_diff = ((fl - fa).abs().mean() / train_std).sort_values(ascending=False)

    forecast_change = {}
    for h, model in models.items():
        diff = model.predict(fl) - model.predict(fa)
        forecast_change[f"d{h}"] = {"mean": round(float(diff.mean()), 3),
                                    "mae": round(float(np.abs(diff).mean()), 3),
                                    "max_abs": round(float(np.abs(diff).max()), 3),
                                    "model_test_rmse": meta["horizons"][f"d{h}"]["test_rmse"]}
        fc = forecast_change[f"d{h}"]
        print(f"day+{h}: forecast with live data vs archive data differs by {fc['mae']:.2f} degC on average "
              f"(max {fc['max_abs']:.2f}); the model's own typical error is {fc['model_test_rmse']:.2f}")

    summary = {
        "period": [str(common.min().date()), str(common.max().date())],
        "days_compared": int(len(common)),
        "ub_temperature_bias": ub_t["bias"], "ub_temperature_mae": ub_t["mae"],
        "forecast_change_mae": {h: v["mae"] for h, v in forecast_change.items()},
        "note": (f"On {len(common)} recent days covered by both sources, the live data (forecast API) "
                 f"and the training data (ERA5 archive) differ by {ub_t['mae']:.1f} degC on average for "
                 f"Ulaanbaatar's daily temperature. Feeding the model live data instead of archive data "
                 f"changes the day+1 forecast by {forecast_change['d1']['mae']:.2f} degC on average, "
                 f"compared with the model's typical error of {forecast_change['d1']['model_test_rmse']:.1f} degC."),
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    out = {"summary": summary, "per_site": per_site, "forecast_change": forecast_change,
           "largest_feature_differences": {f: round(float(v), 3) for f, v in feat_diff.head(8).items()}}
    config.SOURCE_CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    config.SOURCE_CHECK_PATH.write_text(json.dumps(out, indent=1))
    print("\nLargest feature differences (in training standard deviations):")
    for f, v in feat_diff.head(5).items():
        print(f"  {f:26s} {v:.2f}")
    print(f"\n{summary['note']}\nSaved {config.SOURCE_CHECK_PATH}")


if __name__ == "__main__":
    run()
