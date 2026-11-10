"""
export.py: turn the training and evaluation reports into small JSON files
for the website, plus models/meta.json for the API (/api/meta).

Website chart data (static/data/):
  feature_selection.json   number of features vs validation RMSE (+ chosen N, top features)
  bakeoff.json             LightGBM vs XGBoost vs CatBoost, validation RMSE per horizon
  learning_curves.json     train vs validation RMSE per boosting round + early stopping point
  gap_table.json           RMSE and MAE on train / validation / test per horizon
  shap.json                feature importance (mean |SHAP|) per horizon and per group

Run from the repo folder (after src.train and src.evaluate --unlock-test):
  python -m src.export
"""

import datetime as dt
import json
from importlib.metadata import version

import pandas as pd

from src import config
from src.features import fingerprint
from src.train import load_report, model_path, selected_features

TOP_FEATURES = 15       # features shown in the importance charts


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))
    print(f"  wrote {path.relative_to(config.ROOT)}")


def export():
    fs = load_report("feature_selection")
    bake = load_report("bakeoff")
    curves = load_report("learning_curves")
    gap = load_report("gap_table")
    shap = load_report("shap")
    final = load_report("final_models")
    tuning = load_report("tuning")
    if not gap["test_included"]:
        raise SystemExit("Run the one-time test evaluation first: python -m src.evaluate --unlock-test")
    out = config.STATIC_DATA_DIR

    ranked = list(fs["permutation_importance"].items())[:TOP_FEATURES]
    write_json(out / "feature_selection.json", {
        "n_start": fs["n_start"], "n_after_correlation_filter": fs["n_after_correlation_filter"],
        "chosen_n": fs["chosen_n"], "tolerance_degC": fs["tolerance"], "curve": fs["curve"],
        "top_features": [{"feature": f, "importance_degC": round(v["mean"], 3)} for f, v in ranked]})

    write_json(out / "bakeoff.json", {
        "winner": bake["winner"], "settings": bake["settings"],
        "results": {lib: {h: r[h]["rmse"] for h in ("d1", "d2", "d3")} | {"mean": r["mean_rmse"]}
                    for lib, r in bake["results"].items()},
        "fairness_check": bake.get("fairness_check")})

    write_json(out / "learning_curves.json", curves)

    write_json(out / "gap_table.json", {"note": gap["note"], "horizons": gap["horizons"]})

    write_json(out / "shap.json", {
        "period": shap["period"], "groups": shap["groups"],
        "features": {h: dict(list(v.items())[:TOP_FEATURES]) for h, v in shap["features"].items()}})

    ds = pd.read_parquet(config.DATASET_PATH)
    meta = {
        "project": "Forecasting the Coldest Capital on Earth",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "target": "Ulaanbaatar daily mean temperature anomaly, day+1 to day+3 (forecast = anomaly + normal)",
        "library": final["library"],
        "library_version": version(final["library"]),
        "horizons": {h: {"model_file": model_path(final["library"], int(h[1:]), "final").name,
                         "rounds": final["horizons"][h]["rounds_final"],
                         "params": tuning["horizons"][h]["params"],
                         "val_rmse": gap["horizons"][h]["val"]["rmse"],
                         "test_rmse": gap["horizons"][h]["test_final_model"]["rmse"],
                         "test_mae": gap["horizons"][h]["test_final_model"]["mae"]}
                     for h in ("d1", "d2", "d3")},
        "features": selected_features(),
        "trained_on": final["final_period"],
        "data_source": "Open-Meteo archive API, ERA5 (era5_seamless), 5 sites, hourly",
        "sites": {s: {"lat": lat, "lon": lon} for s, (lat, lon) in config.SITES.items()},
        "dataset_fingerprint": fingerprint(ds),
    }
    write_json(config.MODELS_DIR / "meta.json", meta)


if __name__ == "__main__":
    export()
