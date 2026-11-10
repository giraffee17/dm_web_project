"""
evaluate.py: overfitting checks, the ONE-TIME test evaluation, baselines and SHAP.

This is the only file in the project that reads the test set (2023 to 2025).
It is protected by a lock:
  * Without --unlock-test, only train and validation results are computed.
  * The first --unlock-test run records a fingerprint of the final models in
    models/reports/test_unlock_log.json.
  * Running it again with the SAME models simply reproduces the same numbers.
  * Running it with DIFFERENT models is refused: evaluating model after model
    on the test set would quietly turn it into a second validation set.

Outputs (models/reports/):
  gap_table.json   RMSE and MAE on train / validation / test per horizon
  baselines.json   persistence and climatology on validation and test (for the README)
  shap.json        mean |SHAP| per feature and per feature group (final models)

Run from the repo folder:
  python -m src.evaluate                 # train + validation only
  python -m src.evaluate --unlock-test   # the one-time final test evaluation
"""

import argparse
import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd

from src import config
from src.train import (TARGETS, load_model, load_report, mae, model_path, rmse, save_report,
                       selected_features)

LOCK_PATH = config.REPORTS_DIR / "test_unlock_log.json"


def file_sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def models_fingerprint(lib) -> dict:
    """Fingerprints of everything that decides the test result."""
    fp = {}
    for h in TARGETS:
        fp[f"final_d{h}"] = file_sha256(model_path(lib, h, "final"))
        fp[f"train_only_d{h}"] = file_sha256(model_path(lib, h, "train_only"))
    fp["selected_features"] = file_sha256(config.MODELS_DIR / "selected_features.json")
    return fp


def check_test_lock(lib, force: bool):
    fp = models_fingerprint(lib)
    log = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else []
    if log and log[0]["fingerprints"] != fp and not force:
        raise SystemExit(
            f"REFUSED: the test set was already used on {log[0]['time']} to evaluate DIFFERENT models.\n"
            "Testing new models on it would turn it into a second validation set.\n"
            "If this is truly necessary, re-run with --force and explain why in the README.")
    log.append({"time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "fingerprints": fp, "same_as_first": (not log) or log[0]["fingerprints"] == fp,
                "forced": force})
    LOCK_PATH.write_text(json.dumps(log, indent=2))
    if len(log) > 1:
        print(f"Note: test evaluation run #{len(log)}; models identical to the first run: "
              f"{log[-1]['same_as_first']}")


def scores(y, p) -> dict:
    return {"rmse": round(rmse(y, p), 3), "mae": round(mae(y, p), 3)}


def evaluate(unlock_test: bool, force: bool):
    final = load_report("final_models")
    lib, cols = final["library"], selected_features()
    ds = pd.read_parquet(config.DATASET_PATH)
    parts = {"train": ds[ds["split"] == "train"], "val": ds[ds["split"] == "val"]}
    if unlock_test:
        check_test_lock(lib, force)
        parts["test"] = ds[ds["split"] == "test"]      # the only line in the project that reads test rows

    # 1) Gap table: the tuned model trained on 1980 to 2019 scored on every period.
    #    A small train -> validation -> test gap means little overfitting.
    #    Plus the deployed final model (trained on 1980 to 2022) on test.
    gap = {}
    for h, target in TARGETS.items():
        tuned = load_model(lib, h, "train_only")
        row = {name: scores(part[target], tuned.predict(part[cols])) for name, part in parts.items()}
        if unlock_test:
            row["test_final_model"] = scores(parts["test"][target],
                                             load_model(lib, h, "final").predict(parts["test"][cols]))
        gap[f"d{h}"] = row
        print(f"  day+{h}: " + " | ".join(f"{k} RMSE {v['rmse']:.2f} MAE {v['mae']:.2f}" for k, v in row.items()))
    save_report("gap_table", {"library": lib, "test_included": unlock_test,
                              "note": "train/val/test: tuned model trained on 1980-2019; "
                                      "test_final_model: deployed model trained on 1980-2022",
                              "horizons": gap})

    # 2) Baselines (README only): persistence = "tomorrow's anomaly = today's",
    #    climatology = "tomorrow = the normal" (anomaly 0).
    base = {}
    for name in [p for p in ("val", "test") if p in parts]:
        part = parts[name]
        base[name] = {f"d{h}": {"persistence": scores(part[t], part["ub_t_anom"]),
                                "climatology": scores(part[t], np.zeros(len(part))),
                                "model": gap[f"d{h}"]["test_final_model" if name == "test" else name]}
                      for h, t in TARGETS.items()}
    save_report("baselines", base)

    # 3) SHAP on the final models, explaining their predictions for the most
    #    recent period available (test if unlocked, otherwise validation).
    import shap
    explain = parts["test"] if unlock_test else parts["val"]
    groups = feature_groups(cols)
    shap_out = {"period": [str(explain.index.min().date()), str(explain.index.max().date())],
                "groups": {}, "features": {}}
    for h in TARGETS:
        model = load_model(lib, h, "final")
        values = shap.TreeExplainer(model).shap_values(explain[cols])
        mean_abs = pd.Series(np.abs(values).mean(axis=0), index=cols).sort_values(ascending=False)
        shap_out["features"][f"d{h}"] = {f: round(float(v), 4) for f, v in mean_abs.items()}
        shap_out["groups"][f"d{h}"] = {g: round(float(mean_abs[[f for f in cols if groups[f] == g]].sum()), 4)
                                       for g in sorted(set(groups.values()))}
        top = ", ".join(f"{f} {v:.2f}" for f, v in mean_abs.head(4).items())
        print(f"  SHAP day+{h}: {top}")
    shap_out["feature_group"] = groups
    save_report("shap", shap_out)


def feature_groups(cols) -> dict:
    """Plain-language group of each feature, for the 'How it works' section."""
    groups = {}
    for f in cols:
        if f.startswith("doy_"):
            groups[f] = "Season"
        elif f.startswith("ub_") and ("_lag" in f or "_roll" in f):
            groups[f] = "UB recent days"
        elif f.startswith("ub_"):
            groups[f] = "UB today"
        else:
            groups[f] = "Upwind sites"
    return groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the models (test set only with --unlock-test).")
    parser.add_argument("--unlock-test", action="store_true", help="run the one-time test evaluation")
    parser.add_argument("--force", action="store_true", help="override the test lock (explain why in README)")
    args = parser.parse_args()
    evaluate(args.unlock_test, args.force)
