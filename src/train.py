"""
train.py: feature selection, model bake-off, Optuna tuning and final training.

The TEST set (2023 to 2025) is never loaded here. Only evaluate.py may read
it, once, at the very end (see load_train_val below).

Stages (each saves its result to models/reports/, so stages can be re-run alone):
  1. select   a) drop one of each pair of features with |correlation| > 0.95 (train data)
              b) rank the rest by permutation importance on validation
              c) keep the top N where validation error stops improving
  2. bakeoff  LightGBM vs XGBoost vs CatBoost with default settings -> winner
  3. tune     Optuna (TPE sampler, median pruner, fixed seed), one study per horizon,
              winner only, early stopping on validation inside every trial
  4. final    a) tuned model on train (early stopping on val) + learning curves
              b) final model retrained on train + validation (1980 to 2022)

Run from the repo folder:
  python -m src.train                  # all stages
  python -m src.train --stage tune     # one stage
"""

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config
from src.features import feature_columns

TARGETS = {h: f"target_d{h}" for h in config.HORIZONS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validation rows only. The test rows are dropped right here,
    so no training code can ever see them."""
    ds = pd.read_parquet(config.DATASET_PATH)
    return ds[ds["split"] == "train"], ds[ds["split"] == "val"]


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def save_report(name: str, data):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / f"{name}.json").write_text(json.dumps(data, indent=2))


def load_report(name: str):
    path = config.REPORTS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing: run the earlier stage first (python -m src.train).")
    return json.loads(path.read_text())


def make_model(library: str, params: dict | None = None):
    """A model with a fixed seed and thread count, so results are repeatable."""
    params = dict(params or {})
    if library == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(random_state=config.SEED, n_jobs=config.N_JOBS, verbose=-1,
                                 deterministic=True, force_row_wise=True, **params)
    if library == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(random_state=config.SEED, n_jobs=config.N_JOBS,
                                tree_method="hist", **params)
    if library == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(random_seed=config.SEED, thread_count=config.N_JOBS, verbose=0,
                                 allow_writing_files=False, **params)
    raise ValueError(library)


# ---------------------------------------------------------------------------
# Stage 1: feature selection
# ---------------------------------------------------------------------------
def correlation_filter(train: pd.DataFrame, feats: list[str]) -> tuple[list[str], dict]:
    """Go through the features from most to least related to the targets
    (mean |correlation| with day+1, +2, +3 on TRAIN data). Keep a feature
    only if it is not a near-copy (|r| > 0.95) of a feature already kept.
    So of each near-duplicate pair, the more useful one survives."""
    relevance = pd.concat([train[feats].corrwith(train[t]).abs() for t in TARGETS.values()], axis=1).mean(axis=1)
    order = sorted(feats, key=lambda f: (-relevance[f], f))     # ties broken by name: deterministic
    corr = train[feats].corr().abs()
    kept, dropped = [], {}
    for f in order:
        twin = next((k for k in kept if corr.loc[f, k] > config.CORR_THRESHOLD), None)
        if twin is None:
            kept.append(f)
        else:
            dropped[f] = {"kept_instead": twin, "correlation": round(float(corr.loc[f, twin]), 4)}
    return kept, dropped


def stage_select():
    train, val = load_train_val()
    feats = feature_columns(train)

    kept, dropped = correlation_filter(train, feats)
    print(f"a) correlation filter: {len(feats)} -> {len(kept)} features ({len(dropped)} near-duplicates dropped)")

    # b) Permutation importance: shuffle one feature in the VALIDATION data and
    #    see how much the error grows. Big growth = the model relies on it.
    importance = pd.DataFrame(index=kept)
    for h, target in TARGETS.items():
        model = make_model("lightgbm").fit(train[kept], train[target])
        pi = permutation_importance(model, val[kept], val[target], scoring="neg_root_mean_squared_error",
                                    n_repeats=config.PERM_REPEATS, random_state=config.SEED, n_jobs=1)
        importance[f"d{h}"] = pi.importances_mean          # = increase in RMSE (degC)
    importance["mean"] = importance.mean(axis=1)
    ranking = importance.sort_values("mean", ascending=False, kind="mergesort")
    print("b) top 10 by permutation importance (increase in val RMSE, degC):")
    for f, row in ranking.head(10).iterrows():
        print(f"     {f:28s} {row['mean']:.3f}")

    # c) Error vs number of features kept (top N of the ranking).
    order = list(ranking.index)
    grid = sorted({n for n in config.N_GRID if n < len(order)} | {len(order)})
    curve = []
    for n in grid:
        cols = order[:n]
        errs = {f"d{h}": rmse(val[t], make_model("lightgbm").fit(train[cols], train[t]).predict(val[cols]))
                for h, t in TARGETS.items()}
        errs["mean"] = float(np.mean(list(errs.values())))
        curve.append({"n_features": n, **{k: round(v, 4) for k, v in errs.items()}})
        print(f"     N={n:3d}  val RMSE  d1 {errs['d1']:.3f}  d2 {errs['d2']:.3f}  d3 {errs['d3']:.3f}  mean {errs['mean']:.3f}")
    best = min(c["mean"] for c in curve)
    chosen = next(c["n_features"] for c in curve if c["mean"] <= best + config.N_TOLERANCE)
    selected = order[:chosen]
    print(f"c) best mean RMSE {best:.3f}; smallest N within {config.N_TOLERANCE} degC of it: N = {chosen}")

    save_report("feature_selection", {
        "n_start": len(feats), "n_after_correlation_filter": len(kept),
        "correlation_threshold": config.CORR_THRESHOLD, "dropped": dropped,
        "permutation_importance": {f: {k: round(float(v), 5) for k, v in row.items()}
                                   for f, row in ranking.iterrows()},
        "curve": curve, "tolerance": config.N_TOLERANCE, "chosen_n": chosen,
    })
    (config.MODELS_DIR / "selected_features.json").write_text(json.dumps(selected, indent=2))
    print(f"Saved {chosen} selected features to models/selected_features.json")


def selected_features() -> list[str]:
    return json.loads((config.MODELS_DIR / "selected_features.json").read_text())


# ---------------------------------------------------------------------------
# Stage 2: bake-off
# ---------------------------------------------------------------------------
def stage_bakeoff():
    train, val = load_train_val()
    cols = selected_features()
    results = {}
    for lib in config.LIBRARIES:
        results[lib] = {}
        for h, target in TARGETS.items():
            t0 = time.perf_counter()
            model = make_model(lib).fit(train[cols], train[target])
            pred = model.predict(val[cols])
            results[lib][f"d{h}"] = {"rmse": round(rmse(val[target], pred), 4),
                                     "mae": round(mae(val[target], pred), 4),
                                     "seconds": round(time.perf_counter() - t0, 2)}
        results[lib]["mean_rmse"] = round(float(np.mean([results[lib][f"d{h}"]["rmse"] for h in TARGETS])), 4)
        r = results[lib]
        print(f"  {lib:9s} val RMSE  d1 {r['d1']['rmse']:.3f}  d2 {r['d2']['rmse']:.3f}  "
              f"d3 {r['d3']['rmse']:.3f}  mean {r['mean_rmse']:.3f}")
    winner = min(config.LIBRARIES, key=lambda lib: results[lib]["mean_rmse"])
    print(f"Winner (lowest mean validation RMSE): {winner}")

    # Fairness check: the defaults are not a level field (CatBoost builds 1000
    # trees, LightGBM and XGBoost only 100). Here every library keeps its
    # default learning rate but grows as many trees as it needs, stopped by
    # early stopping on validation. The winner should hold in both views.
    fair = {}
    for lib in config.LIBRARIES:
        fair[lib] = {}
        for h, target in TARGETS.items():
            model, best_iter = fit_early_stopping(lib, {}, train[cols], train[target], val[cols], val[target])
            fair[lib][f"d{h}"] = {"rmse": round(rmse(val[target], predict(model, lib, val[cols], best_iter)), 4),
                                  "trees": best_iter}
        fair[lib]["mean_rmse"] = round(float(np.mean([fair[lib][f"d{h}"]["rmse"] for h in TARGETS])), 4)
        print(f"  fairness check, {lib:9s} mean val RMSE {fair[lib]['mean_rmse']:.3f} "
              f"(trees used {[fair[lib][f'd{h}']['trees'] for h in TARGETS]})")
    fair_winner = min(config.LIBRARIES, key=lambda lib: fair[lib]["mean_rmse"])
    print(f"Fairness check winner: {fair_winner} ({'same' if fair_winner == winner else 'DIFFERENT'})")
    save_report("bakeoff", {"settings": "library defaults, seed 42", "results": results, "winner": winner,
                            "fairness_check": {"settings": "default learning rate, early stopping on validation",
                                               "results": fair, "winner": fair_winner}})


# ---------------------------------------------------------------------------
# Fitting with early stopping (same idea for all three libraries)
# ---------------------------------------------------------------------------
class _Pruner:
    """Reports the validation RMSE to Optuna every 50 rounds; stops the
    training early when Optuna's median pruner says the trial is hopeless."""
    def __init__(self, trial, every=50):
        self.trial, self.every, self.pruned = trial, every, False

    def check(self, step: int, value: float) -> bool:
        """Returns True if training should stop."""
        if self.trial is None or step % self.every:
            return False
        self.trial.report(value, step)
        self.pruned = self.trial.should_prune()
        return self.pruned


def fit_early_stopping(lib, params, X, y, Xv, yv, trial=None):
    """Train with up to MAX_ROUNDS rounds; stop when validation RMSE has not
    improved for EARLY_STOPPING rounds. Returns (model, best number of rounds)."""
    pruner = _Pruner(trial)
    if lib == "lightgbm":
        import lightgbm as lgb

        def lgb_callback(env):
            if pruner.check(env.iteration + 1, env.evaluation_result_list[0][2]):
                raise lgb.callback.EarlyStopException(env.iteration, env.evaluation_result_list)
        model = make_model(lib, {"n_estimators": config.MAX_ROUNDS, **params})
        model.fit(X, y, eval_X=(Xv,), eval_y=(yv,), eval_metric="rmse",
                  callbacks=[lgb.early_stopping(config.EARLY_STOPPING, verbose=False), lgb_callback])
        best = model.best_iteration_ or model.n_estimators
    elif lib == "xgboost":
        import xgboost as xgb

        class XgbCallback(xgb.callback.TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):
                return pruner.check(epoch + 1, evals_log["validation_0"]["rmse"][-1])
        model = make_model(lib, {"n_estimators": config.MAX_ROUNDS, "eval_metric": "rmse",
                                 "early_stopping_rounds": config.EARLY_STOPPING,
                                 "callbacks": [XgbCallback()], **params})
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        best = model.best_iteration + 1
    else:
        class CatCallback:
            def after_iteration(self, info):
                return not pruner.check(info.iteration, info.metrics["validation"]["RMSE"][-1])
        model = make_model(lib, {"iterations": config.MAX_ROUNDS,
                                 "early_stopping_rounds": config.EARLY_STOPPING, **params})
        model.fit(X, y, eval_set=(Xv, yv), callbacks=[CatCallback()])
        best = model.get_best_iteration() + 1
    if pruner.pruned:
        raise optuna.TrialPruned()
    return model, int(best)


def predict(model, lib, X, n_rounds=None):
    """Predict using the first n_rounds trees (None = the model's own setting)."""
    if n_rounds is None:
        return model.predict(X)
    if lib == "lightgbm":
        return model.predict(X, num_iteration=n_rounds)
    if lib == "xgboost":
        return model.predict(X, iteration_range=(0, n_rounds))
    return model.predict(X, ntree_end=n_rounds)


def n_rounds_param(lib) -> str:
    return "iterations" if lib == "catboost" else "n_estimators"


# ---------------------------------------------------------------------------
# Stage 3: Optuna tuning (winner only, one study per horizon)
# ---------------------------------------------------------------------------
def search_space(trial, lib) -> dict:
    """The settings Optuna may try. Ranges are wide but sensible for ~15k rows,
    and always contain the library's own defaults (see DEFAULT_TRIAL).
    CatBoost: depth 9 to 10 was tested and was much slower (up to 3 min per
    trial) without being better, so it is excluded."""
    if lib == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.25, log=True),
                "depth": trial.suggest_int("depth", 4, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 30, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 10, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0)}
    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    if lib == "lightgbm":
        return {"learning_rate": lr,
                "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    return {"learning_rate": lr,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True)}


# Trial 0 of every study = the library's own defaults (CatBoost picks its
# learning rate automatically: 0.029 for this data with 8000 rounds max).
# So tuning can never end up worse than the defaults on validation.
# (reg_alpha/reg_lambda default 0 is approximated by 0.001, the bottom of the range.)
DEFAULT_TRIAL = {
    "catboost": {"learning_rate": 0.029, "depth": 6, "l2_leaf_reg": 3.0, "random_strength": 1.0, "subsample": 0.8},
    "lightgbm": {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "subsample": 1.0,
                 "colsample_bytree": 1.0, "reg_alpha": 0.001, "reg_lambda": 0.001},
    "xgboost": {"learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1.0, "subsample": 1.0,
                "colsample_bytree": 1.0, "reg_alpha": 0.001, "reg_lambda": 1.0},
}


def stage_tune():
    train, val = load_train_val()
    cols = selected_features()
    lib = load_report("bakeoff")["winner"]
    storage = f"sqlite:///{config.OPTUNA_DB.as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best = {"library": lib, "trials_per_horizon": config.OPTUNA_TRIALS, "horizons": {}}
    for h, target in TARGETS.items():
        name = f"{lib}_d{h}"
        try:      # start every study fresh, so re-running gives the same result
            optuna.delete_study(study_name=name, storage=storage)
        except KeyError:
            pass
        study = optuna.create_study(
            study_name=name, storage=storage, direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=config.SEED),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=200))
        study.enqueue_trial(DEFAULT_TRIAL[lib])

        def objective(trial):
            params = search_space(trial, lib)
            model, n = fit_early_stopping(lib, params, train[cols], train[target], val[cols], val[target], trial)
            trial.set_user_attr("best_rounds", n)
            return rmse(val[target], predict(model, lib, val[cols], n))

        t0 = time.perf_counter()
        study.optimize(objective, n_trials=config.OPTUNA_TRIALS)
        states = pd.Series([t.state.name for t in study.trials]).value_counts().to_dict()
        bt = study.best_trial
        print(f"  day+{h}: best val RMSE {bt.value:.3f} (trial {bt.number}, {bt.user_attrs['best_rounds']} rounds) "
              f"| {states} | {time.perf_counter() - t0:.0f}s")
        values = [t.value if t.state.name == "COMPLETE" else np.nan for t in study.trials]
        best_so_far = pd.Series(values).cummin().round(4).tolist()
        best["horizons"][f"d{h}"] = {"val_rmse": round(bt.value, 4), "trial": bt.number,
                                     "best_so_far": best_so_far,
                                     "best_rounds": bt.user_attrs["best_rounds"], "params": bt.params,
                                     "trial_states": states}
    save_report("tuning", best)


# ---------------------------------------------------------------------------
# Stage 4: final models
# ---------------------------------------------------------------------------
MODEL_EXT = {"lightgbm": "txt", "xgboost": "json", "catboost": "cbm"}


def model_path(lib, h, kind="final"):
    """kind='final': the deployed model (trained on 1980 to 2022).
    kind='train_only': the tuned model trained on 1980 to 2019 (for the overfitting checks)."""
    folder = config.MODELS_DIR if kind == "final" else config.REPORTS_DIR / "train_only"
    return folder / f"model_d{h}.{MODEL_EXT[lib]}"


def save_model(model, lib, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if lib == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))


def load_model(lib, h, kind="final"):
    path = model_path(lib, h, kind)
    if lib == "lightgbm":
        import lightgbm as lgb
        return lgb.Booster(model_file=str(path))
    model = make_model(lib)
    model.load_model(str(path))
    return model


def staged_rmse(model, lib, X, y, steps) -> list[float]:
    """RMSE after 1, 2, ... trees, at the given step counts."""
    if lib == "catboost":
        out = [rmse(y, p) for p in model.staged_predict(X, eval_period=steps[1] - steps[0], ntree_end=steps[-1])]
        return out[:len(steps)]
    return [rmse(y, predict(model, lib, X, n)) for n in steps]


def stage_final():
    train, val = load_train_val()
    cols = selected_features()
    tuning = load_report("tuning")
    lib = tuning["library"]
    trainval = pd.concat([train, val])
    curves, final_info = {}, {}
    for h, target in TARGETS.items():
        params = dict(tuning["horizons"][f"d{h}"]["params"])

        # a) Tuned model on TRAIN, early stopping on VALIDATION.
        model, best_n = fit_early_stopping(lib, params, train[cols], train[target], val[cols], val[target])

        # Learning curve: grow the same model a bit PAST the stopping point
        # (no early stopping) to show what would happen if we kept going.
        extra = make_model(lib, {**params, n_rounds_param(lib): best_n + config.EARLY_STOPPING})
        extra.fit(train[cols], train[target])
        total = best_n + config.EARLY_STOPPING
        step = max(1, total // 200)
        steps = list(range(step, total + 1, step))
        curves[f"d{h}"] = {"rounds": steps,
                           "train_rmse": [round(v, 4) for v in staged_rmse(extra, lib, train[cols], train[target], steps)],
                           "val_rmse": [round(v, 4) for v in staged_rmse(extra, lib, val[cols], val[target], steps)],
                           "early_stopping_round": best_n}
        tuned = make_model(lib, {**params, n_rounds_param(lib): best_n}).fit(train[cols], train[target])
        save_model(tuned, lib, model_path(lib, h, "train_only"))

        # b) FINAL model: same settings, trained on train + validation. With
        #    ~7.5% more data it gets ~7.5% more rounds (no validation left to stop on).
        n_final = int(round(best_n * len(trainval) / len(train)))
        final = make_model(lib, {**params, n_rounds_param(lib): n_final}).fit(trainval[cols], trainval[target])
        save_model(final, lib, model_path(lib, h, "final"))
        final_info[f"d{h}"] = {"rounds_train_only": best_n, "rounds_final": n_final,
                               "val_rmse_train_only": round(rmse(val[target], tuned.predict(val[cols])), 4)}
        print(f"  day+{h}: stopped at {best_n} rounds (val RMSE {final_info[f'd{h}']['val_rmse_train_only']:.3f}); "
              f"final model: {n_final} rounds on {len(trainval):,} days (1980 to 2022)")
    save_report("learning_curves", curves)
    save_report("final_models", {"library": lib, "features": cols,
                                 "train_only_period": [str(train.index.min().date()), str(train.index.max().date())],
                                 "final_period": [str(trainval.index.min().date()), str(trainval.index.max().date())],
                                 "horizons": final_info})


if __name__ == "__main__":
    stages = {"select": stage_select, "bakeoff": stage_bakeoff, "tune": stage_tune, "final": stage_final}
    parser = argparse.ArgumentParser(description="Phase 3 training pipeline.")
    parser.add_argument("--stage", choices=list(stages), help="run only this stage")
    args = parser.parse_args()
    for name, fn in stages.items():
        if args.stage in (None, name):
            print(f"\n=== Stage: {name} ===")
            fn()
