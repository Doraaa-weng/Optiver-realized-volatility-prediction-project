"""
Training pipeline: load data, build features, run grouped validation, train a
small ensemble of LightGBM models, and save reproducible experiment artifacts.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from data_loading import load_all_train
from feature_engineering import build_features, get_feature_columns
from utils import ensure_artifact_dirs, get_data_dir, rmspe, rmspe_lgb, save_json

try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None


DEFAULT_SEEDS = (42, 52, 62)


def split_by_time_groups(
    time_ids: np.ndarray,
    n_splits: int = 5,
):
    """
    Deterministic grouped CV by time_id.
    `time_id` is treated as a leakage group rather than a true ordered timestamp.
    """
    unique_time_ids = np.array(sorted(pd.unique(time_ids)))
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for grouped validation")
    folds = [fold for fold in np.array_split(unique_time_ids, n_splits) if len(fold) > 0]
    for fold_idx, val_times in enumerate(folds):
        train_folds = [fold for idx, fold in enumerate(folds) if idx != fold_idx]
        train_times = np.concatenate(train_folds) if train_folds else np.array([], dtype=unique_time_ids.dtype)
        yield fold_idx, train_times, val_times


def get_default_params(seed: int) -> dict:
    """Baseline LightGBM params tuned for tabular volatility features."""
    return {
        "objective": "regression",
        "metric": "None",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "max_depth": -1,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_child_samples": 40,
        "lambda_l1": 0.5,
        "lambda_l2": 1.0,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "verbosity": -1,
        "n_jobs": -1,
        "force_col_wise": True,
    }


def build_feature_importance(models, feature_names: list) -> pd.DataFrame:
    """Aggregate gain-based importance across all trained models."""
    rows = []
    for seed, model in models:
        importance = model.feature_importance(importance_type="gain")
        seed_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance_gain": importance,
                "seed": seed,
            }
        )
        rows.append(seed_df)

    importance_df = pd.concat(rows, ignore_index=True)
    summary = (
        importance_df.groupby("feature", as_index=False)["importance_gain"]
        .mean()
        .sort_values("importance_gain", ascending=False)
    )
    return summary


def train_one_seed(
    X: pd.DataFrame,
    y: pd.Series,
    time_ids: np.ndarray,
    seed: int,
    n_splits: int,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose: bool,
):
    """
    Train one seed with group-based CV and a final model on all data.
    Returns OOF predictions, fold summary, and the final all-data model.
    """
    oof = np.full(len(X), np.nan)
    fold_rows = []

    for fold_idx, train_times, val_times in split_by_time_groups(time_ids, n_splits=n_splits):
        train_mask = np.isin(time_ids, train_times)
        val_mask = np.isin(time_ids, val_times)
        X_train, X_val = X.loc[train_mask], X.loc[val_mask]
        y_train, y_val = y.loc[train_mask], y.loc[val_mask]

        train_set = lgb.Dataset(X_train, y_train, feature_name=list(X.columns))
        val_set = lgb.Dataset(X_val, y_val, reference=train_set)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model = lgb.train(
            params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=[val_set],
            valid_names=["val"],
            feval=rmspe_lgb,
            callbacks=callbacks,
        )
        pred = model.predict(X_val)
        oof[val_mask] = pred
        fold_rmspe = rmspe(y_val.values, pred)
        fold_rows.append(
            {
                "seed": seed,
                "fold": fold_idx,
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "train_time_ids": int(len(train_times)),
                "val_time_ids": int(len(val_times)),
                "best_iteration": int(model.best_iteration or num_boost_round),
                "rmspe": float(fold_rmspe),
            }
        )
        if verbose:
            print(f"  Seed {seed} Fold {fold_idx + 1} RMSPE: {fold_rmspe:.6f}")

    train_set_all = lgb.Dataset(X, y, feature_name=list(X.columns))
    best_iteration = int(np.mean([row["best_iteration"] for row in fold_rows]))
    final_model = lgb.train(
        params,
        train_set_all,
        num_boost_round=max(50, best_iteration),
    )
    return oof, pd.DataFrame(fold_rows), final_model


def train_and_validate(
    X: pd.DataFrame,
    y: pd.Series,
    time_ids: np.ndarray,
    n_splits: int = 5,
    params: dict = None,
    seeds=DEFAULT_SEEDS,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    verbose: bool = True,
):
    """
    Group-based CV on `time_id` with multi-seed LightGBM ensembling.
    Returns OOF predictions, CV summary, trained models, and feature names.
    """
    if lgb is None:
        raise ImportError(
            "lightgbm is required and must load successfully. "
            "On macOS, install OpenMP first with: brew install libomp"
        )

    base_params = params or {}
    feature_names = list(X.columns)
    per_seed_oof = []
    fold_metrics = []
    models = []

    for seed in seeds:
        seed_params = get_default_params(seed)
        seed_params.update(base_params)
        seed_oof, seed_fold_metrics, seed_model = train_one_seed(
            X=X,
            y=y,
            time_ids=time_ids,
            seed=seed,
            n_splits=n_splits,
            params=seed_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            verbose=verbose,
        )
        per_seed_oof.append(seed_oof)
        fold_metrics.append(seed_fold_metrics)
        models.append((seed, seed_model))

    oof = np.mean(np.vstack(per_seed_oof), axis=0)
    fold_metrics_df = pd.concat(fold_metrics, ignore_index=True)
    cv_rmspe = rmspe(y.values, oof)
    if verbose:
        print(f"  Ensemble OOF RMSPE: {cv_rmspe:.6f}")

    return oof, cv_rmspe, models, feature_names, fold_metrics_df


def run_train(
    data_dir=None,
    n_splits: int = 5,
    output_dir: str = ".",
    save_model: bool = True,
    seeds=DEFAULT_SEEDS,
    params: dict = None,
):
    """
    Full training pipeline: load -> features -> grouped CV -> save artifacts.
    Returns (cv_rmspe, models, feature_names).
    """
    data_dir = Path(data_dir or get_data_dir())
    artifact_dirs = ensure_artifact_dirs(output_dir)

    train_targets, book, trade = load_all_train(data_dir)
    if book is None and trade is None:
        raise FileNotFoundError(
            "No book or trade parquet found. Add book_train.parquet and trade_train.parquet to the data directory."
        )

    keys = train_targets[["stock_id", "time_id"]]
    feats = build_features(book, trade, keys)
    train_full = feats.merge(
        train_targets[["stock_id", "time_id", "target"]],
        on=["stock_id", "time_id"],
        how="inner",
    )

    feature_cols = get_feature_columns(train_full)
    X = train_full[feature_cols]
    y = train_full["target"]
    time_ids = train_full["time_id"].values

    oof, cv_rmspe, models, feature_names, fold_metrics = train_and_validate(
        X,
        y,
        time_ids,
        n_splits=n_splits,
        params=params,
        seeds=seeds,
        verbose=True,
    )

    if save_model:
        model_paths = []
        for seed, model in models:
            model_path = artifact_dirs["models"] / f"model_seed{seed}.txt"
            model.save_model(str(model_path))
            model_paths.append(model_path.name)

        save_json(artifact_dirs["models"] / "feature_names.json", feature_names)
        save_json(
            artifact_dirs["models"] / "model_manifest.json",
            {
                "model_type": "lightgbm",
                "seeds": list(seeds),
                "model_files": model_paths,
                "feature_file": "feature_names.json",
            },
        )

        oof_df = train_full[["stock_id", "time_id", "target"]].copy()
        oof_df["oof_prediction"] = np.clip(oof, 0, None)
        oof_df.to_csv(artifact_dirs["oof"] / "oof_predictions.csv", index=False)

        fold_metrics.to_csv(artifact_dirs["reports"] / "fold_metrics.csv", index=False)
        feature_importance = build_feature_importance(models, feature_names)
        feature_importance.to_csv(
            artifact_dirs["reports"] / "feature_importance.csv",
            index=False,
        )
        save_json(
            artifact_dirs["reports"] / "training_summary.json",
            {
                "cv_rmspe": float(cv_rmspe),
                "n_rows": int(len(train_full)),
                "n_features": int(len(feature_names)),
                "n_splits": int(n_splits),
                "seeds": list(seeds),
            },
        )
        print(f"Artifacts saved under {artifact_dirs['root']}")

    return cv_rmspe, models, feature_cols


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Optiver volatility model.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.csv and book/trade parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where artifacts/ will be written.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of grouped CV folds.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_train(
        data_dir=args.data_dir,
        n_splits=args.n_splits,
        output_dir=args.output_dir,
    )
