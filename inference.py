"""
Inference: load test data, build features, average predictions from the saved
LightGBM ensemble, and write submission.csv.
"""
import argparse
from pathlib import Path
import json
import pandas as pd

from data_loading import load_all_test
from feature_engineering import build_features
from utils import ensure_artifact_dirs, get_artifact_root, get_data_dir

try:
    import lightgbm as lgb
except (ImportError, OSError):
    lgb = None


def resolve_model_root(model_dir: str = ".") -> Path:
    """Support both a direct artifacts path and a project root."""
    model_dir = Path(model_dir)
    if (model_dir / "models" / "model_manifest.json").exists():
        return model_dir
    artifact_root = get_artifact_root(model_dir)
    if (artifact_root / "models" / "model_manifest.json").exists():
        return artifact_root
    legacy_manifest = model_dir / "artifacts" / "models" / "model_manifest.json"
    if legacy_manifest.exists():
        return legacy_manifest.parent.parent
    raise FileNotFoundError(f"Could not find model manifest under {model_dir}")


def load_models_and_features(model_dir: str = "."):
    """Load the saved LightGBM ensemble and feature names."""
    model_root = resolve_model_root(model_dir)
    models_dir = model_root / "models"
    manifest_path = models_dir / "model_manifest.json"
    with manifest_path.open(encoding="utf-8") as file_obj:
        manifest = json.load(file_obj)

    feat_path = models_dir / manifest["feature_file"]
    with feat_path.open(encoding="utf-8") as file_obj:
        feature_names = json.load(file_obj)

    models = [
        lgb.Booster(model_file=str(models_dir / model_file))
        for model_file in manifest["model_files"]
    ]
    return models, feature_names, model_root


def run_inference(
    data_dir=None,
    model_dir: str = ".",
    output_path: str = None,
):
    """
    Full inference: load test + book_test + trade_test -> features -> predict -> submission.csv.
    submission.csv has exactly: column 1 = row_id, column 2 = target.
    """
    if lgb is None:
        raise ImportError(
            "lightgbm is required and must load successfully. "
            "On macOS, install OpenMP first with: brew install libomp"
        )

    data_dir = Path(data_dir or get_data_dir())
    models, feature_names, model_root = load_models_and_features(model_dir)
    if output_path is None:
        output_path = model_root / "submissions" / "submission.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    test_df, book_test, trade_test = load_all_test(data_dir)
    if book_test is None and trade_test is None:
        raise FileNotFoundError(
            "No book_test or trade_test parquet found. Cannot build features."
        )

    keys = test_df[["stock_id", "time_id"]]
    feats = build_features(book_test, trade_test, keys)

    # Align columns: use same order and set as training; missing -> 0
    for c in feature_names:
        if c not in feats.columns:
            feats[c] = 0
    X = feats[feature_names]

    pred = sum(model.predict(X) for model in models) / len(models)
    feats["target"] = pd.Series(pred).clip(lower=0).values

    # Merge so submission row order matches test.csv (row_id order)
    submission = test_df[["stock_id", "time_id", "row_id"]].merge(
        feats[["stock_id", "time_id", "target"]],
        on=["stock_id", "time_id"],
        how="left",
    )
    submission = submission[["row_id", "target"]].fillna(0)
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission)} rows to {output_path}")
    return submission


def parse_args():
    parser = argparse.ArgumentParser(description="Run Optiver model inference.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing test.csv and book/trade test parquet files.",
    )
    parser.add_argument(
        "--model-dir",
        default=".",
        help="Artifacts root or project directory containing the saved models.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Optional explicit path for the generated submission.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_artifact_dirs(".")
    run_inference(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        output_path=args.output_path,
    )
