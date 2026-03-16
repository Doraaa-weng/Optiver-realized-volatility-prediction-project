"""
Optiver Realized Volatility Prediction - Utilities.
RMSPE metric, paths, and shared config for Kaggle notebook compatibility.
"""
from pathlib import Path
import json
import numpy as np


# ---------- Paths (Kaggle vs local) ----------
def get_data_dir():
    """Data root: Kaggle input dir or current directory."""
    kaggle = Path("/kaggle/input/optiver-realized-volatility-prediction")
    if kaggle.exists():
        return kaggle
    return Path(__file__).resolve().parent


def resolve_path_dir(data_dir=None):
    """Coerce an optional data directory input to a Path."""
    return Path(data_dir) if data_dir is not None else get_data_dir()


def get_artifact_root(output_dir=None):
    """Default artifact root for models, reports, OOF, and submissions."""
    if output_dir is None:
        return Path(__file__).resolve().parent / "artifacts"
    output_dir = Path(output_dir)
    if output_dir.name == "artifacts":
        return output_dir
    return output_dir / "artifacts"


def ensure_artifact_dirs(output_dir=None):
    """Create and return the standard artifact directory layout."""
    artifact_root = get_artifact_root(output_dir)
    paths = {
        "root": artifact_root,
        "models": artifact_root / "models",
        "oof": artifact_root / "oof",
        "reports": artifact_root / "reports",
        "submissions": artifact_root / "submissions",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_train_csv_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "train.csv"


def get_test_csv_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "test.csv"


def get_book_train_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "book_train.parquet"


def get_book_test_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "book_test.parquet"


def get_trade_train_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "trade_train.parquet"


def get_trade_test_path(data_dir=None):
    data_dir = resolve_path_dir(data_dir)
    return data_dir / "trade_test.parquet"


# ---------- Evaluation: RMSPE ----------
def rmspe(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-6) -> float:
    """
    Root Mean Square Percentage Error.
    RMSPE = sqrt( mean( ((y_true - y_pred) / y_true)^2 ) ).
    Uses epsilon to avoid division by zero when y_true is 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Clip denominator to avoid inf
    denom = np.maximum(np.abs(y_true), epsilon)
    pct_errors = (y_true - y_pred) / denom
    return np.sqrt(np.mean(pct_errors ** 2))


def rmspe_lgb(y_pred, y_true):
    """
    LightGBM custom metric: must return (eval_name, eval_result, is_higher_better).
    RMSPE: lower is better -> is_higher_better=False.
    """
    y_true = y_true.get_label()
    return "rmspe", rmspe(y_true, y_pred), False


def save_json(path, payload):
    """Write a JSON payload with UTF-8 encoding and stable formatting."""
    path = Path(path)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)
        file_obj.write("\n")


# ---------- Row ID ----------
def make_row_id(stock_id: int, time_id: int) -> str:
    """Submission row_id format: stock_id-time_id."""
    return f"{stock_id}-{time_id}"
