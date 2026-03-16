"""
Load competition data: train/test CSV and book/trade parquet files.
Works with Kaggle paths or local paths via utils.get_data_dir().
"""
from pathlib import Path
import pandas as pd

from utils import (
    get_data_dir,
    get_train_csv_path,
    get_test_csv_path,
    get_book_train_path,
    get_book_test_path,
    get_trade_train_path,
    get_trade_test_path,
)


def load_train_targets(data_dir=None):
    """Load train.csv: stock_id, time_id, target."""
    path = get_train_csv_path(data_dir)
    return pd.read_csv(path)


def load_test(data_dir=None):
    """Load test.csv: stock_id, time_id, row_id."""
    path = get_test_csv_path(data_dir)
    return pd.read_csv(path)


def _book_train_path(data_dir=None):
    data_dir = data_dir or get_data_dir()
    p = get_book_train_path(data_dir)
    if p.exists() and p.is_file():
        return p
    alt = Path(data_dir) / "synthetic_data" / "book_train.parquet"
    return alt if alt.exists() else p


def load_book_train(data_dir=None):
    """Load book_train.parquet. Returns None if file missing."""
    path = _book_train_path(data_dir)
    if not path.exists() or not path.is_file():
        return None
    return pd.read_parquet(path)


def _book_test_path(data_dir=None):
    data_dir = data_dir or get_data_dir()
    p = get_book_test_path(data_dir)
    if p.exists() and p.is_file():
        return p
    alt = Path(data_dir) / "synthetic_data" / "book_test.parquet"
    return alt if alt.exists() else p


def load_book_test(data_dir=None):
    """Load book_test.parquet. Returns None if file missing."""
    path = _book_test_path(data_dir)
    if not path.exists() or not path.is_file():
        return None
    return pd.read_parquet(path)


def _trade_train_path(data_dir=None):
    data_dir = data_dir or get_data_dir()
    p = get_trade_train_path(data_dir)
    if p.exists() and p.is_file():
        return p
    alt = Path(data_dir) / "synthetic_data" / "trade_train.parquet"
    return alt if alt.exists() else p


def load_trade_train(data_dir=None):
    """Load trade_train.parquet. Returns None if file missing."""
    path = _trade_train_path(data_dir)
    if not path.exists() or not path.is_file():
        return None
    return pd.read_parquet(path)


def _trade_test_path(data_dir=None):
    data_dir = data_dir or get_data_dir()
    p = get_trade_test_path(data_dir)
    if p.exists() and p.is_file():
        return p
    alt = Path(data_dir) / "synthetic_data" / "trade_test.parquet"
    return alt if alt.exists() else p


def load_trade_test(data_dir=None):
    """Load trade_test.parquet. Returns None if file missing."""
    path = _trade_test_path(data_dir)
    if not path.exists() or not path.is_file():
        return None
    return pd.read_parquet(path)


def load_all_train(data_dir=None):
    """
    Load all training data: targets, book, trade.
    Returns (train_targets, book_train, trade_train).
    book_train / trade_train may be None if parquet not found.
    """
    data_dir = data_dir or get_data_dir()
    train = load_train_targets(data_dir)
    book = load_book_train(data_dir)
    trade = load_trade_train(data_dir)
    return train, book, trade


def load_all_test(data_dir=None):
    """
    Load all test data: test rows, book, trade.
    Returns (test_df, book_test, trade_test).
    """
    data_dir = data_dir or get_data_dir()
    test = load_test(data_dir)
    book = load_book_test(data_dir)
    trade = load_trade_test(data_dir)
    return test, book, trade
