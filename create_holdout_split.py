"""
Create a local holdout dataset from the official training data.

This is useful when the public Kaggle test set is only a tiny placeholder.
The script creates a fully local train/test-style directory with:

- train.csv
- test.csv
- sample_submission.csv
- holdout_targets.csv
- book_train.parquet
- book_test.parquet
- trade_train.parquet
- trade_test.parquet

The split is grouped by `time_id` so the same time bucket never appears in both
the local train and local test sets.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loading import load_all_train
from utils import get_data_dir, make_row_id, save_json


def split_time_ids(
    train_df: pd.DataFrame,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[set[int], set[int]]:
    """Deterministically split unique time_ids into train and holdout groups."""
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    unique_time_ids = train_df["time_id"].drop_duplicates().sample(
        frac=1.0,
        random_state=seed,
    )
    n_holdout = max(1, int(len(unique_time_ids) * holdout_fraction))
    holdout_times = set(unique_time_ids.iloc[:n_holdout].tolist())
    train_times = set(unique_time_ids.iloc[n_holdout:].tolist())
    return train_times, holdout_times


def filter_market_data(df: pd.DataFrame | None, keep_times: set[int]) -> pd.DataFrame | None:
    """Filter book/trade data to the selected time_id set."""
    if df is None:
        return None
    return df[df["time_id"].isin(keep_times)].copy()


def build_holdout_files(
    train_targets: pd.DataFrame,
    holdout_times: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train.csv, test.csv, and holdout target files."""
    train_split = train_targets[~train_targets["time_id"].isin(holdout_times)].copy()
    holdout_split = train_targets[train_targets["time_id"].isin(holdout_times)].copy()

    test_split = holdout_split[["stock_id", "time_id"]].copy()
    test_split["row_id"] = [
        make_row_id(int(stock_id), int(time_id))
        for stock_id, time_id in zip(test_split["stock_id"], test_split["time_id"])
    ]
    holdout_targets = holdout_split.merge(
        test_split,
        on=["stock_id", "time_id"],
        how="left",
    )[["row_id", "stock_id", "time_id", "target"]]
    return train_split, test_split, holdout_targets


def write_parquet(df: pd.DataFrame | None, path: Path) -> None:
    """Write parquet only when data exists."""
    if df is not None:
        df.to_parquet(path, index=False)


def create_holdout_split(
    data_dir=None,
    out_dir=None,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """Create a local holdout dataset under `out_dir`."""
    data_dir = Path(data_dir or get_data_dir())
    out_dir = Path(out_dir or (data_dir / "local_holdout"))
    out_dir.mkdir(parents=True, exist_ok=True)

    train_targets, book_train, trade_train = load_all_train(data_dir)
    if book_train is None and trade_train is None:
        raise FileNotFoundError(
            "Need book_train.parquet and/or trade_train.parquet to create a holdout split."
        )

    train_times, holdout_times = split_time_ids(
        train_targets,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    train_split, test_split, holdout_targets = build_holdout_files(
        train_targets,
        holdout_times=holdout_times,
    )

    book_train_split = filter_market_data(book_train, train_times)
    book_test_split = filter_market_data(book_train, holdout_times)
    trade_train_split = filter_market_data(trade_train, train_times)
    trade_test_split = filter_market_data(trade_train, holdout_times)

    train_split.to_csv(out_dir / "train.csv", index=False)
    test_split.to_csv(out_dir / "test.csv", index=False)
    holdout_targets.to_csv(out_dir / "holdout_targets.csv", index=False)
    sample_submission = test_split[["row_id"]].copy()
    sample_submission["target"] = 0.0
    sample_submission.to_csv(out_dir / "sample_submission.csv", index=False)

    write_parquet(book_train_split, out_dir / "book_train.parquet")
    write_parquet(book_test_split, out_dir / "book_test.parquet")
    write_parquet(trade_train_split, out_dir / "trade_train.parquet")
    write_parquet(trade_test_split, out_dir / "trade_test.parquet")

    save_json(
        out_dir / "split_summary.json",
        {
            "source_data_dir": str(data_dir),
            "output_dir": str(out_dir),
            "seed": seed,
            "holdout_fraction": holdout_fraction,
            "train_rows": int(len(train_split)),
            "holdout_rows": int(len(test_split)),
            "train_time_ids": int(len(train_times)),
            "holdout_time_ids": int(len(holdout_times)),
        },
    )
    print(f"Created holdout split under {out_dir}")
    print(f"Train rows: {len(train_split)}")
    print(f"Holdout rows: {len(test_split)}")
    return out_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Create a local holdout split from training data.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing train.csv and book/trade train parquet files.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for the generated local holdout dataset.",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.2,
        help="Fraction of unique time_id groups to place in the holdout split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle unique time_id groups.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    create_holdout_split(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
