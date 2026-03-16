"""
Create minimal synthetic book/trade parquet so the pipeline can run locally
without downloading full Kaggle data. Uses a subset of train (stock_id, time_id)
and all test keys. For real competition, use Kaggle dataset.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from utils import get_data_dir


def make_book_rows(stock_id: int, time_id: int, n_seconds: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed + hash((stock_id, time_id)) % (2**31))
    rows = []
    for s in range(n_seconds):
        bid1, ask1 = 1.0 - rng.uniform(0, 0.01), 1.0 + rng.uniform(0, 0.01)
        bid2, ask2 = bid1 - 0.005, ask1 + 0.005
        bs1, as1 = rng.integers(10, 500, 2)
        bs2, as2 = rng.integers(10, 300, 2)
        rows.append({
            "stock_id": stock_id,
            "time_id": time_id,
            "seconds_in_bucket": s * 60,
            "bid_price1": bid1, "ask_price1": ask1, "bid_price2": bid2, "ask_price2": ask2,
            "bid_size1": bs1, "ask_size1": as1, "bid_size2": bs2, "ask_size2": as2,
        })
    return rows


def make_trade_rows(stock_id: int, time_id: int, n_trades: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed + hash((stock_id, time_id)) % (2**31))
    rows = []
    for _ in range(n_trades):
        rows.append({
            "stock_id": stock_id,
            "time_id": time_id,
            "seconds_in_bucket": rng.integers(0, 600),
            "price": 1.0 + rng.uniform(-0.02, 0.02),
            "size": rng.integers(1, 100),
            "order_count": rng.integers(1, 10),
        })
    return rows


def main(
    data_dir=None,
    max_train_keys: int = 8000,
    out_dir=None,
):
    data_dir = Path(data_dir or get_data_dir())
    out_dir = Path(out_dir or data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    train_keys = train[["stock_id", "time_id"]].drop_duplicates()
    if len(train_keys) > max_train_keys:
        train_keys = train_keys.sample(n=max_train_keys, random_state=42)
    test_keys = test[["stock_id", "time_id"]].drop_duplicates()

    book_train_list = []
    trade_train_list = []
    for _, r in train_keys.iterrows():
        book_train_list.extend(make_book_rows(int(r["stock_id"]), int(r["time_id"])))
        trade_train_list.extend(make_trade_rows(int(r["stock_id"]), int(r["time_id"])))

    book_test_list = []
    trade_test_list = []
    for _, r in test_keys.iterrows():
        book_test_list.extend(make_book_rows(int(r["stock_id"]), int(r["time_id"])))
        trade_test_list.extend(make_trade_rows(int(r["stock_id"]), int(r["time_id"])))

    book_train = pd.DataFrame(book_train_list)
    trade_train = pd.DataFrame(trade_train_list)
    book_test = pd.DataFrame(book_test_list)
    trade_test = pd.DataFrame(trade_test_list)

    out_dir = out_dir / "synthetic_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    book_train.to_parquet(out_dir / "book_train.parquet", index=False)
    trade_train.to_parquet(out_dir / "trade_train.parquet", index=False)
    book_test.to_parquet(out_dir / "book_test.parquet", index=False)
    trade_test.to_parquet(out_dir / "trade_test.parquet", index=False)
    print(f"Written to {out_dir}: book_train {len(book_train)}, book_test {len(book_test)}, "
          f"trade_train {len(trade_train)}, trade_test {len(trade_test)}.")


if __name__ == "__main__":
    main()
