"""
Feature engineering at stock_id + time_id level from order book and trade data.
Adds realized-volatility and bucketed window features that are useful for this
competition's microstructure setting.
"""
import numpy as np
import pandas as pd

WINDOWS = (
    (0, 600, "full"),
    (0, 150, "w0_150"),
    (150, 300, "w150_300"),
    (300, 450, "w300_450"),
    (450, 600, "w450_600"),
)
GROUP_KEYS = ["stock_id", "time_id"]
SECONDS_PER_BUCKET = 600.0


def realized_volatility(series: pd.Series) -> float:
    """Realized volatility from log-return series."""
    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.sum(values ** 2)))


def add_log_return(df: pd.DataFrame, column: str, output_column: str) -> pd.DataFrame:
    """Grouped log return within each (stock_id, time_id)."""
    if column not in df.columns:
        return df
    df[output_column] = (
        np.log(df[column].replace(0, np.nan))
        .groupby([df["stock_id"], df["time_id"]])
        .diff()
    )
    return df


def flatten_columns(df: pd.DataFrame, prefix: str, suffix: str) -> pd.DataFrame:
    """Flatten a pandas MultiIndex after aggregation."""
    out = df.copy()
    out.columns = [
        f"{prefix}_{left}_{right}_{suffix}".strip("_")
        for left, right in out.columns.to_flat_index()
    ]
    return out.reset_index()


# ---------- Book features ----------
def add_wap(book: pd.DataFrame) -> pd.DataFrame:
    """Weighted average price from top of book."""
    book = book.copy()
    book["wap1"] = (
        book["bid_price1"] * book["ask_size1"] + book["ask_price1"] * book["bid_size1"]
    ) / (book["bid_size1"] + book["ask_size1"] + 1e-8)
    book["wap2"] = (
        book["bid_price2"] * book["ask_size2"] + book["ask_price2"] * book["bid_size2"]
    ) / (book["bid_size2"] + book["ask_size2"] + 1e-8)
    return book


def add_spreads_and_imbalance(book: pd.DataFrame) -> pd.DataFrame:
    """Spread, depth, and price/size imbalance at levels 1 and 2."""
    book = book.copy()
    book["spread1"] = book["ask_price1"] - book["bid_price1"]
    book["spread2"] = book["ask_price2"] - book["bid_price2"]
    book["price_spread"] = (book["ask_price1"] - book["bid_price1"]) / (
        (book["ask_price1"] + book["bid_price1"]) / 2.0 + 1e-8
    )
    book["bid_spread"] = book["bid_price1"] - book["bid_price2"]
    book["ask_spread"] = book["ask_price1"] - book["ask_price2"]
    book["size_imbalance1"] = (book["bid_size1"] - book["ask_size1"]) / (
        book["bid_size1"] + book["ask_size1"] + 1e-8
    )
    book["size_imbalance2"] = (book["bid_size2"] - book["ask_size2"]) / (
        book["bid_size2"] + book["ask_size2"] + 1e-8
    )
    book["total_size1"] = book["bid_size1"] + book["ask_size1"]
    book["total_size2"] = book["bid_size2"] + book["ask_size2"]
    book["total_volume"] = (
        book["ask_size1"] + book["ask_size2"] + book["bid_size1"] + book["bid_size2"]
    )
    book["volume_imbalance"] = abs(
        (book["ask_size1"] + book["ask_size2"]) - (book["bid_size1"] + book["bid_size2"])
    )
    book["wap_balance"] = abs(book["wap1"] - book["wap2"])
    return book


def preprocess_book(book: pd.DataFrame) -> pd.DataFrame:
    """Create book-side point-in-time features before aggregation."""
    if book is None or book.empty:
        return pd.DataFrame()

    book = add_wap(book)
    book = add_spreads_and_imbalance(book)
    for source, target in (
        ("wap1", "log_return_wap1"),
        ("wap2", "log_return_wap2"),
        ("bid_price1", "log_return_bid_price1"),
        ("ask_price1", "log_return_ask_price1"),
        ("bid_price2", "log_return_bid_price2"),
        ("ask_price2", "log_return_ask_price2"),
    ):
        book = add_log_return(book, source, target)
    return book


def book_features_per_time(book: pd.DataFrame, suffix: str = "full") -> pd.DataFrame:
    """Aggregate book data by (stock_id, time_id) for one time window."""
    if book is None or book.empty:
        return pd.DataFrame(columns=GROUP_KEYS)

    value_aggs = {
        "wap1": ["mean", "std", "min", "max"],
        "wap2": ["mean", "std", "min", "max"],
        "wap_balance": ["mean", "std"],
        "spread1": ["mean", "std"],
        "spread2": ["mean", "std"],
        "price_spread": ["mean", "std"],
        "bid_spread": ["mean", "std"],
        "ask_spread": ["mean", "std"],
        "size_imbalance1": ["mean", "std"],
        "size_imbalance2": ["mean", "std"],
        "total_size1": ["mean", "sum", "std"],
        "total_size2": ["mean", "sum", "std"],
        "total_volume": ["mean", "sum", "std"],
        "volume_imbalance": ["mean", "std"],
        "seconds_in_bucket": ["count", "max"],
    }
    rv_cols = [
        "log_return_wap1",
        "log_return_wap2",
        "log_return_bid_price1",
        "log_return_ask_price1",
        "log_return_bid_price2",
        "log_return_ask_price2",
    ]

    available_value_aggs = {col: agg for col, agg in value_aggs.items() if col in book.columns}
    group = book.groupby(GROUP_KEYS)
    out = group.agg(available_value_aggs)
    out = flatten_columns(out, "book", suffix)

    for column in rv_cols:
        if column in book.columns:
            rv = group[column].agg(realized_volatility).reset_index()
            out = out.merge(
                rv.rename(columns={column: f"book_{column}_rv_{suffix}"}),
                on=GROUP_KEYS,
                how="left",
            )

    count_col = f"book_seconds_in_bucket_count_{suffix}"
    if count_col in out.columns:
        out[f"book_missing_seconds_ratio_{suffix}"] = 1.0 - (
            out[count_col] / SECONDS_PER_BUCKET
        )
    return out


# ---------- Trade features ----------
def preprocess_trade(trade: pd.DataFrame) -> pd.DataFrame:
    """Create trade-side point-in-time features before aggregation."""
    if trade is None or trade.empty:
        return pd.DataFrame()

    trade = trade.copy()
    if {"price", "size"}.issubset(trade.columns):
        trade["amount"] = trade["price"] * trade["size"]
    if {"size", "order_count"}.issubset(trade.columns):
        trade["orders_per_trade"] = trade["size"] / (trade["order_count"] + 1e-8)
    trade = add_log_return(trade, "price", "log_return_price")
    return trade


def trade_features_per_time(trade: pd.DataFrame, suffix: str = "full") -> pd.DataFrame:
    """Aggregate trade data by (stock_id, time_id) for one time window."""
    if trade is None or trade.empty:
        return pd.DataFrame(columns=GROUP_KEYS)

    value_aggs = {
        "price": ["mean", "std", "min", "max"],
        "size": ["mean", "sum", "std", "count"],
        "order_count": ["mean", "sum", "std"],
        "amount": ["mean", "sum", "std"],
        "orders_per_trade": ["mean", "std"],
        "seconds_in_bucket": ["count", "max"],
    }
    available_value_aggs = {col: agg for col, agg in value_aggs.items() if col in trade.columns}
    group = trade.groupby(GROUP_KEYS)
    out = group.agg(available_value_aggs)
    out = flatten_columns(out, "trade", suffix)

    if "log_return_price" in trade.columns:
        rv = group["log_return_price"].agg(realized_volatility).reset_index()
        out = out.merge(
            rv.rename(columns={"log_return_price": f"trade_log_return_price_rv_{suffix}"}),
            on=GROUP_KEYS,
            how="left",
        )

    count_col = f"trade_seconds_in_bucket_count_{suffix}"
    if count_col in out.columns:
        out[f"trade_missing_seconds_ratio_{suffix}"] = 1.0 - (
            out[count_col] / SECONDS_PER_BUCKET
        )
    if f"trade_size_sum_{suffix}" in out.columns:
        out[f"trade_total_volume_{suffix}"] = out[f"trade_size_sum_{suffix}"]
    if f"trade_order_count_sum_{suffix}" in out.columns:
        out[f"trade_num_orders_{suffix}"] = out[f"trade_order_count_sum_{suffix}"]
    return out


def merge_window_features(df: pd.DataFrame, fn) -> pd.DataFrame:
    """Aggregate one source across full and sub-window slices."""
    features = []
    for start_sec, end_sec, suffix in WINDOWS:
        window_df = df[(df["seconds_in_bucket"] >= start_sec) & (df["seconds_in_bucket"] < end_sec)]
        features.append(fn(window_df, suffix=suffix))

    out = features[0]
    for feature_df in features[1:]:
        out = out.merge(feature_df, on=GROUP_KEYS, how="outer")
    return out


def add_cross_source_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Simple interaction features between book and trade activity."""
    feats = feats.copy()
    if {"trade_total_volume_full", "book_total_volume_sum_full"}.issubset(feats.columns):
        feats["trade_to_book_volume_full"] = feats["trade_total_volume_full"] / (
            feats["book_total_volume_sum_full"] + 1e-8
        )
    if {"trade_num_orders_full", "trade_size_sum_full"}.issubset(feats.columns):
        feats["trade_mean_order_size_full"] = feats["trade_size_sum_full"] / (
            feats["trade_num_orders_full"] + 1e-8
        )
    if {"book_log_return_wap1_rv_full", "trade_log_return_price_rv_full"}.issubset(feats.columns):
        feats["rv_gap_full"] = (
            feats["book_log_return_wap1_rv_full"] - feats["trade_log_return_price_rv_full"]
        )
    return feats


# ---------- Combined feature matrix ----------
def build_features(
    book: pd.DataFrame,
    trade: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature matrix for keys (must have stock_id, time_id).
    keys: e.g. train[['stock_id','time_id']] or test[['stock_id','time_id']].
    Returns keys merged with book and trade features; missing filled with 0 for robustness.
    """
    feats = keys[["stock_id", "time_id"]].drop_duplicates()

    if book is not None and not book.empty:
        book = preprocess_book(book)
        bf = merge_window_features(book, book_features_per_time)
        feats = feats.merge(bf, on=GROUP_KEYS, how="left")

    if trade is not None and not trade.empty:
        trade = preprocess_trade(trade)
        tf = merge_window_features(trade, trade_features_per_time)
        feats = feats.merge(tf, on=GROUP_KEYS, how="left")

    feats = add_cross_source_features(feats)

    feature_cols = [c for c in feats.columns if c not in ("stock_id", "time_id")]
    feats[feature_cols] = feats[feature_cols].fillna(0)
    feats = feats.replace([np.inf, -np.inf], 0)
    return feats


def get_feature_columns(df: pd.DataFrame) -> list:
    """List of column names to use as model features (exclude target/time columns only)."""
    exclude = {"time_id", "target", "row_id"}
    return [c for c in df.columns if c not in exclude]
