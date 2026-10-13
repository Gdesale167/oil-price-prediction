"""Feature engineering. Row t only uses information available up to t-1 (no leakage)."""
import numpy as np
import pandas as pd

from .config import LAGS

FEATURE_COLUMNS = (
    [f"ret_lag_{i}" for i in range(1, LAGS + 1)]
    + ["ma5_gap", "ma20_gap", "vol10", "vol20"]
)


def log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff()


def build_features(prices: pd.Series) -> pd.DataFrame:
    r = log_returns(prices)
    f = pd.DataFrame(index=prices.index)
    for i in range(1, LAGS + 1):
        f[f"ret_lag_{i}"] = r.shift(i)
    last = prices.shift(1)
    f["ma5_gap"] = last / prices.rolling(5).mean().shift(1) - 1
    f["ma20_gap"] = last / prices.rolling(20).mean().shift(1) - 1
    f["vol10"] = r.shift(1).rolling(10).std()
    f["vol20"] = r.shift(1).rolling(20).std()
    return f[FEATURE_COLUMNS]


def make_dataset(prices: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Features X and target y (the log return of day t)."""
    df = build_features(prices).join(log_returns(prices).rename("target")).dropna()
    return df[FEATURE_COLUMNS], df["target"]


def next_row_features(prices: pd.Series) -> pd.DataFrame:
    """Features for the first unseen business day after the last known price."""
    nxt = pd.Timestamp(prices.index[-1]) + pd.offsets.BDay(1)
    extended = pd.concat([prices, pd.Series([prices.iloc[-1]], index=[nxt])])
    return build_features(extended).iloc[[-1]]
