"""Data collection: Yahoo Finance -> SQLite, plus a synthetic generator for tests."""
import logging

import numpy as np
import pandas as pd

from . import database
from .config import DEFAULT_PERIOD, resolve_ticker, seed_dir

log = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """Raised when no price data can be fetched or found locally."""


def fetch_prices(ticker: str, period: str = DEFAULT_PERIOD) -> pd.Series:
    """Download daily closing prices from Yahoo Finance."""
    import yfinance as yf

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise DataUnavailable(f"No data returned for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    # Futures can briefly trade at <= 0 (WTI, April 2020); log returns need > 0.
    s = s[s > 0]
    s.name = "price"
    return s


def collect(ticker: str = "WTI", period: str = DEFAULT_PERIOD) -> int:
    """Fetch the latest prices and store them. Returns number of new rows."""
    symbol = resolve_ticker(ticker)
    prices = fetch_prices(symbol, period)
    new_rows = database.save_prices(symbol, prices)
    log.info("Collected %s: %d rows fetched, %d new", symbol, len(prices), new_rows)
    return new_rows


def seed_path(symbol: str):
    return seed_dir() / f"{symbol.replace('=', '_').replace('^', '')}.csv"


def load_seed(symbol: str) -> pd.Series:
    """Read the bundled CSV snapshot (columns: date, price), if there is one."""
    path = seed_path(symbol)
    if not path.exists():
        return pd.Series(dtype=float, name="price")
    df = pd.read_csv(path, parse_dates=["date"])
    s = pd.Series(df["price"].values, index=pd.DatetimeIndex(df["date"]), name="price")
    return s[s > 0].sort_index()


def get_prices(ticker: str = "WTI", refresh: bool = True,
               period: str = DEFAULT_PERIOD) -> pd.Series:
    """Return stored prices, optionally refreshing from the web first."""
    symbol = resolve_ticker(ticker)
    if refresh:
        try:
            collect(symbol, period)
        except Exception as exc:  # network down, rate limit, ...
            log.warning("Refresh failed for %s: %s. Using stored data.", symbol, exc)
    prices = database.load_prices(symbol)
    seed = load_seed(symbol)  # bundled snapshot, refreshed daily by a GitHub workflow
    if not seed.empty:
        missing = seed[~seed.index.isin(prices.index)]
        if not missing.empty:
            log.info("Adding %d rows from the bundled snapshot for %s", len(missing), symbol)
            database.save_prices(symbol, missing)
            prices = database.load_prices(symbol)
    if prices.empty:
        raise DataUnavailable(
            f"No data available for {symbol}: live fetch failed and no bundled "
            "snapshot exists."
        )
    return prices


def synthetic_prices(n: int = 900, seed: int = 0, start: str = "2020-01-01") -> pd.Series:
    """Random-walk prices with volatility clustering. Used by tests and demos."""
    rng = np.random.default_rng(seed)
    vol = 0.015 + 0.01 * np.abs(np.sin(np.arange(n) / 60))
    returns = rng.normal(0.0002, vol)
    prices = 70 * np.exp(np.cumsum(returns))
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(prices, index=idx, name="price")
