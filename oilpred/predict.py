"""Forecast with a saved model (training one first if none exists)."""
import pandas as pd

from . import database
from .config import resolve_ticker
from .data import get_prices
from .train import load_bundle, train_and_save


def forecast(ticker: str = "WTI", model_name: str = "xgboost", days: int = 30,
             refresh: bool = True) -> pd.DataFrame:
    if not 1 <= days <= 90:
        raise ValueError("days must be between 1 and 90")
    symbol = resolve_ticker(ticker)
    prices = get_prices(symbol, refresh=refresh)
    bundle = load_bundle(symbol, model_name) or train_and_save(
        symbol, model_name, refresh=False
    )
    result = bundle["model"].forecast(prices, days)
    database.log_predictions(symbol, model_name, result)
    return result
