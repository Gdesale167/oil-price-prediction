"""Training, chronological evaluation and model persistence."""
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import database
from .config import MIN_ROWS, model_dir, resolve_ticker
from .data import get_prices
from .models import get_model


def _safe(symbol: str) -> str:
    return symbol.replace("=", "_").replace("^", "")


def model_path(symbol: str, model_name: str):
    return model_dir() / f"{_safe(symbol)}_{model_name}.joblib"


def evaluate(actual: pd.Series, predicted: pd.Series, previous: pd.Series) -> dict:
    err = actual - predicted
    return {
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mape": float((err.abs() / actual).mean() * 100),
        # how often the predicted direction (up/down vs yesterday) was right
        "directional_accuracy": float(
            (np.sign(actual - previous) == np.sign(predicted - previous)).mean() * 100
        ),
        # "tomorrow = today" baseline that every model must beat
        "baseline_mae": float((actual - previous).abs().mean()),
    }


def train_model(prices: pd.Series, model_name: str, test_size: float = 0.2,
                **model_kwargs):
    """Train on the older data, test on the newest, then refit on everything.

    Returns (final_model, metrics, test_frame).
    """
    if len(prices) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} data points, got {len(prices)}")
    if not 0.05 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.05 and 0.5")

    split = int(len(prices) * (1 - test_size))
    model = get_model(model_name, **model_kwargs).fit(prices.iloc[:split])

    pred = model.predict_prices(prices)
    test_idx = pred.index.intersection(prices.index[split:])
    actual = prices.loc[test_idx]
    predicted = pred.loc[test_idx]
    previous = prices.shift(1).loc[test_idx]

    metrics = evaluate(actual, predicted, previous)
    metrics.update(n_train=int(split), n_test=int(len(test_idx)))

    final = get_model(model_name, **model_kwargs).fit(prices)
    final.sigma_ = float(np.std(np.log(actual / predicted)))  # honest out-of-sample band

    test_frame = pd.DataFrame({"actual": actual, "predicted": predicted})
    return final, metrics, test_frame


def train_and_save(ticker: str = "WTI", model_name: str = "xgboost",
                   test_size: float = 0.2, refresh: bool = True, **model_kwargs) -> dict:
    symbol = resolve_ticker(ticker)
    prices = get_prices(symbol, refresh=refresh)
    model, metrics, test_frame = train_model(prices, model_name, test_size, **model_kwargs)
    bundle = {
        "model": model,
        "metrics": metrics,
        "test_frame": test_frame,
        "ticker": symbol,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_price_date": str(prices.index[-1].date()),
    }
    joblib.dump(bundle, model_path(symbol, model_name))
    database.log_run(symbol, model_name, metrics)
    return bundle


def load_bundle(ticker: str, model_name: str) -> dict | None:
    path = model_path(resolve_ticker(ticker), model_name)
    return joblib.load(path) if path.exists() else None
