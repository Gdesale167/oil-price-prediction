import numpy as np
import pytest

from oilpred.models import HAS_TORCH, get_model
from oilpred.train import train_model

MODELS = [
    "ridge",
    "xgboost",
    pytest.param("lstm", marks=pytest.mark.skipif(not HAS_TORCH, reason="no torch")),
]


@pytest.mark.parametrize("name", MODELS)
def test_train_and_forecast(prices, name):
    kwargs = {"epochs": 3} if name == "lstm" else {}
    model, metrics, test_frame = train_model(prices, name, 0.2, **kwargs)

    assert metrics["n_train"] + metrics["n_test"] <= len(prices)
    assert np.isfinite(metrics["mae"]) and metrics["mae"] > 0
    assert 0 <= metrics["directional_accuracy"] <= 100
    assert len(test_frame) == metrics["n_test"]

    fc = model.forecast(prices, 10)
    assert len(fc) == 10
    assert (fc["predicted_price"] > 0).all()
    assert (fc["lower"] < fc["predicted_price"]).all()
    assert (fc["predicted_price"] < fc["upper"]).all()
    assert fc.index[0] > prices.index[-1]


def test_too_little_data_is_rejected(prices):
    with pytest.raises(ValueError):
        train_model(prices.iloc[:50], "ridge")


def test_unknown_model():
    with pytest.raises(ValueError):
        get_model("magic")
