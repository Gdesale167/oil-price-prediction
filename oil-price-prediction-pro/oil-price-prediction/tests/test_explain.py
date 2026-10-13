import pytest

pytest.importorskip("shap")

from oilpred.explain import mean_abs_shap, shap_values  # noqa: E402
from oilpred.features import FEATURE_COLUMNS  # noqa: E402
from oilpred.train import train_model  # noqa: E402


def test_shap_for_xgboost(prices):
    model, _, _ = train_model(prices, "xgboost")
    X, values = shap_values(prices, model, max_rows=100)
    assert values.shape == X.shape
    assert set(mean_abs_shap(X, values).index) == set(FEATURE_COLUMNS)


def test_shap_rejects_other_models(prices):
    model, _, _ = train_model(prices, "ridge")
    with pytest.raises(ValueError):
        shap_values(prices, model)
