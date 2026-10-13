"""Explainable AI with SHAP (works with the XGBoost model)."""
import pandas as pd

from .features import make_dataset
from .models import TabularForecaster


def shap_values(prices: pd.Series, model: TabularForecaster, max_rows: int = 300):
    """Return (X, shap_values) for the most recent `max_rows` observations."""
    import shap

    if not isinstance(model, TabularForecaster) or model.kind != "xgboost":
        raise ValueError("SHAP explanations are available for the xgboost model only")
    X, _ = make_dataset(prices)
    X = X.iloc[-max_rows:]
    values = shap.TreeExplainer(model.model).shap_values(X)
    return X, values


def mean_abs_shap(X: pd.DataFrame, values) -> pd.Series:
    return pd.Series(abs(values).mean(axis=0), index=X.columns).sort_values()
