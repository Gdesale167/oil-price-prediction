"""Forecasting models behind one interface.

Every model predicts the next-day *log return* and converts it back to a price.
Predicting returns (not price levels) avoids the classic problem where tree models
cannot predict outside the price range seen in training.
"""
import numpy as np
import pandas as pd

from .config import LSTM_WINDOW, MODEL_NAMES
from .features import FEATURE_COLUMNS, build_features, log_returns, make_dataset, next_row_features

try:
    import torch
    from torch import nn

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False


class BaseForecaster:
    name = "base"
    sigma_ = 0.02  # std of daily log-return errors, used for forecast bands

    def fit(self, prices: pd.Series) -> "BaseForecaster":
        raise NotImplementedError

    def predict_returns(self, prices: pd.Series) -> pd.Series:
        """One-step-ahead predicted log return for every date with enough history."""
        raise NotImplementedError

    def predict_next_return(self, prices: pd.Series) -> float:
        raise NotImplementedError

    def predict_prices(self, prices: pd.Series) -> pd.Series:
        r = self.predict_returns(prices)
        return prices.shift(1).reindex(r.index) * np.exp(r)

    def forecast(self, prices: pd.Series, days: int) -> pd.DataFrame:
        """Recursive multi-day forecast with an approximate 95% band."""
        history = prices.copy()
        dates = pd.bdate_range(prices.index[-1] + pd.offsets.BDay(1), periods=days)
        rows = []
        for step, date in enumerate(dates, start=1):
            r = self.predict_next_return(history)
            price = float(history.iloc[-1] * np.exp(r))
            history.loc[date] = price
            band = 1.96 * self.sigma_ * np.sqrt(step)
            rows.append((date, price, price * np.exp(-band), price * np.exp(band)))
        out = pd.DataFrame(rows, columns=["date", "predicted_price", "lower", "upper"])
        return out.set_index("date")


class TabularForecaster(BaseForecaster):
    """Ridge regression or XGBoost on lagged returns, moving averages and volatility."""

    def __init__(self, kind: str = "xgboost", **kwargs):
        self.kind = kind
        self.name = kind
        self.model = self._build(kind, kwargs)

    @staticmethod
    def _build(kind: str, kwargs: dict):
        if kind == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            return make_pipeline(StandardScaler(), Ridge(alpha=kwargs.get("alpha", 10.0)))
        if kind == "xgboost":
            from xgboost import XGBRegressor

            params = dict(
                n_estimators=300, max_depth=3, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=1,
            )
            params.update(kwargs)
            return XGBRegressor(**params)
        raise ValueError(f"Unknown tabular model: {kind}")

    def fit(self, prices):
        X, y = make_dataset(prices)
        self.model.fit(X, y)
        self.sigma_ = float(np.std(y - self.model.predict(X)))
        return self

    def predict_returns(self, prices):
        X = build_features(prices).dropna()
        return pd.Series(self.model.predict(X), index=X.index)

    def predict_next_return(self, prices):
        X = next_row_features(prices)
        return float(self.model.predict(X[FEATURE_COLUMNS])[0])


if HAS_TORCH:

    class _LSTMNet(nn.Module):
        def __init__(self, hidden: int = 32):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)


class LSTMForecaster(BaseForecaster):
    """LSTM over the last `window` daily log returns (PyTorch, CPU)."""

    name = "lstm"

    def __init__(self, window: int = LSTM_WINDOW, hidden: int = 32, epochs: int = 30,
                 lr: float = 1e-3, batch_size: int = 64, seed: int = 42):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is not installed. Run: pip install torch")
        self.window, self.hidden, self.epochs = window, hidden, epochs
        self.lr, self.batch_size, self.seed = lr, batch_size, seed
        self.scale_ = 1.0
        self.net = None

    def _windows(self, z: np.ndarray):
        w = self.window
        X = np.stack([z[i - w:i] for i in range(w, len(z))])
        return X[:, :, None].astype(np.float32), z[w:].astype(np.float32)

    def _predict_z(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.from_numpy(X)).numpy()

    def fit(self, prices):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        r = log_returns(prices).dropna().values
        self.scale_ = float(r.std() + 1e-9)
        X, y = self._windows(r / self.scale_)
        self.net = _LSTMNet(self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(len(Xt))
            for s in range(0, len(order), self.batch_size):
                idx = torch.from_numpy(order[s:s + self.batch_size])
                opt.zero_grad()
                loss = loss_fn(self.net(Xt[idx]), yt[idx])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        self.sigma_ = float(np.std(y - self._predict_z(X)) * self.scale_)
        return self

    def predict_returns(self, prices):
        r = log_returns(prices).dropna()
        X, _ = self._windows(r.values / self.scale_)
        return pd.Series(self._predict_z(X) * self.scale_, index=r.index[self.window:])

    def predict_next_return(self, prices):
        r = log_returns(prices).dropna().values[-self.window:] / self.scale_
        x = r[None, :, None].astype(np.float32)
        return float(self._predict_z(x)[0] * self.scale_)


def available_models() -> list[str]:
    return [m for m in MODEL_NAMES if m != "lstm" or HAS_TORCH]


def get_model(name: str, **kwargs) -> BaseForecaster:
    if name in ("ridge", "xgboost"):
        return TabularForecaster(name, **kwargs)
    if name == "lstm":
        return LSTMForecaster(**kwargs)
    raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
