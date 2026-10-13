"""FastAPI service.  Run:  uvicorn api.main:app --reload   (docs at /docs)"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from oilpred import __version__, database
from oilpred.config import MODEL_NAMES, TICKERS, resolve_ticker
from oilpred.data import DataUnavailable, collect, get_prices
from oilpred.models import available_models
from oilpred.predict import forecast
from oilpred.train import train_and_save

app = FastAPI(
    title="Oil Price Prediction API",
    version=__version__,
    description="Collect crude oil prices, train models and serve forecasts.",
)


class CollectRequest(BaseModel):
    ticker: str = "WTI"
    period: str = "1mo"


class TrainRequest(BaseModel):
    ticker: str = "WTI"
    model: str = "xgboost"
    test_size: float = Field(0.2, ge=0.05, le=0.5)
    refresh: bool = True


def _guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except DataUnavailable as exc:
        raise HTTPException(503, str(exc))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc))


def _check_model(name: str) -> None:
    if name not in MODEL_NAMES:
        raise HTTPException(400, f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    if name not in available_models():
        raise HTTPException(400, f"Model '{name}' is not available (PyTorch missing)")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "models": available_models()}


@app.get("/tickers")
def tickers():
    return TICKERS


@app.post("/collect")
def collect_prices(req: CollectRequest):
    new_rows = _guard(collect, req.ticker, req.period)
    return {"ticker": resolve_ticker(req.ticker), "new_rows": new_rows}


@app.get("/prices/{ticker}")
def prices(ticker: str, limit: int = Query(30, ge=1, le=2000), refresh: bool = False):
    series = _guard(get_prices, ticker, refresh)
    tail = series.tail(limit)
    return {
        "ticker": resolve_ticker(ticker),
        "count": len(tail),
        "prices": [{"date": d.strftime("%Y-%m-%d"), "price": round(p, 4)}
                   for d, p in tail.items()],
    }


@app.post("/train")
def train(req: TrainRequest):
    _check_model(req.model)
    bundle = _guard(train_and_save, req.ticker, req.model, req.test_size, req.refresh)
    return {"ticker": bundle["ticker"], "model": req.model,
            "trained_at": bundle["trained_at"], "metrics": bundle["metrics"]}


@app.get("/predict")
def predict(ticker: str = "WTI", model: str = "xgboost",
            days: int = Query(7, ge=1, le=90), refresh: bool = True):
    _check_model(model)
    fc = _guard(forecast, ticker, model, days, refresh)
    return {
        "ticker": resolve_ticker(ticker),
        "model": model,
        "forecast": [
            {"date": d.strftime("%Y-%m-%d"),
             "predicted_price": round(r.predicted_price, 4),
             "lower": round(r.lower, 4), "upper": round(r.upper, 4)}
            for d, r in fc.iterrows()
        ],
        "disclaimer": "Educational use only. Not financial advice.",
    }


@app.get("/runs")
def runs(limit: int = Query(10, ge=1, le=100)):
    return database.recent_runs(limit).to_dict(orient="records")
