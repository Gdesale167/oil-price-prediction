[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://oil-price-gunjan.streamlit.app)
[![CI](https://github.com/Gdesale167/oil-price-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Gdesale167/oil-price-prediction/actions)

# 🛢️ Oil Price Prediction: end-to-end ML project

Collects crude oil prices (WTI and Brent), trains three forecasting models, explains them
with SHAP and serves predictions through a REST API and a Streamlit dashboard.
Containerised with Docker and tested with GitHub Actions CI.

| Area | What is implemented |
|---|---|
| Time-series forecasting | Next-day log-return model, recursive multi-day forecast, 95% uncertainty band |
| Deep learning | LSTM in PyTorch over the last 30 daily returns |
| Gradient boosting | XGBoost on lagged returns, moving-average gaps and volatility |
| Baseline | Ridge regression, plus the naive "tomorrow = today" benchmark |
| Explainable AI | SHAP (TreeExplainer): bar chart and beeswarm in the dashboard |
| Database | SQLite: prices, prediction log and model run history |
| Real-time collection | `python -m oilpred.collect --loop` (Yahoo Finance to SQLite) |
| API | FastAPI with validation and auto docs at `/docs` |
| Dashboard | Streamlit: forecast, model comparison, SHAP, database view |
| Docker | One image, `docker-compose` runs API, dashboard and collector |
| CI/CD | GitHub Actions: ruff lint, pytest, Docker build |
| Tests | 17 tests: leakage check, DB, all 3 models, SHAP, API |

## Project structure
```
oil-price-prediction/
├── oilpred/               core package
│   ├── config.py          settings
│   ├── database.py        SQLite layer
│   ├── data.py            Yahoo Finance collector
│   ├── features.py        leakage-safe feature engineering
│   ├── models.py          Ridge / XGBoost / LSTM behind one interface
│   ├── train.py           chronological split, metrics, model saving
│   ├── predict.py         forecasting
│   ├── explain.py         SHAP
│   └── collect.py         collector CLI
├── api/main.py            FastAPI service
├── streamlit_app.py       dashboard
├── tests/                 pytest suite
├── .github/workflows/     CI pipeline
├── Dockerfile, docker-compose.yml
└── requirements.txt
```

## Run locally

Use **Python 3.11 or 3.12**. Newer versions (3.14) may not have wheels for PyTorch
or SHAP yet.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

streamlit run streamlit_app.py    # dashboard  -> http://localhost:8501
uvicorn api.main:app --reload     # API docs   -> http://localhost:8000/docs
python -m oilpred.collect         # fetch latest prices once
pytest                            # run tests
```

## Run with Docker
```bash
docker compose up --build
```
Dashboard on `:8501`, API on `:8000`, and a collector that refreshes prices every hour.

## API examples
```bash
curl "http://localhost:8000/predict?ticker=WTI&model=xgboost&days=7"
curl -X POST http://localhost:8000/train -H "Content-Type: application/json" \
     -d '{"ticker": "Brent", "model": "lstm"}'
curl http://localhost:8000/runs
```

## Design decisions (worth mentioning in interviews)
- **Predict returns, not prices.** Tree models cannot extrapolate beyond the price range
  they saw in training; log returns are stationary, so this problem disappears.
- **No leakage.** Features for day *t* use data up to *t-1* only; a unit test enforces it.
- **Chronological split** (never shuffled), with a final refit on all data.
- **Honest baseline.** Every model is compared with "tomorrow = today". Oil is close to a
  random walk, so beating it by a small margin is a realistic result.
- **Forecast bands** come from out-of-sample error and widen with the horizon.
- Negative prices (WTI, April 2020) are dropped because log returns need prices > 0.

## Data note
Live prices come from Yahoo Finance. Some cloud hosts (including Streamlit Community
Cloud) are rate-limited by Yahoo, so a CSV snapshot in `data/seed/` is used as a
fallback and the dashboard shows the date of the data it is using. Locally and in
Docker the collector refreshes prices normally.

## Limitations
Prices depend on geopolitics, OPEC decisions and news that are not in this data.
Multi-day recursive forecasts converge toward a flat line. This is an educational
project, not financial advice.

## Deploy the dashboard (free)
Push to GitHub, then on share.streamlit.io choose `streamlit_app.py`.
