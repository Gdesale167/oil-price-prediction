"""SQLite persistence: prices, prediction log and model run history."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

import pandas as pd

from .config import db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,
    price  REAL NOT NULL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    model TEXT NOT NULL,
    target_date TEXT NOT NULL,
    predicted_price REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    ticker TEXT NOT NULL,
    model TEXT NOT NULL,
    mae REAL, rmse REAL, mape REAL,
    directional_accuracy REAL, baseline_mae REAL,
    n_train INTEGER, n_test INTEGER
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_prices(ticker: str, prices: pd.Series) -> int:
    """Upsert prices. Returns the number of rows that were new."""
    before = count_prices(ticker)
    rows = [
        (ticker, pd.Timestamp(d).strftime("%Y-%m-%d"), float(p))
        for d, p in prices.items()
    ]
    with connect() as conn:
        conn.executemany("INSERT OR REPLACE INTO prices VALUES (?, ?, ?)", rows)
    return count_prices(ticker) - before


def count_prices(ticker: str) -> int:
    with connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM prices WHERE ticker = ?", (ticker,)
        ).fetchone()[0]


def load_prices(ticker: str) -> pd.Series:
    with connect() as conn:
        df = pd.read_sql_query(
            "SELECT date, price FROM prices WHERE ticker = ? ORDER BY date",
            conn,
            params=(ticker,),
        )
    s = pd.Series(df["price"].values, index=pd.to_datetime(df["date"]), name="price")
    s.index.name = "date"
    return s


def log_predictions(ticker: str, model: str, forecast: pd.DataFrame) -> None:
    now = _now()
    rows = [
        (now, ticker, model, pd.Timestamp(d).strftime("%Y-%m-%d"), float(p))
        for d, p in forecast["predicted_price"].items()
    ]
    with connect() as conn:
        conn.executemany(
            "INSERT INTO predictions "
            "(created_at, ticker, model, target_date, predicted_price) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def log_run(ticker: str, model: str, m: dict) -> None:
    with connect() as conn:
        conn.execute(
            "INSERT INTO model_runs (created_at, ticker, model, mae, rmse, mape, "
            "directional_accuracy, baseline_mae, n_train, n_test) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _now(), ticker, model, m["mae"], m["rmse"], m["mape"],
                m["directional_accuracy"], m["baseline_mae"], m["n_train"], m["n_test"],
            ),
        )


def recent_runs(limit: int = 20) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM model_runs ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )


def recent_predictions(limit: int = 50) -> pd.DataFrame:
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
