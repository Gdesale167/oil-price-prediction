"""Fetch the latest prices and merge them into data/seed/*.csv (used by the daily workflow).

    python tools/update_seed.py
"""
import sys
from pathlib import Path

import pandas as pd

TICKERS = {"CL=F": "CL_F.csv", "BZ=F": "BZ_F.csv"}  # WTI, Brent
SEED_DIR = Path(__file__).resolve().parent.parent / "data" / "seed"


def merge(old: pd.Series, new: pd.Series) -> pd.Series:
    """Combine stored and fresh prices; fresh values win on the same date."""
    both = pd.concat([old, new])
    both = both[~both.index.duplicated(keep="last")].sort_index()
    return both[both > 0]


def read_csv(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.Series(df["price"].values, index=pd.DatetimeIndex(df["date"]))


def fetch(symbol: str, period: str) -> pd.Series:
    import yfinance as yf

    df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError("no data returned")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def main() -> int:
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for symbol, filename in TICKERS.items():
        path = SEED_DIR / filename
        old = read_csv(path)
        try:
            new = fetch(symbol, "1y" if len(old) else "5y")
        except Exception as exc:
            print(f"{symbol}: FAILED ({exc})")
            continue
        merged = merge(old, new)
        out = merged.round(4).rename("price").rename_axis("date").reset_index()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        out.to_csv(path, index=False)
        print(f"{symbol}: {len(old)} -> {len(merged)} rows, last date {out['date'].iloc[-1]}")
        ok += 1
    return 0 if ok else 1  # fail the workflow if nothing could be fetched


if __name__ == "__main__":
    sys.exit(main())
