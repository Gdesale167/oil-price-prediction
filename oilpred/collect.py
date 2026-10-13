"""Real-time data collector.

    python -m oilpred.collect                      # one-off, WTI and Brent
    python -m oilpred.collect --loop --interval 3600
"""
import argparse
import logging
import time

from .config import TICKERS
from .data import collect


def run_once(tickers: list[str], period: str) -> None:
    for t in tickers:
        try:
            new = collect(t, period)
            print(f"{t}: {new} new rows")
        except Exception as exc:
            print(f"{t}: failed ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect oil prices into SQLite")
    parser.add_argument("--tickers", nargs="+", default=list(TICKERS))
    parser.add_argument("--period", default="1mo")
    parser.add_argument("--loop", action="store_true", help="keep collecting")
    parser.add_argument("--interval", type=int, default=3600, help="seconds between runs")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    while True:
        run_once(args.tickers, args.period)
        if not args.loop:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
