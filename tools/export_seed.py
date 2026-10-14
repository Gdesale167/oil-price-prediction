"""Export stored prices from data/oil.db to data/seed/*.csv (standard library only).

    py tools\\export_seed.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
db = root / "data" / "oil.db"
out_dir = root / "data" / "seed"

if not db.exists():
    sys.exit(f"Database not found: {db}")

con = sqlite3.connect(db)
tickers = [r[0] for r in con.execute("SELECT DISTINCT ticker FROM prices")]
if not tickers:
    sys.exit("The database has no prices yet.")

out_dir.mkdir(parents=True, exist_ok=True)
for t in tickers:
    rows = con.execute(
        "SELECT date, price FROM prices WHERE ticker = ? ORDER BY date", (t,)
    ).fetchall()
    path = out_dir / f"{t.replace('=', '_').replace('^', '')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "price"])
        w.writerows(rows)
    print(f"{t}: {len(rows)} rows, last date {rows[-1][0]} -> {path.name}")
