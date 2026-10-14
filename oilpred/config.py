"""Central configuration. Paths can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TICKERS = {"WTI": "CL=F", "Brent": "BZ=F"}
MODEL_NAMES = ["ridge", "xgboost", "lstm"]

LAGS = 10            # number of lagged returns used as features
LSTM_WINDOW = 30     # sequence length for the LSTM
MIN_ROWS = 150       # minimum history needed to train
DEFAULT_PERIOD = "5y"


def db_path() -> Path:
    return Path(os.getenv("OILPRED_DB", ROOT / "data" / "oil.db"))


def seed_dir() -> Path:
    """Folder with bundled CSV snapshots, used when live data is unavailable."""
    return Path(os.getenv("OILPRED_SEED", ROOT / "data" / "seed"))


def model_dir() -> Path:
    path = Path(os.getenv("OILPRED_MODELS", ROOT / "models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_ticker(name: str) -> str:
    """Accept 'WTI', 'Brent' or a raw Yahoo symbol like 'CL=F'."""
    return TICKERS.get(name, name)
