import pytest

from oilpred import database
from oilpred.data import synthetic_prices


@pytest.fixture(autouse=True)
def isolated_storage(tmp_path, monkeypatch):
    """Every test gets its own SQLite file and model folder."""
    monkeypatch.setenv("OILPRED_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("OILPRED_MODELS", str(tmp_path / "models"))


@pytest.fixture
def prices():
    return synthetic_prices(n=700, seed=1)


@pytest.fixture
def stored_prices(prices):
    database.save_prices("CL=F", prices)
    return prices
