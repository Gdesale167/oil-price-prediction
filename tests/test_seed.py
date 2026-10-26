import pytest

from oilpred import database
from oilpred.data import DataUnavailable, get_prices


def test_seed_snapshot_is_used_when_database_is_empty(prices, tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setenv("OILPRED_SEED", str(seed))
    prices.rename("price").rename_axis("date").reset_index().to_csv(
        seed / "CL_F.csv", index=False
    )

    out = get_prices("WTI", refresh=False)
    assert len(out) == len(prices)
    assert len(database.load_prices("CL=F")) == len(prices)  # saved for next time


def test_no_data_and_no_seed_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OILPRED_SEED", str(tmp_path / "empty"))
    with pytest.raises(DataUnavailable):
        get_prices("WTI", refresh=False)


def test_newer_seed_rows_are_added_to_existing_database(prices, tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    seed.mkdir()
    monkeypatch.setenv("OILPRED_SEED", str(seed))
    database.save_prices("CL=F", prices.iloc[:-5])           # database is 5 days behind
    prices.rename("price").rename_axis("date").reset_index().to_csv(
        seed / "CL_F.csv", index=False
    )
    out = get_prices("WTI", refresh=False)
    assert out.index[-1] == prices.index[-1]
    assert len(out) == len(prices)
