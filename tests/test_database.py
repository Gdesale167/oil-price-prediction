from oilpred import database


def test_save_is_idempotent(prices):
    assert database.save_prices("CL=F", prices) == len(prices)
    assert database.save_prices("CL=F", prices) == 0   # no duplicates
    loaded = database.load_prices("CL=F")
    assert len(loaded) == len(prices)
    assert abs(loaded.iloc[-1] - prices.iloc[-1]) < 1e-9


def test_tickers_are_separate(prices):
    database.save_prices("CL=F", prices)
    assert database.load_prices("BZ=F").empty
