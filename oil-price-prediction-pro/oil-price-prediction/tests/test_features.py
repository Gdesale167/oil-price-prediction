import numpy as np

from oilpred.features import FEATURE_COLUMNS, build_features, make_dataset, next_row_features


def test_no_lookahead_leakage(prices):
    """Changing the last price must not change the features of the last row."""
    original = build_features(prices).iloc[-1]
    altered = prices.copy()
    altered.iloc[-1] *= 5
    assert np.allclose(original, build_features(altered).iloc[-1], equal_nan=True)


def test_dataset_shape_and_no_nans(prices):
    X, y = make_dataset(prices)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y) > 600
    assert not X.isna().any().any()


def test_next_row_uses_last_known_data(prices):
    row = next_row_features(prices)
    assert row.shape == (1, len(FEATURE_COLUMNS))
    assert row.index[0] > prices.index[-1]
    assert not row.isna().any().any()
