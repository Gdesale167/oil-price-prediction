from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_prices_endpoint(stored_prices):
    r = client.get("/prices/WTI", params={"limit": 5})
    assert r.status_code == 200
    assert r.json()["count"] == 5


def test_train_then_predict(stored_prices):
    r = client.post("/train", json={"ticker": "WTI", "model": "ridge", "refresh": False})
    assert r.status_code == 200
    assert "mae" in r.json()["metrics"]

    r = client.get("/predict", params={"ticker": "WTI", "model": "ridge",
                                       "days": 5, "refresh": False})
    assert r.status_code == 200
    assert len(r.json()["forecast"]) == 5

    assert len(client.get("/runs").json()) == 1


def test_validation_errors(stored_prices):
    assert client.get("/predict", params={"model": "magic", "refresh": False}).status_code == 400
    assert client.get("/predict", params={"days": 500, "refresh": False}).status_code == 422


def test_no_data_returns_503():
    r = client.get("/prices/NOPE", params={"refresh": False})
    assert r.status_code == 503
