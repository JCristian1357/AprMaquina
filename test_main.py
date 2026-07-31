from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_returns_service_status():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Wine Profile Classifier"


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_structure_for_valid_payload():
    payload = {
        "alcohol": 13.0,
        "malic_acid": 2.3,
        "color_intensity": 5.0,
        "flavanoids": 2.0,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {"cluster", "profile_name", "description", "is_mock", "status"}
    assert data["cluster"] in {0, 1, 2}
    assert isinstance(data["is_mock"], bool)
    assert data["status"] == "ok"
    assert isinstance(data["description"], str)
    assert data["description"]


def test_predict_rejects_invalid_payload():
    invalid_payload = {
        "alcohol": 0,
        "malic_acid": 2.3,
        "color_intensity": 5.0,
        "flavanoids": 2.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422
