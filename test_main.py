import json
import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_returns_frontend_html():
    """Verifica que el endpoint raíz sirva la interfaz web HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<!DOCTYPE html>" in response.text


def test_api_info_returns_service_status():
    """Verifica que el endpoint /api/info retorne el estado del servicio en JSON."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "Wine Profile Classifier"


def test_health_returns_ok():
    """Verifica el estado de salud de la API."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_structure_for_valid_payload():
    """Verifica que la predicción retorne la estructura completa esperada por el frontend."""
    payload = {
        "alcohol": 13.0,
        "malic_acid": 2.3,
        "color_intensity": 5.0,
        "flavanoids": 2.0,
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert set(data.keys()) == {
        "cluster",
        "profile_name",
        "description",
        "is_mock",
        "status",
    }
    assert data["cluster"] in {0, 1, 2}
    assert isinstance(data["is_mock"], bool)
    assert data["status"] == "ok"
    assert isinstance(data["description"], str)
    assert data["description"]


def test_predict_rejects_invalid_payload():
    """Verifica que la API rechace valores no válidos (<= 0 o tipos incorrectos) con HTTP 422."""
    invalid_payload = {
        "alcohol": 0,
        "malic_acid": 2.3,
        "color_intensity": 5.0,
        "flavanoids": 2.0,
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_calidad_metrica_modelo_mlops():
    """Prueba de MLOps: Audita que el archivo de métricas exista y cumpla el umbral."""
    ruta_metricas = os.path.join("modelo", "metricas_modelo.json")

    assert os.path.exists(
        ruta_metricas
    ), "El archivo modelo/metricas_modelo.json no existe."

    with open(ruta_metricas, "r") as f:
        metricas = json.load(f)

    assert "silhouette_score" in metricas
    assert isinstance(metricas["silhouette_score"], float)
    assert (
        metricas["silhouette_score"] >= 0.35
    ), f"Calidad insuficiente: Silhouette Score es {metricas['silhouette_score']}"
