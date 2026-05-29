from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_endpoint_raiz_operacional():
    """Verifica que la página de inicio responda correctamente."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "operational"

def test_endpoint_health_check():
    """Verifica el estado de salud de la API."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediccion_exitosa():
    """Verifica que el modelo reciba datos y devuelva una predicción válida."""
    payload = {
        "MedInc": 8.32,
        "HouseAge": 41.0,
        "AveRooms": 6.98
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    assert "estimated_price" in response.json()
    assert isinstance(response.json()["estimated_price"], float)
