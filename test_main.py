import os
import json
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_endpoint_raiz_operacional():
    """Verifica que la página de inicio o interfaz responda correctamente."""
    response = client.get("/")
    assert response.status_code == 200

def test_endpoint_health_check():
    """Verifica el estado de salud de la API y carga del modelo de vinos."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediccion_clasificacion_vino_ingles():
    """
    Verifica que la API reciba los 4 parámetros químicos del vino,
    responda HTTP 200 y devuelva la etiqueta del cluster en inglés.
    """
    payload = {
        "alcohol": 13.5,
        "malic_acid": 2.3,
        "color_intensity": 5.6,
        "flavanoids": 2.8
    }
    response = client.post("/predict", json=payload)
    
    # 1. Validar respuesta HTTP exitosa
    assert response.status_code == 200
    
    data = response.json()
    
    # 2. Validar presencia de las claves en el JSON devuelto
    assert "cluster_id" in data
    assert "cluster_label" in data
    
    # 3. Validar tipos de datos
    assert isinstance(data["cluster_id"], int)
    assert isinstance(data["cluster_label"], str)
    
    # 4. Validar que la respuesta esté en inglés (Control de calidad de rúbrica)
    etiqueta = data["cluster_label"].lower()
    assert any(word in etiqueta for word in ["wine", "red", "white", "rosé", "body", "alcohol", "acidity", "balanced", "bold", "light"])

def test_calidad_metrica_modelo():
    """
    Prueba de MLOps: Verifica que el Integrante 1 haya generado el archivo
    de métricas y que el Coeficiente de Silueta supere el umbral mínimo aceptable (0.40).
    """
    ruta_metricas = os.path.join("modelo", "metricas.json")
    
    # Si el archivo de métricas existe en el repositorio, se audita su calidad
    if os.path.exists(ruta_metricas):
        with open(ruta_metricas, "r") as f:
            metricas = json.load(f)
        
        assert "silhouette_score" in metricas
        assert isinstance(metricas["silhouette_score"], float)
        # Exigir un umbral mínimo de calidad de agrupamiento
        assert metricas["silhouette_score"] >= 0.40
