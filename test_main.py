from fastapi.testclient import TestClient
from main import app  # Importa la aplicación FastAPI del Integrante 2

# Creamos un cliente de pruebas basado en nuestra app
client = TestClient(app)

def test_prediccion_exitosa():
    # 1. Definimos un caso de prueba con datos simulados válidos
    # MedInc: 4.5 ($45k de ingresos), HouseAge: 15 años, AveRooms: 5.2 habitaciones
    payload = {
        "MedInc": 4.5,
        "HouseAge": 15.0,
        "AveRooms": 5.2
    }
    
    # 2. Enviamos una petición POST simulada a la ruta /predict
    response = client.post("/predict", json=payload)
    
    # 3. VERIFICACIONES DE MANTENIMIENTO (Si alguna falla, el pipeline da Rojo ❌)
    
    # Asegurar que la API responda exitosamente (Código HTTP 200 OK)
    assert response.status_code == 200
    
    # Asegurar que la respuesta devuelva la clave exacta del precio
    assert "precio_predicho" in response.json()
    
    # Asegurar que el resultado sea un número decimal (float) y no un texto o error
    assert isinstance(response.json()["precio_predicho"], float)

def test_ruta_inicio():
    # Prueba adicional para verificar que la página de bienvenida funcione
    response = client.get("/")
    assert response.status_code == 200
    assert "mensaje" in response.json()
