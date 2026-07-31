API de Mae Learning para predecir precios de casas basándose en el dataset California Housing. Construida con **FastAPI** y **scikit-learn**.

---

## 📋 Descripción

Esta API proporciona un servicio de predicción de precios de casas utilizando características como:
- **MedInc**: Ingreso medio de la zona
- **HouseAge**: Edad de la vivienda en años
- **AveRooms**: Promedio de habitaciones por casa

### Características principales

✅ **Modelo en dos capas**: Carga modelo real si existe (`model/model.pkl`), si no, utiliza modelo simulado  
✅ **Validación de datos**: Schemas Pydantic para entrada/salida  
✅ **Documentación automática**: Swagger UI en `/docs`  
✅ **Health checks**: Endpoints para monitoreo  
✅ **Logging detallado**: Trazabilidad completa de operaciones  
✅ **Manejo robusto de errores**: Excepciones informativas  

---

## 🚀 Instalación

### Requisitos previos
- Python 3.8+
- pip (administrador de paquetes de Python)

### Pasos de instalación

1. **Navega al directorio del proyecto**:
   ```bash
   cd project
   ```

2. **Crea un entorno virtual** (recomendado):
   ```bash
   python -m venv venv
   ```

3. **Activa el entorno virtual**:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/macOS**:
     ```bash
     source venv/bin/activate
     ```

4. **Instala las dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎯 Uso

### Iniciar la API

Ejecuta el servidor de desarrollo con Uvicorn:

```bash
uvicorn main:app --reload
```

**Opciones útiles**:
- `--reload`: Reinicia el servidor automáticamente cuando hay cambios en el código
- `--port 8000`: Especifica el puerto (default es 8000)
- `--host 0.0.0.0`: Permite acceso desde otros equipos

**Salida esperada**:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### Acceder a la API

**URL base**: `http://localhost:8000`

#### 1. Documentación Interactiva (Swagger UI)
Abre tu navegador y ve a:
```
http://localhost:8000/docs
```

Aquí puedes:
- Ver todos los endpoints disponibles
- Probar los endpoints interactivamente
- Ver ejemplos de solicitud/respuesta

#### 2. Documentación Alternativa (ReDoc)
```
http://localhost:8000/redoc
```

---

## 📡 Endpoints

### 1. Estado de la API

**GET** `/`

Retorna el estado actual de la API.

**Respuesta exitosa (200)**:
```json
{
  "status": "operational",
  "model_loaded": "Mock",
  "version": "1.0.0"
}
```

---

### 2. Predicción de Precio

**POST** `/predict`

Realiza una predicción del precio de una casa.

**Cuerpo de la solicitud (JSON)**:
```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984127
}
```

**Respuesta exitosa (200)**:
```json
{
  "estimated_price": 4.526123,
  "model_type": "Mock",
  "message": "Predicción realizada con modelo simulado (respaldo)"
}
```

**Errores posibles**:
- `422 Unprocessable Entity`: Datos de entrada inválidos
- `500 Internal Server Error`: Error del servidor

---

### 3. Health Check

**GET** `/health`

Verifica que la API esté operativa.

**Respuesta**:
```json
{
  "status": "healthy",
  "model_status": "Mock"
}
```

---

## 🧪 Ejemplos de Uso

### Con cURL

```bash
# Hacer una predicción
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{
    \"MedInc\": 8.3252,
    \"HouseAge\": 41.0,
    \"AveRooms\": 6.984127
  }"
```

### Con Python (requests)

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.984127
}

response = requests.post(url, json=data)
print(response.json())
```

### Con Python (httpx asincrónico)

```python
import httpx
import asyncio

async def predict():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/predict",
            json={
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127
            }
        )
        print(response.json())

asyncio.run(predict())
```

---

## 🔄 Integración con Modelo Real

### Estructura esperada para el modelo real

Cuando esté listo el modelo real entrenado, guárdalo como:
```
project/
└── model/
    └── model.pkl  ← Coloca el modelo aquí
```

El modelo real debe:
1. Ser compatible con `joblib.load()`
2. Tener un método `.predict()` que acepte un diccionario o similar con keys `medinc`, `houseage`, `averrooms`
3. Retornar un valor numérico (precio predicho)

**Nota**: La API automáticamente cargará el modelo real si existe en `model/model.pkl`. Si no existe, usará el modelo simulado.

---

## 📁 Estructura del Proyecto

```
project/
│
├── main.py                    # Código principal de la API (FastAPI)
├── requirements.txt           # Dependencias del proyecto
│
├── model/
│   ├── __init__.py           # (Opcional) Inicializador del paquete
│   └── fake_model.py         # Modelo simulado para desarrollo
│
└── README.md                  # Este archivo
```

---

## 🛠️ Tecnologías Utilizadas

| Tecnología | Versión | Propósito |
|----------|---------|----------|
| FastAPI | 0.104.1 | Framework web asincrónico |
| Uvicorn | 0.24.0 | Servidor ASGI |
| scikit-learn | 1.3.2 | Machine Learning |
| Pydantic | 2.5.0 | Validación de datos |
| joblib | 1.3.2 | Serialización de modelos |

---

## 📊 Logging y Monitoreo

La API genera logs detallados en la consola. Ejemplos:

```
2026-05-29 10:15:32 - __main__ - INFO - Iniciando API de Predicción de Precios de Casas
2026-05-29 10:15:32 - __main__ - INFO - Intentando cargar modelo real desde: model/model.pkl
2026-05-29 10:15:32 - __main__ - WARNING - Archivo de modelo no encontrado en model/model.pkl
2026-05-29 10:15:32 - __main__ - INFO - ✓ Modelo simulado (FakeHouseModel) inicializado correctamente
2026-05-29 10:15:35 - __main__ - INFO - Solicitud de predicción recibida - MedInc: 8.3252, HouseAge: 41.0, AveRooms: 6.984127
2026-05-29 10:15:35 - __main__ - INFO - Predicción exitosa: $4.526123 (Mock)
```

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'fastapi'`
**Solución**: Instala las dependencias con `pip install -r requirements.txt`

### Error: `Port 8000 is already in use`
**Solución**: Usa otro puerto con `uvicorn main:app --reload --port 8001`

### Error: `FileNotFoundError: [Errno 2] No such file or directory: 'model/model.pkl'`
**Solución**: Es normal. Significa que el modelo real aún no está disponible. La API usa el modelo simulado automáticamente.

### La API no responde a las solicitudes
**Solución**: Asegúrate de que el servidor esté corriendo y accesible en `http://localhost:8000`

---

## 🔐 Seguridad y Consideraciones de Producción

Para desplegar en producción:

1. **Desactiva el `reload`**: `uvicorn main:app` (sin `--reload`)
2. **Usa HTTPS**: Configura certificados SSL/TLS
3. **Variables de entorno**: Usa `.env` para secretos y configuración
4. **Límites de rate**: Implementa throttling para evitar abuso
5. **Autenticación**: Agrega claves API o JWT según necesidad
6. **CORS**: Configura Cross-Origin Resource Sharing si es necesario
7. **Monitoreo**: Implementa APM (Application Performance Monitoring)

---

## 📝 Notas Importantes

- El modelo simulado usa una ecuación matemática simple con coeficientes fijos
- Los precios están en escala de cientos de miles (California Housing dataset)
- La API está optimizada para desarrollo y demostración
- Se recomienda usar un ASGI server como Gunicorn+Uvicorn para producción

---

## 👨‍💻 Autor

Desarrollado como parte del Proyecto Final de Aprendizaje de Máquina.

---


## 📞 Soporte

Para reportar bugs o sugerencias, por favor crea un issue o contacta al equipo de desarrollo.

---

**¡Disfruta usando la API de Predicción de Precios de Casas! 🚀**
