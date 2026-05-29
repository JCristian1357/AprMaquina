"""
API de Machine Learning para Predicción de Precios de Casas (California Housing).
Desarrollado con FastAPI y scikit-learn.

La API intenta cargar un modelo real desde 'modelo/modelo_casas.pkl'.
Si el archivo no existe o falla, utiliza un simulador local integrado como respaldo.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib

# Configurar logging para desarrollo y producción
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Rutas del modelo corregidas para el repositorio corporativo
MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_casas.pkl")

# Variable global para el modelo activo y su tipo
active_model = None
model_type = None  # "Real" o "Mock"


# ============================================================================
# Modelos Pydantic (Esquemas de datos con namespaces protegidos limpios)
# ============================================================================


class HousePredictionInput(BaseModel):
    """Esquema de entrada para la predicción de precios de casas."""
    MedInc: float = Field(..., ge=0, description="Ingreso medio de la zona")
    HouseAge: float = Field(..., ge=0, description="Edad de la vivienda en años")
    AveRooms: float = Field(..., ge=0, description="Promedio de habitaciones por casa")

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 8.3252,
                "HouseAge": 41.0,
                "AveRooms": 6.984127,
            }
        }
    }


class HousePredictionOutput(BaseModel):
    """Esquema de salida para la predicción de precios."""
    estimated_price: float = Field(..., description="Precio estimado de la casa")
    model_type: str = Field(..., description="Tipo de modelo utilizado")
    message: str = Field(..., description="Mensaje informativo sobre la predicción")

    model_config = {
        "protected_namespaces": (),  # Quita los warnings de Pydantic v2
        "json_schema_extra": {
            "example": {
                "estimated_price": 4.526,
                "model_type": "Mock",
                "message": "Predicción realizada con modelo simulado (respaldo)",
            }
        }
    }


class APIStatus(BaseModel):
    """Esquema de estado de la API."""
    status: str = Field(..., description="Estado de la API")
    model_loaded: str = Field(..., description="Tipo de modelo cargado")
    version: str = Field(..., description="Versión de la API")

    model_config = {
        "protected_namespaces": (),  # Quita los warnings de Pydantic v2
    }


# ============================================================================
# Funciones de inicialización y carga de modelos
# ============================================================================


def load_model():
    """
    Intenta cargar el modelo real de la carpeta corporativa.
    Si no existe o falla por compatibilidad, activa un simulador indestructible.
    """
    global active_model, model_type

    # 1. Intentar cargar el modelo real de scikit-learn
    if os.path.exists(MODEL_PATH):
        try:
            logger.info(f"Intentando cargar modelo real desde: {MODEL_PATH}")
            model = joblib.load(MODEL_PATH)
            logger.info("✓ Modelo real cargado exitosamente")
            return model, "Real"
        except Exception as e:
            logger.error(f"Error al cargar modelo real desde {MODEL_PATH}: {str(e)}")
            logger.info("Cambiando a simulador de contingencia...")
    else:
        logger.warning(f"Archivo de modelo no encontrado en {MODEL_PATH}. Pasando a modo simulación...")

    # 2. RESPALDO ABSOLUTO: Simulador local integrado (Indestructible para QA)
    logger.warning("Activando modelo simulado de contingencia integrado para QA")
    
    class ContingencyModel:
        def predict(self, data):
            # Simula una predicción matemática usando el valor de MedInc
            return [float(data[0][0] * 35000.0)]
            
    return ContingencyModel(), "Mock"


# ============================================================================
# Event handlers del ciclo de vida de la aplicación
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    global active_model, model_type
    logger.info("=" * 60)
    logger.info("Iniciando API de Predicción de Precios de Casas")
    logger.info("=" * 60)

    active_model, model_type = load_model()
    logger.info(f"Modelo activo en inicio: {model_type}")
    logger.info("API lista para recibir solicitudes")

    yield

    logger.info("Apagando API...")
    logger.info("=" * 60)


# ============================================================================
# Inicialización de FastAPI
# ============================================================================

app = FastAPI(
    title="API de Predicción de Precios de Casas",
    description="API de Machine Learning para predecir precios de casas basándose en el dataset California Housing",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# Rutas (Endpoints)
# ============================================================================


@app.get("/", response_model=APIStatus, tags=["Status"], summary="Estado de la API")
async def root():
    return APIStatus(
        status="operational",
        model_loaded=model_type or "No model loaded",
        version="1.0.0",
    )


@app.post("/predict", response_model=HousePredictionOutput, tags=["Predictions"], summary="Predecir precio de casa")
async def predict(input_data: HousePredictionInput) -> HousePredictionOutput:
    global active_model, model_type
    
    # Rescate de última instancia si la variable global quedara vacía
    if active_model is None:
        logger.warning("Rescate de última instancia ejecutado")
        class ContingencyModel:
            def predict(self, data): return [150000.0]
        active_model = ContingencyModel()
        model_type = "Mock"

    try:
        logger.info(f"Solicitud de predicción recibida - MedInc: {input_data.MedInc}")

        # Estructura de matriz bidimensional nativa para Scikit-Learn [[MedInc, HouseAge, AveRooms]]
        prediction_input = [[input_data.MedInc, input_data.HouseAge, input_data.AveRooms]]

        # Realizar la predicción matemática
        prediction_result = active_model.predict(prediction_input)

        # Extraer el valor numérico de forma segura si viene dentro de un array de NumPy o lista
        if hasattr(prediction_result, "__len__") and not isinstance(prediction_result, dict):
            predicted_price = prediction_result[0]
        else:
            predicted_price = prediction_result

        # Configurar mensaje informativo según el origen de la predicción
        if model_type == "Real":
            message = "Predicción realizada con modelo entrenado (scikit-learn)"
        else:
            message = "Predicción realizada con modelo simulado (respaldo)"

        logger.info(f"Predicción exitosa: ${predicted_price:.6f} ({model_type})")

        return HousePredictionOutput(
            estimated_price=round(float(predicted_price), 6),
            model_type=model_type,
            message=message,
        )

    except ValueError as e:
        logger.error(f"Error de validación en predicción: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error al procesar los datos: {str(e)}")
    except Exception as e:
        logger.error(f"Error inesperado durante la predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error interno en la predicción: {str(e)}")


@app.get("/health", tags=["Status"], summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "model_status": model_type or "not loaded",
    }


# ============================================================================
# Entrada de la aplicación (Ejecución directa fuera de Docker)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
