import os
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

MODEL_DIR = "modelo"
MODEL_PATH = os.path.join(MODEL_DIR, "modelo_vinos.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_vinos.pkl")

PROFILE_MAP = {
    0: {
        "name": "Bold & High-Alcohol Red Wine",
        "description": "This profile fits a rich, full-bodied red wine with strong structure and a high alcohol character.",
    },
    1: {
        "name": "Light & High-Acidity White/Rosé Wine",
        "description": "This profile is suitable for a fresher, brighter style with crisp acidity and a lighter body.",
    },
    2: {
        "name": "Balanced & Moderate-Body Wine",
        "description": "This profile is ideal for a well-rounded wine with balanced flavor intensity and moderate body.",
    },
}


class WineInput(BaseModel):
    alcohol: float = Field(..., gt=0, description="Alcohol percentage")
    malic_acid: float = Field(..., gt=0, description="Malic acid level")
    color_intensity: float = Field(..., gt=0, description="Color intensity")
    flavanoids: float = Field(..., gt=0, description="Flavanoids level")

    model_config = {
        "json_schema_extra": {
            "example": {
                "alcohol": 13.0,
                "malic_acid": 2.3,
                "color_intensity": 5.0,
                "flavanoids": 2.0,
            }
        }
    }


class PredictionOutput(BaseModel):
    cluster: int
    profile_name: str
    description: str
    is_mock: bool
    status: str


class RootResponse(BaseModel):
    status: str
    service: str
    docs: str


model = None
scaler = None
is_mock = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, is_mock
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            is_mock = False
        else:
            raise FileNotFoundError("Model artifacts not found")
    except Exception:
        model = None
        scaler = None
        is_mock = True
    yield


app = FastAPI(
    title="Wine Profile Classifier",
    description="FastAPI-based wine clustering classifier using a mock-trained KMeans model",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/app", StaticFiles(directory="static", html=True), name="static")


@app.get("/", response_model=RootResponse)
async def root():
    return {
        "status": "ok",
        "service": "Wine Profile Classifier",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
async def predict(payload: WineInput):
    global model, scaler, is_mock

    try:
        if model is None or scaler is None:
            raise RuntimeError("Using mock fallback")

        features = [[payload.alcohol, payload.malic_acid, payload.color_intensity, payload.flavanoids]]
        scaled_features = scaler.transform(features)
        cluster_id = int(model.predict(scaled_features)[0])
        profile = PROFILE_MAP.get(cluster_id, PROFILE_MAP[2])
        return PredictionOutput(
            cluster=cluster_id,
            profile_name=profile["name"],
            description=profile["description"],
            is_mock=False,
            status="ok",
        )
    except Exception:
        cluster_id = 0
        profile = PROFILE_MAP[cluster_id]
        return PredictionOutput(
            cluster=cluster_id,
            profile_name=profile["name"],
            description=profile["description"],
            is_mock=True,
            status="ok",
        )
