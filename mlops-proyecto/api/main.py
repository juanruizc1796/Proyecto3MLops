import os
import socket
import mlflow
import mlflow.pyfunc
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# =====================================
# CONFIGURACIÓN
# =====================================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes_readmission_xgb")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/opt/airflow/data/models/current/model.pkl")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(
    title="API de Predicción de Diabetes",
    version="1.1.1",
    description="Servicio FastAPI que carga el modelo desde MLflow o desde respaldo local (modo forzado o fallback)."
)

# =====================================
# MÉTRICAS PROMETHEUS
# =====================================
REQUEST_COUNT = Counter(
    "api_request_count",
    "Número total de requests agrupados por endpoint, método y estado",
    ["endpoint", "method", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latencia de requests por endpoint y método",
    ["endpoint", "method"]
)
PREDICTION_COUNT = Counter(
    "api_prediction_count",
    "Conteo de predicciones por clase",
    ["predicted_class"]
)

# =====================================
# CARGA DE MODELO
# =====================================
model = None
model_version = None
model_source = "none"  # 'mlflow' o 'local'


def load_production_model():
    """Carga el modelo desde MLflow si es accesible; si no, usa respaldo local."""
    global model, model_version, model_source

    # 🔹 Robustez ante distintos formatos de variable de entorno
    raw_force_local = os.getenv("FORCE_LOCAL_MODEL", "false")
    force_local = str(raw_force_local).strip().lower() in ["true", "1", "yes"]

    if force_local:
        print(f"[MODEL] ⚙️ Modo forzado a local (FORCE_LOCAL_MODEL={raw_force_local}). Saltando conexión a MLflow.")
        try:
            if not os.path.exists(LOCAL_MODEL_PATH):
                raise FileNotFoundError(f"No existe el archivo {LOCAL_MODEL_PATH}")
            model = joblib.load(LOCAL_MODEL_PATH)
            model_version = "local"
            model_source = "local"
            print(f"[MODEL] Modelo local cargado correctamente desde {LOCAL_MODEL_PATH}")
        except Exception as e:
            print(f"[MODEL] Error cargando modelo local: {e}")
            model = None
            model_version = None
            model_source = "none"
        return  # 👈 Detiene aquí, no intenta MLflow

    # 🔹 Si no se fuerza local, se prueba MLflow
    try:
        socket.gethostbyname("mlflow")
        mlflow_accessible = True
    except socket.gaierror:
        mlflow_accessible = False

    if mlflow_accessible:
        try:
            print(f"[MODEL] Intentando cargar desde MLflow: {MODEL_NAME}/{MODEL_STAGE}")
            client = mlflow.tracking.MlflowClient()
            model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            model = mlflow.pyfunc.load_model(model_uri)
            latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
            model_version = latest[0].version if latest else "unknown"
            model_source = "mlflow"
            print(f"[MODEL] Cargado desde MLflow: {MODEL_NAME} v{model_version}")
            return
        except Exception as e:
            print(f"[MODEL] Error cargando desde MLflow: {e}")

    # 🔹 Fallback local si MLflow falla o no está disponible
    print(f"[MODEL] Saltando MLflow (no disponible). Cargando modelo local...")
    try:
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise FileNotFoundError(f"No existe el archivo {LOCAL_MODEL_PATH}")
        model = joblib.load(LOCAL_MODEL_PATH)
        model_version = "local"
        model_source = "local"
        print(f"[MODEL] Modelo local cargado correctamente desde {LOCAL_MODEL_PATH}")
    except Exception as e2:
        print(f"[MODEL] Error cargando modelo local: {e2}")
        model = None
        model_version = None
        model_source = "none"


@app.on_event("startup")
def startup_event():
    load_production_model()

# =====================================
# SCHEMAS
# =====================================
class PredictionInput(BaseModel):
    features: dict


class PredictionOutput(BaseModel):
    prediction: float
    predicted_class: int
    model_version: str
    model_source: str


# =====================================
# MIDDLEWARE MÉTRICAS
# =====================================
class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        method = request.method
        if path == "/metrics":
            return await call_next(request)

        with REQUEST_LATENCY.labels(path, method).time():
            response = await call_next(request)

        REQUEST_COUNT.labels(path, method, str(response.status_code)).inc()
        return response


app.add_middleware(MetricsMiddleware)

# =====================================
# ENDPOINTS
# =====================================
@app.get("/health")
def health():
    if model is None:
        return {"status": "degraded", "message": "Modelo no cargado"}
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": str(model_version),
        "model_source": model_source
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MODEL_NAME,
        "model_stage": MODEL_STAGE,
        "model_version": str(model_version),
        "model_source": model_source,
        "local_model_path": LOCAL_MODEL_PATH
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado")
    try:
        df = pd.DataFrame([payload.features])
        pred = model.predict(df)
        pred_value = float(pred[0])
        predicted_class = int(pred_value >= 0.5)
        PREDICTION_COUNT.labels(str(predicted_class)).inc()
        return PredictionOutput(
            prediction=pred_value,
            predicted_class=predicted_class,
            model_version=str(model_version),
            model_source=model_source
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error durante la predicción: {e}")


@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
