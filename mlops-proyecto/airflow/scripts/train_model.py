import os
import glob
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import shutil

RANDOM_STATE = 31102025


# ===============================================================
# Función auxiliar para limpieza
# ===============================================================
def replace_missing_values(df):
    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = df["gender"].replace("Unknown/Invalid", "Male")
    return df


# ===============================================================
# Función principal
# ===============================================================
def train_model():
    """
    Entrena un modelo incremental acumulando todos los sets (train, val, test)
    de los chunks previos.
    Cada ejecución genera una nueva versión del modelo en MLflow.
    """
    data_dir = "/opt/airflow/data/processed"
    state_file = "/opt/airflow/data/state.json"

    if not os.path.exists(state_file):
        raise FileNotFoundError("No se encontró state.json. Ejecuta primero la ingesta y el preprocesamiento.")

    with open(state_file, "r") as f:
        state = json.load(f)
    current_chunk = state["last_chunk"]

    # Buscar todos los sets disponibles
    X_train_files = sorted(glob.glob(os.path.join(data_dir, "X_train_*.parquet")))
    y_train_files = sorted(glob.glob(os.path.join(data_dir, "y_train_*.parquet")))
    X_val_files = sorted(glob.glob(os.path.join(data_dir, "X_val_*.parquet")))
    y_val_files = sorted(glob.glob(os.path.join(data_dir, "y_val_*.parquet")))
    X_test_files = sorted(glob.glob(os.path.join(data_dir, "X_test_*.parquet")))
    y_test_files = sorted(glob.glob(os.path.join(data_dir, "y_test_*.parquet")))

    if not X_train_files or not y_train_files or not X_val_files or not y_val_files:
        raise FileNotFoundError("No hay suficientes archivos procesados para entrenar el modelo.")

    print(f"📊 Entrenando modelo incremental hasta chunk {current_chunk:03d}")

    # ===============================================================
    # Alinear correctamente X e y chunk por chunk
    # ===============================================================
    def safe_concat(X_files, y_files):
        X_list, y_list = [], []
        for xf, yf in zip(X_files, y_files):
            X = pd.read_parquet(xf)
            y = pd.read_parquet(yf).squeeze()
            if len(X) != len(y):
                print(f"⚠️ Tamaño inconsistente en {xf}: X={len(X)}, y={len(y)}. Se ajusta al mínimo común.")
                n = min(len(X), len(y))
                X, y = X.iloc[:n], y.iloc[:n]
            X_list.append(X)
            y_list.append(y)
        return pd.concat(X_list, ignore_index=True), pd.concat(y_list, ignore_index=True)

    X_train, y_train = safe_concat(X_train_files, y_train_files)
    X_val, y_val = safe_concat(X_val_files, y_val_files)

    # Usar todos los test acumulados
    X_test = pd.concat([pd.read_parquet(f) for f in X_test_files], ignore_index=True)
    y_test = pd.concat([pd.read_parquet(f).squeeze() for f in y_test_files], ignore_index=True)

    # Aplanar por seguridad
    y_train = np.ravel(y_train)
    y_val = np.ravel(y_val)
    y_test = np.ravel(y_test)

    print(f"Dimensiones finales:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # MLflow setup
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment("diabetes_xgb_incremental")

    # Clasificador y pipeline
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]

    xgb = XGBClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2
    )

    numeric_cols = X_train.select_dtypes(include=np.number).columns
    categorical_cols = X_train.select_dtypes(exclude=np.number).columns

    pipeline = Pipeline([
        ("cleaner", FunctionTransformer(replace_missing_values)),
        ("preprocess", ColumnTransformer([
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols)
        ])),
        ("clf", xgb)
    ])

    # Entrenamiento y logging robusto
    try:
        with mlflow.start_run(run_name=f"chunk_{current_chunk:03d}") as run:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0)
            }

            mlflow.log_params({
                "current_chunk": current_chunk,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "n_features": X_train.shape[1],
            })

            mlflow.log_metrics(metrics)

            # ===========================================================
            # Guardar modelo localmente y subirlo como artefacto
            # ===========================================================
            local_model_path = f"/opt/airflow/data/models/model_chunk_{current_chunk:03d}"

            # Si ya existe, eliminarlo completamente
            if os.path.exists(local_model_path):
                print(f"⚠️ Eliminando modelo previo en {local_model_path}...")
                shutil.rmtree(local_model_path)

            # Crear carpeta limpia y guardar
            os.makedirs(local_model_path, exist_ok=True)
            mlflow.sklearn.save_model(pipeline, local_model_path)
            mlflow.log_artifacts(local_model_path, artifact_path="model")

            print(f"Modelo chunk {current_chunk:03d} registrado en MLflow con métricas: {metrics}")

    except Exception as e:
        print(f"Error durante el registro en MLflow: {e}")
    finally:
        mlflow.end_run()
