import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 31102025


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica transformaciones adicionales sobre el dataset."""
    df = df.copy()

    df["total_visits"] = (
        df["number_inpatient"].fillna(0)
        + df["number_emergency"].fillna(0)
        + df["number_outpatient"].fillna(0)
    )

    df["multi_inpatient"] = (df["number_inpatient"].fillna(0) > 0).astype(int)
    return df


def preprocess_and_split():
    """
    Procesa el último chunk generado en la fase de ingesta.
    Aplica limpieza, feature engineering y guarda los sets train/val/test
    identificados por el número de chunk.
    """
    interim_dir = "/opt/airflow/data/interim"
    output_dir = "/opt/airflow/data/processed"
    state_file = "/opt/airflow/data/state.json"

    os.makedirs(interim_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(state_file):
        raise FileNotFoundError(" No se encontró state.json. Ejecuta primero el DAG de ingesta.")

    with open(state_file, "r") as f:
        state = json.load(f)
    chunk_id = state["last_chunk"]

    file_path = os.path.join(interim_dir, f"chunk_{chunk_id:03d}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f" No existe el archivo {file_path}. Ejecuta primero la ingesta.")

    print(f"⚙️ Procesando archivo: {file_path}")

    # === Leer datos ===
    df = pd.read_csv(file_path)

    # === Target mapping ===
    df["readmitted_mapped"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 1})
    df = df.dropna(subset=["readmitted_mapped"])

    # === Feature engineering ===
    df = feature_engineering(df)

    # === Separar X / y ===
    y = df["readmitted_mapped"]
    X = df.drop(columns=["readmitted_mapped", "readmitted"], errors="ignore")

    # === Split ===
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=RANDOM_STATE, stratify=y_trainval
    )

    # === Evitar sobrescribir si ya existe ese chunk ===
    for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]:
        file_name = os.path.join(output_dir, f"{name}_{chunk_id:03d}.parquet")
        if os.path.exists(file_name):
            print(f" Archivo {file_name} ya existe. Se omite este chunk.")
            return  # No reprocesa ni sobreescribe

    # === Guardar archivos con sufijo del chunk actual ===
    X_train.to_parquet(os.path.join(output_dir, f"X_train_{chunk_id:03d}.parquet"))
    X_val.to_parquet(os.path.join(output_dir, f"X_val_{chunk_id:03d}.parquet"))
    X_test.to_parquet(os.path.join(output_dir, f"X_test_{chunk_id:03d}.parquet"))

    y_train.to_frame(name="target").to_parquet(os.path.join(output_dir, f"y_train_{chunk_id:03d}.parquet"))
    y_val.to_frame(name="target").to_parquet(os.path.join(output_dir, f"y_val_{chunk_id:03d}.parquet"))
    y_test.to_frame(name="target").to_parquet(os.path.join(output_dir, f"y_test_{chunk_id:03d}.parquet"))

    print(f" División train/val/test completada para chunk {chunk_id}")
    print(f" Archivos guardados en: {output_dir}")

    # === Actualizar el estado (opcional, por si el pipeline no lo hace luego) ===
    state["last_processed_chunk"] = chunk_id
    with open(state_file, "w") as f:
        json.dump(state, f)
    print(f"🧾 Estado actualizado: último chunk procesado = {chunk_id}")
