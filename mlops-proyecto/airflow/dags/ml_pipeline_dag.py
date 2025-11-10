"""
DAG unificado: ingesta → preprocesamiento → entrenamiento
Ejecuta todo el pipeline MLOps completo cada 3 minutos.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
import sys
import os

# === Configuración del entorno ===
sys.path.append("/opt/airflow/scripts")

# Importar las funciones principales
from ingest_data import ingest_csv_chunks
from preprocess_split import preprocess_and_split
from train_model import train_model

# === Parámetros generales ===
default_args = {
    "owner": "juan_ruiz",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# === Definición del DAG ===
with DAG(
    dag_id="ml_pipeline_dag",
    description="Pipeline completa: ingesta → preprocesamiento → entrenamiento y registro en MLflow",
    default_args=default_args,
    start_date=datetime(2025, 11, 1),
    schedule_interval="*/3 * * * *",  # cada 3 minutos
    catchup=False,
    tags=["mlops", "pipeline", "unified"],
) as dag:

    # === Agrupamos las tareas de ingesta y preprocesamiento ===
    with TaskGroup("data_pipeline", tooltip="Pipeline de datos") as data_group:

        # Ingesta de datos
        ingest_task = PythonOperator(
            task_id="ingest_csv_chunks",
            python_callable=ingest_csv_chunks,
        )

        # Preprocesamiento y split
        preprocess_task = PythonOperator(
            task_id="preprocess_and_split",
            python_callable=preprocess_and_split,
        )

        ingest_task >> preprocess_task

    # === Entrenamiento del modelo ===
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # === Flujo final ===
    data_group >> train_task
