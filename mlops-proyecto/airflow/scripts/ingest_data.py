import os
import pandas as pd
import json

# ===============================================================
# CONFIGURACIÓN GENERAL
# ===============================================================
CHUNK_SIZE = 15000
DATA_PATH = "/opt/airflow/data/raw/diabetic_data.csv"
OUTPUT_DIR = "/opt/airflow/data/interim"
STATE_FILE = "/opt/airflow/data/state.json"


# ===============================================================
# FUNCIÓN PRINCIPAL
# ===============================================================
def ingest_csv_chunks():
    """
    Lee el CSV en partes de 15.000 filas y guarda un nuevo chunk en cada ejecución.
    Guarda el progreso en un archivo de estado para continuar desde donde se quedó.
    Si el último bloque tiene menos filas, igual se guarda.
    """

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Leer estado anterior
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        last_chunk = state.get("last_chunk", -1)
    else:
        last_chunk = -1

    # Calcular el siguiente bloque
    next_chunk = last_chunk + 1
    start_row = next_chunk * CHUNK_SIZE

    # Calcular total de filas (sin cabecera)
    total_rows = sum(1 for _ in open(DATA_PATH)) - 1
    if start_row >= total_rows:
        print("✅ No hay más filas por procesar. Ingesta completada totalmente.")
        return None

    print(f"Procesando chunk {next_chunk} desde la fila {start_row} (de {total_rows})")

    # Leer bloque (manteniendo encabezado)
    df_chunk = pd.read_csv(DATA_PATH, skiprows=range(1, start_row + 1), nrows=CHUNK_SIZE)

    if df_chunk.empty:
        print("No se encontraron más filas. Finalizando ingesta.")
        return None

    # Guardar chunk correctamente numerado
    output_file = os.path.join(OUTPUT_DIR, f"chunk_{next_chunk:03d}.csv")
    df_chunk.to_csv(output_file, index=False)
    print(f"Guardado {output_file} con {len(df_chunk)} registros.")

    # Actualizar archivo de estado
    with open(STATE_FILE, "w") as f:
        json.dump({"last_chunk": next_chunk}, f)

    print(f"Estado actualizado: último chunk = {next_chunk}")

    # Mostrar progreso
    remaining = total_rows - (start_row + len(df_chunk))
    if remaining > 0:
        print(f"Quedan {remaining} filas por procesar ({remaining / CHUNK_SIZE:.2f} chunks aprox).")
    else:
        print("Todos los datos fueron divididos correctamente en chunks.")


# ===============================================================
# EJECUCIÓN LOCAL (solo si se corre manualmente)
# ===============================================================
if __name__ == "__main__":
    ingest_csv_chunks()
