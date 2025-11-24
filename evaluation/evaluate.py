import sys
import asyncio
import platform

# --- FIX CRÍTICO PARA WINDOWS Y GRPC ---
# Esto debe ir ANTES de cualquier otra importación que use asyncio o grpc
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
# ---------------------------------------

import pandas as pd
import logging
import time
import requests
import os
import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluator")

load_dotenv("../.env")

# Cargar configuración
config_path = "../config/settings.yaml"
if not os.path.exists(config_path):
    # Fallback si se ejecuta desde dentro de evaluation/
    config_path = "config/settings.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Dataset de Prueba
TEST_DATA = [
    {"question": "Who is Monica Lewinsky?", "ground_truth": "American activist and former White House intern."},
    {"question": "What happened in Brazil in 1999?", "ground_truth": "Events related to Brazil in 1999."}
]


def run_eval():
    logger.info(f"--- Iniciando Evaluación con Juez: {config['llm']['model_name']} ---")

    # 1. Recopilación de respuestas (Fase Síncrona)
    full_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    logger.info("Generando respuestas del RAG (Orquestador)...")
    for item in TEST_DATA:
        q = item["question"]
        try:
            # Llamada al orquestador local
            res = requests.post("http://localhost:8000/rag", json={"question": q, "mode": "hybrid"}, timeout=20)

            if res.status_code == 200:
                resp = res.json()
                full_data["question"].append(q)
                full_data["answer"].append(resp["answer"])

                # Validación de contextos: debe ser lista de strings
                ctx = resp.get("context_used", [])
                if isinstance(ctx, list):
                    full_data["contexts"].append([str(c) for c in ctx])
                else:
                    full_data["contexts"].append([])

                full_data["ground_truth"].append(item["ground_truth"])
            else:
                logger.warning(f"Fallo en API para '{q}': {res.status_code}")
        except Exception as e:
            logger.error(f"Error de conexión con Orquestador: {e}")

    if not full_data["question"]:
        logger.error("No hay datos válidos para evaluar. Abortando.")
        return

    # 2. Configuración del Juez Ragas (Gemini)
    # Nota: Configuramos timeout alto y reintentos en el cliente de Google si es posible
    judge_llm = ChatGoogleGenerativeAI(
        model=config['llm']['model_name'],
        temperature=0,
        google_api_key='AIzaSyAAW0n_q49iGXZ8aBJ2A3edmZgr6msIC8c',
        timeout=60
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Ejecución por Lotes (Batching)
    logger.info("Iniciando evaluación (Mode: Safe Batching)...")

    batch_size = 1
    total_items = len(full_data["question"])
    dfs = []

    for i in range(0, total_items, batch_size):
        # Crear lote actual
        batch_data = {
            key: full_data[key][i: i + batch_size]
            for key in full_data
        }

        # Convertir a Dataset Ragas
        dataset_batch = Dataset.from_dict(batch_data)

        logger.info(f"Evaluando lote {i + 1}/{total_items}...")

        try:
            # Ejecutar evaluación
            results_batch = evaluate(
                dataset=dataset_batch,
                metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                llm=judge_llm,
                embeddings=embeddings,
                raise_exceptions=False
            )

            dfs.append(results_batch.to_pandas())

            # Pausa de seguridad para cuota
            if i + batch_size < total_items:
                logger.info("Pausando 35s para liberar cuota API...")
                time.sleep(35)

        except Exception as e:
            logger.error(f"Error crítico en lote {i}: {e}")

    # 4. Guardar Resultados
    if dfs:
        final_df = pd.concat(dfs, ignore_index=True)
        print("\n=== RESULTADOS CONSOLIDADOS ===")
        print(final_df.mean(numeric_only=True))

        final_df.to_csv("ragas_metrics.csv", index=False)
        logger.info("Resultados guardados en 'ragas_metrics.csv'.")
    else:
        logger.warning("La evaluación finalizó sin resultados.")


if __name__ == "__main__":
    run_eval()