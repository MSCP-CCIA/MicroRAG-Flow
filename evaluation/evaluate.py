import sys
import asyncio
import platform
import os
import time
import yaml
import pandas as pd
import logging
import requests
import json  # <--- NUEVO IMPORT
import random  # <--- NUEVO IMPORT
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- FIX WINDOWS ---
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Logging limpio (sin emojis)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluator")

# Cargar entorno y config
load_dotenv("../.env")

config_path = "../config/settings.yaml"
if not os.path.exists(config_path):
    config_path = "config/settings.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

OUTPUT_FILE = "ragas_metrics_vector.csv"
DATASET_FILE = "test_dataset_synthetic.json"  # <--- Archivo fuente

# --- CARGA Y SELECCIÓN ALEATORIA DE DATOS ---
TEST_DATA = []

if os.path.exists(DATASET_FILE):
    try:
        with open(DATASET_FILE, "r", encoding="utf-8") as f:
            full_dataset = json.load(f)

        if isinstance(full_dataset, list) and len(full_dataset) > 0:
            # Seleccionar 10 aleatorios (o todos si hay menos de 10)
            sample_size = len(full_dataset)
            TEST_DATA = random.sample(full_dataset, sample_size)
            logger.info(f"[INIT] Se seleccionaron {len(TEST_DATA)} preguntas aleatorias de '{DATASET_FILE}'.")
        else:
            logger.warning(f"[WARN] El archivo '{DATASET_FILE}' no contiene una lista válida.")
    except Exception as e:
        logger.error(f"[ERROR] No se pudo cargar el dataset: {e}")
else:
    logger.warning(f"[WARN] No se encontró el archivo '{DATASET_FILE}'. Se usará lista vacía.")


# --- CLASE PARCHE PARA CORREGIR JSON DE MISTRAL ---
class JSONFixMistralAI(ChatMistralAI):
    """
    Wrapper que corrige el error comun de Mistral donde escapa comillas simples (\')
    dentro del JSON, lo cual rompe el parser de Ragas.
    """

    def _fix_json(self, text: str) -> str:
        # Reemplaza la secuencia ilegal \' por '
        return text.replace("\\'", "'")

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        res = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in res.generations:
            clean = self._fix_json(gen.text)
            gen.text = clean
            if gen.message:
                gen.message.content = clean
        return res

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        res = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for gen in res.generations:
            clean = self._fix_json(gen.text)
            gen.text = clean
            if gen.message:
                gen.message.content = clean
        return res


# --------------------------------------------------

def load_processed_questions():
    if os.path.exists(OUTPUT_FILE):
        try:
            df = pd.read_csv(OUTPUT_FILE)
            return set(df["question"].tolist())
        except Exception:
            return set()
    return set()


def run_eval():
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if not mistral_api_key:
        logger.error("[ERROR] MISTRAL_API_KEY no encontrada en .env")
        return

    logger.info("--- Iniciando Evaluacion Resiliente con Juez Mistral ---")

    processed_set = load_processed_questions()

    # Filtramos de los 10 seleccionados aleatoriamente, cuáles ya fueron procesados antes
    questions_to_process = [item for item in TEST_DATA if item["question"] not in processed_set]

    if not questions_to_process:
        logger.info("[LISTO] Las preguntas seleccionadas ya han sido evaluadas previamente. Terminado.")
        return

    logger.info(
        f"Muestra aleatoria: {len(TEST_DATA)} | Ya procesadas (histórico): {len(processed_set)} | A procesar ahora: {len(questions_to_process)}")

    # Generar respuestas del RAG
    full_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    logger.info("Generando respuestas nuevas del RAG...")
    for item in questions_to_process:
        q = item["question"]
        try:
            res = requests.post(
                "http://localhost:8000/rag",
                json={"question": q, "mode": "vector"},
                timeout=120
            )

            if res.status_code == 200:
                resp = res.json()
                full_data["question"].append(q)
                full_data["answer"].append(resp["answer"])

                ctx = resp.get("context_used", [])
                full_data["contexts"].append([str(c) for c in ctx] if isinstance(ctx, list) else [])

                full_data["ground_truth"].append(item["ground_truth"])
            else:
                logger.warning(f"[WARN] Fallo API RAG para '{q}': {res.status_code}")
        except Exception as e:
            logger.error(f"[ERROR] Error conectando a Orquestador: {e}")

    if not full_data["question"]:
        logger.warning("No se generaron respuestas validas para evaluar.")
        return

    # --- CONFIGURACION DEL JUEZ CON EL PARCHE ---
    # Usamos nuestra clase personalizada JSONFixMistralAI en lugar de ChatMistralAI
    judge_llm = JSONFixMistralAI(
        model="mistral-large-latest",
        temperature=0,
        mistral_api_key=mistral_api_key,
        timeout=120,
        max_retries=3
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Evaluacion Batch
    batch_size = 1
    total_items = len(full_data["question"])

    for i in range(0, total_items, batch_size):
        batch_data = {key: full_data[key][i: i + batch_size] for key in full_data}
        dataset_batch = Dataset.from_dict(batch_data)
        current_q = batch_data["question"][0]

        logger.info(f"[EVAL] Evaluando: '{current_q}' ({i + 1}/{total_items})...")

        max_retries = 5
        for attempt in range(max_retries):
            try:
                results_batch = evaluate(
                    dataset=dataset_batch,
                    metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
                    llm=judge_llm,
                    embeddings=embeddings,
                    raise_exceptions=True
                )

                df_batch = results_batch.to_pandas()

                use_header = not os.path.exists(OUTPUT_FILE)
                df_batch.to_csv(OUTPUT_FILE, mode='a', header=use_header, index=False)

                logger.info(f"[GUARDADO] CSV actualizado. Pausando 2s...")
                time.sleep(5)
                break

            except Exception as e:
                wait_time = (attempt + 1) * 10
                logger.error(f"[ERROR] Fallo en evaluacion (Intento {attempt + 1}): {e}")
                logger.warning(f"[ESPERA] Esperando {wait_time} segundos...")
                time.sleep(wait_time)

                if attempt == max_retries - 1:
                    logger.error(f"[OMITIDO] Se salto la pregunta '{current_q}' tras multiples fallos.")

    logger.info("=== Evaluacion completada. Revisa 'ragas_metrics.csv' ===")


if __name__ == "__main__":
    run_eval()