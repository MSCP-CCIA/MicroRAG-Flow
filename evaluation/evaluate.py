import pandas as pd
import logging
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import requests
import os
import yaml
from dotenv import load_dotenv

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Evaluator")

load_dotenv("../.env")

# Cargar configuración para leer el modelo correcto
with open("../config/settings.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dataset de Prueba (Ejemplo)
TEST_DATA = [
    {"question": "Who is Monica Lewinsky?", "ground_truth": "American activist and former White House intern."},
    {"question": "What happened in Brazil in 1999?", "ground_truth": "Events related to Brazil in 1999."}
]


def run_eval():
    logger.info(f"--- Iniciando Evaluación con Juez: {config['llm']['model_name']} ---")
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    # 1. Generación (Cliente HTTP al Orquestador)
    for item in TEST_DATA:
        q = item["question"]
        try:
            # El orquestador usará Gemini 2.5 Flash según el YAML
            res = requests.post("http://localhost:8000/rag", json={"question": q, "mode": "hybrid"})

            if res.status_code == 200:
                resp = res.json()
                data["question"].append(q)
                data["answer"].append(resp["answer"])
                data["contexts"].append(resp.get("context_used", []))
                data["ground_truth"].append(item["ground_truth"])
            else:
                logger.warning(f"Fallo en API para: {q}")
        except Exception as e:
            logger.error(f"Error de conexión: {e}")

    if not data["question"]:
        logger.error("No hay datos para evaluar.")
        return

    # 2. Configuración del Juez Ragas
    # Usamos Gemini 2.5 Flash explícitamente leyendo del config
    judge_llm = ChatGoogleGenerativeAI(
        model=config['llm']['model_name'],  # Aquí tomará 'gemini-2.5-flash'
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Ejecución
    dataset = Dataset.from_dict(data)
    results = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=judge_llm,
        embeddings=embeddings
    )

    logger.info(f"\n=== RESULTADOS (Gemini 2.5 Flash) ===\n{results}")
    results.to_pandas().to_csv("ragas_metrics.csv", index=False)
    logger.info("Reporte guardado.")


if __name__ == "__main__":
    run_eval()