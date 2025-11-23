import os
import yaml
import logging
import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChromaService")

load_dotenv("../../.env")
with open("../../config/settings.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Chroma Vector Service")

# Inicializar Embeddings
logger.info(f"Cargando modelo embeddings: {config['embeddings']['model_name']}")
embedding_fn = HuggingFaceEmbeddings(
    model_name=config['embeddings']['model_name'],
    model_kwargs={'device': config['embeddings']['device']},
    encode_kwargs={'normalize_embeddings': True}
)

# Conexión a EC2
vector_db = None
try:
    host = os.getenv("CHROMA_HOST", config['chroma']['host'])
    port = int(os.getenv("CHROMA_PORT", config['chroma']['port']))
    logger.info(f"Conectando a ChromaDB Remoto en {host}:{port}...")

    http_client = chromadb.HttpClient(
        host=host,
        port=port,
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    vector_db = Chroma(
        client=http_client,
        collection_name=config['chroma']['collection_name'],
        embedding_function=embedding_fn
    )
    logger.info("Conectado a ChromaDB")
except Exception as e:
    logger.critical(f"Fallo conexión Chroma: {e}", exc_info=True)


class SearchRequest(BaseModel):
    query: str
    k: int = 4


@app.post("/search")
def search_vectors(request: SearchRequest):
    if not vector_db:
        raise HTTPException(status_code=503, detail="ChromaDB no disponible")

    try:
        logger.info(f"Buscando vectores para: '{request.query}'")
        results = vector_db.similarity_search(request.query, k=request.k)
        logger.info(f"Encontrados {len(results)} documentos.")
        return {"results": [doc.page_content for doc in results]}
    except Exception as e:
        logger.error(f"Error en búsqueda vectorial: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar: uvicorn main:app --port 8001