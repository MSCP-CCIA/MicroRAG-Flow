import os
import yaml
import logging
import chromadb
import re
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb.config import Settings
from dotenv import load_dotenv

# --- Configuración de Logs ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ChromaService")

load_dotenv("../../.env")
with open("../../config/settings.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Chroma Vector Service")

# --- 1. CARGAR SPACY ---
try:
    logger.info("Cargando modelo Spacy...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
except Exception as e:
    logger.warning(f"Modelo no encontrado ({e}). Descargando...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- 2. Inicializar Embeddings y Chroma ---
embedding_fn = HuggingFaceEmbeddings(
    model_name=config['embeddings']['model_name'],
    model_kwargs={'device': config['embeddings']['device']},
    encode_kwargs={'normalize_embeddings': True}
)

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


# --- HERRAMIENTAS DE LIMPIEZA ---
def clean_entity(text: str) -> str:
    """Limpia prefijos comunes y caracteres raros"""
    text = re.sub(r'^[\W_]+', '', text)
    stopwords = ["de ", "del ", "el ", "la ", "los ", "las ", "of ", "the ", "about ", "image ", "picture ", "show ",
                 "me "]
    text_lower = text.lower()
    for sw in stopwords:
        if text_lower.startswith(sw):
            text = text[len(sw):]
            break
    return text.strip()


@app.post("/search")
def search_vectors(request: SearchRequest):
    if not vector_db:
        raise HTTPException(status_code=503, detail="ChromaDB no disponible")

    try:
        # A. Extracción de Entidades
        doc = nlp(request.query)
        raw_entities = [ent.text for ent in doc.ents]
        cleaned_entities = []

        for ent in raw_entities:
            clean = clean_entity(ent)
            if len(clean) > 2:
                cleaned_entities.append(clean)

        cleaned_entities = list(set(cleaned_entities))

        if not cleaned_entities:
            fallback = clean_entity(request.query)
            if len(fallback) > 3:
                cleaned_entities = [fallback]
                logger.info(f"⚠️ Spacy falló, usando fallback manual: {cleaned_entities}")

        logger.info(f"🔍 Buscando en Vectores para entidades: {cleaned_entities}")

        # B. Búsqueda Vectorial "Ampliada" (CORREGIDO: MAYOR ALCANCE)
        # Usamos max(50, k*10) para asegurar que traemos suficientes candidatos
        fetch_k = max(50, request.k * 10)

        results_with_score = vector_db.similarity_search_with_score(request.query, k=fetch_k)

        final_results = []

        # Threshold base
        SCORE_THRESHOLD = 1.4

        for doc, score in results_with_score:
            title = doc.metadata.get('title', 'Sin título')
            content = doc.page_content
            doc_type = doc.metadata.get('type', 'unknown')

            doc_text_full = (str(title) + " " + str(content)).lower()

            # C. VERIFICACIÓN: ¿Contiene alguna de las entidades buscadas?
            has_keyword_match = False
            if cleaned_entities:
                for entity in cleaned_entities:
                    # Tokenización simple para evitar falsos positivos parciales
                    if entity.lower() in doc_text_full:
                        has_keyword_match = True
                        break
            else:
                has_keyword_match = True

                # REGLAS DE DECISIÓN:
            if score < 0.8:
                final_results.append(doc)
                logger.info(f"✅ Aceptado por Vector Puro (Score: {score:.3f}): {title}")

            elif score < SCORE_THRESHOLD and has_keyword_match:
                final_results.append(doc)
                logger.info(f"✅ Aceptado Híbrido (Score: {score:.3f} + Match): {title}")

            elif doc_type == 'image' and has_keyword_match:
                final_results.append(doc)
                logger.info(f"📸 Imagen Rescatada por Título (Score: {score:.3f}): {title}")

            else:
                # Logueamos solo los primeros descartes para no saturar consola
                logger.info(f"🗑 Descartado (Score: {score:.3f} | Match: {has_keyword_match}): {title}")
                pass

            if len(final_results) >= request.k:
                break

        logger.info(f"Enviando {len(final_results)} documentos finales.")
        return {"results": final_results}

    except Exception as e:
        logger.error(f"Error en búsqueda vectorial: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))