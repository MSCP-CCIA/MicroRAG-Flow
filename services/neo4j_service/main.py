import yaml
import os
import spacy
import logging
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# --- Configuración de Logger ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Neo4jService")

load_dotenv("../../.env")

with open("../../config/settings.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Neo4j Graph Service")

# 1. Cargar NLP
try:
    logger.info("Cargando modelo Spacy...")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
except Exception as e:
    logger.warning(f"Modelo no encontrado ({e}). Descargando...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. Conexión Neo4j
uri = os.getenv("NEO4J_URI", config['neo4j']['uri'])
if uri.startswith("tcp://"):
    uri = uri.replace("tcp://", "bolt://")
user = config['neo4j']['user']
password = os.getenv("NEO4J_PASSWORD")

driver = None
try:
    driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
    driver.verify_connectivity()
    logger.info("Conectado exitosamente a Neo4j")
except Exception as e:
    logger.critical(f"Error fatal conectando a Neo4j: {e}", exc_info=True)


class QueryRequest(BaseModel):
    query: str
    limit: int = 10


def clean_entity(text: str) -> str:
    """Limpia prefijos comunes y caracteres raros"""
    text = re.sub(r'^[\W_]+', '', text)
    stopwords = ["de ", "del ", "el ", "la ", "los ", "las ", "of ", "the ", "about ", "image ", "picture "]
    text_lower = text.lower()
    for sw in stopwords:
        if text_lower.startswith(sw):
            text = text[len(sw):]
            break
    return text.strip()


@app.post("/search")
def search_graph(request: QueryRequest):
    if not driver:
        raise HTTPException(status_code=503, detail="Conexión a Neo4j no establecida")

    # A. NLP
    doc = nlp(request.query)
    raw_entities = [ent.text for ent in doc.ents]
    cleaned_entities = []

    for ent in raw_entities:
        clean = clean_entity(ent)
        if len(clean) > 2:
            cleaned_entities.append(clean)

    cleaned_entities = list(set(cleaned_entities))

    if not cleaned_entities:
        # FALLBACK: Si Spacy no encuentra nada (ej: "Foto Rick"), intentamos usar la query entera limpia
        fallback = clean_entity(request.query)
        if len(fallback) > 3:
            cleaned_entities = [fallback]
            logger.info(f"Spacy falló, usando fallback manual: {cleaned_entities}")
        else:
            return {"results": [], "message": "No valid entities found"}

    logger.info(f"Buscando: {cleaned_entities}")

    # B. Consulta Cypher MEJORADA (Entidad O Título)
    cypher_query = """
    MATCH (d:Document)
    WHERE 
        // 1. Coincidencia por relación NLP (MENTIONS)
        EXISTS {
            MATCH (d)-[:MENTIONS]->(e:Entity)
            WHERE toLower(e.name) IN [name IN $names | toLower(name)]
        }
        OR
        // 2. Coincidencia directa por TÍTULO (Fuzzy Search)
        ANY(name IN $names WHERE toLower(d.title) CONTAINS toLower(name))

    RETURN DISTINCT d.id AS id, 
                    d.type AS type,
                    d.url AS url,
                    d.path AS path, 
                    d.text AS content,
                    d.title AS title
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(cypher_query, names=cleaned_entities, limit=request.limit)

            records = []
            for record in result:
                records.append({
                    "content": record["content"],
                    "metadata": {
                        "id": record["id"],
                        "type": record["type"],
                        "url": record["url"],
                        "path": record["path"],
                        "title": record["title"]
                    }
                })

        logger.info(f"Recuperados {len(records)} docs (Grafo + Título).")
        return {"results": records}

    except Exception as e:
        logger.error(f"Error Cypher: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))