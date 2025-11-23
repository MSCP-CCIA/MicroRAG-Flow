import yaml
import os
import spacy
import logging
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

# Cargar config
with open("../../config/settings.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="Neo4j Graph Service")

# 1. Cargar NLP
try:
    logger.info("Cargando modelo Spacy...")
    # Deshabilitamos componentes innecesarios para un endpoint rápido
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
except Exception as e:
    logger.warning(f"Modelo no encontrado ({e}). Descargando...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# 2. Preparar Credenciales
uri = os.getenv("NEO4J_URI", config['neo4j']['uri'])
if uri.startswith("tcp://"):
    uri = uri.replace("tcp://", "bolt://")

user = config['neo4j']['user']
password = os.getenv("NEO4J_PASSWORD")

logger.info(f"Configurando conexión Neo4j -> URI: {uri} | User: {user}")

# 3. Conexión Robusta
driver = None
try:
    driver = GraphDatabase.driver(
        uri,
        auth=basic_auth(user, password)
    )
    driver.verify_connectivity()
    logger.info(" Conectado exitosamente a Neo4j")
except Exception as e:
    logger.critical(f"Error fatal conectando a Neo4j: {e}", exc_info=True)
    # No levantamos excepción aquí para permitir que la API arranque y reporte el error luego


class QueryRequest(BaseModel):
    query: str
    limit: int = 3


@app.post("/search")
def search_graph(request: QueryRequest):
    if not driver:
        raise HTTPException(status_code=503, detail="Conexión a Neo4j no establecida")

    # A. Extraer entidades
    doc = nlp(request.query)
    entities = [ent.text for ent in doc.ents]

    if not entities:
        logger.info(f"No se encontraron entidades en la query: '{request.query}'")
        return {"results": [], "message": "No entities found"}

    logger.info(f"Entidades extraídas: {entities}")

    # B. Consulta Cypher (MODIFICADA para devolver metadata)
    cypher_query = """
    MATCH (e:Entity)<-[:MENTIONS]-(d:Document)
    WHERE e.name IN $names
    RETURN DISTINCT d.id AS id, 
                    d.type AS type,
                    d.url AS url,
                    d.path AS path, 
                    d.text AS content
    LIMIT $limit
    """

    try:
        with driver.session() as session:
            result = session.run(cypher_query, names=entities, limit=request.limit)

            # C. Procesar y estructurar la respuesta con metadata
            records_with_metadata = []
            for record in result:
                # El campo 'path' es clave para las imágenes; lo incluimos si existe
                path_value = record["path"] if record["path"] else None

                records_with_metadata.append({
                    "content": record["content"],
                    "metadata": {
                        "id": record["id"],
                        "type": record["type"],
                        "url": record["url"],
                        "path": path_value
                    }
                })

        logger.info(f"Recuperados {len(records_with_metadata)} contextos del grafo.")
        return {"results": records_with_metadata}

    except Exception as e:
        logger.error(f"Error ejecutando consulta Cypher: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar: uvicorn main:app --port 8002