import os
import yaml
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

# --- Carga de Configuración ---
load_dotenv("../../.env")
CONFIG_PATH = "../../config/settings.yaml"

with open(CONFIG_PATH, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="RAG Orchestrator")

# --- Configuración del LLM (Gemini) ---
try:
    logger.info(f"Iniciando LLM: {config['llm']['model_name']}")
    llm = ChatGoogleGenerativeAI(
        model=config['llm']['model_name'],
        temperature=config['llm']['temperature'],
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    logger.critical(f"Error configurando Gemini: {e}")
    raise e

# --- Prompt Template ---
# Se mejora el prompt para que cite las fuentes si están disponibles en el contexto
prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto y preciso. Responde a la pregunta basándote ÚNICAMENTE en el contexto proporcionado a continuación.
Si la información no está en el contexto, indica que no lo sabes.

Contexto Recuperado:
{context}

Pregunta: {question}

Respuesta:
""")


class RagRequest(BaseModel):
    question: str
    mode: str = "hybrid"  # Opciones: vector, graph, hybrid


def format_doc(content: str, metadata: dict, source_type: str) -> str:
    """Formatea un documento para inyectarlo en el contexto del LLM"""
    title = metadata.get("title", "Desconocido")
    doc_type = metadata.get("type", "text")
    return f"[{source_type.upper()} - {doc_type}] Título: {title}\nContenido: {content}"


@app.post("/rag")
def rag_pipeline(req: RagRequest):
    logger.info(f"Solicitud recibida: '{req.question}' [Modo: {req.mode}]")

    urls = config['microservices_urls']
    context_entries = []  # Lista de cadenas de texto formateadas
    raw_docs = []  # Para devolver al frontend si es necesario

    # ---------------------------------------------------------
    # 1. VECTOR SERVICE (ChromaDB)
    # ---------------------------------------------------------
    if req.mode in ["vector", "hybrid"]:
        try:
            res = requests.post(urls['chroma_api'], json={"query": req.question, "k": 4}, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                logger.info(f"Vector Service: {len(results)} docs recuperados")

                for doc in results:
                    # Chroma devuelve objetos serializados con 'page_content' y 'metadata'
                    content = doc.get("page_content", "")
                    metadata = doc.get("metadata", {})

                    if content:
                        formatted = format_doc(content, metadata, "VECTOR")
                        context_entries.append(formatted)
                        raw_docs.append(content)
            else:
                logger.warning(f"Vector Service Error: {res.status_code} - {res.text}")
        except Exception as e:
            logger.error(f"Vector Service inalcanzable: {e}")

    # ---------------------------------------------------------
    # 2. GRAPH SERVICE (Neo4j)
    # ---------------------------------------------------------
    if req.mode in ["graph", "hybrid"]:
        try:
            res = requests.post(urls['neo4j_api'], json={"query": req.question, "limit": 3}, timeout=10)
            if res.status_code == 200:
                results = res.json().get("results", [])
                logger.info(f"Graph Service: {len(results)} docs recuperados")

                for item in results:
                    # Neo4j Service devuelve estructura custom: {'content': ..., 'metadata': ...}
                    content = item.get("content", "")
                    metadata = item.get("metadata", {})

                    if content:
                        formatted = format_doc(content, metadata, "GRAFO")
                        context_entries.append(formatted)
                        raw_docs.append(content)
            else:
                logger.warning(f"Graph Service Error: {res.status_code} - {res.text}")
        except Exception as e:
            logger.error(f"Graph Service inalcanzable: {e}")

    # ---------------------------------------------------------
    # 3. PROCESAMIENTO Y GENERACIÓN
    # ---------------------------------------------------------

    # Deduplicación simple basada en el contenido exacto del string formateado
    unique_context = list(set(context_entries))

    if not unique_context:
        logger.warning("Sin contexto útil recuperado de ningún servicio.")
        return {
            "answer": "No encontré información relevante en las bases de datos (Vectorial o Grafo) para responder tu pregunta.",
            "context_used": []
        }

    context_str = "\n\n".join(unique_context)

    try:
        logger.info("Generando respuesta con Gemini...")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": req.question})
        logger.info("Respuesta generada exitosamente.")

        return {
            "answer": answer,
            "context_used": list(set(raw_docs))  # Devolvemos raw docs para mostrar en UI
        }
    except Exception as e:
        logger.error(f"Fallo en generación con Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generando respuesta: {str(e)}")

# Ejecutar: uvicorn main:app --port 8000

# Ejecutar: uvicorn main:app --port 8000