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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

load_dotenv("../../.env")
with open("../../config/settings.yaml", "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="RAG Orchestrator")

# Configurar Gemini
try:
    logger.info(f"Iniciando LLM: {config['llm']['model_name']}")
    llm = ChatGoogleGenerativeAI(
        model=config['llm']['model_name'],
        temperature=config['llm']['temperature'],
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
except Exception as e:
    logger.critical(f"Error configurando Gemini: {e}")

prompt = ChatPromptTemplate.from_template("""
Eres un asistente experto. Responde basándote SOLO en el contexto siguiente:

Contexto:
{context}

Pregunta: {question}
""")


class RagRequest(BaseModel):
    question: str
    mode: str = "hybrid"


@app.post("/rag")
def rag_pipeline(req: RagRequest):
    logger.info(f"Solicitud recibida: {req.question} [Modo: {req.mode}]")
    context_docs = []
    urls = config['microservices_urls']

    # 1. Vector Service
    if req.mode in ["vector", "hybrid"]:
        try:
            res = requests.post(urls['chroma_api'], json={"query": req.question, "k": 4}, timeout=5)
            if res.status_code == 200:
                docs = res.json().get("results", [])
                context_docs.extend(docs)
                logger.info(f"Vector Service: {len(docs)} docs recuperados")
            else:
                logger.warning(f"Vector Service Error: {res.status_code}")
        except Exception as e:
            logger.error(f"Vector Service inalcanzable: {e}")

    # 2. Graph Service
    if req.mode in ["graph", "hybrid"]:
        try:
            res = requests.post(urls['neo4j_api'], json={"query": req.question, "limit": 3}, timeout=5)
            if res.status_code == 200:
                docs = res.json().get("results", [])
                context_docs.extend(docs)
                logger.info(f"Graph Service: {len(docs)} docs recuperados")
            else:
                logger.warning(f"Graph Service Error: {res.status_code}")
        except Exception as e:
            logger.error(f"Graph Service inalcanzable: {e}")

    # 3. Generación
    unique_context = list(set(context_docs))
    context_str = "\n\n".join(unique_context)

    if not context_str:
        logger.warning("Sin contexto útil recuperado.")
        return {"answer": "No encontré información relevante en las bases de datos.", "context_used": []}

    try:
        logger.info("Generando respuesta con Gemini...")
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"context": context_str, "question": req.question})
        logger.info("Respuesta generada OK.")
        return {"answer": answer, "context_used": unique_context}
    except Exception as e:
        logger.error(f"Fallo en Gemini: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Ejecutar: uvicorn main:app --port 8000