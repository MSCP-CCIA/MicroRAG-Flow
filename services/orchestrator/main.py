import os
import yaml
import requests
import logging
import boto3
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# --- CONFIGURACIÓN PRINCIPAL ---
# Pon esto en False cuando quieras usar S3 en producción
USE_LOCAL_IMAGES = True
# Ajusta esta ruta a tu carpeta de imágenes local real
LOCAL_IMAGE_PATH = "../../data/images/final_dataset_images/final_dataset_images"

# --- Logging y Configuración ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Orchestrator")

load_dotenv("../../.env")
CONFIG_PATH = "../../config/settings.yaml"

with open(CONFIG_PATH, "r", encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = FastAPI(title="RAG Orchestrator")

# --- Clientes (LLM y S3) ---
llm = ChatGoogleGenerativeAI(
    model=config['llm']['model_name'],
    temperature=config['llm']['temperature'],
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

try:
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION")
    )
    BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
except Exception:
    s3_client = None
    BUCKET_NAME = None

# --- MEMORIA VOLÁTIL (RAM) ---
CHAT_HISTORY = {}


# --- MODELO DE DATOS ---
class RagRequest(BaseModel):
    question: str
    mode: str = "hybrid"
    session_id: str = "default"


# --- PROMPT PARA REESCRITURA DE PREGUNTA ---
contextualize_q_system_prompt = """Given a chat history and the user's last question 
(which may refer to the context of the history), formulate an independent question 
that can be understood without the history. Do NOT answer the question, just rephrase it if necessary 
or return it as is if it is already explicit."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


# --- FUNCIONES DE IMAGEN ---
def get_image_from_s3(path_key: str) -> str:
    if not s3_client or not BUCKET_NAME: return None
    try:
        # Limpieza simple por si viene basura de Mac
        if path_key.startswith("._"): path_key = path_key[2:]
        if not path_key.startswith("images/"): path_key = f"images/{path_key}"

        response = s3_client.get_object(Bucket=BUCKET_NAME, Key=path_key)
        return base64.b64encode(response['Body'].read()).decode('utf-8')
    except Exception as e:
        logger.error(f"S3 Error: {e}")
        return None


def get_image_from_local(path_key: str) -> str:
    if not path_key: return None
    try:
        filename = os.path.basename(path_key)
        if filename.startswith("._"): filename = filename[2:]

        full_path = os.path.abspath(os.path.join(LOCAL_IMAGE_PATH, filename))

        if not os.path.exists(full_path):
            logger.warning(f"Imagen no encontrada localmente: {full_path}")
            return None

        with open(full_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Local Image Error: {e}")
        return None


def get_image(path_key: str) -> str:
    """Selector inteligente de fuente de imágenes"""
    if USE_LOCAL_IMAGES:
        return get_image_from_local(path_key)
    return get_image_from_s3(path_key)


# --- PIPELINE PRINCIPAL ---
@app.post("/rag")
async def rag_pipeline(req: RagRequest):
    # 1. Gestión de Sesión
    session_id = req.session_id
    history = CHAT_HISTORY.get(session_id, [])

    # 2. Reescritura de Pregunta (Query Rewriting)
    query_to_search = req.question
    if history:
        try:
            logger.info("Reformulando pregunta...")
            query_to_search = contextualize_q_chain.invoke({
                "chat_history": history,
                "question": req.question
            })
            logger.info(f"Original: '{req.question}' | Buscada: '{query_to_search}'")
        except Exception as e:
            logger.error(f"Error reformulando: {e}")

    # 3. Recuperación (Retrieval)
    urls = config['microservices_urls']
    raw_results = []

    # Búsqueda Vectorial
    if req.mode in ["vector", "hybrid"]:
        try:
            res = requests.post(urls['chroma_api'], json={"query": query_to_search, "k": 10}, timeout=5)
            if res.status_code == 200: raw_results.extend(res.json().get("results", []))
        except Exception:
            pass

    # Búsqueda en Grafo
    if req.mode in ["graph", "hybrid"]:
        try:
            res = requests.post(urls['neo4j_api'], json={"query": query_to_search, "limit": 10}, timeout=5)
            if res.status_code == 200: raw_results.extend(res.json().get("results", []))
        except Exception:
            pass

    # 4. Procesamiento de Resultados
    seen_ids = set()
    text_context = []
    images_b64 = []
    display_context = []

    for doc in raw_results:
        meta = doc.get("metadata", {})
        content = doc.get("page_content") or doc.get("content") or ""
        did = meta.get("id")

        if did in seen_ids: continue
        if did: seen_ids.add(did)

        # Manejo de Imágenes
        if meta.get("type") == "image":
            path = meta.get("path")
            img_data = get_image(path)  # Usa la función selectora

            if img_data:
                images_b64.append(img_data)
                text_context.append(f"[IMAGEN ADJUNTA]: {content}")
            else:
                text_context.append(f"[IMAGEN NO DISPONIBLE]: {content}")
        else:
            text_context.append(f"[TEXTO]: {content}")
            display_context.append(content)

    # 5. Generación con Gemini
    formatted_context = "\n".join(text_context)

    prompt_text = f"""Answer the user's question using the context provided (text and images).    If you don't know the answer, say so honestly..

        Contexto:
        {formatted_context}

        Question: {query_to_search}
        """

    message_parts = [{"type": "text", "text": prompt_text}]
    for img in images_b64:
        message_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

    try:
        ai_msg = llm.invoke([HumanMessage(content=message_parts)])
        answer = ai_msg.content
    except Exception as e:
        logger.error(f"Error Gemini: {e}")
        answer = "Lo siento, tuve un problema generando la respuesta."

    # 6. Actualizar Memoria
    if session_id not in CHAT_HISTORY: CHAT_HISTORY[session_id] = []
    CHAT_HISTORY[session_id].extend([
        HumanMessage(content=req.question),
        AIMessage(content=answer)
    ])
    # Limitar historial (Window Buffer)
    if len(CHAT_HISTORY[session_id]) > 10:
        CHAT_HISTORY[session_id] = CHAT_HISTORY[session_id][-10:]

    return {
        "answer": answer,
        "context_used": display_context,
        "images": images_b64
    }

# Ejecutar: uvicorn main:app --port 8000