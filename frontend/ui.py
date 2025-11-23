import streamlit as st
import requests
import logging

# Logger simple para frontend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Frontend")

st.set_page_config(page_title="MicroRAG-Flow", layout="wide")
st.title("🧬 MicroRAG-Flow: Gemini Powered")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Estrategia:", ("hybrid", "vector", "graph"))
    st.info("Backend en puerto 8000")

question = st.chat_input("Escribe tu pregunta...")

if question:
    st.chat_message("user").write(question)
    logger.info(f"Pregunta enviada: {question}")

    with st.spinner("Consultando Orquestador..."):
        try:
            res = requests.post(
                "http://localhost:8000/rag",
                json={"question": question, "mode": mode}
            )
            if res.status_code == 200:
                data = res.json()
                st.chat_message("assistant").write(data["answer"])

                with st.expander("Contexto Recuperado"):
                    for i, doc in enumerate(data.get("context_used", [])):
                        st.text(f"--- Doc {i + 1} ---\n{doc[:200]}...")

                logger.info("Respuesta renderizada correctamente.")
            else:
                st.error(f"Error del servidor: {res.text}")
                logger.error(f"Error HTTP: {res.status_code}")
        except Exception as e:
            st.error(f"No se pudo conectar al backend: {e}")
            logger.error(f"Excepción de conexión: {e}")