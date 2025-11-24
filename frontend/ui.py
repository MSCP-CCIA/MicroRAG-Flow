import streamlit as st
import requests
import logging
import uuid  # <--- 1. IMPORTAR UUID

# Logger simple para frontend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Frontend")

st.set_page_config(page_title="MicroRAG-Flow", layout="wide")
st.title("🧬 MicroRAG-Flow: Gemini Powered")

# --- 2. GESTIÓN DE SESIÓN ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())  # Generar ID único
    logger.info(f"Nueva sesión iniciada: {st.session_state.session_id}")

with st.sidebar:
    st.header("Configuración")
    mode = st.radio("Estrategia:", ("hybrid", "vector", "graph"))
    st.info(f"Session ID: {st.session_state.session_id[:8]}...")  # Visualizar ID (opcional)

    # Botón para limpiar memoria
    if st.button("Borrar Historial"):
        st.session_state.messages = []
        # Aquí podrías llamar a un endpoint del backend para borrar la memoria del servidor también
        st.rerun()

question = st.chat_input("Escribe tu pregunta...")

# Inicializar historial visual en Streamlit si no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial visual
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:
    # Agregar pregunta al historial visual
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    logger.info(f"Pregunta enviada: {question}")

    with st.spinner("Consultando Orquestador Multimodal..."):
        try:
            # --- 3. ENVIAR SESSION_ID AL BACKEND ---
            res = requests.post(
                "http://localhost:8000/rag",
                json={
                    "question": question,
                    "mode": mode,
                    "session_id": st.session_state.session_id  # <--- NUEVO CAMPO
                }
            )

            if res.status_code == 200:
                data = res.json()
                answer = data["answer"]

                # Agregar respuesta al historial visual
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.write(answer)

                # Mostrar Contexto e Imágenes (Tu código actual)
                images = data.get("images", [])
                if images:
                    with st.expander(f"Imágenes ({len(images)})", expanded=True):
                        cols = st.columns(min(len(images), 3))
                        for idx, img_b64 in enumerate(images):
                            import base64

                            img_bytes = base64.b64decode(img_b64)
                            cols[idx % 3].image(img_bytes, caption=f"Img {idx + 1}", width='stretch')

                with st.expander("Contexto de Texto"):
                    for i, doc in enumerate(data.get("context_used", [])):
                        st.text(f"--- Doc {i + 1} ---\n{doc[:200]}...")

            else:
                st.error(f"Error del servidor: {res.text}")
        except Exception as e:
            st.error(f"No se pudo conectar al backend: {e}")