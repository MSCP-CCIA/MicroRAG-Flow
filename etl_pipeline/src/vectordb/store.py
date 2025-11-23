import torch
import chromadb  # <--- Nueva importación necesaria
from chromadb.config import Settings  # <--- Nueva importación
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.utils.logger import setup_logger

logger = setup_logger("VectorDB")


class VectorStoreService:
    def __init__(self, config: Dict):
        self.chroma_cfg = config['chroma']  # Leemos la nueva sección del yaml
        self.emb_config = config['embeddings']

        # Configuración de Dispositivo (Igual que antes)
        device = self.emb_config['device']
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, cambiando a CPU")
            device = "cpu"

        logger.info(f"Iniciando modelo de embeddings en {device}...")
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=self.emb_config['model_name'],
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': self.emb_config['batch_size']}
        )

    def ingest_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No hay documentos para ingerir.")
            return

        logger.info(f"Preparando ingesta de {len(documents)} documentos...")
        batch_size = 5000

        # --- LÓGICA DE CONEXIÓN HÍBRIDA ---
        if self.chroma_cfg['mode'] == 'http':
            # MODO AWS / REMOTO
            logger.info(f"Conectando a ChromaDB Remoto en {self.chroma_cfg['host']}:{self.chroma_cfg['port']}...")
            try:
                # Cliente HTTP para conectar a la EC2
                http_client = chromadb.HttpClient(
                    host=self.chroma_cfg['host'],
                    port=self.chroma_cfg['port'],
                    settings=Settings(allow_reset=True, anonymized_telemetry=False)
                )

                # Inicializamos LangChain Chroma con el cliente remoto
                vector_db = Chroma(
                    client=http_client,
                    collection_name=self.chroma_cfg['collection_name'],
                    embedding_function=self.embedding_fn
                )
            except Exception as e:
                logger.error(f" Error conectando a Chroma Remoto: {e}")
                raise e
        else:
            # MODO LOCAL (Legacy)
            logger.info("Usando persistencia local en disco...")
            vector_db = Chroma(
                persist_directory="./chroma_db_storage",  # Hardcode o leer de config antiguo
                collection_name=self.chroma_cfg['collection_name'],
                embedding_function=self.embedding_fn
            )

        # --- INGESTA POR LOTES (Igual que antes) ---
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i: i + batch_size]
            vector_db.add_documents(batch)
            logger.info(f"Enviado lote {min(i + batch_size, total)}/{total} a la nube")

        logger.info("Ingesta vectorial remota completada.")