import spacy
import time
import random
import concurrent.futures
from neo4j import GraphDatabase
from neo4j.exceptions import TransientError, ServiceUnavailable
from typing import List, Dict
from langchain_core.documents import Document
from src.utils.logger import setup_logger

logger = setup_logger("GraphDB")


class KnowledgeGraphBuilder:
    def __init__(self, config: Dict, uri, user, password):
        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=100,  # Pool grande para hilos locales
            connection_acquisition_timeout=60
        )
        logger.info("Cargando modelo NLP (CPU Local)...")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
        except:
            raise ImportError("Modelo Spacy no encontrado.")

    def close(self):
        self.driver.close()

    def _prepare_doc_data(self, doc: Document) -> Dict:
        """Procesamiento intensivo de CPU (Spacy)"""
        text = doc.page_content
        meta = doc.metadata
        # Limitamos caracteres para velocidad, ajusta si necesitas mas precision
        spacy_doc = self.nlp(text[:50000])
        entities = [{"text": ent.text, "label": ent.label_} for ent in spacy_doc.ents]

        return {
            "id": meta.get('id'),
            "title": meta.get('title', 'Sin Título'),
            "type": meta.get('type'),
            "url": meta.get('url', ''),
            "text_content": text,
            "entities": entities
        }

    def _write_batch_task(self, batch_data):
        """Escribe un lote en Neo4j con reintentos"""
        query = """
        UNWIND $batch AS row
        MERGE (d:Document {id: row.id})
        SET d.title = row.title, 
            d.type = row.type, 
            d.url = row.url, 
            d.text = row.text_content

        WITH d, row
        UNWIND row.entities as ent
        MERGE (e:Entity {name: ent.text})
        ON CREATE SET e.type = ent.label
        MERGE (d)-[:MENTIONS]->(e)
        """

        max_retries = 10
        base_delay = 0.5

        for attempt in range(max_retries):
            try:
                with self.driver.session() as session:
                    session.run(query, batch=batch_data)
                return True
            except (TransientError, ServiceUnavailable) as e:
                sleep_time = base_delay * (1.5 ** attempt) + random.uniform(0, 0.5)
                # Logueamos solo si falla muchas veces para no ensuciar consola
                if attempt > 2:
                    logger.warning(f"[WARN] Reintentando lote por bloqueo DB ({attempt + 1}/{max_retries})...")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"[ERROR] Fallo fatal en lote: {e}")
                return False
        return False

    def build_graph_from_documents(self, documents: List[Document], batch_size=1000):
        total = len(documents)
        MAX_WORKERS = 8  # 8 Hilos es ideal para un i7 en local

        logger.info(f"Iniciando carga LOCAL MULTIHILO de {total} documentos.")
        logger.info(f"Configuracion: {MAX_WORKERS} Hilos | Lotes de {batch_size}")

        # 1. Checkpoint
        logger.info("Consultando IDs existentes en local...")
        try:
            with self.driver.session() as session:
                result = session.run("MATCH (d:Document) RETURN d.id as id")
                existing_ids = set([record["id"] for record in result])
            logger.info(f"Encontrados {len(existing_ids)} ya cargados. Se omitiran.")
        except:
            existing_ids = set()

        current_batch = []
        futures = []

        # Usamos ThreadPoolExecutor para paralelizar Spacy y Escritura
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

            for i, doc in enumerate(documents):
                if doc.metadata.get('id') in existing_ids:
                    continue

                # Esto corre en el hilo principal, pero es rapido
                # Si el cuello de botella es Spacy, podriamos mover esto adentro del task,
                # pero esta estructura es mas segura para la memoria.
                processed_doc = self._prepare_doc_data(doc)
                current_batch.append(processed_doc)

                if len(current_batch) >= batch_size:
                    batch_to_send = list(current_batch)
                    current_batch = []

                    future = executor.submit(self._write_batch_task, batch_to_send)
                    futures.append(future)

                    # Limpiar memoria de futuros terminados
                    futures = [f for f in futures if not f.done()]

                    if i % 2000 == 0:
                        print(f" > Procesando documento {i}/{total}...", end='\r')

            # Lote final
            if current_batch:
                executor.submit(self._write_batch_task, current_batch)

            logger.info("Todos los lotes enviados a la cola. Esperando finalizacion...")
            concurrent.futures.wait(futures)

        logger.info("=== CARGA LOCAL FINALIZADA ===")