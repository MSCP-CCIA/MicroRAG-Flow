import os
import yaml
from dotenv import load_dotenv
from src.utils.logger import setup_logger
from src.etl.loader import DataLoader
from src.etl.transformer import DataTransformer
from src.graphdb.builder import KnowledgeGraphBuilder

# Cargar variables de entorno
load_dotenv()
logger = setup_logger("Main")


def load_config(path="config/settings.yaml"):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    logger.info("=== INICIANDO CARGA LOCAL (NEO4J) ===")
    config = load_config()

    # 1. CARGA
    logger.info("1. Cargando datos desde disco...")
    loader = DataLoader(config)
    df_texts = loader.load_texts()
    df_tables = loader.load_tables()
    df_images = loader.load_images()

    # 2. TRANSFORMACION
    logger.info("2. Transformando datos...")
    transformer = DataTransformer()
    docs_text = transformer.process_texts_to_documents(df_texts)
    docs_tables = transformer.process_tables_to_documents(df_tables)
    docs_images = transformer.process_images_to_documents(df_images)

    all_docs = docs_text + docs_tables + docs_images
    logger.info(f"Total documentos a procesar: {len(all_docs)}")

    # 3. CHROMA (SALTADO)
    # ... (Ignorado porque ya está listo) ...

    # 4. NEO4J LOCAL
    try:
        neo4j_cfg = config['neo4j']
        neo4j_pass = os.getenv("NEO4J_PASSWORD")

        if not neo4j_pass:
            raise ValueError("NEO4J_PASSWORD no definido en .env")

        logger.info(f"Conectando a Neo4j Local en: {neo4j_cfg['uri']}")

        graph_builder = KnowledgeGraphBuilder(
            config,
            uri=neo4j_cfg['uri'],
            user=neo4j_cfg['user'],
            password=neo4j_pass
        )

        # Ejecucion
        start_time = os.times()[4]
        graph_builder.build_graph_from_documents(all_docs)
        graph_builder.close()

        end_time = os.times()[4]
        logger.info(f"Tiempo total de carga: {end_time - start_time:.2f} segundos")

    except Exception as e:
        logger.error(f"[ERROR] Fallo en Neo4j: {e}")

    logger.info("=== LISTO. AHORA PUEDES EXPORTAR LA DATA ===")


if __name__ == "__main__":
    main()