# src/rag_engines.py
import spacy
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase


class ChromaRAG:
    def __init__(self, config):
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=config['embeddings']['model_name'],
            model_kwargs={'device': config['embeddings']['device']},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.db = Chroma(
            persist_directory=config['paths']['chroma_db'],
            embedding_function=self.embedding_fn,
            collection_name="multimodal_rag"
        )

    def retrieve(self, query, k=3):
        """Recupera contexto basado en similitud vectorial"""
        results = self.db.similarity_search_with_score(query, k=k)
        # Retorna solo el texto de los documentos encontrados
        return [doc.page_content for doc, score in results]


class Neo4jRAG:
    def __init__(self, config, password):
        self.driver = GraphDatabase.driver(
            config['neo4j']['uri'],
            auth=(config['neo4j']['user'], password)
        )
        self.nlp = spacy.load("en_core_web_sm")

    def retrieve(self, query):
        """Recupera contexto basado en entidades conectadas"""
        # 1. Extraer entidades de la pregunta
        doc = self.nlp(query)
        entidades = [ent.text for ent in doc.ents]

        if not entidades:
            return []  # No encontró entidades, no puede buscar en el grafo

        # 2. Buscar en Neo4j
        # 👇 CAMBIO: Usamos coalesce() para que nunca devuelva NULL
        cypher_query = """
            MATCH (e:Entity)<-[:MENTIONS]-(d:Document)
            WHERE e.name IN $nombres
            RETURN d.title + ': ' + coalesce(d.text, 'Sin contenido de texto disponible') as contexto
            LIMIT 3
            """
        with self.driver.session() as session:
            result = session.run(cypher_query, nombres=entidades)
            return [record["contexto"] for record in result]

    def close(self):
        self.driver.close()