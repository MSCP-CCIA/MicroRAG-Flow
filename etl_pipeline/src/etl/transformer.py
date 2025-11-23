from typing import List
from langchain_core.documents import Document
from src.utils.logger import setup_logger

logger = setup_logger("ETL_Transformer")


class DataTransformer:
    @staticmethod
    def process_tables_to_documents(tables_df) -> List[Document]:
        """Convierte DataFrames de tablas en Documentos LangChain serializados"""
        logger.info("Transformando tablas a formato textual...")
        documents = []

        # Iteramos (Polars es rápido, pero la lógica custom requiere iteración o map_elements)
        # Para dataset masivo, usar map_elements de Polars sería mejor, aquí usamos iteración simple por claridad
        for row in tables_df.iter_rows(named=True):
            try:
                table_data = row['table']
                header = [col['column_name'] for col in table_data['header']]

                # Serialización simple
                text_representation = f"Tabla: {row['title']}. "
                rows_text = []
                for t_row in table_data['table_rows']:
                    row_str = ", ".join([f"{h}: {val['text']}" for h, val in zip(header, t_row)])
                    rows_text.append(row_str)

                text_representation += " | ".join(rows_text)

                doc = Document(
                    page_content=text_representation,
                    metadata={"id": row['id'], "title": row['title'], "type": "table", "url": row['url']}
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error procesando tabla {row.get('id', 'unknown')}: {e}")
                continue

        return documents

    @staticmethod
    def process_texts_to_documents(texts_df) -> List[Document]:
        logger.info("Transformando textos a Documentos...")
        return [
            Document(
                page_content=row['text'],
                metadata={"id": row['id'], "title": row['title'], "type": "text", "url": row['url']}
            )
            for row in texts_df.iter_rows(named=True)
        ]

    @staticmethod
    def process_images_to_documents(images_df) -> List[Document]:
        logger.info("Transformando metadatos de imágenes a Documentos...")
        documents = []

        # 🔴 ERROR COMÚN: for row in images_df: (Esto itera columnas)
        # 🟢 CORRECTO: .iter_rows(named=True) (Esto itera filas como diccionarios)
        for row in images_df.iter_rows(named=True):
            try:
                # Usamos .get() por seguridad, aunque con named=True row es un dict
                title = row.get('title') or "Imagen sin título"
                caption = row.get('caption', '') or row.get('description', '') or ""

                text_representation = f"Imagen titulada: {title}. {caption}"

                doc = Document(
                    page_content=text_representation,
                    metadata={
                        "id": row.get('id', 'unknown'),
                        "title": title,
                        "type": "image",
                        "url": row.get('url', ''),
                        "path": row.get('path', '')
                    }
                )
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Error procesando imagen: {e}")
                continue

        return documents