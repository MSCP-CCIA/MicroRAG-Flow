import os
import polars as pl
from typing import Dict
from src.utils.logger import setup_logger

# Inicializamos el logger
logger = setup_logger("ETL_Loader")


class DataLoader:
    def __init__(self, config: Dict):
        """
        Inicializa el cargador con la configuración del archivo settings.yaml
        """
        self.data_path = config['paths']['raw_data']
        self.files = config['files']

        # Leemos el modo de ejecución (test o prod)
        self.mode = config['execution']['mode']
        self.limit = config['execution']['test_limit']

    def _load_file(self, filename: str, description: str) -> pl.DataFrame:
        """
        Método interno genérico para cargar archivos JSONL.gz de forma eficiente.
        Usa 'scan_ndjson' para no saturar la RAM antes de filtrar.
        """
        file_path = os.path.join(self.data_path, filename)

        # Verificación de seguridad
        if not os.path.exists(file_path):
            logger.error(f"No se encontró el archivo en: {file_path}")
            raise FileNotFoundError(f"Falta el archivo {filename}. Revisa la carpeta data/raw.")

        logger.info(f"Cargando {description} desde {file_path}...")

        try:
            # Polars scan es 'lazy': prepara la lectura sin ejecutarla todavía
            lazy_df = pl.scan_ndjson(file_path)

            # Aplicamos el límite SOLO si estamos en modo test
            if self.mode == "test":
                logger.warning(f"MODO TEST ACTIVO: Cargando solo los primeros {self.limit} registros.")
                lazy_df = lazy_df.limit(self.limit)

            # .collect() ejecuta la lectura optimizada en paralelo
            return lazy_df.collect()

        except Exception as e:
            logger.error(f"Error leyendo {filename}: {e}")
            raise e

    def load_texts(self) -> pl.DataFrame:
        """Carga el dataset de textos"""
        return self._load_file(self.files['texts'], "textos")

    def load_tables(self) -> pl.DataFrame:
        """Carga el dataset de tablas"""
        return self._load_file(self.files['tables'], "tablas")

    def load_images(self) -> pl.DataFrame:
        """Carga el dataset de metadatos de imágenes"""
        return self._load_file(self.files['images'], "metadatos de imágenes")