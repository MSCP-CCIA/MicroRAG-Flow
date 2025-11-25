import os
import sys
import yaml
import gzip
import json
import random
import time  # <--- IMPORTANTE
import logging
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("DataGenerator")

load_dotenv("../.env")

# Cargar configuración
config_path = "../config/settings.yaml"
if not os.path.exists(config_path):
    config_path = "config/settings.yaml"

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Configuración del LLM Generador
llm = ChatGoogleGenerativeAI(
    model=config['llm']['model_name'],
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    # Añadimos reintentos automáticos por si acaso
    max_retries=5,
    request_timeout=60
)


# Estructura de salida deseada
class QAPair(BaseModel):
    question: str = Field(description="Una pregunta clara basada en el contexto")
    ground_truth: str = Field(description="La respuesta exacta y concisa basada en el contexto")


parser = JsonOutputParser(pydantic_object=QAPair)


def get_random_line_from_gz(filepath: str, sample_size=1000) -> Dict:
    """Lee un archivo GZ grande y selecciona una linea aleatoria sin cargarlo todo en RAM"""
    candidates = []
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i > sample_size: break
                candidates.append(json.loads(line))

        if not candidates:
            return None
        return random.choice(candidates)
    except Exception as e:
        logger.error(f"[ERROR] Leyendo archivo {filepath}: {e}")
        return None


def generate_qa_from_context(context_text: str, context_type: str) -> Dict:
    """Usa Gemini para crear una pregunta basada en el texto proporcionado"""

    prompt = ChatPromptTemplate.from_template(
        """Eres un experto creando datasets de evaluacion.
        Tu tarea es generar UN prompt de usuario y su respuesta (ground_truth) basada EXCLUSIVAMENTE en el siguiente contexto.

        Tipo de Contexto: {type}
        Contexto:
        {context}

        Requisitos:
        1. El prompt debe ser especifico y claro, lo mas similar posible a una consulta de un usuario real.
        2. La respuesta (ground_truth) debe ser extraida del contexto.
        3. Si es una tabla, pregunta por un valor especifico.
        
        {format_instructions}
        """
    )

    chain = prompt | llm | parser

    try:
        return chain.invoke({
            "type": context_type,
            "context": context_text,
            "format_instructions": parser.get_format_instructions()
        })
    except Exception as e:
        logger.warning(f"Fallo generando pregunta: {e}")
        return None


def main():
    TARGET_SIZE = 10
    generated_data = []

    # Rutas a los datos crudos
    base_path = "../" + config['paths']['raw_data']
    files = {
        "text": os.path.join(base_path, config['files']['texts']),
        "table": os.path.join(base_path, config['files']['tables']),
        "image": os.path.join(base_path, config['files']['images'])
    }

    logger.info(f"--- Generando Dataset Sintetico de {TARGET_SIZE} preguntas ---")
    logger.info("NOTA: Se aplicará una pausa de 10s entre preguntas para respetar la cuota gratuita.")

    while len(generated_data) < TARGET_SIZE:
        # 1. Elegir aleatoriamente el tipo de dato
        source_type = random.choices(["text", "table", "image"], weights=[0.5, 0.3, 0.2])[0]
        filepath = files[source_type]

        # 2. Obtener dato crudo
        item = get_random_line_from_gz(filepath)
        if not item:
            continue

        # 3. Preparar contexto
        context_str = ""
        if source_type == "text":
            context_str = item.get('text', '')
        elif source_type == "table":
            table = item.get('table', {})
            header = [c['column_name'] for c in table.get('header', [])]
            rows = table.get('table_rows', [])
            sample_rows = random.sample(rows, min(len(rows), 3))
            context_str = f"Titulo Tabla: {item.get('title')}. Columns: {header}. Rows: {sample_rows}"
        elif source_type == "image":
            context_str = f"Titulo Imagen: {item.get('title')}. Descripcion: {item.get('caption')}"

        if len(context_str) < 50:
            continue

        # 4. Generar QA
        logger.info(f"Generando par QA ({len(generated_data) + 1}/{TARGET_SIZE}) tipo: {source_type.upper()}...")

        qa_pair = generate_qa_from_context(context_str, source_type)

        if qa_pair:
            qa_pair['source_type'] = source_type
            qa_pair['source_id'] = item.get('id')
            generated_data.append(qa_pair)

            # --- PAUSA DE SEGURIDAD ---
            # Dormimos 10 segundos para garantizar no pasar de 6 peticiones/minuto
            # Esto evitará el error 429 ResourceExhausted
            time.sleep(10)

    # 5. Guardar
    output_path = "test_dataset_synthetic.json"
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(generated_data, f, indent=4, ensure_ascii=False)

    logger.info(f"[EXITO] Dataset generado en '{output_path}'")


if __name__ == "__main__":
    main()