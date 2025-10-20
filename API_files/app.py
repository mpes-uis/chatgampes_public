from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from pydantic import BaseModel
from typing import Dict, Any
import uuid
from datetime import datetime, timezone
import logging
import pyodbc
from dotenv import load_dotenv
import os
from elasticsearch import Elasticsearch, ApiError, TransportError
from elasticsearch.exceptions import NotFoundError
import json
import sys
import time



# Load environment variables
load_dotenv()

# Elasticsearch connection
# Configurações do Elasticsearch
elasticsearch_host = os.getenv('ELASTICSEARCH_HOST')
elasticsearch_user = os.getenv('ELASTICSEARCH_USER')
elasticsearch_pwd = os.getenv('ELASTICSEARCH_PASSWORD')
elasticsearch_hosts = os.getenv('ELASTICSEARCH_HOSTS')

# Split the hosts string into a list
elasticsearch_hosts_list = elasticsearch_hosts.split(',')

# Conecta ao Elasticsearch
es = Elasticsearch(
    elasticsearch_hosts_list,
    basic_auth=(elasticsearch_user, elasticsearch_pwd),
)

# Define the Elasticsearch index for storing responses
index_responses = os.getenv('ELASTICSEARCH_INDEX_RESPONSES')

# Database connection details
connection_string = os.getenv("SQL_SERVER_CNXN_STR_IA")

# URLs das APIs para OCR
url_api_ocr_gampes = os.getenv("URL_API_OCR_GAMPES")
url_api_ocr_mni = os.getenv("URL_API_OCR_MNI")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#logging.info(elasticsearch_host, elasticsearch_user)

# Função para criar uma nova conexão com o banco de dados
def get_db_connection():
    """Cria uma nova conexão com o banco de dados"""
    return pyodbc.connect(connection_string)

# Modify the insert_into_fila_processamento function
def insert_into_fila_processamento(id_elasticsearch: str, payload_json: dict, status: int, agente: int):
    """Insere dados na tabela fila_processamento_agentes"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            query = """
            INSERT INTO IA.dbo.fila_processamento_agentes 
            (id_elasticsearch, payload, status, data_criacao, id_agente)
            VALUES (?, ?, ?, GETDATE(), ?)
            """
            
            # Convert the dictionary to a JSON string
            payload_str = json.dumps(payload_json, ensure_ascii=False)
            cursor.execute(query, id_elasticsearch, payload_str, status, agente)
            conn.commit()
            logging.info(f"Dados inseridos na tabela com sucesso. ID Elasticsearch: {id_elasticsearch}")
            
    except Exception as e:
        logging.error(f"Erro ao inserir dados na tabela: {e}")
        raise

# Function to check database connection
def check_db_connection():
    """Checks the database connection."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            logging.info("Database connection successful.")
            return True
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        return False

# Function to check Elasticsearch connection
def check_elasticsearch_connection():
    """Checks the Elasticsearch connection."""
    try:
        if es.ping():
            logging.info("Elasticsearch connection successful.")
            return True
        else:
            logging.error("Elasticsearch connection failed: ping returned False.")
            return False
    except Exception as e:
        logging.error(f"Elasticsearch connection failed: {e}")
        return False

app = FastAPI()

@app.on_event("startup")
def startup_event():
    """Application startup event handler."""
    # Check database connection
    if not check_db_connection():
        logging.error("Application startup failed: could not connect to the database.")
        sys.exit(1)

    # Check Elasticsearch connection
    if not check_elasticsearch_connection():
        logging.error("Application startup failed: could not connect to Elasticsearch.")
        sys.exit(1)

@app.post("/rag", status_code=202) # 202 Accepted is more appropriate for async tasks
async def rag_async_trigger(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    if not es:
        raise HTTPException(status_code=503, detail="Elasticsearch service unavailable. Cannot process request.")

    task_id = str(uuid.uuid4())
    
    # Construct the complete URL that will be returned
    elasticsearch_url = f"{elasticsearch_host}/{index_responses}/_doc/{task_id}"
    
    # Initial document creation in Elasticsearch
    initial_doc_body = {
        "status": 202,
        "texto_resposta": "processando requisição",
        "texto_aux": payload.get('texto_prompt', ''), # Storing original payload for reference
        "data_criacao": datetime.now(timezone.utc).isoformat(),
    }
    try:
        es.index(
            index=index_responses,
            id=task_id,
            document=initial_doc_body
        )
        logging.info(f"Initial Elasticsearch document {task_id} created for new RAG task.")
    except (ApiError, TransportError) as e:
        logging.error(f"Failed to create initial Elasticsearch document for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate task tracking in Elasticsearch.")
    except Exception as e: # Catch any other unexpected error during ES init
        logging.error(f"Unexpected error creating initial Elasticsearch document for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error initiating task tracking.")

    #background_tasks.add_task(process_rag_task, task_id, payload, es, connection_string)
    insert_into_fila_processamento(task_id, payload, 102, 101)
    
    return {
        "task_id": task_id,
        "url": elasticsearch_url,
        "message": "Requisição recebida e processamento iniciado em segundo plano."
    }

@app.get("/rag/status/{task_id}")
async def get_rag_status(task_id: str):
    if not es:
        raise HTTPException(status_code=503, detail="Elasticsearch service unavailable. Cannot retrieve status.")
    try:
        doc = es.get(index=index_responses, id=task_id)
        return doc["_source"]
    except NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' não encontrado.")
    except (ApiError, TransportError) as e:
        logging.error(f"Elasticsearch error fetching status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Erro ao consultar o status da tarefa no Elasticsearch.")
    except Exception as e:
        logging.error(f"Unexpected error fetching status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Erro inesperado ao consultar o status da tarefa.")


class EvalData(BaseModel):
    id: str
    eval: bool
    info: str

@app.post("/evaluate", status_code=status.HTTP_201_CREATED)
async def save_evaluation(data: EvalData):
    try:
        with pyodbc.connect(connection_string) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO IA.dbo.rag_gampes_eval (id, eval, info) VALUES (?, ?, ?)",
                    (data.id, int(data.eval), data.info)
                )
                conn.commit()
        return {"message": "Evaluation saved successfully"}
    
    except pyodbc.IntegrityError:
        raise HTTPException(
            status_code=400,
            detail="Duplicate ID - This evaluation already exists"
        )
        
    except pyodbc.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )


# To run the API, use the following command:
#uvicorn app:app --reload
# or
#uvicorn app:app --host 0.0.0.0 --port 8000 --reload
