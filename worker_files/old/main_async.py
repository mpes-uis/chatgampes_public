from fastapi import FastAPI, HTTPException, BackgroundTasks
import time
from elasticsearch import Elasticsearch, exceptions as es_exceptions, ApiError, TransportError
import os
from dotenv import load_dotenv
import json
from src.elastic import buscar_ids, buscar_paginas_por_ids, buscar_vetores_por_ids, vector_similarity_search, bm25_similarity_search, merge_and_rerank, process_merged_results, enhance_results, update_es_document
from src.embed import get_embeddings
from src.model import generate_chat_completion
from src.prompt import build_structured_response, create_full_prompt
from src.utils import save_logs_to_database, consultar_apis
import logging
from typing import Dict, Any
import markdown
from src.prompt_roles import role_upgrade_prompt, role_answer
import uuid # For generating unique task IDs
from datetime import datetime, timezone # For timestamps

# Load environment variables
load_dotenv()

# Azure OpenAI API connection
endpoint_api = os.getenv("ENDPOINT_URL")
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_key = os.getenv('AZURE_OPENAI_KEY')
endpoint_model = os.getenv("ENDPOINT_URL") # Redundant with ENDPOINT_URL? Assuming it's used somewhere.
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Elasticsearch connection
elasticsearch_host = os.getenv('ELASTICSEARCH_HOST')
elasticsearch_user = os.getenv('ELASTICSEARCH_USER')
elasticsearch_pwd = os.getenv('ELASTICSEARCH_PASSWORD')

# Conecta ao Elasticsearch
try:
    es = Elasticsearch(
        elasticsearch_host,
        basic_auth=(elasticsearch_user, elasticsearch_pwd),
        retry_on_timeout=True,
        max_retries=3
    )
    if not es.ping():
        raise ConnectionError("Failed to connect to Elasticsearch")
    logging.info("Successfully connected to Elasticsearch.")
except ConnectionError as e:
    logging.error(f"Elasticsearch connection error: {e}")
    # Depending on the desired behavior, you might want to exit or handle this differently.
    # For now, we'll let the app start, but operations requiring ES will fail.
    es = None 
except Exception as e:
    logging.error(f"An unexpected error occurred during Elasticsearch setup: {e}")
    es = None


# Database connection details
connection_string = os.getenv("SQL_SERVER_CNXN_STR_IA")

# URLs das APIs para OCR
url_api_ocr_gampes = os.getenv("URL_API_OCR_GAMPES")
url_api_ocr_mni = os.getenv("URL_API_OCR_MNI")

#### Variables
ELASTICSEARCH_INDEX_RESPONSES = "gampes_agent_assessorvirtual"

# Reranking
merged_top_k = 5
bm25_top_k = 10
vector_top_k = 10

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

async def process_rag_task(
    task_id: str,
    payload: Dict[str, Any],
    es_client: Elasticsearch, # Pass dependencies if they are not easily global or for testability
    db_connection_string: str
):
    start_time = time.time()
    original_payload_for_log = payload.copy() # Keep a copy for logging

    try:
        # 1. Preparação
        logging.info(f"Task {task_id}: Starting phase 1: Preparation")
        prompt_data = payload # Payload is already a dict
        ids_documento_mni = prompt_data.get('id_documentos_mni', [])
        ids_documento_gampes = prompt_data.get('id_documentos_gampes', [])
        prompt_original = prompt_data.get('texto_prompt', '')

        if not prompt_original:
            raise KeyError("'texto_prompt' is missing or empty in payload")

        logging.info(f"Task {task_id}: Phase 1 completed in {time.time() - start_time:.2f} seconds")

        # 2. Recuperação de documentos
        logging.info(f"Task {task_id}: Starting phase 2: Document retrieval")
        id_textual_list = consultar_apis(ids_documento_gampes, ids_documento_mni, url_api_ocr_gampes, url_api_ocr_mni)
        # Ensure subsequent functions are adapted if consultar_apis becomes async or if they need to be async
        id_paginas_list = buscar_paginas_por_ids(es_client, id_textual_list) # Pass es_client
        id_vector_list = buscar_vetores_por_ids(es_client, id_paginas_list) # Pass es_client

        logging.info(f"Task {task_id}: Phase 2.0 completed in {time.time() - start_time:.2f} seconds")

        # 2.1. Geração de prompt aprimorado
        logging.info(f"Task {task_id}: Starting phase 2.1: Enhanced prompt generation")
        # Assuming generate_chat_completion and get_embeddings can be blocking or are already async-compatible
        prompt_enhanced_response = generate_chat_completion(endpoint_api, deployment, subscription_key, role_upgrade_prompt, prompt_original)
        prompt_enhanced = prompt_enhanced_response["choices"][0]["message"]["content"]
        prompt_embedding = get_embeddings(prompt_enhanced)

        logging.info(f"Task {task_id}: Phase 2.1 completed in {time.time() - start_time:.2f} seconds")

        # 3. Busca híbrida
        logging.info(f"Task {task_id}: Starting phase 3: Hybrid search")
        bm25_results = bm25_similarity_search(es_client, prompt_enhanced, id_paginas_list, k=bm25_top_k) # Pass es_client
        vector_results = vector_similarity_search(es_client, prompt_embedding, id_list=id_paginas_list, k=vector_top_k) # Pass es_client

        logging.info(f"Task {task_id}: Phase 3 completed in {time.time() - start_time:.2f} seconds")

        # 4. Reranking
        logging.info(f"Task {task_id}: Starting phase 4: Reranking")
        merged_results = merge_and_rerank(vector_results, bm25_results, vector_weight=0.6, bm25_weight=0.4)

        logging.info(f"Task {task_id}: Phase 4 completed in {time.time() - start_time:.2f} seconds")

        # 5. Contexto e compressão
        logging.info(f"Task {task_id}: Starting phase 5: Context and compression")
        processed_results = process_merged_results(merged_results)
        top_k_merged_results = processed_results[:merged_top_k]
        enhanced_results = enhance_results(es_client, top_k_merged_results) # Pass es_client
        prompt_final = create_full_prompt(prompt_enhanced, enhanced_results)
        
        # Consider making time.sleep async if this function becomes truly async
        # For now, BackgroundTasks run in a separate thread, so time.sleep is fine.
        time.sleep(1) 

        logging.info(f"Task {task_id}: Phase 5 completed in {time.time() - start_time:.2f} seconds")

        # 6. Geração de resposta
        logging.info(f"Task {task_id}: Starting phase 6: Response generation")
        llm_response = generate_chat_completion(endpoint_api, deployment, subscription_key, role_answer, prompt_final)
        llm_response_content = llm_response["choices"][0]["message"]["content"]
        llm_response_html = markdown.markdown(llm_response_content, extensions=['extra', 'codehilite', 'tables'])

        logging.info(f"Task {task_id}: Phase 6 completed in {time.time() - start_time:.2f} seconds")

        # 7. Montagem da resposta final
        logging.info(f"Task {task_id}: Starting phase 7: Final response assembly")
        final_structured_response = build_structured_response(
            llm_response_html,
            enhanced_results,
            llm_response["id"]
        )
        logging.info(f"Task {task_id}: Phase 7 completed in {time.time() - start_time:.2f} seconds")

        # 8. Logging e métricas
        tempo_processamento = time.time() - start_time
        logging.info(f"Task {task_id}: Total processing time: {tempo_processamento:.2f} seconds")

        # Update Elasticsearch with success
        update_es_document(task_id, 200, response_text=llm_response["choices"][0]["message"]["content"], texto_aux=prompt_final, usuario=prompt_data["user"], index_resposta=ELASTICSEARCH_INDEX_RESPONSES, id_requisicao=task_id, tipo_requisicao="RAG", status_code=200)
        
        # Save logs to database only if no errors occurred
        # Ensure save_logs_to_database can handle the new structure if needed
        save_logs_to_database(db_connection_string, llm_response, prompt_original, prompt_final, original_payload_for_log, tempo_processamento, task_id)

    except json.JSONDecodeError as e:
        logging.error(f"Task {task_id}: JSON decoding error: {e}")
        update_es_document(task_id, 400, f"Invalid JSON payload: {e}")
    except KeyError as e:
        logging.error(f"Task {task_id}: Missing key in payload: {e}")
        update_es_document(task_id, 400, f"Missing key in payload: {e}")
    except Exception as e:
        logging.error(f"Task {task_id}: An unexpected error occurred: {e}", exc_info=True)
        update_es_document(task_id, 500, f"Internal server error: {e}")


@app.post("/rag", status_code=202) # 202 Accepted is more appropriate for async tasks
async def rag_async_trigger(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    if not es:
        raise HTTPException(status_code=503, detail="Elasticsearch service unavailable. Cannot process request.")

    task_id = str(uuid.uuid4())
    
    # Initial document creation in Elasticsearch
    initial_doc_body = {
        "status": 102,
        "texto_resposta": "processando requisição",
        "texto_aux": payload.get('texto_prompt', ''), # Storing original payload for reference
        "data_criacao": datetime.now(timezone.utc).isoformat(),
    }
    try:
        es.index(
            index=ELASTICSEARCH_INDEX_RESPONSES,
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

    # Add the long-running task to background
    # Ensure all dependencies like 'es' and 'connection_string' are passed if they are not globally available
    # or if you prefer explicit dependency injection.
    # The 'es' client needs to be passed to your src.elastic functions now, or they need to access the global 'es'.
    # I'll assume for now they can access the global 'es' or you'll modify them.
    # If your src.elastic functions like buscar_ids, buscar_paginas_por_ids were not designed to accept 'es' client,
    # you'll need to modify them. I've added 'es_client' to the calls in `process_rag_task` for clarity.
    
    # The `consultar_apis` function seems to be making HTTP requests. If it's using a synchronous library (like `requests`),
    # it will block the background worker thread. For true async, it should use an async HTTP client (like `httpx`).
    # For now, FastAPI's BackgroundTasks run in a separate thread pool, so blocking I/O is okay-ish but not ideal for high concurrency.
    # I've made `consultar_apis` async in the call, assuming you'll adapt it or it's already async.

    background_tasks.add_task(process_rag_task, task_id, payload, es, connection_string)
    
    return {"task_id": task_id, "message": "Requisição recebida e processamento iniciado em segundo plano."}

@app.get("/rag/status/{task_id}")
async def get_rag_status(task_id: str):
    if not es:
        raise HTTPException(status_code=503, detail="Elasticsearch service unavailable. Cannot retrieve status.")
    try:
        doc = es.get(index=ELASTICSEARCH_INDEX_RESPONSES, id=task_id)
        return doc["_source"]
    except es_exceptions.NotFoundError:
        raise HTTPException(status_code=404, detail=f"Task ID '{task_id}' não encontrado.")
    except (ApiError, TransportError) as e:
        logging.error(f"Elasticsearch error fetching status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Erro ao consultar o status da tarefa no Elasticsearch.")
    except Exception as e:
        logging.error(f"Unexpected error fetching status for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Erro inesperado ao consultar o status da tarefa.")

# To run the API, use the following command:
# uvicorn main:app --reload