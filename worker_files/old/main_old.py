########## LET`S RAG!!! ##########
# 
# This script is the main script for the retrieval of the RAG model.
# 
# The script is divided into the following sections:
# 
# 1. Preparation
# 2. Document retrieval
# 3. Hybrid search
# 4. Reranking
# 5. Context and compression
# 6. Response generation
# 7. Final response assembly
# 8. Logging and metrics
#
#
# 
#### Packages

import time
from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
import json
from src.elastic import buscar_ids, buscar_paginas_por_ids, buscar_vetores_por_ids, vector_similarity_search, bm25_similarity_search, merge_and_rerank, process_merged_results, enhance_results
from src.embed import get_embeddings
from src.model import generate_chat_completion
from src.prompt import build_structured_response, create_full_prompt
from src.utils import save_logs_to_database
import logging


#### Constants

# Load environment variables
load_dotenv()

# Azure OpenAI API connection
endpoint_api = os.getenv("ENDPOINT_URL")
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_key = os.getenv('AZURE_OPENAI_KEY')
endpoint_model = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")


# Elasticsearch connection
elasticsearch_host = os.getenv('ELASTICSEARCH_HOST')
elasticsearch_user = os.getenv('ELASTICSEARCH_USER')
elasticsearch_pwd = os.getenv('ELASTICSEARCH_PASSWORD')

try:
    es = Elasticsearch(
        elasticsearch_host,
        basic_auth=(elasticsearch_user, elasticsearch_pwd),
        request_timeout=60 # Aumentar timeout se necessário para queries longas
    )
    es.info() # Test connection
    logging.info("Conexão com Elasticsearch estabelecida com sucesso.")
except Exception as e:
    logging.error(f"Falha ao conectar ao Elasticsearch: {e}")
    exit() # Sair se não conseguir conectar

# Database connection details
connection_string = os.getenv("SQL_SERVER_CNXN_STR_IA")
print(connection_string)

#### Variables

# Prompt
role_upgrade_prompt = "Você é um especialista em análise de documentos jurídicos extensos como inquéritos policiais, autos de prisão, etc. Sua tarefa é aprimorar consultas para facilitar a recuperação de informações desses documentos. Dada a seguinte consulta do usuário: Tarefas: Refine e Expanda: Reescreva a consulta para torná-la mais detalhada e abrangente, incluindo possíveis sinônimos e termos jurídicos relacionados. Por exemplo, se a consulta for - Quem são as vítimas? -, considere incluir variações como - identificação das vítimas -, - pessoas lesadas -, ou - partes prejudicadas -. Contextualize: Considere que os documentos podem conter informações distribuídas em diferentes seções e com terminologias variadas. Justifique: Após a reformulação, forneça uma breve explicação das mudanças realizadas para melhorar a recuperação de informações. Retorno: Forneça a consulta refinada apenas e mais nada."
role_answer = "Você é um assistente jurídico especializado em auxiliar promotores de justiça na análise de processos. Sua função é fornecer respostas somente com base nos documentos fornecidos. Se a informação não estiver nos documentos, responda claramente que não há referência disponível. Não tente adivinhar ou inferir respostas além do conteúdo recuperado. Mantenha um tom formal e objetivo, adequado ao ambiente jurídico. Se a pergunta for irrelevante ao contexto processual, informe educadamente que sua função é auxiliar exclusivamente na análise dos documentos."
#prompt = "Explain the concept of machine learning in simple terms."

# Reranking
merged_top_k = 5
bm25_top_k = 10
vector_top_k = 10



#### Functions
import json
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_and_respond(payload: Dict[str, Any]) -> Dict[str, Any]:
    start_time = time.time()
    response = {}
    try:
        # 1. Preparação
        logging.info("Starting phase 1: Preparation")
        prompt_data = json.loads(json.dumps(payload))
        ids_documento_mni = prompt_data['id_documentos_mni']
        ids_documento_gampes = prompt_data['id_documentos_gampes']
        prompt_original = prompt_data['texto_prompt']

        logging.info(f"Phase 1 completed in {time.time() - start_time} seconds")

        # 2. Recuperação de documentos
        logging.info("Starting phase 2: Document retrieval")
        id_textual_list = buscar_ids(ids_documento_gampes, ids_documento_mni)
        id_paginas_list = buscar_paginas_por_ids(id_textual_list)
        id_vector_list = buscar_vetores_por_ids(id_paginas_list)

        logging.info(f"Phase 2.0 completed in {time.time() - start_time} seconds")

        # 2.1. Geração de prompt aprimorado
        logging.info("Starting phase 2.1: Enhanced prompt generation")
        prompt_enhanced_response = generate_chat_completion(endpoint_api, deployment, subscription_key, role_upgrade_prompt, prompt_original)
        prompt_enhanced = prompt_enhanced_response["choices"][0]["message"]["content"]
        prompt_embedding = get_embeddings(prompt_enhanced)

        logging.info(f"Phase 2.1 completed in {time.time() - start_time} seconds")

        # 3. Busca híbrida
        logging.info("Starting phase 3: Hybrid search")
        bm25_results = bm25_similarity_search(prompt_enhanced, id_paginas_list, k=bm25_top_k)
        vector_results = vector_similarity_search(prompt_embedding, id_list=id_paginas_list, k=vector_top_k)

        logging.info(f"Phase 3 completed in {time.time() - start_time} seconds")

        # 4. Reranking
        logging.info("Starting phase 4: Reranking")
        merged_results = merge_and_rerank(vector_results, bm25_results, vector_weight=0.6, bm25_weight=0.4)

        logging.info(f"Phase 4 completed in {time.time() - start_time} seconds")

        # 5. Contexto e compressão
        logging.info("Starting phase 5: Context and compression")
        processed_results = process_merged_results(merged_results)
        top_k_merged_results = processed_results[:merged_top_k]
        enhanced_results = enhance_results(top_k_merged_results)
        prompt_final = create_full_prompt(prompt_enhanced, enhanced_results)
        time.sleep(1) # Pause for 1 second to avoid rate limiting

        logging.info(f"Phase 5 completed in {time.time() - start_time} seconds")

        # 6. Geração de resposta
        logging.info("Starting phase 6: Response generation")
        llm_response = generate_chat_completion(endpoint_api, deployment, subscription_key, role_answer, prompt_final)

        logging.info(f"Phase 6 completed in {time.time() - start_time} seconds")

        # 7. Montagem da resposta final
        logging.info("Starting phase 7: Final response assembly")
        response = build_structured_response(
            llm_response["choices"][0]["message"]["content"],
            enhanced_results,
            llm_response["id"]
        )

        logging.info(f"Phase 7 completed in {time.time() - start_time} seconds")

        # 8. Logging e métricas
        tempo_processamento = time.time() - start_time
        logging.info(f"Total processing time: {tempo_processamento} seconds")

        # Save logs to database only if no errors occurred
        save_logs_to_database(connection_string, llm_response, prompt_original, prompt_final, prompt_data, tempo_processamento)

    except json.JSONDecodeError as e:
        logging.error(f"JSON decoding error: {e}")
        response = {"error": "Invalid JSON payload", "code": 400}
    except KeyError as e:
        logging.error(f"Missing key in payload: {e}")
        response = {"error": f"Missing key in payload: {e}", "code": 400}
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        response = {"error": "Internal server error", "code": 500}

    return response

# Example usage
payload = {
  "texto_prompt": "quem sao os envolvidos do processo?",
  "id_documentos_mni": [23204850, 23204851, 23204846, 23204845, 23204849, 23204848, 23204847, 23204844],
  "id_documentos_gampes": [251735, 7340129],
  "idfuncao": "987",
  "idorgao": "456",
  "user": "fulano"
}


response = retrieve_and_respond(payload)
logging.info(response)