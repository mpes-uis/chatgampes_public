from elasticsearch import Elasticsearch
import os
from dotenv import load_dotenv
from src.embed import get_embeddings
import logging
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Azure OpenAI API connection
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
azure_key = os.getenv('AZURE_OPENAI_KEY')

# Elasticsearch connection
elasticsearch_host = os.getenv('ELASTICSEARCH_HOST')
es = Elasticsearch(elasticsearch_host)

def buscar_ids(ids_documento_gampes, ids_documento_mni):
    """
    Search for document IDs in Elasticsearch based on GAMPES and MNI document IDs.

    Args:
        ids_documento_gampes (list): List of GAMPES document IDs to search for.
        ids_documento_mni (list): List of MNI document IDs to search for.

    Returns:
        list: List of found document IDs.
    """
    logging.info("Searching for document IDs.")
    ids_encontrados = []

    try:
        for doc_id in ids_documento_gampes:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"id_documento_gampes": doc_id}},
                            {"match": {"fonte": "GAMPES"}}
                        ]
                    }
                },
                "size": 1000 # Adjust size as needed
            }
            response = es.search(index="gampes_textual", body=query)
            for hit in response['hits']['hits']:
                ids_encontrados.append(hit['_id'])

        for doc_id in ids_documento_mni:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"match": {"id_documento_mni": doc_id}},
                            {"match": {"fonte": "MNI"}}
                        ]
                    }
                },
                "size": 1000 # Adjust size as needed
            }
            response = es.search(index="gampes_textual", body=query)
            for hit in response['hits']['hits']:
                ids_encontrados.append(hit['_id'])

        logging.info("Document IDs found successfully.")
    except Exception as e:
        logging.error(f"Error searching for document IDs: {e}")

    return ids_encontrados


def buscar_paginas_por_ids(ids, es):
    """
    Search for pages in Elasticsearch based on textual IDs.

    Args:
        ids (list): List of textual IDs to search for.

    Returns:
        list: List of found page IDs.
    """
    logging.info("Searching for pages by IDs.")
    documentos_encontrados = []

    try:
        for id_textual in ids:
            query = {
                "query": {
                    "match": {
                        "id_textual": id_textual
                    }
                },
                "size": 1000 # Adjust size as needed
            }
            resposta = es.search(index="gampes_textual_paginas", body=query)

            if resposta["hits"]["total"]["value"] > 0:
                for hit in resposta["hits"]["hits"]:
                    documentos_encontrados.append(hit["_id"])

        logging.info("Pages found successfully.")
    except Exception as e:
        logging.error(f"Error searching for pages by IDs: {e}")

    return documentos_encontrados


def buscar_vetores_por_ids(ids_paginas, es: Elasticsearch, key, endpoint, index_vector="gampes_vector_small"):
    """
    Search for vectors in Elasticsearch based on page IDs. If not found, create new vectors.

    Args:
        ids_paginas (list): List of page IDs to search for.

    Returns:
        list: List of found or newly created document IDs.
    """
    logging.info("Searching for vectors by page IDs.")
    ids_documentos = []

    if not isinstance(es, Elasticsearch):
        logging.error("The 'es' parameter must be an instance of Elasticsearch.")
        return []

    try:
        for id_pagina in ids_paginas:
            query = {
                "query": {
                    "match": {
                        "id_pagina": id_pagina
                    }
                },
                "size": 1000 # Adjust as needed
            }

            response = es.search(index=index_vector, body=query)

            if response['hits']['total']['value'] > 0:
                ids_documentos.append(response['hits']['hits'][0]['_id'])
            else:
                embedding = get_embeddings(id_pagina, key, endpoint)
                new_doc = {
                    "id_pagina": id_pagina,
                    "embedding": embedding
                }
                es.index(index=index_vector, body=new_doc)
                logging.info(f"New document created for id_pagina: {id_pagina}")

    except Exception as e:
        logging.error(f"Error searching for vectors by page IDs: {e}")

    return ids_documentos


import logging # Make sure logging is imported


def vector_similarity_search(es, prompt_embedding, id_list, k=5, similarity_threshold=0.7):
    """
    Perform a vector similarity search in Elasticsearch.

    Args:
        es (Elasticsearch): The Elasticsearch client instance. # Added es type hint
        prompt_embedding (list): The embedding vector of the prompt.
        id_list (list): List of page IDs to search within.
        k (int): Number of top results to return.
        similarity_threshold (float): Minimum similarity score threshold.

    Returns:
        list: List of tuples containing document IDs and their similarity scores.
    """
    logging.info("Performing vector similarity search.")

    # --- Verification Step ---
    expected_dims = 1536 # Define expected dimension based on your mapping
    if not isinstance(prompt_embedding, list) or len(prompt_embedding) != expected_dims:
        logging.error(f"Prompt embedding dimension mismatch or invalid type. "
                      f"Expected list of {expected_dims} floats, got {type(prompt_embedding)} "
                      f"with length {len(prompt_embedding) if isinstance(prompt_embedding, list) else 'N/A'}.")
        return [] # Return empty list or raise an error
    # --- End Verification ---

    try:
        # Consider using min_score for efficiency if your ES version supports it well with script_score
        # min_score_adjusted = similarity_threshold + 1.0

        query = {
            "size": k,
            # "min_score": min_score_adjusted, # Optional: Filter by score directly in ES
            "query": {
                "script_score": {
                    "query": {
                        # Use bool/filter context for non-scoring queries like terms
                        "bool": {
                            "filter": [
                                {"terms": {"id_pagina": id_list}},
                                # Optional but good practice: ensure embedding field exists
                                # {"exists": {"field": "embedding"}}
                            ]
                        }
                    },
                    "script": {
                        # Ensure the embedding field exists and is valid before calculating similarity
                        # This adds robustness but might slightly impact performance. Test if needed.
                        "source": """
                            if (doc['embedding'] == null || doc['embedding'].size() == 0) {
                                return 0; // Or some other default score for docs missing embedding
                            }
                            return cosineSimilarity(params.query_vector, 'embedding') + 1.0;
                        """,
                        "params": {"query_vector": prompt_embedding}
                    }
                }
            }
        }

        response = es.search(index="gampes_vector_small", body=query)

        results = []
        for hit in response["hits"]["hits"]:
            # Check if _score exists and is valid before processing
            if hit.get("_score") is not None:
                score = hit["_score"] - 1.0
                # Filter results based on the original cosine similarity score
                if score >= similarity_threshold:
                   # Check if _source and id_pagina exist
                   if hit.get("_source") and hit["_source"].get("id_pagina"):
                       results.append((hit["_source"]["id_pagina"], score))
                   else:
                       logging.warning(f"Hit missing _source or id_pagina: {hit.get('_id')}")
            else:
                logging.warning(f"Hit missing _score: {hit.get('_id')}")


        # Sort results descending by score (Elasticsearch usually does, but good to ensure)
        results.sort(key=lambda x: x[1], reverse=True)

        logging.info(f"Vector similarity search completed. Found {len(results)} results above threshold.")
        return results # Return filtered results directly

    except Exception as e:
        # Log the specific query body that caused the error for easier debugging
        logging.error(f"Error performing vector similarity search: {e}", exc_info=True)
        logging.error(f"Query Body: {query}") # Log the query
        return []


def bm25_similarity_search(es, prompt, id_list, k=5):
    """
    Perform a BM25 similarity search in Elasticsearch.

    Args:
        es (Elasticsearch): The Elasticsearch client instance.
        prompt (str): The search prompt.
        id_list (list): List of page IDs to search within.
        k (int): Number of top results to return.

    Returns:
        list: List of tuples containing document IDs and their BM25 scores.
    """
    logging.info("Performing BM25 similarity search.")
    try:
        query = {
            "query": {
                "bool": {
                    "must": [{
                        "multi_match": {
                            "query": prompt,
                            "fields": ["texto"],
                            "type": "best_fields",
                            "tie_breaker": 0.3
                        }
                    }],
                    "filter": [{"terms": {"_id": id_list}}]
                }
            },
            "size": k
        }
        
        response = es.search(index="gampes_textual_paginas", body=query)
        top_results = [(hit["_id"], hit["_score"]) for hit in response['hits']['hits']]
        
        logging.info("BM25 similarity search completed successfully.")
        return top_results
    except Exception as e:
        logging.error(f"Error performing BM25 similarity search: {e}")
        return []


def merge_and_rerank(vector_results, bm25_results, vector_weight=0.5, bm25_weight=0.5):
    """
    Merge and rerank results from vector and BM25 similarity searches.

    Args:
        vector_results (list): List of tuples containing document IDs and vector similarity scores.
        bm25_results (list): List of tuples containing document IDs and BM25 scores.
        vector_weight (float): Weight for vector similarity scores.
        bm25_weight (float): Weight for BM25 scores.

    Returns:
        list: List of tuples containing document IDs and their combined scores.
    """
    logging.info("Merging and reranking results.")
    combined_scores = {}
    
    for doc_id, score in vector_results:
        combined_scores[doc_id] = {
            'vector': score,
            'bm25': 0.0
        }
    
    for doc_id, score in bm25_results:
        if doc_id in combined_scores:
            combined_scores[doc_id]['bm25'] = score
        else:
            combined_scores[doc_id] = {
                'vector': 0.0,
                'bm25': score
            }
    
    merged = []
    for doc_id, scores in combined_scores.items():
        total_score = (scores['vector'] * vector_weight) + (scores['bm25'] * bm25_weight)
        merged.append((doc_id, total_score))
    
    merged.sort(key=lambda x: x[1], reverse=True)
    logging.info("Merging and reranking completed successfully.")
    return merged


def merge_and_rerank_rrf(vector_results, bm25_results, k=60):
    """
    Merge and rerank results using Reciprocal Rank Fusion (RRF).

    Args:
        vector_results (list): List of tuples (doc_id, score) from vector search.
                               Can be empty if all scores were below threshold.
        bm25_results (list): List of tuples (doc_id, score) from BM25 search.
        k (int): Parameter for RRF, controlling the influence of rank.
                 Lower k gives more weight to top ranks. Defaults to 60.

    Returns:
        list: List of tuples (doc_id, rrf_score) sorted by RRF score descending.
              Returns an empty list if both input lists are empty.
    """
    logging.info(f"Merging and reranking results using RRF (k={k}).")
    
    # Handle cases where one or both lists might be empty
    if not vector_results and not bm25_results:
        logging.warning("Both vector_results and bm25_results are empty. Returning empty list.")
        return []
        
    rrf_scores = defaultdict(float)

    # Process vector results (if any)
    if vector_results:
        logging.info(f"Processing {len(vector_results)} vector results.")
        for rank, (doc_id, _) in enumerate(vector_results):
            # Rank starts at 0, RRF uses 1-based rank, hence rank + 1
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    else:
         logging.info("vector_results list is empty.")


    # Process BM25 results (if any)
    if bm25_results:
        logging.info(f"Processing {len(bm25_results)} BM25 results.")
        for rank, (doc_id, _) in enumerate(bm25_results):
            # Rank starts at 0, RRF uses 1-based rank, hence rank + 1
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    else:
        logging.info("bm25_results list is empty.") # Should generally not happen unless query is bad

    # Convert the defaultdict to a list of tuples and sort
    merged = list(rrf_scores.items())
    merged.sort(key=lambda x: x[1], reverse=True)

    logging.info(f"RRF merging and reranking completed. Found {len(merged)} unique documents.")
    return merged


def get_document_fields(es, _id):
    """
    Retrieve document fields from Elasticsearch based on document ID.

    Args:
        _id (str): The document ID.

    Returns:
        dict: Dictionary containing document fields (id_textual, pagina, texto) or None if not found.
    """
    logging.info(f"Retrieving document fields for ID: {_id}")
    try:
        index_name = "gampes_textual_paginas"
        response = es.get(index=index_name, id=_id)

        if response['found']:
            id_textual = response['_source']['id_textual']
            pagina = response['_source']['pagina']
            texto = response['_source']['texto']

            return {
                'id_textual': id_textual,
                'pagina': pagina,
                'texto': texto
            }
        else:
            return None
    except Exception as e:
        logging.error(f"Error retrieving document fields for ID {_id}: {e}")
        return None

def process_merged_results(es, merged_results):
    """
    Process merged results to retrieve document fields.

    Args:
        merged_results (list): List of tuples containing document IDs and their combined scores.

    Returns:
        list: List of dictionaries containing document fields and scores.
    """
    logging.info("Processing merged results.")
    processed_results = []

    for doc_id, score in merged_results:
        document_fields = get_document_fields(es, doc_id)
        
        if document_fields:
            id_pagina = doc_id
            id_textual = document_fields['id_textual']
            pagina = document_fields['pagina']
            texto = document_fields['texto']
            
            processed_results.append({
                'id_pagina': id_pagina,
                'id_textual': id_textual,
                'pagina': pagina,
                'texto': texto,
                'score': score
            })
    
    logging.info("Merged results processed successfully.")
    return processed_results

def enhance_results(es, final_results, es_index="gampes_textual"):
    """
    Enhance results by fetching additional data from Elasticsearch.

    Args:
        final_results (list): List of dictionaries containing document fields and scores.
        es_index (str): Elasticsearch index to fetch additional data from.

    Returns:
        list: List of dictionaries containing enriched document fields and scores.
    """
    logging.info("Enhancing results.")
    enriched_results = []
    
    for result in final_results:
        id_textual = result['id_textual']
        
        try:
            response = es.get(index=es_index, id=id_textual)
            source = response['_source']
            id_documento_gampes = source.get('id_documento_gampes', None)
            id_documento_mni = source.get('id_identificador_MNI', None)
        except Exception as e:
            logging.error(f"Error fetching data from Elasticsearch for id_textual {id_textual}: {e}")
            id_documento_gampes = None
            id_documento_mni = None
        
        enriched_result = result.copy()
        enriched_result['id_documento_gampes'] = id_documento_gampes
        enriched_result['id_documento_mni'] = id_documento_mni
        
        enriched_results.append(enriched_result)
    
    return enriched_results


def update_document(
    id,                # Sempre precisa de um ID para update
    es,
    index="gampes_agent_assessorvirtual",	
    id_requisicao=None,
    texto_resposta=None,
    texto_aux=None,
    status=None,
    mensagem_erro=None,
    data_criacao=None,
    usuario=None,
    tipo_requisicao=None
):
    # Prepara o dicionário apenas com campos não-nulos
    fields = [
        ('id_requisicao', id_requisicao),
        ('texto_resposta', texto_resposta),
        ('texto_aux', texto_aux),
        ('status', status),
        ('mensagem_erro', mensagem_erro),
        ('data_criacao', data_criacao),
        ('usuario', usuario),
        ('tipo_requisicao', tipo_requisicao)
    ]
    body = {"doc": {key: value for key, value in fields if value is not None}}

    if not body["doc"]:
        print("Nenhum campo para atualizar.")
        return

    response = es.update(index=index, id=id, body=body)
    return response
