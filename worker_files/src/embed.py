import os
import requests
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, NotFoundError
from typing import List
import logging

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

def get_embeddings(text, key, endpoint):
    """
    Generate embeddings for the given text using Azure OpenAI API.

    Args:
        text (str): The text to generate embeddings for.

    Returns:
        list: The generated embedding vector if successful, None otherwise.
    """

    logging.info("Generating embeddings.")
    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }
    payload = {"input": text}
    
    response = requests.post(endpoint, headers=headers, json=payload)
    
    if response.status_code == 200:
        logging.info("Embeddings generated successfully.")
        return response.json()["data"][0]["embedding"]
    else:
        logging.error(f"Error generating embedding: {response.status_code} - {response.text}")
        return None



def process_and_store_embedding(es_client: Elasticsearch, document_id: str) -> str:
    """
    Process and store the embedding for a single document in Elasticsearch.

    Args:
        es_client (Elasticsearch): The Elasticsearch client.
        document_id (str): The ID of the document to process.

    Returns:
        str: The ID of the stored document in the 'gampes_vector_small' index.
    """
    logging.info(f"Processing and storing embedding for document ID: {document_id}")
    try:
        existing_doc = es_client.search(
            index="gampes_vector_small",
            body={
                "query": {
                    "term": {
                        "id_pagina": document_id
                    }
                }
            }
        )
        
        if existing_doc['hits']['total']['value'] > 0:
            logging.info(f"Document with id_pagina {document_id} already exists in gampes_vector_small.")
            return existing_doc['hits']['hits'][0]['_id']
    
    except NotFoundError:
        pass
    
    doc = es_client.get(
        index="gampes_textual_paginas",
        id=document_id
    )
    
    texto_content = doc['_source']['texto']
    embedding_vector = get_embeddings(texto_content)
    
    if embedding_vector is None:
        raise ValueError("Failed to generate embedding")
    
    result = es_client.index(
        index="gampes_vector_small",
        document={
            "id_pagina": document_id,
            "embedding": embedding_vector
        }
    )
    
    logging.info(f"Embedding stored successfully for document ID: {document_id}")
    return result['_id']

def process_and_store_embeddings(es_client: Elasticsearch, document_ids: List[str]) -> List[str]:
    """
    Process and store embeddings for multiple documents in Elasticsearch.

    Args:
        es_client (Elasticsearch): The Elasticsearch client.
        document_ids (List[str]): The list of document IDs from gampes_textual_paginas to process.

    Returns:
        List[str]: The list of IDs of the stored documents in the 'gampes_vector_small' index.
    """
    logging.info("Processing and storing embeddings for multiple documents.")
    new_doc_ids = []
    
    for document_id in document_ids:
        try:
            existing_doc = es_client.search(
                index="gampes_vector_small",
                body={
                    "query": {
                        "term": {
                            "id_pagina": document_id
                        }
                    }
                }
            )
            
            if existing_doc['hits']['total']['value'] > 0:
                logging.info(f"Document with id_pagina {document_id} already exists in gampes_vector_small.")
                new_doc_ids.append(existing_doc['hits']['hits'][0]['_id'])
                continue
        
        except NotFoundError:
            pass
        
        try:
            doc = es_client.get(
                index="gampes_textual_paginas",
                id=document_id
            )
            
            texto_content = doc['_source']['texto']
            embedding_vector = get_embeddings(texto_content)
            
            if embedding_vector is None:
                logging.error(f"Failed to generate embedding for document {document_id}")
                continue
            
            result = es_client.index(
                index="gampes_vector_small",
                document={
                    "id_pagina": document_id,
                    "embedding": embedding_vector
                }
            )
            
            new_doc_ids.append(result['_id'])
        
        except Exception as e:
            logging.error(f"Error processing document {document_id}: {e}")
            continue
    
    logging.info("Embeddings processed and stored successfully for multiple documents.")
    return new_doc_ids