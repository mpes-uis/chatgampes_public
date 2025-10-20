import pyodbc
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
import requests
import time

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection details
connection_string = os.getenv("SQL_SERVER_CNXN_STR_IA")

# URLs das APIs para OCR
url_api_ocr_gampes = os.getenv("URL_API_OCR_GAMPES")
url_api_ocr_mni = os.getenv("URL_API_OCR_MNI")

def save_logs_to_database(connection_string, llm_response, prompt_original, prompt_final, prompt_data, tempo_processamento):
    """
    Save logs to the database.

    Args:
        connection_string (str): The database connection string.
        llm_response (dict): The response from the language model.
        prompt_original (str): The original prompt text.
        prompt_final (str): The final prompt text.
        prompt_data (dict): Additional data related to the prompt.
        tempo_processamento (float): Processing time in seconds.

    Returns:
        None
    """
    try:
        # Connect to the database
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Prepare the SQL query
        query = """
        INSERT INTO IA.dbo.rag_gampes (
            id, data, prompt_original, prompt_final, resposta, 
            prompt_tokens, completion_tokens, total_tokens, user_gampes, 
            idfuncao, idorgao, modelo, tempo_processamento
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Extract values from the LLM response and other variables
        values = (
            llm_response["id"],  # Unique ID from the LLM response
            datetime.now(),  # Current timestamp
            prompt_original,  # Original prompt text
            prompt_final,  # Final prompt text
            llm_response["choices"][0]["message"]["content"],  # LLM response
            llm_response["usage"]["prompt_tokens"],  # Prompt tokens
            llm_response["usage"]["completion_tokens"],  # Completion tokens
            llm_response["usage"]["total_tokens"],  # Total tokens
            prompt_data["user"],  # User who made the request
            prompt_data["idfuncao"],  # Function ID
            prompt_data["idorgao"],  # Organization ID
            llm_response["model"],  # Model used
            tempo_processamento  # Processing time in seconds
        )

        # Execute the query
        cursor.execute(query, values)
        conn.commit()
        logging.info("Logs saved to database successfully.")

    except Exception as e:
        logging.error(f"Error saving logs to database: {e}")

    finally:
        # Close the connection
        cursor.close()
        conn.close()


def consultar_apis(ids_documento_gampes, ids_documento_mni, url_api_ocr_gampes, url_api_ocr_mni):
    """
    Consult APIs and return document IDs.

    Args:
        ids_documento_gampes (list): List of document IDs for GAMPES.
        ids_documento_mni (list): List of document IDs for MNI.
        url_api_ocr_gampes (str): URL for the GAMPES OCR API.
        url_api_ocr_mni (str): URL for the MNI OCR API.

    Returns:
        list: Combined list of document IDs from both APIs.
    """
    def consultar_api(url, document_ids):
        payload = {"document_ids": document_ids}
        headers = {"Content-Type": "application/json"}
        tentativas = 0
        while tentativas < 3:
            try:
                response = requests.post(url, json=payload, headers=headers)
                if response.status_code == 200:
                    resultados = response.json().get("resultados", {})
                    return list(resultados.values())
                else:
                    print(f"Erro na requisição para {url}: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"Erro ao fazer a requisição para {url}: {e}")
            tentativas += 1
            if tentativas < 3:
                time.sleep(2)
        return []

    _ids_gampes = consultar_api(url_api_ocr_gampes, ids_documento_gampes)
    _ids_mni = consultar_api(url_api_ocr_mni, ids_documento_mni)
    return _ids_gampes + _ids_mni


def proximo_da_fila(connection_string):
    cnxn = pyodbc.connect(connection_string)
    cursor = cnxn.cursor()
    query_select = (
        "SELECT TOP 1 id, id_elasticsearch, tentativas, payload "
        "FROM fila_processamento_agentes "
        "WHERE status = 102 AND id_agente = 101 "
        "ORDER BY data_criacao ASC"
    )
    cursor.execute(query_select)
    row = cursor.fetchone()

    if row:
        query_update = (
            "UPDATE fila_processamento_agentes "
            "SET data_inicio_processamento = ?, status = ? "
            "WHERE id = ?"
        )
        cursor.execute(query_update, (datetime.now(), 202, row.id))
        cnxn.commit()

    cursor.close()
    cnxn.close()
    return row


def update_fila(id_value, connection_string, status, data_fim_processamento=None, erro_mensagem=None, tentativas=None):
    import pyodbc
    try:
        cnxn = pyodbc.connect(connection_string)
        cursor = cnxn.cursor()
        # Build the update query dynamically based on which optional parameters are provided
        update_fields = ["status = ?"]
        params = [status]
        if data_fim_processamento is not None:
            update_fields.append("data_fim_processamento = ?")
            params.append(data_fim_processamento)
        if erro_mensagem is not None:
            update_fields.append("erro_mensagem = ?")
            params.append(erro_mensagem)
        if tentativas is not None:
            update_fields.append("tentativas = ?")
            params.append(tentativas)
        params.append(id_value)
        query = f"UPDATE fila_processamento_agentes SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, params)
        cnxn.commit()
        cursor.close()
        cnxn.close()
        return True
    except Exception as e:
        return str(e)
