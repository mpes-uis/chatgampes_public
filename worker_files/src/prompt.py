import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_full_prompt(prompt, enriched_dict):
    """
    Create a full prompt by appending information from the enriched dictionary.

    Args:
        prompt (str): The initial prompt string.
        enriched_dict (list): A list of dictionaries containing document information.

    Returns:
        str: The full prompt with appended document information.
    """
    logging.info("Creating full prompt.")
    full_prompt = prompt + "\n\n Fontes de informação:\n\n"
    
    for result in enriched_dict:
        full_prompt += f"Documento: GAMPES ID n: {result['id_documento_gampes']} PJe ID n: {result['id_documento_mni']}\nPagina {result['pagina']}:\n{result['texto']}\n\n"
    
    logging.info("Full prompt created successfully.")
    return full_prompt

def build_structured_response(content, fonte, id_prompt):
    """
    Build a structured response with content and sources.

    Args:
        content (str): The main content of the response.
        fonte (list): A list of dictionaries containing source information.
        id_prompt (str): The ID of the prompt.

    Returns:
        dict: A dictionary representing the structured response.
    """
    logging.info("Building structured response.")
    # Initialize the response structure
    response = {
        "id": id_prompt,
        "content": content,
        "sources": {}
    }

    # Iterate over the sources and add them to the response
    for index, source in enumerate(fonte, start=1):
        source_key = f"fonte_{index}"
        response["sources"][source_key] = {
            "pagina": source["pagina"],
            "score": source["score"],
            "id_documento_gampes": source["id_documento_gampes"],
            "id_documento_mni": source["id_documento_mni"]
        }

    logging.info("Structured response built successfully.")
    return response