import os  
from openai import AzureOpenAI  
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_chat_completion(endpoint, deployment, subscription_key, role, prompt):
    """
    Generate a chat completion using Azure OpenAI.

    Args:
        endpoint (str): The Azure OpenAI endpoint URL.
        deployment (str): The deployment name of the model.
        subscription_key (str): The subscription key for Azure OpenAI.
        role (str): The role description for the system message.
        prompt (str): The user prompt for the chat completion.

    Returns:
        dict: The completion result as a dictionary, or None if an error occurs.
    """
    try:
        logging.info("Initializing Azure OpenAI client.")
        # Initialize the Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=subscription_key,
            api_version="2024-05-01-preview",
        )

        # Prepare the chat prompt with the provided role and prompt
        chat_prompt = [
            {
                "role": "system",
                "content": role  # Change this to a string
            },
            {
                "role": "user",
                "content": prompt  # Change this to a string
            }
        ]

        logging.info("Generating chat completion.")
        # Generate the completion
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=4096,  # Aumente para o máximo que o modelo suporta (ajuste conforme necessário)
            temperature=0.2,  # Reduza para respostas mais determinísticas e coerentes
            top_p=1.0,        # Permita que o modelo explore todas as possibilidades
            frequency_penalty=0,  # Sem penalização de frequência
            presence_penalty=0,   # Sem penalização de presença
            stop=None,         # Sem paradas forçadas
            stream=False       # Desative o streaming para obter a resposta completa de uma vez
        )

        logging.info("Chat completion generated successfully.")
        # Return the completion result as JSON
        print(completion.to_dict())
        return completion.to_dict()

    except Exception as e:
        if "maximum context length" in str(e) or "context_length_exceeded" in str(e):
            logging.error("Exceeded token limit. Consider reducing input size.")
        else:
            logging.error(f"Error generating chat completion: {e}")
