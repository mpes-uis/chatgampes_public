# RAG GAMPES

Agente RAG para documentos do sistema GAMPES.

## Documentaçao

https://github.com/mpes-uis/rag_gampes/blob/main/doc.md#:~:text=doc.md

## Pré-requisitos

- Docker
- Docker Compose
- SQL Server
- ElasticSearch
- APIs do OCR_APP_MPES

## Como rodar o container

1. Clone o repositório:

    ```sh
    git clone https://github.com/seu-usuario/rag_gampes.git
    cd rag_gampes
    ```

2. Construa a imagem Docker:

    ```sh
    docker build -t rag_gampes .
    ```

3. Rode o container:

    ```sh
    docker run -d -p 8080:80 --name rag_gampes_container rag_gampes
    ```

4. Verifique se o container está rodando:

    ```sh
    docker ps
    ```

## Como consumir a API

A API estará disponível em `http://localhost:8080/rag`.

### Exemplo de requisição

Você pode usar a ferramenta `curl` ou qualquer cliente HTTP para fazer uma requisição à API. Aqui está um exemplo usando `curl`:

```sh
curl -X POST "http://localhost:8080/rag" -H "Content-Type: application/json" -d '{
    "texto_prompt": "quem sao os envolvidos do processo?",
    "id_documentos_mni": [23204850, 23204851, 23204846, 23204845, 23204849, 23204848, 23204847, 23204844],
    "id_documentos_gampes": [251735, 7340129],
    "idfuncao": "987",
    "idorgao": "456",
    "user": "fulano"
}'
```

Usando python

```python
import requests

# Define the URL for the FastAPI endpoint
url = "http://localhost:8080/rag"

# Define the payload
payload = {
    "texto_prompt": "quem sao os envolvidos do processo?",
    "id_documentos_mni": [23204850, 23204851, 23204846, 23204845, 23204849, 23204848, 23204847, 23204844],
    "id_documentos_gampes": [251735, 7340129],
    "idfuncao": "987",
    "idorgao": "456",
    "user": "fulano"
}

# Send a POST request to the endpoint
response = requests.post(url, json=payload)

# Print the response
print(response.status_code)
print(response.json())
```






### Exemplo de resposta

A resposta será um JSON contendo o status e os resultados da consulta. Aqui está um exemplo de resposta:

{
    "id": "unique_id",
    "content": "Resposta gerada pelo modelo",
    "sources": {
        "fonte_1": {
            "pagina": 1,
            "score": 0.95,
            "id_documento_gampes": 251735,
            "id_documento_mni": 23204850
        },
        "fonte_2": {
            "pagina": 2,
            "score": 0.90,
            "id_documento_gampes": 7340129,
            "id_documento_mni": 23204851
        }
    }
}

