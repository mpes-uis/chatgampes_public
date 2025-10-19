# RAG Gampes Chatbot

Este repositório contém o código para um chatbot de Geração Aumentada por Recuperação (RAG), dividido em dois componentes principais: uma **API** para receber requisições e um **Worker** para processá-las de forma assíncrona.

## Arquitetura

O sistema é projetado para lidar com requisições de chat de forma robusta e escalável, seguindo o fluxo abaixo:

1.  **Recebimento da Requisição**: A **API**, construída com FastAPI, recebe uma requisição de chat através de um endpoint POST.
2.  **Criação da Tarefa**: A API cria um registro inicial no Elasticsearch com o status "processando" e insere uma nova tarefa na tabela `fila_processamento_agentes` em um banco de dados SQL Server. A API retorna imediatamente um `task_id` e uma URL para consultar o status da tarefa.
3.  **Processamento Assíncrono**: O **Worker** monitora continuamente a fila de tarefas no SQL Server. Ao encontrar uma nova tarefa, ele a inicia.
4.  **Pipeline RAG**: Para cada tarefa, o Worker executa um pipeline de RAG completo:
    *   **Recuperação de Documentos**: Busca documentos e páginas relevantes de fontes de dados (APIs OCR, Elasticsearch).
    *   **Geração de Prompt Aprimorado**: Utiliza um modelo de linguagem para refinar e otimizar o prompt original do usuário.
    *   **Busca Híbrida**: Realiza buscas de similaridade BM25 (baseada em palavras-chave) e vetorial para encontrar os trechos mais relevantes nos documentos.
    *   **Reranking**: Combina e reclassifica os resultados das buscas para selecionar os melhores candidatos.
    *   **Geração de Resposta**: Monta um prompt final com o contexto recuperado e o envia para um modelo de linguagem para gerar a resposta.
5.  **Armazenamento e Atualização**: A resposta final é armazenada no Elasticsearch, e o status da tarefa é atualizado no SQL Server para "concluído". O usuário pode consultar o resultado final através da URL de status fornecida pela API.

## Funcionalidades

*   **Processamento Assíncrono**: Permite que a API responda rapidamente enquanto o processamento pesado é feito em segundo plano.
*   **Busca Híbrida e Reranking**: Combina o melhor das buscas por palavra-chave e semântica para maior precisão na recuperação de informações.
*   **Escalabilidade**: A arquitetura com fila permite escalar o número de workers para lidar com uma carga maior de requisições.
*   **Monitoramento de Tarefas**: Endpoints para verificar o status e o resultado de cada requisição.
*   **Logging Detalhado**: Registra as fases do pipeline e métricas de desempenho no banco de dados.

## Tecnologias Utilizadas

*   **Backend**: Python
*   **API**: FastAPI, Uvicorn
*   **Worker**: Python (loop de processamento)
*   **Banco de Dados**: SQL Server (para a fila de tarefas e logs)
*   **Busca e Armazenamento de Respostas**: Elasticsearch
*   **Modelos de Linguagem**: Azure OpenAI
*   **Containerização**: Docker

## Pré-requisitos

*   Docker e Docker Compose
*   Acesso a um servidor SQL Server
*   Acesso a um cluster Elasticsearch
*   Credenciais para as APIs do Azure OpenAI

## Instalação e Execução

### 1. Configuração do Ambiente

1.  **Clone o repositório**:
    ```sh
    git clone https://github.com/seu-usuario/rag_gampes_chatbot.git
    cd rag_gampes_chatbot
    ```

2.  **Crie os arquivos de ambiente**:
    *   Na raiz do projeto, crie um arquivo `.env` a partir do exemplo `worker/.env_exemple`.
    *   Preencha as variáveis de ambiente com as credenciais e endpoints corretos para o SQL Server, Elasticsearch e Azure OpenAI.

### 2. Executando com Docker

O projeto é configurado para ser executado com Docker.

1.  **Construa as imagens da API e do Worker**:
    ```sh
    docker build -t rag-gampes-api -f API/dockerfile .
    docker build -t rag-gampes-worker -f worker/Dockerfile .
    ```

2.  **Inicie os containers**:
    Você pode usar `docker run` para iniciar cada serviço individualmente ou criar um arquivo `docker-compose.yml` para orquestrar os dois.

    *   **Para a API**:
        ```sh
        docker run -d --name rag_api_container --env-file .env -p 8000:8000 rag-gampes-api
        ```
    *   **Para o Worker**:
        ```sh
        docker run -d --name rag_worker_container --env-file .env rag-gampes-worker
        ```

## Como Usar a API

### 1. Enviar uma Requisição de Chat

Envie uma requisição `POST` para o endpoint `/rag` com o payload contendo os detalhes da sua pergunta.

*   **URL**: `http://localhost:8000/rag`
*   **Método**: `POST`
*   **Body** (exemplo):
    ```json
    {
      "id_documentos_mni": [123, 456],
      "id_documentos_gampes": [789],
      "texto_prompt": "Qual o procedimento para solicitar férias?",
      "user": "nome.usuario"
    }
    ```

A API responderá imediatamente com um `task_id`:

```json
{
  "task_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "url": "http://<elasticsearch_host>/<index_name>/_doc/a1b2c3d4-e5f6-7890-1234-567890abcdef",
  "message": "Requisição recebida e processamento iniciado em segundo plano."
}
```

### 2. Consultar o Status da Tarefa

Use o `task_id` recebido para verificar o andamento e o resultado da sua requisição.

*   **URL**: `http://localhost:8000/rag/status/{task_id}`
*   **Método**: `GET`

A resposta conterá o status (`202` para processando, `200` para concluído) e, quando finalizado, o resultado completo.

## Estrutura do Projeto

```
/
├── API/                # Contém a aplicação FastAPI para receber requisições.
│   ├── app.py          # Lógica principal da API.
│   ├── dockerfile      # Dockerfile para a API.
│   └── requirements.txt
│
├── worker/             # Contém o worker para processamento assíncrono.
│   ├── main.py         # Lógica principal do worker e do pipeline RAG.
│   ├── Dockerfile      # Dockerfile para o worker.
│   ├── src/            # Módulos do pipeline RAG (Elastic, Model, Prompt, etc.).
│   └── requirements.txt
│
├── .gitignore
├── README.md           # Este arquivo.
└── LICENSE
```