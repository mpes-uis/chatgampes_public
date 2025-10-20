
## Documentação da Aplicação RAG para Chatbot GAMPES (MPES)

### 1. Visão Geral

Esta aplicação FastAPI implementa um chatbot baseado na arquitetura RAG (Retrieval-Augmented Generation). Seu objetivo principal é auxiliar usuários do sistema GAMPES (Ministério Público do Estado do Espírito Santo - MPES) na análise de documentos jurídicos, como inquéritos e processos, fornecendo respostas baseadas no conteúdo desses documentos.

A aplicação funciona como um serviço backend que:
1.  Recebe uma consulta do usuário e identificadores de documentos relevantes (dos sistemas GAMPES e MNI).
2.  Recupera trechos de texto pertinentes (chunks/páginas) desses documentos armazenados no Elasticsearch.
3.  Utiliza um modelo de linguagem grande (LLM), especificamente o Azure OpenAI, para aprimorar a consulta inicial e gerar uma resposta final fundamentada nos trechos recuperados.
4.  Combina busca lexical (BM25) e busca semântica (vetorial) para uma recuperação de informação mais robusta (busca híbrida).
5.  Busca informações adicionais sobre os documentos no Elasticsearch.
6.  Registra detalhes da interação e métricas de desempenho em um banco de dados SQL Server.
7.  É projetada para rodar em um container Docker, facilitando o deploy e o gerenciamento.

### 2. Tecnologias Principais

*   **FastAPI:** Framework web Python para construir a API RESTful.
*   **Elasticsearch:** Banco de dados NoSQL utilizado para:
    *   Armazenar os textos dos documentos, divididos em páginas/chunks.
    *   Indexar os textos para busca lexical (BM25).
    *   Armazenar embeddings vetoriais das páginas para busca semântica.
    *   Armazenar metadados dos documentos/páginas.
*   **Azure OpenAI:** Serviço de LLM utilizado para:
    *   Refinar a consulta do usuário (Prompt Enhancement).
    *   Gerar a resposta final com base no contexto recuperado (Generation).
*   **SQL Server:** Banco de dados relacional usado para armazenar logs de execução, métricas e potencialmente metadados adicionais (configurado via `SQL_SERVER_CNXN_STR_IA`).
*   **Docker:** Plataforma de containerização para empacotar e executar a aplicação.
*   **Python Libraries:** `elasticsearch-py`, `python-dotenv`, `requests` (implícito em `src.utils.consultar_apis`), `openai` (implícito em `src.model.generate_chat_completion`), `markdown`.

### 3. Configuração

A aplicação utiliza variáveis de ambiente para configuração, carregadas de um arquivo `.env`. As principais variáveis incluem:

*   **Azure OpenAI:**
    *   `ENDPOINT_URL` / `AZURE_OPENAI_ENDPOINT`: Endpoint da API do Azure OpenAI.
    *   `AZURE_OPENAI_KEY` / `AZURE_OPENAI_API_KEY`: Chave de API para autenticação.
    *   `DEPLOYMENT_NAME`: Nome do deployment do modelo no Azure OpenAI.
*   **Elasticsearch:**
    *   `ELASTICSEARCH_HOST`: Endereço do cluster Elasticsearch.
    *   `ELASTICSEARCH_USER`: Usuário para autenticação.
    *   `ELASTICSEARCH_PASSWORD`: Senha para autenticação.
*   **SQL Server:**
    *   `SQL_SERVER_CNXN_STR_IA`: String de conexão para o banco de dados SQL Server.
*   **APIs de OCR:**
    *   `URL_API_OCR_GAMPES`: URL da API para consulta/OCR de documentos GAMPES.
    *   `URL_API_OCR_MNI`: URL da API para consulta/OCR de documentos MNI.

### 4. Fluxo de Execução da API (`POST /rag`)

O endpoint `/rag` é o coração da aplicação e segue um pipeline RAG estruturado em várias fases:

**Payload de Entrada:** A API espera um payload JSON contendo:
*   `id_documentos_mni`: Lista de IDs de documentos do sistema MNI.
*   `id_documentos_gampes`: Lista de IDs de documentos do sistema GAMPES.
*   `texto_prompt`: A pergunta/consulta original do usuário.

**Fases do Processamento:**

1.  **Fase 1: Preparação**
    *   Recebe e valida o payload JSON.
    *   Extrai os IDs dos documentos e a consulta original.
    *   Registra o início do processamento e o tempo.

2.  **Fase 2: Identificação de Conteúdo e Páginas Relevantes**
    *   **Consulta a APIs Externas (`consultar_apis`):** Utiliza as URLs `URL_API_OCR_GAMPES` e `URL_API_OCR_MNI` para, possivelmente, obter IDs textuais ou garantir que o OCR dos documentos especificados (`id_documentos_gampes`, `id_documentos_mni`) foi realizado e está disponível no Elasticsearch. O resultado é uma lista de IDs (`id_textual_list`) que representam os conteúdos textuais relevantes no Elasticsearch.
    *   **Busca de Páginas (`buscar_paginas_por_ids`):** Consulta o Elasticsearch para encontrar os identificadores únicos das páginas/chunks (`id_paginas_list`) que pertencem aos IDs textuais recuperados na etapa anterior. Isso define o escopo da busca dentro do Elasticsearch.
    *   *(Opcional/Implícito: `buscar_vetores_por_ids` pode ser usado internamente nas buscas ou para pré-filtragem, mas as buscas subsequentes usam `id_paginas_list` como filtro principal). Porém a função varre as páginas e vetoriza as que não estavam vetorizadas ainda.*

3.  **Fase 2.1: Aprimoramento do Prompt (Consulta)**
    *   Envia a consulta original do usuário (`prompt_original`) para o Azure OpenAI (`generate_chat_completion`) junto com um prompt de sistema (`role_upgrade_prompt`) que instrui o modelo a refinar e expandir a consulta, tornando-a mais detalhada e incluindo termos jurídicos relevantes.
    *   Recebe a consulta aprimorada (`prompt_enhanced`).
    *   Gera o embedding vetorial da consulta aprimorada (`get_embeddings`) para uso na busca semântica.

4.  **Fase 3: Busca Híbrida (Recuperação)**
    *   **Busca Lexical (BM25):** Realiza uma busca BM25 (`bm25_similarity_search`) no Elasticsearch usando a consulta aprimorada (`prompt_enhanced`) contra as páginas identificadas (`id_paginas_list`). Retorna os `bm25_top_k` resultados mais relevantes com base na frequência e raridade dos termos.
    *   **Busca Semântica (Vetorial):** Realiza uma busca por similaridade de cosseno (`vector_similarity_search`) no Elasticsearch usando o embedding da consulta aprimorada (`prompt_embedding`) contra os vetores das páginas identificadas (`id_paginas_list`). Retorna os `vector_top_k` resultados vetoriais mais próximos semanticamente.

5.  **Fase 4: Reranking**
    *   Combina os resultados das buscas BM25 e vetorial (`merge_and_rerank`).
    *   Aplica um algoritmo de reranking (Reciprocal Rank Fusion - RRF implícito pela função) com pesos configuráveis (`vector_weight`, `bm25_weight`) para gerar uma lista unificada e reordenada de páginas/chunks relevantes.

6.  **Fase 5: Construção do Contexto e Compressão**
    *   Processa os resultados combinados e reordenados (`process_merged_results`), possivelmente extraindo o texto ou formatando.
    *   Seleciona os `merged_top_k` resultados mais relevantes após o reranking.
    *   Enriquece esses resultados (`enhance_results`), buscando no Elasticsearch (e potencialmente no SQL Server) informações adicionais como o texto completo da página, metadados do documento (nome, tipo), número da página, etc.
    *   Monta o prompt final (`create_full_prompt`) para o LLM, que inclui:
        *   A consulta aprimorada (`prompt_enhanced`).
        *   O contexto recuperado e enriquecido (trechos de texto das `enhanced_results`).
        *   Instruções específicas sobre como gerar a resposta (implícito no `role_answer` usado na próxima fase).
    *   Inclui uma pausa (`time.sleep(1)`) para evitar limites de taxa da API do LLM.

7.  **Fase 6: Geração da Resposta (LLM)**
    *   Envia o prompt final (`prompt_final`) para o Azure OpenAI (`generate_chat_completion`), usando o prompt de sistema `role_answer`. Este prompt instrui o modelo a agir como um assistente jurídico, responder *somente* com base no contexto fornecido, citar as fontes (ID do documento e página), e manter um tom formal.
    *   Recebe a resposta gerada pelo LLM (`llm_response`).
    *   Converte a resposta (assumida como Markdown) para HTML (`markdown.markdown`) para exibição facilitada no frontend do GAMPES.

8.  **Fase 7: Montagem da Resposta Final**
    *   Estrutura a resposta final (`build_structured_response`) em um formato JSON padronizado, contendo:
        *   A resposta do LLM em formato HTML.
        *   A lista de fontes (documentos/páginas) utilizadas para gerar a resposta (`enhanced_results`, que contêm metadados como ID do documento, página, etc.).
        *   O ID da resposta do LLM (`llm_response["id"]`).

9.  **Fase 8: Logging e Métricas**
    *   Calcula o tempo total de processamento da requisição.
    *   Se nenhuma exceção ocorreu durante o processo, salva informações detalhadas da execução no banco de dados SQL Server (`save_logs_to_database`). Isso inclui:
        *   A resposta do LLM.
        *   O prompt original do usuário.
        *   O prompt final enviado ao LLM (com contexto).
        *   O payload original da requisição.
        *   O tempo de processamento.

10. **Retorno:** Retorna a resposta JSON estruturada para o cliente (sistema GAMPES).

### 5. Módulos Auxiliares (`src.*`)

*   **`src.elastic`:** Contém funções para interagir com o Elasticsearch (buscar IDs, páginas, vetores, realizar buscas BM25 e vetorial, processar e enriquecer resultados).
*   **`src.embed`:** Responsável por gerar os embeddings vetoriais das consultas usando um modelo apropriado (provavelmente via API ou biblioteca local).
*   **`src.model`:** Encapsula a lógica de interação com o modelo LLM (Azure OpenAI), como a função `generate_chat_completion`.
*   **`src.prompt`:** Contém funções para construir os prompts enviados ao LLM, incluindo a formatação do contexto e a montagem da resposta final estruturada.
*   **`src.utils`:** Funções utilitárias diversas, como `save_logs_to_database` para persistir logs no SQL Server e `consultar_apis` para interagir com as APIs de OCR/documentos.

### 6. Tratamento de Erros

A aplicação implementa tratamento de erros básico:
*   Captura `json.JSONDecodeError` para payloads inválidos (retorna HTTP 400).
*   Captura `KeyError` para chaves ausentes no payload (retorna HTTP 400).
*   Captura exceções genéricas (`Exception`) para outros erros inesperados durante o processamento (retorna HTTP 500).
*   Erros são logados usando o módulo `logging`.
*   Importante: Os logs só são salvos no banco de dados SQL Server se o processamento for concluído *sem* exceções.

### 7. Deployment e Execução

*   **Docker:** A aplicação é projetada para ser executada dentro de um container Docker, o que requer um `Dockerfile` (não fornecido aqui) e uma imagem construída.
*   **Execução Local (Desenvolvimento):** O comando `uvicorn main:app --reload` pode ser usado para iniciar o servidor FastAPI localmente, geralmente para fins de desenvolvimento e teste. O `--reload` monitora alterações nos arquivos e reinicia o servidor automaticamente.

---

Faça o que você ama e ame o que você faz S2
