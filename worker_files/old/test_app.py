import gradio as gr
import requests
import json

def make_rag_request(texto_prompt, id_documentos_mni, id_documentos_gampes, idfuncao, idorgao, user):
    # Convert string inputs to appropriate types
    id_docs_mni = [int(id.strip()) for id in id_documentos_mni.split(',') if id.strip()]
    id_docs_gampes = [int(id.strip()) for id in id_documentos_gampes.split(',') if id.strip()]
    
    # Define the URL for the FastAPI endpoint
    url = "http://localhost:8080/rag"
    
    # Define the payload
    payload = {
        "texto_prompt": texto_prompt,
        "id_documentos_mni": id_docs_mni,
        "id_documentos_gampes": id_docs_gampes,
        "idfuncao": idfuncao,
        "idorgao": idorgao,
        "user": user
    }
    
    try:
        # Send a POST request to the endpoint
        response = requests.post(url, json=payload, timeout=30)
        
        # Return the status code and response as formatted JSON
        return (
            f"Status Code: {response.status_code}",
            json.dumps(response.json(), indent=4, ensure_ascii=False)
        )
    except requests.exceptions.ConnectionError:
        return "Connection Error", "Failed to connect to the server. Please ensure the API is running at http://localhost:8080"
    except requests.exceptions.Timeout:
        return "Timeout Error", "The request timed out. The server took too long to respond."
    except requests.exceptions.RequestException as e:
        return f"Request Error: {type(e).__name__}", str(e)
    except json.JSONDecodeError:
        return f"Status Code: {response.status_code}", "Response is not valid JSON"
    except Exception as e:
        return f"Error: {type(e).__name__}", str(e)

# Create the Gradio interface
with gr.Blocks(title="RAG API Interface", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG API Interface")
    gr.Markdown("Enter the parameters and submit to query the RAG endpoint.")
    
    with gr.Row():
        with gr.Column():
            # Input fields
            texto_input = gr.Textbox(
                label="Texto Prompt",
                placeholder="Ex: quem sao os envolvidos do processo?",
                lines=3
            )
            
            mni_input = gr.Textbox(
                label="ID Documentos MNI (comma-separated integers)",
                placeholder="Ex: 23204850, 23204851, 23204846",
                lines=2
            )
            
            gampes_input = gr.Textbox(
                label="ID Documentos GAMPES (comma-separated integers)",
                placeholder="Ex: 251735, 7340129",
                lines=2
            )
            
            with gr.Row():
                idfuncao_input = gr.Textbox(
                    label="ID Função",
                    placeholder="Ex: 987"
                )
                
                idorgao_input = gr.Textbox(
                    label="ID Órgão",
                    placeholder="Ex: 456"
                )
            
            user_input = gr.Textbox(
                label="Usuário",
                placeholder="Ex: fulano"
            )
            
            submit_btn = gr.Button("Submit Request", variant="primary")
    
    # Output areas
    status_output = gr.Textbox(label="Status", lines=1)
    response_output = gr.Code(language="json", label="Response", lines=20)
    
    # Set up the submit action
    submit_btn.click(
        fn=make_rag_request,
        inputs=[
            texto_input,
            mni_input,
            gampes_input,
            idfuncao_input,
            idorgao_input,
            user_input
        ],
        outputs=[status_output, response_output]
    )
    
    # Add example inputs
    gr.Examples(
        examples=[
            [
                "quem sao os envolvidos do processo?",
                "23204850, 23204851, 23204846, 23204845, 23204849, 23204848, 23204847, 23204844",
                "251735, 7340129",
                "987",
                "456",
                "fulano"
            ]
        ],
        inputs=[
            texto_input,
            mni_input,
            gampes_input,
            idfuncao_input,
            idorgao_input,
            user_input
        ]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()