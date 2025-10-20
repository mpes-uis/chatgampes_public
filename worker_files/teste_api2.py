import requests
import time

# URL base da API
#BASE_URL = "http://localhost:8000"
BASE_URL = "http://localhost:2001"

# Payload conforme especificado
payload = {
    "texto_prompt": "escreva uma denuncia para o processo",
    "id_documentos_mni": [23039571, 22948832, 22948833, 22948836, 22948837, 22948839, 22948841, 22948843, 22948844, 25651396, 22948840, 22948834, 22948835, 22948838, 22948842, 25651397],
    "id_documentos_gampes": [7892868],
    "idfuncao": "987",
    "idorgao": "456",
    "user": "fulano"
}

# 1. Disparar o processamento assíncrono
response = requests.post(f"{BASE_URL}/rag", json=payload)

if response.status_code == 202:
    task_data = response.json()
    task_id = task_data["task_id"]
    print(f"Tarefa iniciada com ID: {task_id}")
    print(f"URL para acompanhamento: {task_data['url']}")
    
    # 2. Verificar o status periodicamente (polling)
    while True:
        status_response = requests.get(f"{BASE_URL}/rag/status/{task_id}")
        status_data = status_response.json()
        
        print(f"Status atual: {status_data['status']}")
        
        if status_data['status'] != 102:  # 102 = processando
            print("Processamento concluído!")
            print(f"Resposta final: {status_data['texto_resposta']}")
            break
            
        time.sleep(2)  # Aguarda 2 segundos entre as verificações
else:
    print(f"Erro ao iniciar a tarefa: {response.status_code} - {response.text}")

status_response = requests.get(f"{BASE_URL}/rag/status/{task_id}")
print(status_response.json())