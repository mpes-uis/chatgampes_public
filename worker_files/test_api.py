import requests

# Define the URL for the FastAPI endpoint
url = "http://ebino.mpes.gov.br:2001/rag/"

# Define the payload
payload = {
    "texto_prompt": "escreva uma denuncia para o processo",
    "id_documentos_mni": [23039571, 22948832, 22948833, 22948836, 22948837, 22948839, 22948841, 22948843, 22948844, 25651396, 22948840, 22948834, 22948835, 22948838, 22948842, 25651397],
    "id_documentos_gampes": [7892868],
    "idfuncao": "987",
    "idorgao": "456",
    "user": "fulano"
}

# Send a POST request to the endpoint
response = requests.post(url, json=payload)

# Print the response
print(response.status_code)
print(response.json())