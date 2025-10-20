import requests
import time
import json
import sys

# Base URL of the API (change this to your actual API URL)
#BASE_URL = "http://ebino.mpes.gov.br:2001"  # Update if your API is hosted elsewhere
BASE_URL = "http://localhost:8000"  # Change to your API URL if needed


def print_response(response):
    """Helper function to print response details"""
    print(f"Status Code: {response.status_code}")
    try:
        print("Response Body:")
        print(json.dumps(response.json(), indent=2))
    except ValueError:
        print("Response Text:")
        print(response.text)

# Test the /rag endpoint
def test_rag_endpoint():
    print("Testing /rag endpoint...")
    
    payload = {
        "texto_prompt": "escreva uma denuncia para o processo",
        "id_documentos_mni": [23039571, 22948832, 22948833, 22948836, 22948837, 
                              22948839, 22948841, 22948843, 22948844, 25651396, 
                              22948840, 22948834, 22948835, 22948838, 22948842, 25651397],
        "id_documentos_gampes": [7892868, 2767571, 8534093, 2767556, 8970345],
        "idfuncao": "987",
        "idorgao": "456",
        "user": "fulano",
        "info": "Test payload for RAG endpoint"
    }
    
    try:
        print("Sending payload:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(f"{BASE_URL}/rag", json=payload)
        print_response(response)
        
        if response.status_code != 202:
            print(f"Unexpected status code: {response.status_code}")
            return None
        
        result = response.json()
        task_id = result.get("task_id")
        
        if not task_id:
            print("No task_id received in response")
            return None
            
        print(f"Task ID received: {task_id}")
        return task_id
        
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return None

# Test the /rag/status/{task_id} endpoint
def test_status_endpoint(task_id):
    if not task_id:
        print("No task_id provided, skipping status check")
        return
    
    print(f"\nTesting /rag/status/{task_id} endpoint...")
    
    try:
        # Initial check
        print("Initial status check:")
        response = requests.get(f"{BASE_URL}/rag/status/{task_id}")
        print_response(response)
        
        # Wait and check again
        print("\nWaiting 5 seconds and checking again...")
        time.sleep(5)
        
        response = requests.get(f"{BASE_URL}/rag/status/{task_id}")
        print_response(response)
        
    except requests.exceptions.RequestException as e:
        print(f"Status check failed: {str(e)}")

# Test the /evaluate endpoint
def test_evaluate_endpoint(task_id, eval_result=True, info="Test evaluation"):
    if not task_id:
        print("No task_id provided, skipping evaluation")
        return
    
    print(f"\nTesting /evaluate endpoint for task {task_id}...")
    
    payload = {
        "id": task_id,
        "eval": eval_result,
        "info": info
    }
    
    try:
        print("Sending evaluation:")
        print(json.dumps(payload, indent=2))
        
        response = requests.post(f"{BASE_URL}/evaluate", json=payload)
        print_response(response)
        
    except requests.exceptions.RequestException as e:
        print(f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    print("Starting API tests...\n")
    
    # Test RAG endpoint
    task_id = test_rag_endpoint()
    
    # Only proceed with other tests if we got a task_id
    if task_id:
        # Test status endpoint
        test_status_endpoint(task_id)
        
        # Test evaluation endpoint with positive evaluation
        test_evaluate_endpoint(task_id, True, "The response was accurate and helpful")
        
        # Test evaluation endpoint with negative evaluation
        test_evaluate_endpoint(task_id, False, "The response was not relevant to the query")
    else:
        print("\nInitial test failed, not proceeding with subsequent tests")
    
    print("\nTesting complete")