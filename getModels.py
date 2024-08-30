import requests
import os

api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

def get_models():
    response = requests.get(url, headers=headers)
    models = response.json().get('data', [])

    # Filter models based on active status and context window size
    filtered_models = [
        model for model in models
        if model['active'] and model['context_window'] > 8000 and 'guard' not in model['id']
    ]
    
    return filtered_models