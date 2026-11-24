import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
TOKEN = os.getenv("ESIOS_API_KEY")

url = "https://api.esios.ree.es/indicators"

headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}

params = {
        
}

r = requests.get(url, headers=headers, params=params)
r.raise_for_status()
data = r.json()
print(json.dumps(data, indent=4, ensure_ascii=False))



