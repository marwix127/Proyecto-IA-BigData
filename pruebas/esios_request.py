import json
import os
import requests
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
TOKEN = os.getenv("ESIOS_API_KEY")

url = "https://api.esios.ree.es/indicators/460"

headers = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}

params = {
    "start_date": "2026-10-01T00:00:00Z",
    "end_date": "2026-10-31T23:59:59Z",
    "time_trunc": "hour",     
    "time_agg": "average",   
    "geo_ids[]": 3,           
}

r = requests.get(url, headers=headers, params=params)
r.raise_for_status()
data = r.json()

valores = data["indicator"]["values"]

carpeta = Path("data/raw/esios")
carpeta.mkdir(parents=True, exist_ok=True)

fichero = carpeta / "demanda_2026-10.json"
with open(fichero, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Guardado en {fichero}")