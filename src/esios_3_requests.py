import os
import json
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("ESIOS_API_KEY")

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": TOKEN,
}

INDICADORES = {
    460: "prevision_demanda",
    541: "prevision_eolica",
    542: "prevision_solar",
}

INICIO = "2026-10-01T00:00:00Z"
FIN = "2026-10-31T23:59:59Z"
MES = "2026-10"

carpeta = Path("data/raw/esios")
carpeta.mkdir(parents=True, exist_ok=True)

for id_ind, nombre in INDICADORES.items():
    params = {
        "start_date": INICIO,
        "end_date": FIN,
        "time_trunc": "hour",
        "time_agg": "average",
        # sin geo_ids: comprobamos abajo qué zonas llegan
    }
    r = requests.get(f"https://api.esios.ree.es/indicators/{id_ind}",
                     headers=HEADERS, params=params)
    r.raise_for_status()
    data = r.json()

    valores = data["indicator"]["values"]
    zonas = sorted({(v.get("geo_id"), v.get("geo_name")) for v in valores})
    unidad = data["indicator"].get("magnitud")

    print(f"{id_ind} {nombre}: {len(valores)} valores | zonas: {zonas} | unidad: {unidad}")

    fichero = carpeta / f"{nombre}_{MES}.json"
    with open(fichero, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)