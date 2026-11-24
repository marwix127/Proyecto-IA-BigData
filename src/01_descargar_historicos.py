

import calendar
import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# --- Configuración -----------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent   # raíz del repositorio
RAW = RAIZ / "data" / "raw"

INICIO = date(2024, 1, 1)
FIN = date(2026, 10, 31)

load_dotenv(RAIZ / ".env")
ESIOS_TOKEN = os.getenv("ESIOS_API_KEY")

ESIOS_HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": ESIOS_TOKEN,
}

# id del indicador -> (nombre del fichero, filtro de zona en la petición)
ESIOS_INDICADORES = {
    600: ("precio", 3),               # 3 = España
    460: ("prevision_demanda", None), # peninsulares: sin filtro, se filtra al cargar
    541: ("prevision_eolica", None),
    542: ("prevision_solar", None),
}

# grupo de puntos.json -> variable horaria de Open-Meteo
OPENMETEO_VARIABLES = {
    "temperatura": "temperature_2m",
    "viento": "wind_speed_100m",
    "solar": "shortwave_radiation",
}

PAUSA = 1.0       # segundos entre peticiones, para no saturar las APIs
REINTENTOS = 3


# --- Utilidades --------------------------------------------------------------

def pedir(url, params, headers=None):
    """GET con reintentos. Devuelve el JSON o lanza el último error."""
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"    intento {intento} fallido: {e}")
            if intento == REINTENTOS:
                raise
            time.sleep(5 * intento)


def guardar(data, fichero):
    fichero.parent.mkdir(parents=True, exist_ok=True)
    with open(fichero, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def meses(inicio, fin):
    """Genera (año, mes) desde inicio hasta fin, ambos incluidos."""
    a, m = inicio.year, inicio.month
    while (a, m) <= (fin.year, fin.month):
        yield a, m
        a, m = (a + 1, 1) if m == 12 else (a, m + 1)


# --- ESIOS -------------------------------------------------------------------

def descargar_esios():
    if not ESIOS_TOKEN:
        raise SystemExit("Falta ESIOS_TOKEN en el fichero .env")

    for id_ind, (nombre, geo) in ESIOS_INDICADORES.items():
        print(f"\nESIOS {id_ind} ({nombre})")
        for a, m in meses(INICIO, FIN):
            fichero = RAW / "esios" / f"{nombre}_{a}-{m:02d}.json"
            if fichero.exists():
                continue

            ultimo_dia = calendar.monthrange(a, m)[1]
            params = {
                "start_date": f"{a}-{m:02d}-01T00:00:00Z",
                "end_date": f"{a}-{m:02d}-{ultimo_dia}T23:59:59Z",
                "time_trunc": "hour",
                "time_agg": "average",
            }
            if geo is not None:
                params["geo_ids[]"] = geo

            data = pedir(f"https://api.esios.ree.es/indicators/{id_ind}",
                         params, ESIOS_HEADERS)
            guardar(data, fichero)

            n = len(data["indicator"]["values"])
            esperado = ultimo_dia * 24
            aviso = "" if n == esperado else f"  <-- esperado {esperado}"
            print(f"  {a}-{m:02d}: {n} valores{aviso}")
            time.sleep(PAUSA)


# --- Open-Meteo --------------------------------------------------------------

def descargar_openmeteo():
    with open(RAIZ / "config" / "ciudades.json", encoding="utf-8") as f:
        puntos = json.load(f)["puntos"]

    for grupo, variable in OPENMETEO_VARIABLES.items():
        del_grupo = [p for p in puntos if p["grupo"] == grupo]
        print(f"\nOpen-Meteo {grupo} ({variable}): {len(del_grupo)} puntos")

        for a in range(INICIO.year, FIN.year + 1):
            fichero = RAW / "openmeteo" / f"{grupo}_{a}.json"
            if fichero.exists():
                continue

            inicio = max(INICIO, date(a, 1, 1))
            fin = min(FIN, date(a, 12, 31))
            params = {
                "latitude": ",".join(str(p["latitud"]) for p in del_grupo),
                "longitude": ",".join(str(p["longitud"]) for p in del_grupo),
                "start_date": inicio.isoformat(),
                "end_date": fin.isoformat(),
                "hourly": variable,
                "timezone": "GMT",
            }
            data = pedir("https://archive-api.open-meteo.com/v1/archive", params)
            if isinstance(data, dict):   # un solo punto -> lista de uno
                data = [data]
            guardar(data, fichero)

            esperado = ((fin - inicio).days + 1) * 24
            for p, res in zip(del_grupo, data):
                valores = res["hourly"][variable]
                nulos = sum(v is None for v in valores)
                aviso = "" if len(valores) == esperado else f"  <-- esperado {esperado}"
                print(f"  {a} {p['id']}: {len(valores)} valores, {nulos} nulos{aviso}")
            time.sleep(PAUSA)


if __name__ == "__main__":
    descargar_esios()
    descargar_openmeteo()
    print("\nDescarga terminada.")