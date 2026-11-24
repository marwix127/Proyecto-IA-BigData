"""
Diagnóstico de los resultados del modelo.

Responde a tres preguntas:
  1. ¿Cambia el nivel del precio entre entrenamiento y prueba?
  2. ¿Alguna variable cambia de escala o comportamiento con el tiempo?
  3. ¿Los modelos tienen sesgo (se quedan siempre cortos o largos) y en qué meses fallan?

Uso (desde cualquier carpeta, después de entrenar.py):
    python src/diagnostico.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
FEATURES = RAIZ / "data" / "features" / "dataset"
RESULTADOS = RAIZ / "data" / "results"
INICIO_PRUEBA = pd.Timestamp("2025-11-01", tz="UTC")

COLUMNAS = ["precio", "demanda_prev", "eolica_prev", "solar_prev",
            "temp_media", "viento_medio", "radiacion_media"]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

df = pd.read_parquet(FEATURES).sort_values("hora_utc")
if df["hora_utc"].dt.tz is None:
    df["hora_utc"] = df["hora_utc"].dt.tz_localize("UTC")
mensual = df.set_index("hora_utc")[COLUMNAS].resample("MS").mean().round(1)
mensual.index = mensual.index.strftime("%Y-%m")

# 1 y 2. Medias mensuales del precio y de cada variable
print("=== 1-2. Media mensual de cada variable ===")
print("(busca saltos bruscos de un mes a otro, sobre todo en oct-2025 y dic-2025)\n")
print(mensual.to_string())

entreno = df[df["hora_utc"] < INICIO_PRUEBA]["precio"]
prueba = df[df["hora_utc"] >= INICIO_PRUEBA]["precio"]
print(f"\nPrecio medio entrenamiento: {entreno.mean():.1f} €/MWh (desv. {entreno.std():.1f})")
print(f"Precio medio prueba:        {prueba.mean():.1f} €/MWh (desv. {prueba.std():.1f})")
print(f"Horas de prueba por encima del máximo de entrenamiento: "
      f"{(prueba > entreno.max()).sum()}")

# 3. Sesgo y error por mes de cada modelo
pred = pd.read_parquet(RESULTADOS / "predicciones.parquet")
if pred["hora_utc"].dt.tz is None:
    pred["hora_utc"] = pred["hora_utc"].dt.tz_localize("UTC")
modelos = [c for c in pred.columns if c.startswith("pred_")]

print("\n=== 3a. Sesgo en prueba (media de predicción - real, €/MWh) ===")
print("(negativo = el modelo se queda corto; positivo = se pasa)\n")
for m in modelos:
    print(f"  {m:<18} {(pred[m] - pred['precio']).mean():+.1f}")

print("\n=== 3b. MAE por mes en prueba (€/MWh) ===\n")
pred["mes"] = pred["hora_utc"].dt.strftime("%Y-%m")
mae_mes = pd.DataFrame({
    m: (pred[m] - pred["precio"]).abs().groupby(pred["mes"]).mean() for m in modelos
}).round(1)
print(mae_mes.to_string())

print("\n=== 3c. MAE por hora local en prueba (€/MWh) ===\n")
hora = pred["hora_utc"].dt.tz_convert("Europe/Madrid").dt.hour
mae_hora = pd.DataFrame({
    m: (pred[m] - pred["precio"]).abs().groupby(hora).mean() for m in modelos
}).round(1)
print(mae_hora.to_string())

# Gráficas
fig, ax = plt.subplots(figsize=(12, 4))
mensual["precio"].plot(ax=ax, marker="o")
ax.axvline(list(mensual.index).index(INICIO_PRUEBA.strftime("%Y-%m")), color="red", ls="--",
           label="inicio prueba")
ax.set_title("Precio medio mensual")
ax.set_ylabel("€/MWh")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTADOS / "diag_precio_mensual.png", dpi=120)

semana = pred[pred["hora_utc"] < pred["hora_utc"].min() + pd.Timedelta(days=14)]
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(semana["hora_utc"], semana["precio"], label="real", color="black", lw=2)
for m in modelos:
    ax.plot(semana["hora_utc"], semana[m], label=m, lw=1)
ax.set_title("Primeras dos semanas de prueba: real frente a predicciones")
ax.set_ylabel("€/MWh")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTADOS / "diag_predicciones.png", dpi=120)

print(f"\nGráficas guardadas en {RESULTADOS}")
