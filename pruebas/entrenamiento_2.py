"""
Entrenamiento y evaluación de los modelos.

Lee data/features/dataset y compara 5 configuraciones:
  1. Referencia: precio de ayer a la misma hora
  2. Ridge con todas las variables
  3. XGBoost con calendario + retardos
  4. XGBoost con calendario + retardos + meteorología
  5. XGBoost con calendario + retardos + meteorología + previsiones de REE

División temporal: entrenamiento hasta el 30/06/2026, prueba del 01/07 al 31/10/2026.

Resultados en data/results/:
  metricas.csv, predicciones.parquet, importancia_variables.csv, modelo_xgb.json

Uso (desde cualquier carpeta):
    python src/entrenar.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# --- Configuración -----------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
FEATURES = RAIZ / "data" / "features" / "dataset"
RESULTADOS = RAIZ / "data" / "results"

INICIO_PRUEBA = pd.Timestamp("2026-07-01", tz="UTC")
OBJETIVO = "precio"
SEMILLA = 42

# Si es True, los modelos aprenden la desviación respecto a la media de los 7 días
# anteriores (precio - precio_media_7d) y luego se les suma ese nivel. Así se adaptan
# a cambios de nivel del precio que no aparecen en el entrenamiento.
PREDECIR_DESVIACION = True

CALENDARIO = ["hora_local", "dia_semana", "mes", "es_finde", "incluye_autoconsumo"]
RETARDOS = ["precio_lag_24", "precio_lag_168", "precio_media_7d"]
METEO = ["temp_media", "viento_medio", "radiacion_media"]
PREVISIONES = ["demanda_prev", "eolica_prev", "solar_prev"]

CONJUNTOS = {
    "calendario + retardos": CALENDARIO + RETARDOS,
    "+ meteorología": CALENDARIO + RETARDOS + METEO,
    "+ previsiones REE": CALENDARIO + RETARDOS + METEO + PREVISIONES,
}
TODAS = CONJUNTOS["+ previsiones REE"]


# --- Datos -------------------------------------------------------------------

def cargar_datos():
    df = pd.read_parquet(FEATURES).sort_values("hora_utc").reset_index(drop=True)
    if df["hora_utc"].dt.tz is None:
        df["hora_utc"] = df["hora_utc"].dt.tz_localize("UTC")

    # Huecos sueltos en las previsiones: interpolación lineal (máx. 3 horas seguidas)
    df[PREVISIONES] = df[PREVISIONES].interpolate(limit=3)

    # Nivel reciente: media de las 168 h que terminan 24 h antes (todas ya conocidas
    # cuando se predice el día siguiente). Se calcula antes de quitar filas, con todas
    # las horas seguidas.
    df["precio_media_7d"] = df[OBJETIVO].shift(24).rolling(168, min_periods=150).mean()

    antes = len(df)
    df = df.dropna(subset=[OBJETIVO] + TODAS)   # precio sin dato y primeras 168 h
    print(f"Filas utilizables: {len(df)} de {antes}")

    entreno = df[df["hora_utc"] < INICIO_PRUEBA]
    prueba = df[df["hora_utc"] >= INICIO_PRUEBA]
    print(f"Entrenamiento: {len(entreno)} filas "
          f"({entreno['hora_utc'].min():%d/%m/%Y} - {entreno['hora_utc'].max():%d/%m/%Y})")
    print(f"Prueba:        {len(prueba)} filas "
          f"({prueba['hora_utc'].min():%d/%m/%Y} - {prueba['hora_utc'].max():%d/%m/%Y})")
    return entreno, prueba


def metricas(nombre, variables, real, pred):
    return {
        "modelo": nombre,
        "variables": variables,
        "MAE": round(mean_absolute_error(real, pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(real, pred))), 2),
    }


# --- Modelos -----------------------------------------------------------------

def crear_ridge():
    # Las variables de calendario son categorías: la hora 23 no es "23 veces" la hora 1
    categoricas = ["hora_local", "dia_semana", "mes"]
    numericas = [c for c in TODAS if c not in categoricas]
    preparacion = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
        ("num", StandardScaler(), numericas),
    ])
    return Pipeline([("prep", preparacion), ("ridge", Ridge(alpha=1.0))])


def ajustar_xgboost(X, y):
    """Busca hiperparámetros con validación temporal sobre el entrenamiento."""
    busqueda = RandomizedSearchCV(
        XGBRegressor(objective="reg:squarederror", random_state=SEMILLA, n_jobs=-1),
        param_distributions={
            "n_estimators": [200, 400, 600, 800],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.02, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "min_child_weight": [1, 5, 10],
        },
        n_iter=20,
        cv=TimeSeriesSplit(n_splits=4),   # cada pliegue valida con datos posteriores
        scoring="neg_mean_absolute_error",
        random_state=SEMILLA,
        verbose=0,
    )
    busqueda.fit(X, y)
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"MAE en validación cruzada: {-busqueda.best_score_:.2f} €/MWh")
    return busqueda.best_params_


# --- Ejecución ---------------------------------------------------------------

if __name__ == "__main__":
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    entreno, prueba = cargar_datos()
    y_prueba = prueba[OBJETIVO]

    # Lo que aprenden los modelos y cómo se vuelve a €/MWh
    if PREDECIR_DESVIACION:
        print("Objetivo: desviación respecto a la media de los 7 días anteriores")
        y_entreno = entreno[OBJETIVO] - entreno["precio_media_7d"]
        nivel_prueba = prueba["precio_media_7d"].values
    else:
        print("Objetivo: precio absoluto")
        y_entreno = entreno[OBJETIVO]
        nivel_prueba = 0.0

    resultados = []
    predicciones = prueba[["hora_utc", OBJETIVO]].copy()

    # 1. Referencia
    pred = prueba["precio_lag_24"]
    resultados.append(metricas("Referencia (precio de ayer)", "precio_lag_24", y_prueba, pred))
    predicciones["pred_referencia"] = pred.values

    # 2. Ridge
    ridge = crear_ridge().fit(entreno[TODAS], y_entreno)
    pred = ridge.predict(prueba[TODAS]) + nivel_prueba
    resultados.append(metricas("Ridge", "todas", y_prueba, pred))
    predicciones["pred_ridge"] = pred

    # 3-5. XGBoost: hiperparámetros ajustados una vez con todas las variables
    # y reutilizados en los tres conjuntos, para que la comparación sea justa
    print("\nAjustando XGBoost (puede tardar unos minutos)...")
    params = ajustar_xgboost(entreno[TODAS], y_entreno)

    for nombre, columnas in CONJUNTOS.items():
        modelo = XGBRegressor(objective="reg:squarederror", random_state=SEMILLA,
                              n_jobs=-1, **params)
        modelo.fit(entreno[columnas], y_entreno)
        pred = modelo.predict(prueba[columnas]) + nivel_prueba
        resultados.append(metricas("XGBoost", nombre, y_prueba, pred))
        if nombre == "+ previsiones REE":
            predicciones["pred_xgboost"] = pred
            modelo.save_model(RESULTADOS / "modelo_xgb.json")
            importancia = (
                pd.DataFrame({"variable": columnas, "importancia": modelo.feature_importances_})
                .sort_values("importancia", ascending=False)
            )

    # Guardar
    tabla = pd.DataFrame(resultados)
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    predicciones.to_parquet(RESULTADOS / "predicciones.parquet", index=False)
    importancia.to_csv(RESULTADOS / "importancia_variables.csv", index=False)

    print("\n--- Resultados en el conjunto de prueba (€/MWh) ---")
    print(tabla.to_string(index=False))
    print("\n--- Importancia de variables (XGBoost completo) ---")
    print(importancia.to_string(index=False))
    print(f"\nGuardado en {RESULTADOS}")
