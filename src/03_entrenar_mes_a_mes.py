"""
Entrenamiento y evaluación de los modelos.

Lee data/features/dataset y compara 5 configuraciones:
  1. Referencia: precio de ayer a la misma hora
  2. Ridge con todas las variables
  3. XGBoost con calendario + retardos
  4. XGBoost con calendario + retardos + meteorología
  5. XGBoost con calendario + retardos + meteorología + previsiones de REE

Periodo de prueba: del 01/07/2026 al 31/10/2026.

Dos formas de entrenar (variable MODO):
  - "fijo":         se entrena una vez con todo hasta el 30/06/2026 y se predicen los 4 meses.
  - "walk_forward": se reentrena cada mes con todo lo anterior. Para predecir agosto se
                    entrena hasta el 31/07, para septiembre hasta el 31/08, etc.
                    Es como se usaría el modelo en la realidad.

Resultados en data/results/:
  metricas.csv, metricas_por_mes.csv, predicciones.parquet,
  importancia_variables.csv, modelo_xgb.json

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

MODO = "walk_forward"          # "walk_forward" o "fijo"
PREDECIR_DESVIACION = True     # aprender precio - precio_media_7d

INICIO_PRUEBA = pd.Timestamp("2025-11-01", tz="UTC")
FIN_PRUEBA = pd.Timestamp("2026-11-01", tz="UTC")   # exclusivo
OBJETIVO = "precio"
SEMILLA = 42

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

# Nombre de la columna de predicción de cada configuración
MODELOS = {
    "pred_referencia": ("Referencia (precio de ayer)", "precio_lag_24"),
    "pred_ridge": ("Ridge", "todas"),
    "pred_xgb_calendario": ("XGBoost", "calendario + retardos"),
    "pred_xgb_meteo": ("XGBoost", "+ meteorología"),
    "pred_xgboost": ("XGBoost", "+ previsiones REE"),
}
COLUMNA_XGB = {
    "calendario + retardos": "pred_xgb_calendario",
    "+ meteorología": "pred_xgb_meteo",
    "+ previsiones REE": "pred_xgboost",
}


# --- Datos -------------------------------------------------------------------

def cargar_datos():
    df = pd.read_parquet(FEATURES).sort_values("hora_utc").reset_index(drop=True)
    if df["hora_utc"].dt.tz is None:
        df["hora_utc"] = df["hora_utc"].dt.tz_localize("UTC")

    # Huecos sueltos en las previsiones: interpolación lineal (máx. 3 horas seguidas)
    df[PREVISIONES] = df[PREVISIONES].interpolate(limit=3)

    # Nivel reciente: media de las 168 h que terminan 24 h antes (ya conocidas al
    # predecir el día siguiente). Se calcula antes de quitar filas.
    df["precio_media_7d"] = df[OBJETIVO].shift(24).rolling(168, min_periods=150).mean()

    antes = len(df)
    df = df.dropna(subset=[OBJETIVO] + TODAS)
    df = df[df["hora_utc"] < FIN_PRUEBA]
    print(f"Filas utilizables: {len(df)} de {antes}")
    return df


def periodos_prueba():
    """Lista de (inicio, fin) de cada bloque que se predice."""
    if MODO == "fijo":
        return [(INICIO_PRUEBA, FIN_PRUEBA)]
    inicios = pd.date_range(INICIO_PRUEBA, FIN_PRUEBA, freq="MS", inclusive="left")
    return [(i, i + pd.offsets.MonthBegin(1)) for i in inicios]


def objetivo_y_nivel(entreno, prueba):
    if PREDECIR_DESVIACION:
        return entreno[OBJETIVO] - entreno["precio_media_7d"], prueba["precio_media_7d"].values
    return entreno[OBJETIVO], 0.0


def metricas(real, pred):
    return (round(mean_absolute_error(real, pred), 2),
            round(float(np.sqrt(mean_squared_error(real, pred))), 2),
            round(float(np.mean(pred - real)), 2))


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
    """Busca hiperparámetros con validación temporal sobre el entrenamiento inicial."""
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
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_mean_absolute_error",
        random_state=SEMILLA,
        verbose=0,
    )
    busqueda.fit(X, y)
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"MAE en validación cruzada: {-busqueda.best_score_:.2f} €/MWh")
    return busqueda.best_params_


def predecir_bloque(entreno, prueba, params):
    """Entrena todos los modelos con `entreno` y devuelve sus predicciones en `prueba`."""
    y_entreno, nivel = objetivo_y_nivel(entreno, prueba)
    salida = prueba[["hora_utc", OBJETIVO]].copy()

    salida["pred_referencia"] = prueba["precio_lag_24"].values

    ridge = crear_ridge().fit(entreno[TODAS], y_entreno)
    salida["pred_ridge"] = ridge.predict(prueba[TODAS]) + nivel

    modelo_final = None
    for nombre, columnas in CONJUNTOS.items():
        modelo = XGBRegressor(objective="reg:squarederror", random_state=SEMILLA,
                              n_jobs=-1, **params)
        modelo.fit(entreno[columnas], y_entreno)
        salida[COLUMNA_XGB[nombre]] = modelo.predict(prueba[columnas]) + nivel
        if nombre == "+ previsiones REE":
            modelo_final = modelo
    return salida, modelo_final


# --- Ejecución ---------------------------------------------------------------

if __name__ == "__main__":
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    print(f"Modo: {MODO} | objetivo: "
          f"{'desviación respecto a la media de 7 días' if PREDECIR_DESVIACION else 'precio absoluto'}")

    # Hiperparámetros: se ajustan una vez con los datos anteriores a la prueba
    # y se reutilizan en todos los reentrenamientos (es lo más rápido y evita
    # usar información del periodo de prueba)
    print("\nAjustando XGBoost (puede tardar unos minutos)...")
    inicial = df[df["hora_utc"] < INICIO_PRUEBA]
    y_inicial, _ = objetivo_y_nivel(inicial, inicial)
    params = ajustar_xgboost(inicial[TODAS], y_inicial)

    bloques = []
    for inicio, fin in periodos_prueba():
        entreno = df[df["hora_utc"] < inicio]
        prueba = df[(df["hora_utc"] >= inicio) & (df["hora_utc"] < fin)]
        print(f"\nBloque {inicio:%m/%Y}: entrena con {len(entreno)} filas "
              f"(hasta {entreno['hora_utc'].max():%d/%m/%Y}), predice {len(prueba)}")
        salida, modelo_final = predecir_bloque(entreno, prueba, params)
        bloques.append(salida)

    predicciones = pd.concat(bloques, ignore_index=True)

    # Métricas globales
    filas = []
    for col, (modelo, variables) in MODELOS.items():
        mae, rmse, sesgo = metricas(predicciones[OBJETIVO], predicciones[col])
        filas.append({"modelo": modelo, "variables": variables,
                      "MAE": mae, "RMSE": rmse, "sesgo": sesgo})
    tabla = pd.DataFrame(filas)

    # Métricas por mes (MAE)
    mes = predicciones["hora_utc"].dt.strftime("%Y-%m")
    por_mes = pd.DataFrame({
        col: (predicciones[col] - predicciones[OBJETIVO]).abs().groupby(mes).mean()
        for col in MODELOS
    }).round(1)

    # Importancia de variables del último XGBoost completo entrenado
    importancia = (
        pd.DataFrame({"variable": TODAS, "importancia": modelo_final.feature_importances_})
        .sort_values("importancia", ascending=False)
    )

    # Guardar
    tabla.to_csv(RESULTADOS / "metricas.csv", index=False)
    por_mes.to_csv(RESULTADOS / "metricas_por_mes.csv")
    predicciones.to_parquet(RESULTADOS / "predicciones.parquet", index=False)
    importancia.to_csv(RESULTADOS / "importancia_variables.csv", index=False)
    modelo_final.save_model(RESULTADOS / "modelo_xgb.json")

    pd.set_option("display.width", 200)
    print("\n--- Resultados en prueba (€/MWh) ---")
    print("(sesgo negativo = el modelo se queda corto)\n")
    print(tabla.to_string(index=False))
    print("\n--- MAE por mes (€/MWh) ---\n")
    print(por_mes.to_string())
    print("\n--- Importancia de variables (último XGBoost completo) ---\n")
    print(importancia.to_string(index=False))
    print(f"\nGuardado en {RESULTADOS}")
