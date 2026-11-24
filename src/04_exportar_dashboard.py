"""
Exporta los datos del cuadro de mando de Power BI a dashboard/data/.

Ficheros (CSV con ; y coma decimal, para Power BI en español):
  predicciones.csv      una fila por hora y modelo: real, predicción y error
  historico.csv         una fila por hora con precio y variables (ene 2024 - oct 2026)
  importancia.csv       importancia de variables del XGBoost completo
  metricas.csv          tabla final de MAE, RMSE y sesgo (copia de data/results)

Las fechas se exportan en hora de Madrid y sin zona horaria, que es como
las entiende Power BI.

Uso (después de entrenar.py):
    python src/exportar_dashboard.py
"""

from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "data" / "results"
FEATURES = RAIZ / "data" / "features" / "dataset"
SALIDA = RAIZ / "dashboard" / "data"

NOMBRES = {
    "pred_referencia": "Referencia (precio de ayer)",
    "pred_ridge": "Ridge",
    "pred_xgb_calendario": "XGBoost · calendario",
    "pred_xgb_meteo": "XGBoost · + meteorología",
    "pred_xgboost": "XGBoost · completo",
}
# Orden en el que aparecerán los modelos en los gráficos
ORDEN = {nombre: i for i, nombre in enumerate(NOMBRES.values(), start=1)}

NOMBRES_VARIABLES = {
    "solar_prev": "Previsión solar (REE)",
    "eolica_prev": "Previsión eólica (REE)",
    "demanda_prev": "Previsión demanda (REE)",
    "radiacion_media": "Radiación solar",
    "viento_medio": "Viento",
    "temp_media": "Temperatura",
    "precio_lag_24": "Precio de ayer",
    "precio_lag_168": "Precio hace una semana",
    "precio_media_7d": "Media de precio 7 días",
    "hora_local": "Hora del día",
    "dia_semana": "Día de la semana",
    "mes": "Mes",
    "es_finde": "Fin de semana",
    "incluye_autoconsumo": "Incluye autoconsumo",
}


def a_hora_local(serie):
    if serie.dt.tz is None:
        serie = serie.dt.tz_localize("UTC")
    return serie.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)


def guardar(df, nombre):
    ruta = SALIDA / nombre
    df.to_csv(ruta, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    print(f"  {nombre:<18} {len(df):>7} filas")


if __name__ == "__main__":
    SALIDA.mkdir(parents=True, exist_ok=True)
    print(f"Exportando a {SALIDA}")

    # Predicciones en formato largo: facilita filtrar por modelo en Power BI
    pred = pd.read_parquet(RESULTADOS / "predicciones.parquet")
    pred["fecha_hora"] = a_hora_local(pred["hora_utc"])
    largo = pred.melt(
        id_vars=["fecha_hora", "precio"],
        value_vars=[c for c in NOMBRES if c in pred.columns],
        var_name="columna", value_name="prediccion",
    )
    largo["modelo"] = largo["columna"].map(NOMBRES)
    largo["orden_modelo"] = largo["modelo"].map(ORDEN)
    largo["error"] = largo["prediccion"] - largo["precio"]
    largo["error_abs"] = largo["error"].abs()
    largo["fecha"] = largo["fecha_hora"].dt.date
    largo["hora"] = largo["fecha_hora"].dt.hour
    largo["mes"] = largo["fecha_hora"].dt.strftime("%Y-%m")
    largo = largo.rename(columns={"precio": "precio_real"})
    largo = largo[["fecha_hora", "fecha", "hora", "mes", "modelo", "orden_modelo",
                   "precio_real", "prediccion", "error", "error_abs"]].round(
        {"precio_real": 2, "prediccion": 2, "error": 2, "error_abs": 2})
    guardar(largo, "predicciones.csv")

    # Histórico completo con las variables, para la vista de contexto y la curva de pato
    hist = pd.read_parquet(FEATURES)
    hist["fecha_hora"] = a_hora_local(hist["hora_utc"])
    hist["fecha"] = hist["fecha_hora"].dt.date
    hist["hora"] = hist["fecha_hora"].dt.hour
    hist["mes"] = hist["fecha_hora"].dt.strftime("%Y-%m")
    hist = hist[["fecha_hora", "fecha", "hora", "mes", "precio", "demanda_prev",
                 "eolica_prev", "solar_prev", "temp_media", "viento_medio",
                 "radiacion_media"]]
    hist = hist.round({c: 2 for c in hist.columns if hist[c].dtype.kind == "f"})
    guardar(hist, "historico.csv")

    # Importancia de variables con nombres legibles
    imp = pd.read_csv(RESULTADOS / "importancia_variables.csv")
    imp["variable"] = imp["variable"].map(NOMBRES_VARIABLES).fillna(imp["variable"])
    imp["importancia"] = (imp["importancia"] * 100).round(1)   # en %
    guardar(imp, "importancia.csv")

    guardar(pd.read_csv(RESULTADOS / "metricas.csv"), "metricas.csv")
    print("Listo.")
