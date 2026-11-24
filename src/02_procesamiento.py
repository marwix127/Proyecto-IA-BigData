"""
Pipeline de procesamiento con PySpark.

raw (JSON de ESIOS y Open-Meteo)
  -> clean    : una fila por hora UTC con precio, previsiones y meteorología
  -> features : clean + variables de calendario y retardos del precio

Uso (desde cualquier carpeta):
    python src/procesar.py
"""

from pathlib import Path

from pyspark.sql import SparkSession, functions as F, Window

# --- Configuración -----------------------------------------------------------

RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
CLEAN = RAIZ / "data" / "clean" / "dataset"
FEATURES = RAIZ / "data" / "features" / "dataset"

INICIO = "2024-01-01 00:00:00"
FIN = "2026-10-31 23:00:00"

# Fecha desde la que la previsión de demanda (460) incluye el autoconsumo
FECHA_AUTOCONSUMO = "2025-12-11 00:00:00"

spark = (
    SparkSession.builder
    .appName("tfm-precio-electricidad")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")   # todo el pipeline en UTC
    .config("spark.ui.showConsoleProgress", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")


# --- Lectura de ESIOS --------------------------------------------------------

def leer_esios(nombre, columna, zona=None):
    """Lee todos los meses de un indicador y devuelve (hora_utc, columna)."""
    ruta = str(RAW / "esios" / f"{nombre}_*.json")
    df = (
        spark.read.option("multiLine", True).json(ruta)
        .select(F.explode("indicator.values").alias("v"))
        .select("v.*")
    )

    zonas = [r["geo_name"] for r in df.select("geo_name").distinct().collect()]
    print(f"ESIOS {nombre}: zonas encontradas {zonas}")
    if zona:
        df = df.filter(F.col("geo_name") == zona)
    elif len(zonas) > 1:
        raise SystemExit(f"{nombre} trae varias zonas {zonas}: indica cuál usar con zona=...")

    return (
        df.withColumn("hora_utc", F.col("datetime_utc").cast("timestamp"))
        .groupBy("hora_utc")
        .agg(F.avg("value").alias(columna))
    )


# --- Lectura de Open-Meteo ---------------------------------------------------

def leer_openmeteo(grupo, variable, columna):
    """Lee todos los años de un grupo y devuelve la media de sus puntos por hora."""
    # Solo ficheros anuales (grupo_2024.json...), no los de prueba mensuales
    ruta = str(RAW / "openmeteo" / f"{grupo}_[0-9][0-9][0-9][0-9].json")
    df = (
        spark.read.option("multiLine", True).json(ruta)   # una fila por punto
        .select(
            F.col("hourly.time").alias("t"),
            F.col(f"hourly.{variable}").alias("v"),
        )
        .select(F.explode(F.arrays_zip("t", "v")).alias("z"))
        .select(
            F.to_timestamp("z.t", "yyyy-MM-dd'T'HH:mm").alias("hora_utc"),
            F.col("z.v").cast("double").alias("valor"),
        )
    )
    n_puntos = df.groupBy("hora_utc").count().agg(F.max("count")).first()[0]
    print(f"Open-Meteo {grupo}: {n_puntos} puntos por hora")

    return df.groupBy("hora_utc").agg(F.avg("valor").alias(columna))


# --- Capa clean --------------------------------------------------------------

def construir_clean():
    # Todas las horas esperadas: si a una fuente le falta una hora, saldrá como nulo
    horas = spark.sql(f"""
        SELECT explode(sequence(
            to_timestamp('{INICIO}'), to_timestamp('{FIN}'), interval 1 hour
        )) AS hora_utc
    """)

    fuentes = [
        leer_esios("precio", "precio"),
        leer_esios("prevision_demanda", "demanda_prev"),
        leer_esios("prevision_eolica", "eolica_prev"),
        leer_esios("prevision_solar", "solar_prev"),
        leer_openmeteo("temperatura", "temperature_2m", "temp_media"),
        leer_openmeteo("viento", "wind_speed_100m", "viento_medio"),
        leer_openmeteo("solar", "shortwave_radiation", "radiacion_media"),
    ]

    df = horas
    for fuente in fuentes:
        df = df.join(fuente, on="hora_utc", how="left")
    return df.orderBy("hora_utc")


def controles_calidad(df):
    print("\n--- Controles de calidad ---")
    total = df.count()
    esperado = spark.sql(f"""
        SELECT size(sequence(to_timestamp('{INICIO}'), to_timestamp('{FIN}'),
                    interval 1 hour)) AS n
    """).first()["n"]
    print(f"Filas: {total} (esperado {esperado})")
    print(f"Horas duplicadas: {total - df.select('hora_utc').distinct().count()}")

    print("Nulos por columna:")
    nulos = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).first()
    for c in df.columns:
        print(f"  {c:<16} {nulos[c]}")

    print(f"Precios <= 0: {df.filter(F.col('precio') <= 0).count()}")
    print(f"Viento negativo: {df.filter(F.col('viento_medio') < 0).count()}")
    print(f"Radiación negativa: {df.filter(F.col('radiacion_media') < 0).count()}")


# --- Capa features -----------------------------------------------------------

def construir_features(df):
    # El consumo sigue la hora local, así que el calendario se calcula en hora de Madrid
    local = F.from_utc_timestamp("hora_utc", "Europe/Madrid")
    orden = Window.orderBy("hora_utc")   # aviso de "una sola partición": normal con este volumen

    return (
        df.withColumn("hora_local", F.hour(local))
        .withColumn("dia_semana", F.dayofweek(local))   # 1 = domingo, 7 = sábado
        .withColumn("mes", F.month(local))
        .withColumn("es_finde", F.dayofweek(local).isin(1, 7).cast("int"))
        .withColumn(
            "incluye_autoconsumo",
            (F.col("hora_utc") >= F.to_timestamp(F.lit(FECHA_AUTOCONSUMO))).cast("int"),
        )
        # Precio de la misma hora el día anterior y la semana anterior:
        # ya se conocen cuando se predice el día siguiente
        .withColumn("precio_lag_24", F.lag("precio", 24).over(orden))
        .withColumn("precio_lag_168", F.lag("precio", 168).over(orden))
    )


# --- Ejecución ---------------------------------------------------------------

if __name__ == "__main__":
    clean = construir_clean().cache()
    controles_calidad(clean)
    clean.coalesce(1).write.mode("overwrite").parquet(str(CLEAN))
    print(f"\nGuardado clean en {CLEAN}")

    features = construir_features(clean)
    features.coalesce(1).write.mode("overwrite").parquet(str(FEATURES))
    print(f"Guardado features en {FEATURES}")

    features.show(5, truncate=False)
    spark.stop()