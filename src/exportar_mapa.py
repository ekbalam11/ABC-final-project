"""
exportar_mapa.py
────────────────
Ejecutar UNA VEZ para generar data/mapa_predicciones.csv
que consume el dashboard Streamlit.

Requiere:
  - models/xgb_uhi.pkl
  - data/processed/dataset_modeling.csv
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ── Rutas ────────────────────────────────────────────────────────────────────
MODEL_PATH = Path("../models/xgb_uhi.pkl")
CSV_IN     = Path("../data/processed/dataset_modeling.csv")
CSV_OUT    = Path("../data/processed/mapa_predicciones.csv")

# ── Features exactos usados en el entrenamiento ──────────────────────────────
BASE_FEATURES = [
    'elevation',
    'mndwi', 'nbr', 'ndbi', 'ndmi', 'ndvi', 'ndwi',
    'nir', 'swir1', 'swir2',
    'season'
]

# ── Carga ────────────────────────────────────────────────────────────────────
print("Cargando modelo y datos...")
modelo = joblib.load(MODEL_PATH)
df     = pd.read_csv(CSV_IN)

print(f"  Dataset: {len(df):,} filas · {len(df.columns)} columnas")
print(f"  Columnas: {df.columns.tolist()}")

# ── One-hot encoding de season (igual que en el entrenamiento) ────────────────
df_enc = pd.get_dummies(df[BASE_FEATURES], columns=['season'], drop_first=True)

print(f"\n  Features tras encoding: {df_enc.columns.tolist()}")
print(f"  Total features: {len(df_enc.columns)} (esperados: 13)")

# ── Predicciones ─────────────────────────────────────────────────────────────
filas_validas = df_enc.notna().all(axis=1)
print(f"\n  Filas con features completos: {filas_validas.sum():,} / {len(df):,}")

df_valid  = df[filas_validas].copy()
X         = df_enc[filas_validas]

df_valid["pred_label"] = modelo.predict(X)
df_valid["pred_label"] = df_valid["pred_label"].map({1: "riesgo", 0: "no_riesgo"})
df_valid["pred_proba"] = modelo.predict_proba(X)[:, 1]
df_valid["real_label"] = df_valid["uhi_risk"].map({1: "riesgo", 0: "no_riesgo"})

# ── Clasificar cada punto: TP / TN / FP / FN ─────────────────────────────────
def clasificar(row):
    r, p = row["real_label"], row["pred_label"]
    if r == "riesgo"    and p == "riesgo":    return "TP"
    if r == "no_riesgo" and p == "no_riesgo": return "TN"
    if r == "no_riesgo" and p == "riesgo":    return "FP"
    return "FN"

df_valid["resultado"] = df_valid.apply(clasificar, axis=1)

print(f"\n  Distribución de resultados:")
print(df_valid["resultado"].value_counts().to_string())

# ── Exportar ──────────────────────────────────────────────────────────────────
cols_export = [
    "latitude", "longitude",
    "pred_label", "pred_proba",
    "real_label", "resultado",
    "elevation",
    "mndwi", "nbr", "ndbi", "ndmi", "ndvi", "ndwi",
    "nir", "swir1", "swir2",
    "season",
]

df_valid[cols_export].to_csv(CSV_OUT, index=False)

print(f"\n✅ Exportado: {CSV_OUT}")
print(f"   {len(df_valid):,} filas · {len(cols_export)} columnas")