# -*- coding: utf-8 -*-
"""
Guardar el StandardScaler entrenado con las columnas Likert del dataset.
Ejecutar UNA SOLA VEZ para generar modelo_hibrido_final/scaler.pkl
"""
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Cargar dataset
df = pd.read_csv("dataset_recalibrado.csv")
df = df.loc[:, ~df.columns.duplicated()]

# Mismas columnas que se usaron en el entrenamiento
COLS_EXCLUIR = ["id", "texto", "texto_original",
                "nivel_sisco", "puntaje_total",
                "n_palabras", "fuente", "etiquetado_por"]
COLS_LIKERT = [c for c in df.columns if c not in COLS_EXCLUIR]

print(f"Columnas Likert: {len(COLS_LIKERT)}")
print(f"Columnas: {COLS_LIKERT}")

# Extraer datos Likert
X_likert = df[COLS_LIKERT].apply(pd.to_numeric, errors="coerce").fillna(0)

# Entrenar scaler (igual que en el notebook original)
scaler = StandardScaler()
scaler.fit(X_likert)

# Guardar
with open("modelo_hibrido_final/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n[OK] Scaler guardado en modelo_hibrido_final/scaler.pkl")
print(f"   Media: {scaler.mean_[:5]}...")
print(f"   Std:   {scaler.scale_[:5]}...")
