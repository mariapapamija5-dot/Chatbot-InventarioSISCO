# -*- coding: utf-8 -*-
"""
Modelo Híbrido BETO + Likert — Detección de Estrés Académico SISCO

ANTES DE CORRER:
1. Entorno de ejecución → Cambiar tipo → T4 GPU → Guardar
2. Sube dataset_final_balanceado.csv al panel 
3. Corre cada celda con Shift + Enter
"""

# ══════════════════════════════════════════════════════════════
# CELDA 1 — Instalación
# ══════════════════════════════════════════════════════════════
# !pip install transformers torch scikit-learn seaborn matplotlib accelerate -q
# print(' Instalado')


# ══════════════════════════════════════════════════════════════
# CELDA 2 — Importaciones
# ══════════════════════════════════════════════════════════════
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score,
                              classification_report)
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Dispositivo: {str(device).upper()}")
if str(device) == 'cpu':
    print("  Sin GPU — ve a Entorno de ejecución → Cambiar tipo → T4 GPU")


# ══════════════════════════════════════════════════════════════
# CELDA 3 — Cargar dataset
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# IMPORTANTE: verifica que subiste dataset_final_balanceado.csv
# Si el archivo tiene otro nombre, cámbialo aquí:
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("dataset_recalibrado.csv")
df = df.loc[:, ~df.columns.duplicated()]
df = df.dropna(subset=["texto", "nivel_sisco"]).reset_index(drop=True)

print(f" Dataset cargado: {len(df)} registros")
print(f"\nDistribución nivel_sisco:")
for nivel, n in df["nivel_sisco"].value_counts().items():
    pct = n/len(df)*100
    barra = '█' * int(pct/3)
    print(f"  {nivel:<10} {n:>4}  ({pct:.1f}%)  {barra}")

print(f"\nFuentes:")
for fuente, n in df["fuente"].value_counts().items():
    print(f"  {fuente:<25} {n}")


# ══════════════════════════════════════════════════════════════
# CELDA 4 — Separar reales y sintéticos ANTES del split
# ══════════════════════════════════════════════════════════════
# CORRECCIÓN CLAVE: test SOLO con datos reales
# Los sintéticos solo van en entrenamiento

df_real = df[df["fuente"] == "real"].copy().reset_index(drop=True)
df_sint = df[df["fuente"] != "real"].copy().reset_index(drop=True)

print(f"Reales:     {len(df_real)}")
print(f"Sintéticos: {len(df_sint)}")

# Columnas Likert
COLS_EXCLUIR = ["id", "texto", "texto_original",
                "nivel_sisco", "puntaje_total",
                "n_palabras", "fuente", "etiquetado_por"]
COLS_LIKERT = [c for c in df.columns if c not in COLS_EXCLUIR]
print(f"Columnas Likert: {len(COLS_LIKERT)}")

# Split 80/20 SOLO sobre reales
X_text_real_train, X_text_test, \
X_likert_real_train, X_likert_test, \
y_real_train, y_test = train_test_split(
    df_real["texto"],
    df_real[COLS_LIKERT].apply(pd.to_numeric, errors="coerce").fillna(0),
    df_real["nivel_sisco"],
    test_size=0.20,
    random_state=42,
    stratify=df_real["nivel_sisco"]
)

# Agregar sintéticos al train
X_text_sint    = df_sint["texto"]
X_likert_sint  = df_sint[COLS_LIKERT].apply(pd.to_numeric, errors="coerce").fillna(0)
y_sint         = df_sint["nivel_sisco"]

X_text_train   = pd.concat([X_text_real_train, X_text_sint]).reset_index(drop=True)
X_likert_train = pd.concat([X_likert_real_train, X_likert_sint]).reset_index(drop=True)
y_train        = pd.concat([y_real_train, y_sint]).reset_index(drop=True)

print(f"\nTrain: {len(X_text_train)} (reales:{len(X_text_real_train)} + sint:{len(X_text_sint)})")
print(f"Test (solo reales): {len(X_text_test)}")
print(f"\nDistribución train:")
for nivel, n in y_train.value_counts().items():
    print(f"  {nivel:<10} {n}")
print(f"\nDistribución test:")
for nivel, n in y_test.value_counts().items():
    print(f"  {nivel:<10} {n}")


# ══════════════════════════════════════════════════════════════
# CELDA 5 — Codificar etiquetas
# ══════════════════════════════════════════════════════════════
le = LabelEncoder()
le.fit(df["nivel_sisco"])  # fit sobre todo para consistencia

y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

label_inv = dict(enumerate(le.classes_))
nombres_clases = [label_inv[i] for i in sorted(label_inv)]
print(f"Mapa etiquetas: {label_inv}")


# ══════════════════════════════════════════════════════════════
# CELDA 6 — Normalizar Likert
# ══════════════════════════════════════════════════════════════
scaler = StandardScaler()
X_likert_train_sc = scaler.fit_transform(X_likert_train)
X_likert_test_sc  = scaler.transform(X_likert_test)
print(f" Likert normalizado — shape train: {X_likert_train_sc.shape}")


# ══════════════════════════════════════════════════════════════
# CELDA 7 — Tokenizar con BETO
# ══════════════════════════════════════════════════════════════
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
print(" Descargando BETO...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(" Tokenizador listo")

print("Tokenizando textos...")
train_enc = tokenizer(
    list(X_text_train.astype(str)),
    padding=True, truncation=True,
    max_length=128, return_tensors="pt"
)
test_enc = tokenizer(
    list(X_text_test.astype(str)),
    padding=True, truncation=True,
    max_length=128, return_tensors="pt"
)
print(f" Train: {train_enc['input_ids'].shape} | Test: {test_enc['input_ids'].shape}")


# ══════════════════════════════════════════════════════════════
# CELDA 8 — Convertir a tensores
# ══════════════════════════════════════════════════════════════
X_likert_train_t = torch.tensor(X_likert_train_sc, dtype=torch.float32)
X_likert_test_t  = torch.tensor(X_likert_test_sc,  dtype=torch.float32)
y_train_t        = torch.tensor(y_train_enc, dtype=torch.long)
y_test_t         = torch.tensor(y_test_enc,  dtype=torch.long)
print(" Tensores listos")


# ══════════════════════════════════════════════════════════════
# CELDA 9 — Modelo Híbrido (arquitectura mejorada)
# ══════════════════════════════════════════════════════════════
class ModeloHibrido(nn.Module):
    def __init__(self, num_likert, num_clases=3):
        super().__init__()

        # Rama texto — BETO
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768

        # Rama Likert — más profunda con BatchNorm
        self.fc_likert = nn.Sequential(
            nn.Linear(num_likert, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Clasificador final
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_clases)
        )

    def forward(self, input_ids, attention_mask, likert):
        # Texto → BETO → token [CLS]
        bert_out  = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]  # [batch, 768]

        # Likert → red densa
        likert_out = self.fc_likert(likert)  # [batch, 64]

        # Concatenar → clasificador
        combinado = torch.cat([cls_token, likert_out], dim=1)  # [batch, 832]
        return self.classifier(combinado)  # [batch, 3]


model = ModeloHibrido(num_likert=len(COLS_LIKERT)).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f" Modelo híbrido listo — Parámetros: {total_params:,}")


# ══════════════════════════════════════════════════════════════
# CELDA 10 — Class Weights y DataLoader
# ══════════════════════════════════════════════════════════════
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_enc),
    y=y_train_enc
)
class_weights_t = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)

print("Class weights:")
for clase, w in zip(le.classes_, class_weights):
    print(f"  {clase:<10} {w:.4f}")

train_dataset = TensorDataset(
    train_enc["input_ids"],
    train_enc["attention_mask"],
    X_likert_train_t,
    y_train_t
)
test_dataset = TensorDataset(
    test_enc["input_ids"],
    test_enc["attention_mask"],
    X_likert_test_t,
    y_test_t
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=8)
print(f"\n DataLoader — Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")


# ══════════════════════════════════════════════════════════════
# CELDA 11 — Optimizador con learning rates diferenciados
# CORRECCIÓN: BETO necesita lr más pequeño que las capas nuevas
# ══════════════════════════════════════════════════════════════
optimizer = torch.optim.AdamW([
    {"params": model.bert.parameters(),       "lr": 2e-5},  # BETO lr pequeño
    {"params": model.fc_likert.parameters(),  "lr": 1e-3},  # Likert lr mayor
    {"params": model.classifier.parameters(), "lr": 1e-3},  # Clasificador
], weight_decay=0.01)

from transformers import get_linear_schedule_with_warmup

EPOCHS       = 10  # CORRECCIÓN: de 3 a 10 épocas
total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * 0.1)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"   Optimizador listo")
print(f"   Épocas:  {EPOCHS}")
print(f"   Pasos:   {total_steps}")
print(f"   Warmup:  {warmup_steps}")


# ══════════════════════════════════════════════════════════════
# CELDA 12 — Entrenamiento (~10-15 min con GPU)
# ══════════════════════════════════════════════════════════════
losses         = []
val_accuracies = []
mejor_acc      = 0
mejor_epoch    = 0

print(f" Entrenando {EPOCHS} épocas...\n")

for epoch in range(EPOCHS):
    # ── TRAIN ──────────────────────────────────────────────────
    model.train()
    total_loss = 0

    for batch in train_loader:
        input_ids, attention_mask, likert, labels = [b.to(device) for b in batch]

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        likert=likert)
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    losses.append(avg_loss)

    # ── VALIDACIÓN ─────────────────────────────────────────────
    model.eval()
    val_preds = []
    val_real  = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, likert, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            likert=likert)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_real.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_real, val_preds)
    val_accuracies.append(val_acc)

    if val_acc > mejor_acc:
        mejor_acc   = val_acc
        mejor_epoch = epoch + 1
        torch.save(model.state_dict(), "mejor_modelo_hibrido.pt")

    print(f"Época {epoch+1:>2}/{EPOCHS} | Loss: {avg_loss:.4f} | "
          f"Val acc: {val_acc:.2%}"
          + (" ← mejor" if val_acc == mejor_acc else ""))

print(f"\n Mejor accuracy: {mejor_acc:.2%} en época {mejor_epoch}")


# ══════════════════════════════════════════════════════════════
# CELDA 13 — Cargar mejor modelo y evaluación final
# ══════════════════════════════════════════════════════════════
model.load_state_dict(torch.load("mejor_modelo_hibrido.pt"))
model.eval()

predicciones = []
reales       = []

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, likert, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        likert=likert)
        preds = torch.argmax(outputs, dim=1)
        predicciones.extend(preds.cpu().numpy())
        reales.extend(labels.cpu().numpy())

acc = accuracy_score(reales, predicciones)
print(f"\n{'='*55}")
print(f"  MODELO HÍBRIDO — RESULTADOS FINALES")
print(f"  Accuracy: {acc:.2%}")
print(f"{'='*55}")
print(classification_report(reales, predicciones, target_names=nombres_clases))


# ══════════════════════════════════════════════════════════════
# CELDA 14 — Gráfica: Pérdida y Accuracy por época
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(range(1, EPOCHS+1), losses, 'b-o', markersize=5)
axes[0].set_title("Pérdida del entrenamiento", fontweight='bold')
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), [v*100 for v in val_accuracies], 'g-o', markersize=5)
axes[1].axhline(y=70, color='gray', linestyle='--', alpha=0.6, label='Umbral 70%')
axes[1].set_title("Accuracy de validación por época", fontweight='bold')
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("grafica_entrenamiento.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Guardada: grafica_entrenamiento.png")


# ══════════════════════════════════════════════════════════════
# CELDA 15 — Gráfica: Matriz de confusión
# ══════════════════════════════════════════════════════════════
cm = confusion_matrix(reales, predicciones)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=nombres_clases,
            yticklabels=nombres_clases,
            annot_kws={"size": 14})
plt.title(f"Matriz de Confusión — Modelo Híbrido\nAccuracy: {acc:.2%}",
          fontweight='bold', fontsize=13)
plt.ylabel("Nivel real", fontsize=11)
plt.xlabel("Nivel predicho", fontsize=11)
plt.tight_layout()
plt.savefig("grafica_matriz_confusion.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Guardada: grafica_matriz_confusion.png")


# ══════════════════════════════════════════════════════════════
# CELDA 16 — Gráfica: Precision, Recall, F1 por clase
# ══════════════════════════════════════════════════════════════
report = classification_report(reales, predicciones,
                                target_names=nombres_clases,
                                output_dict=True)

metricas = {c: {"Precision": report[c]["precision"],
                "Recall":    report[c]["recall"],
                "F1-Score":  report[c]["f1-score"]}
            for c in nombres_clases}
df_met = pd.DataFrame(metricas).T

x = np.arange(len(nombres_clases))
ancho = 0.25

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x-ancho, df_met["Precision"], ancho, label="Precision", color="#3498db", edgecolor="black")
b2 = ax.bar(x,       df_met["Recall"],    ancho, label="Recall",    color="#e67e22", edgecolor="black")
b3 = ax.bar(x+ancho, df_met["F1-Score"],  ancho, label="F1-Score",  color="#2ecc71", edgecolor="black")

for bs in [b1, b2, b3]:
    for b in bs:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h+0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(nombres_clases, fontsize=12)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Valor", fontsize=11)
ax.set_title("Precision, Recall y F1-Score por Nivel SISCO\nModelo Híbrido BETO + Likert",
             fontweight='bold', fontsize=13)
ax.legend(fontsize=10)
ax.axhline(y=0.70, color="gray", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("grafica_metricas_por_clase.png", dpi=150, bbox_inches='tight')
plt.show()
print(" Guardada: grafica_metricas_por_clase.png")


# ══════════════════════════════════════════════════════════════
# CELDA 17 — Guardar modelo
# ══════════════════════════════════════════════════════════════
import os, json

os.makedirs("./modelo_hibrido_final", exist_ok=True)
torch.save(model.state_dict(), "./modelo_hibrido_final/pesos.pt")
tokenizer.save_pretrained("./modelo_hibrido_final/tokenizador")

config = {
    "label_map":     label_inv,
    "cols_likert":   COLS_LIKERT,
    "num_likert":    len(COLS_LIKERT),
    "num_clases":    3,
    "accuracy_test": round(acc, 4),
    "mejor_epoca":   mejor_epoch
}
with open("./modelo_hibrido_final/config.json", "w") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)

print(" Modelo guardado en ./modelo_hibrido_final/")
print("   Descárgalo desde el panel  de Colab")


# ══════════════════════════════════════════════════════════════
# CELDA 18 — Predicción en tiempo real (chatbot)
# ══════════════════════════════════════════════════════════════
import re

JERGA = {
    "parce":"amigo", "parcero":"amigo", "mamado":"cansado",
    "berraco":"difícil", "bacano":"bien", "chévere":"bien",
    "xq":"porque", "bn":"bien", "re mal":"muy mal"
}

def normalizar(texto):
    t = str(texto).lower()
    for k, v in JERGA.items():
        t = t.replace(k, v)
    t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def predecir(texto, likert_values=None):
    """
    texto:         respuesta del estudiante
    likert_values: lista de 35 valores Likert (-3 a 3)
                   Si no se proveen, usa 0 (solo texto)
    """
    model.eval()
    texto_norm = normalizar(texto)

    enc = tokenizer(
        texto_norm, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )

    if likert_values is None:
        likert_values = [0] * len(COLS_LIKERT)

    likert_sc = scaler.transform([likert_values])
    likert_t  = torch.tensor(likert_sc, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            likert=likert_t
        )

    probs = torch.softmax(logits, dim=1)[0]
    idx   = torch.argmax(probs).item()
    nivel = label_inv[idx]
    emoji = {"alto":"🔴", "moderado":"🟡", "bajo":"🟢"}

    print(f'\n "{texto[:70]}"')
    print(f"   {emoji[nivel]} Nivel: {nivel.upper()} — Confianza: {probs[idx].item():.1%}")
    print(f"   Probabilidades:")
    for i, clase in enumerate(le.classes_):
        print(f"     {clase:<10} {probs[i].item():.1%}")
    return nivel

# Pruebas
print("="*55)
print("  PRUEBAS DEL CHATBOT")
print("="*55)

predecir("No puedo dormir, tengo mucha ansiedad con los parciales y siento que voy a reprobar")
predecir("Estoy tranquilo, me preparé bien y confío en mis capacidades para este semestre")
predecir("Tengo algo de nervios pero creo que puedo manejarlo con organización")
