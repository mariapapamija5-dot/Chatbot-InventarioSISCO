# -*- coding: utf-8 -*-
"""
SISCO — Chatbot de Detección de Estrés Académico
Backend Flask con Modelo Híbrido BETO + Likert
"""
import os
import re
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModel

# ══════════════════════════════════════════════════════════════
# Configuración
# ══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo_hibrido_final")
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

app = Flask(__name__)
app.secret_key = "sisco_fup_2026_secretkey"

# ══════════════════════════════════════════════════════════════
# Cargar configuración del modelo
# ══════════════════════════════════════════════════════════════
with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    model_config = json.load(f)

LABEL_MAP = model_config["label_map"]      # {"0": "alto", "1": "bajo", "2": "moderado"}
COLS_LIKERT = model_config["cols_likert"]   # 35 columnas
NUM_LIKERT = model_config["num_likert"]     # 35
NUM_CLASES = model_config["num_clases"]     # 3

print(f"[SISCO] Config cargada: {NUM_LIKERT} cols Likert, {NUM_CLASES} clases")

# ══════════════════════════════════════════════════════════════
# Cargar StandardScaler
# ══════════════════════════════════════════════════════════════
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("[SISCO] Scaler cargado")

# ══════════════════════════════════════════════════════════════
# Cargar tokenizador BETO
# ══════════════════════════════════════════════════════════════
tokenizer_path = os.path.join(MODEL_DIR, "tokenizador")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("[SISCO] Tokenizador BETO cargado")

# ══════════════════════════════════════════════════════════════
# Definir arquitectura del modelo (idéntica al entrenamiento)
# ══════════════════════════════════════════════════════════════
class ModeloHibrido(nn.Module):
    def __init__(self, num_likert, num_clases=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.bert.config.hidden_size  # 768

        self.fc_likert = nn.Sequential(
            nn.Linear(num_likert, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_clases)
        )

    def forward(self, input_ids, attention_mask, likert):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = bert_out.last_hidden_state[:, 0, :]
        likert_out = self.fc_likert(likert)
        combinado = torch.cat([cls_token, likert_out], dim=1)
        return self.classifier(combinado)

# ══════════════════════════════════════════════════════════════
# Cargar pesos del modelo entrenado
# ══════════════════════════════════════════════════════════════
device = torch.device("cpu")  # Inferencia en CPU para el servidor
model = ModeloHibrido(num_likert=NUM_LIKERT, num_clases=NUM_CLASES).to(device)
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "pesos.pt"), map_location=device)
)
model.eval()
print("[SISCO] Modelo hibrido cargado (CPU)")

# ══════════════════════════════════════════════════════════════
# Importar estimador de Likert y diccionario de datos
# ══════════════════════════════════════════════════════════════
from estimador_likert import estimar_likert_completo
from diccionario_datos import normalizar_para_beto

# ══════════════════════════════════════════════════════════════
# Preguntas conversacionales SISCO
# ══════════════════════════════════════════════════════════════
PREGUNTAS = {
    1: ("Estresores academicos",
        "Entiendo. Ahora me gustaría que nos enfoquemos en el origen de esto. "
        "¿Qué tareas o dinámicas de la universidad te están generando esa presión? "
        "(Por ejemplo: no entender algunos temas, demasiados trabajos grupales, "
        "miedo a perder una materia, etc.)."),

    2: ("Sintomas fisicos",
        "Entiendo. Ahora me gustaría saber si has notado algún cambio físico "
        "últimamente. ¿Has tenido problemas para dormir, te sientes cansado/a "
        "con frecuencia, dolores de cabeza, problemas de estómago o somnolencia "
        "durante el día?"),

    3: ("Sintomas psicologicos",
        "Gracias por contarme. ¿Y emocionalmente cómo te has sentido? "
        "¿Has experimentado ansiedad, nervios, tristeza, dificultad para "
        "concentrarte, irritabilidad o ganas de dejar todo?"),

    4: ("Comportamiento",
        "Comprendo. ¿Has notado cambios en tu comportamiento? Por ejemplo: "
        "aislarte de los demás, desgano para ir a clase, cambios en tu "
        "alimentación, faltar a clases o conflictos con otras personas."),

    5: ("Afrontamiento",
        "Última pregunta. Cuando te sientes estresado/a, ¿qué haces para "
        "manejarlo? Por ejemplo: hablar con alguien, organizarte mejor, "
        "hacer ejercicio, buscar ayuda profesional, rezar, escuchar música "
        "o algún pasatiempo."),
}

RECOMENDACIONES = {
    "bajo": (
        "Tu nivel de estrés académico parece ser manejable. Sigue aplicando "
        "tus estrategias de afrontamiento y mantén tus hábitos saludables. "
        "Si en algún momento sientes que la presión aumenta, no dudes en "
        "buscar apoyo."
    ),
    "moderado": (
        "Presentas un nivel moderado de estrés académico. Es importante que "
        "organices mejor tus tiempos, descanses lo suficiente y busques apoyo "
        "cuando lo necesites. Considera hablar con un compañero de confianza "
        "o acudir a la oficina de bienestar universitario."
    ),
    "alto": (
        "Tu nivel de estrés académico es alto. Te recomendamos acudir a la "
        "oficina de bienestar universitario de la FUP para recibir orientación "
        "profesional. No estás solo/a en esto — pedir ayuda es un acto de "
        "fortaleza. También puedes hablar con un docente de confianza."
    ),
}

# ══════════════════════════════════════════════════════════════
# Scoring basado en reglas (complementa al modelo neural)
# ══════════════════════════════════════════════════════════════
def _scoring_reglas_likert(likert_values):
    """
    Genera probabilidades de cada nivel basándose directamente
    en los 35 valores Likert estimados, sin pasar por el modelo.
    Actúa como "segunda opinión" para reforzar la confianza.

    Lógica SISCO:
    - Columnas 0-8:   estresores (9 cols) — más alto = más estrés
    - Columnas 9-14:  síntomas físicos (6 cols) — más alto = más estrés
    - Columnas 15-21: síntomas psicológicos (7 cols) — más alto = más estrés
    - Columnas 22-26: comportamiento (5 cols) — más alto = más estrés
    - Columnas 27-34: afrontamiento (8 cols) — más alto = MENOS estrés
    """
    vals = np.array(likert_values, dtype=float)

    # Separar dimensiones
    estresores = vals[0:9]       # más alto = más estrés
    sint_fisicos = vals[9:15]    # más alto = más estrés
    sint_psico = vals[15:22]     # más alto = más estrés
    comportamiento = vals[22:27] # más alto = más estrés
    afrontamiento = vals[27:35]  # más alto = MENOS estrés (invertir)

    # Función auxiliar para ignorar los 0s al promediar
    def _mean_nonzero(arr):
        non_zeros = arr[arr != 0]
        if len(non_zeros) == 0:
            return 0.0
        # Dar más peso a los valores extremos (positivos o negativos)
        # para que 1 solo síntoma grave cuente bastante
        return np.mean(non_zeros)

    # Calcular score de estrés normalizado (-1 a 1)
    # Promedio de las dimensiones de estrés (normalizadas a [-1, 1])
    score_estresores = _mean_nonzero(estresores) / 3.0
    score_fisicos = _mean_nonzero(sint_fisicos) / 3.0
    score_psico = _mean_nonzero(sint_psico) / 3.0
    score_comport = _mean_nonzero(comportamiento) / 3.0
    score_afront = -_mean_nonzero(afrontamiento) / 3.0  # invertido

    # Score compuesto ponderado (las dimensiones psicológicas pesan más)
    score_total = (
        0.20 * score_estresores +
        0.15 * score_fisicos +
        0.25 * score_psico +
        0.20 * score_comport +
        0.20 * score_afront
    )

    # Convertir score a probabilidades suavizadas
    # score_total ∈ [-1, 1] → mapear a distribución de probabilidades
    if score_total > 0.3:
        # Tendencia alta
        p_alto = 0.5 + score_total * 0.4
        p_moderado = 0.3 - score_total * 0.1
        p_bajo = 0.2 - score_total * 0.15
    elif score_total < -0.3:
        # Tendencia baja
        p_bajo = 0.5 + abs(score_total) * 0.4
        p_moderado = 0.3 - abs(score_total) * 0.1
        p_alto = 0.2 - abs(score_total) * 0.15
    else:
        # Zona moderada
        p_moderado = 0.5 + abs(score_total) * 0.2
        p_alto = 0.25 + score_total * 0.2
        p_bajo = 0.25 - score_total * 0.2

    # Normalizar para que sumen 1 y clampear a [0.01, 0.98]
    p_alto = max(0.01, p_alto)
    p_bajo = max(0.01, p_bajo)
    p_moderado = max(0.01, p_moderado)
    total = p_alto + p_bajo + p_moderado

    return {
        "alto": p_alto / total,
        "bajo": p_bajo / total,
        "moderado": p_moderado / total,
    }


# ══════════════════════════════════════════════════════════════
# Función de predicción con ensemble (modelo + reglas)
# ══════════════════════════════════════════════════════════════
# Hiperparámetros de calibración
TEMPERATURA = 0.7      # < 1 agudiza las probabilidades del softmax
PESO_MODELO = 0.70     # Peso del modelo neural en el ensemble
PESO_REGLAS = 0.30     # Peso del scoring basado en reglas

def predecir_con_modelo(texto_completo, likert_values):
    """
    Predice el nivel de estrés usando ensemble:
    1. Modelo híbrido BETO + Likert (70%)
    2. Scoring basado en reglas sobre Likert (30%)

    Incluye temperature scaling para agudizar las probabilidades.

    Args:
        texto_completo: Texto concatenado de todas las respuestas
        likert_values: Lista de 35 valores Likert estimados (-3 a 3)

    Returns:
        (nivel, confianza, probabilidades_dict)
    """
    texto_norm = normalizar_para_beto(texto_completo)

    # Tokenizar con BETO
    enc = tokenizer(
        texto_norm, padding=True, truncation=True,
        max_length=128, return_tensors="pt"
    )

    # Normalizar Likert con el scaler
    likert_sc = scaler.transform([likert_values])
    likert_t = torch.tensor(likert_sc, dtype=torch.float32).to(device)

    # ── Predicción del modelo neural ──────────────────────────
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            likert=likert_t
        )

    # Temperature scaling: dividir logits por T antes del softmax
    logits_calibrados = logits / TEMPERATURA
    probs_modelo = torch.softmax(logits_calibrados, dim=1)[0]

    # Extraer probabilidades del modelo por clase
    probas_modelo = {}
    for i, clase in LABEL_MAP.items():
        probas_modelo[clase] = probs_modelo[int(i)].item()

    # ── Scoring basado en reglas ──────────────────────────────
    probas_reglas = _scoring_reglas_likert(likert_values)

    # ── Ensemble: combinar modelo + reglas ────────────────────
    probas_final = {}
    for clase in ["alto", "bajo", "moderado"]:
        probas_final[clase] = (
            PESO_MODELO * probas_modelo.get(clase, 0) +
            PESO_REGLAS * probas_reglas.get(clase, 0)
        )

    # Normalizar para que sumen exactamente 1
    total = sum(probas_final.values())
    for clase in probas_final:
        probas_final[clase] = probas_final[clase] / total

    # Determinar nivel y confianza
    nivel = max(probas_final, key=probas_final.get)
    confianza = probas_final[nivel]

    # Redondear para la respuesta
    probas_redondeadas = {c: round(p, 4) for c, p in probas_final.items()}

    return nivel, confianza, probas_redondeadas

# ══════════════════════════════════════════════════════════════
# Rutas Flask
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    # Inicializar sesión
    session["paso"] = 0
    session["respuestas"] = {}
    return render_template("index.html")


@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    texto = data.get("texto", "").strip()

    if not texto or len(texto.split()) < 3:
        return jsonify({"error": "Texto demasiado corto. Escribe al menos 3 palabras."}), 400

    # Obtener estado de la conversación
    paso_actual = session.get("paso", 0)
    respuestas = session.get("respuestas", {})

    # Guardar respuesta actual
    respuestas[str(paso_actual)] = texto
    session["respuestas"] = respuestas

    # Avanzar al siguiente paso
    paso_siguiente = paso_actual + 1

    if paso_siguiente <= 5:
        # Hay más preguntas
        session["paso"] = paso_siguiente
        dimension, pregunta = PREGUNTAS[paso_siguiente]

        return jsonify({
            "tipo": "pregunta",
            "pregunta": pregunta,
            "dimension": dimension,
            "paso": paso_siguiente,
            "total_pasos": 5,
        })

    else:
        # Ya se completaron todas las preguntas → predecir
        # Concatenar todas las respuestas
        texto_completo = " ".join(
            respuestas.get(str(i), "") for i in range(6)
        )

        # Estimar valores Likert desde las respuestas
        respuestas_int = {int(k): v for k, v in respuestas.items()}
        likert_values = estimar_likert_completo(respuestas_int)

        # Predecir con el modelo
        nivel, confianza, probas = predecir_con_modelo(texto_completo, likert_values)

        # Reiniciar sesión para nuevo análisis
        session["paso"] = 0
        session["respuestas"] = {}

        return jsonify({
            "tipo": "resultado",
            "nivel": nivel,
            "confianza": round(confianza, 4),
            "probas": probas,
            "recomendacion": RECOMENDACIONES.get(nivel, ""),
        })


@app.route("/reiniciar", methods=["POST"])
def reiniciar():
    session["paso"] = 0
    session["respuestas"] = {}
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════
# Ejecutar servidor
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SISCO - Chatbot de Estres Academico")
    print("  Modelo Hibrido BETO + Likert")
    print(f"  Accuracy entrenamiento: {model_config.get('accuracy_test', 'N/A')}")
    print("=" * 55)
    print("  Abriendo en http://localhost:5000")
    print("=" * 55 + "\n")
    app.run(debug=False, port=5000)
