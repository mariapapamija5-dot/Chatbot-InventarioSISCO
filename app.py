# -*- coding: utf-8 -*-
"""
SISCO — Chatbot de Detección de Estrés Académico
Backend Flask con Modelo Híbrido BETO + Likert
"""
import os
import re
import json
import pickle
import uuid
import sqlite3
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModel


# Configuración
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo_hibrido_final")
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
DB_PATH = os.path.join(BASE_DIR, "sisco_chatbot_v2.db")

# ══════════════════════════════════════════════════════════════
# Configuración Base de Datos SQLite
# ══════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre_usuario TEXT UNIQUE,
            pin TEXT,
            edad INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sesiones (
            id TEXT PRIMARY KEY,
            usuario_id INTEGER,
            fecha_inicio TIMESTAMP,
            nivel_final TEXT,
            confianza REAL,
            completado BOOLEAN,
            FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS respuestas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sesion_id TEXT,
            paso INTEGER,
            texto TEXT,
            sentimiento TEXT,
            FOREIGN KEY(sesion_id) REFERENCES sesiones(id)
        )
    ''')
    conn.commit()
    conn.close()

# Inicializar DB al arrancar
init_db()

app = Flask(__name__)
app.secret_key = "sisco_fup_2026_secretkey"


# Cargar configuración del modelo

with open(os.path.join(MODEL_DIR, "config.json"), "r", encoding="utf-8") as f:
    model_config = json.load(f)

LABEL_MAP = model_config["label_map"]      # {"0": "alto", "1": "bajo", "2": "moderado"}
COLS_LIKERT = model_config["cols_likert"]   # 35 columnas
NUM_LIKERT = model_config["num_likert"]     # 35
NUM_CLASES = model_config["num_clases"]     # 3

print(f"[SISCO] Config cargada: {NUM_LIKERT} cols Likert, {NUM_CLASES} clases")


# Cargar StandardScaler

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
print("[SISCO] Scaler cargado")


# Cargar tokenizador BETO

tokenizer_path = os.path.join(MODEL_DIR, "tokenizador")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
print("[SISCO] Tokenizador BETO cargado")


# Definir arquitectura del modelo (idéntica al entrenamiento)

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


# ID de Google Drive (¡DEBES REEMPLAZAR ESTO CON EL ID DE TU ARCHIVO pesos.pt!)
# Ejemplo: si el enlace es https://drive.google.com/file/d/1A2B3C/view, el ID es 1A2B3C
GOOGLE_DRIVE_PESOS_ID = "1ot-_JNAaSGg_FS0eoqrLLLsM8FD8iBEM"

# Cargar pesos del modelo entrenado

device = torch.device("cpu")  # Inferencia en CPU para el servidor
model = ModeloHibrido(num_likert=NUM_LIKERT, num_clases=NUM_CLASES).to(device)

pesos_path = os.path.join(MODEL_DIR, "pesos.pt")
if not os.path.exists(pesos_path):
    print("[SISCO] No se encontraron los pesos locales. Descargando desde Google Drive (esto puede tardar unos minutos)...")
    import gdown
    url = f'https://drive.google.com/uc?id={GOOGLE_DRIVE_PESOS_ID}'
    gdown.download(url, pesos_path, quiet=False)

model.load_state_dict(
    torch.load(pesos_path, map_location=device)
)
model.eval()
print("[SISCO] Modelo hibrido cargado (CPU)")


# Importar estimador de Likert y diccionario de datos

from estimador_likert import estimar_likert_completo
from diccionario_datos import normalizar_para_beto
from analizador_sentimiento import analizar_sentimiento_general

# ══════════════════════════════════════════════════════════════
# Preguntas conversacionales SISCO (Dinámicas por sentimiento)
# ══════════════════════════════════════════════════════════════
PREGUNTAS = {
    1: {
        "dimension": "Estresores academicos",
        "positivo": "Me alegra saber que te has sentido tranquilo/a. Aún así, la universidad siempre presenta retos. ¿Qué tareas o dinámicas suelen exigirte más esfuerzo o causarte algo de presión? (Por ejemplo: trabajos grupales, parciales, exponer, etc.).",
        "negativo": "Entiendo, lamento que te sientas así. Ahora me gustaría que nos enfoquemos en el origen de esto. ¿Qué tareas o dinámicas de la universidad te están generando esa presión? (Por ejemplo: no entender algunos temas, demasiados trabajos grupales, miedo a perder una materia, etc.).",
        "neutral": "Entiendo. Ahora me gustaría que nos enfoquemos en tu contexto académico. ¿Qué tareas o dinámicas de la universidad te están generando mayor presión o exigencia?"
    },
    2: {
        "dimension": "Sintomas fisicos",
        "positivo": "Entendido. Cambiando un poco de tema, me gustaría saber sobre tu salud física. A veces el cuerpo reacciona sin darnos cuenta. ¿Has tenido problemas para dormir, te sientes cansado/a, con dolores de cabeza o problemas de estómago últimamente?",
        "negativo": "Comprendo. Es muy común que esa presión afecte el cuerpo. ¿Has notado algún cambio físico últimamente? ¿Has tenido problemas para dormir, te sientes cansado/a con frecuencia, dolores de cabeza o problemas de estómago?",
        "neutral": "Entiendo. Ahora me gustaría saber si has notado algún cambio físico últimamente. ¿Has tenido problemas para dormir, te sientes cansado/a con frecuencia, dolores de cabeza, problemas de estómago o somnolencia?"
    },
    3: {
        "dimension": "Sintomas psicologicos",
        "positivo": "Gracias por la información. ¿Y emocionalmente cómo te has sentido en general? ¿Has experimentado algo de ansiedad, nervios, dificultad para concentrarte o irritabilidad, aunque sea leve?",
        "negativo": "Gracias por contarme, sé que no siempre es fácil hablar de esto. ¿Y emocionalmente cómo te has sentido? ¿Has experimentado mucha ansiedad, nervios, tristeza, dificultad para concentrarte, irritabilidad o ganas de dejar todo?",
        "neutral": "Gracias por contarme. ¿Y emocionalmente cómo te has sentido? ¿Has experimentado ansiedad, nervios, tristeza, dificultad para concentrarte o irritabilidad?"
    },
    4: {
        "dimension": "Comportamiento",
        "positivo": "Comprendo. En cuanto a tu rutina diaria, ¿has notado cambios en tu comportamiento? Por ejemplo: desgano para ir a clase, cambios en tu alimentación o aislamiento social.",
        "negativo": "Entiendo profundamente. Todo esto a veces nos hace cambiar nuestra rutina. ¿Has notado cambios en tu comportamiento? Por ejemplo: aislarte de los demás, desgano total para ir a clase, cambios drásticos en tu alimentación o conflictos con otras personas.",
        "neutral": "Comprendo. ¿Has notado cambios en tu comportamiento? Por ejemplo: aislarte de los demás, desgano para ir a clase, cambios en tu alimentación, faltar a clases o conflictos con otras personas."
    },
    5: {
        "dimension": "Afrontamiento",
        "positivo": "Última pregunta. Dado que te sientes relativamente bien, ¿qué estrategias usas normalmente para mantener el equilibrio y manejar el estrés? (ej. hacer ejercicio, organizar tu tiempo, salir con amigos, pasatiempos).",
        "negativo": "Última pregunta. Cuando te sientes así de abrumado/a, ¿qué intentas hacer para manejarlo? Por ejemplo: hablar con alguien, organizar mejor tu tiempo, buscar ayuda profesional, rezar o escuchar música.",
        "neutral": "Última pregunta. Cuando te sientes estresado/a, ¿qué haces para manejarlo? Por ejemplo: hablar con alguien, organizarte mejor, hacer ejercicio, buscar ayuda profesional, rezar o algún pasatiempo."
    },
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


# Scoring basado en reglas (complementa al modelo neural)

def generar_insight_dinamico(texto, paso):
    t = str(texto).lower()
    insight = ""
    consejo = ""
    
    # Detectar si el usuario está pidiendo ayuda explícitamente
    palabras_ayuda = ["recomend", "consejo", "que hago", "qué hago", "ayuda", "tip", "que me sugieres", "que puedo hacer", "qué puedo hacer"]
    pide_consejo = any(p in t for p in palabras_ayuda)
    
    # Dependiendo de la pregunta que acaba de responder (paso)
    if paso == 0 or paso == 1: # Estresores o pregunta inicial
        if "parciales" in t or "examen" in t or "evaluacion" in t:
            insight = "Es completamente comprensible, los periodos de exámenes siempre traen una carga de presión muy alta. "
            consejo = "Como sugerencia: dividir los temas por días y estudiar en bloques de 45 minutos ayuda mucho."
        elif "tiempo" in t or "trabajos" in t or "muchas cosas" in t:
            insight = "Se nota que estás lidiando con muchas responsabilidades a la vez, es normal sentirse abrumado. "
            consejo = "Podría serte útil usar una agenda o hacer una lista de prioridades para tener más control."
        elif "profesor" in t or "docente" in t or "entender" in t:
            insight = "Es frustrante cuando la dinámica con las clases o docentes no es la mejor. "
            consejo = "No dudes en pedir tutorías o apoyarte en tus compañeros de clase."
            
    elif paso == 2: # Acaban de responder sobre síntomas físicos
        if "cansad" in t or "sueño" in t or "dormir" in t or "fatiga" in t or "agotad" in t:
            insight = "Noto que mencionas cansancio. Es muy duro para el cuerpo mantener ese ritmo. "
            consejo = "Recuerda que dormir bien (7-8 horas) es el pilar fundamental para recuperarte."
        elif "cabeza" in t or "migraña" in t or "duele" in t:
            insight = "Los dolores de cabeza son una señal clara de que tu cuerpo está acumulando mucha tensión. "
            consejo = "Intenta hacer pausas activas cada hora, alejándote por completo de las pantallas."
        elif "estomago" in t or "estómago" in t or "comer" in t or "apetito" in t or "digestion" in t:
            insight = "El estrés muchas veces se refleja directamente en nuestro estómago, es una reacción muy natural. "
            consejo = "Trata de no saltarte las comidas principales, incluso si sientes que no tienes mucho tiempo."
            
    elif paso == 3: # Acaban de responder sobre síntomas psicológicos
        if "ansiedad" in t or "nervios" in t or "angustia" in t or "desesper" in t:
            insight = "La ansiedad puede ser muy limitante y abrumadora, lamento que estés pasando por eso. "
            consejo = "Hacer ejercicios de respiración profunda (inhalar 4 segundos, retener 4, exhalar 6) ayuda a calmar el sistema nervioso."
        elif "triste" in t or "llorar" in t or "depre" in t or "desanim" in t or "mal" in t:
            insight = "Es completamente válido sentirse triste bajo tanta presión académica, tus emociones son importantes. "
            consejo = "A veces ayuda mucho escribir lo que sientes o desahogarte con alguien de absoluta confianza."
        elif "concentrar" in t or "mente" in t or "blanco" in t:
            insight = "Cuando hay saturación, es muy frustrante sentir que la mente no da más. "
            consejo = "Si te pasa, cierra los libros 15 minutos, camina un poco y luego retoma. No fuerces tu cerebro."
            
    elif paso == 4: # Acaban de responder sobre comportamiento
        if "aislad" in t or "sol" in t or "encerr" in t or "salir" in t or "hablar" in t:
            insight = "Aislarse es un reflejo protector muy común cuando nos sentimos sobrepasados. "
            consejo = "Intentar conversar aunque sea 10 minutos con un amigo puede aliviar mucho esa sensación."
        elif "discut" in t or "pele" in t or "irritable" in t or "mal genio" in t:
            insight = "Es normal que el estrés acumulado a veces salga en forma de irritabilidad con los demás. "
            consejo = "Date espacios a solas para respirar hondo antes de reaccionar si sientes que vas a explotar."
    
    resultado = insight
    if pide_consejo and consejo:
        resultado += f"<br><br>💡 <b>Consejo:</b> {consejo}"
        
    if resultado:
        return f"<em>{resultado}</em><br><br>"
    return ""

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



# Función de predicción con ensemble (modelo + reglas)

# Hiperparámetros de calibración
TEMPERATURA = 0.7      # < 1 agudiza las probabilidades del softmax
PESO_MODELO = 0.85     # Peso del modelo neural en el ensemble (aumentado para mayor fluidez humana)
PESO_REGLAS = 0.15     # Peso del scoring basado en reglas (reducido para no penalizar tan fuerte por palabras clave aisladas)

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

    # Predicción del modelo neural
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
# ══════════════════════════════════════════════════════════════
# Rutas de Autenticación
# ══════════════════════════════════════════════════════════════

@app.route("/login")
def login_page():
    return render_template("login.html")

@app.route("/api/auth", methods=["POST"])
def auth():
    data = request.json
    usuario = data.get("usuario", "").strip()
    pin = data.get("pin", "").strip()
    edad = data.get("edad", "")

    if not usuario or not pin or len(pin) != 4 or not pin.isdigit():
        return jsonify({"error": "Usuario inválido o el PIN debe ser de 4 dígitos."}), 400

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Buscar si el usuario existe
    cursor.execute("SELECT id, pin FROM usuarios WHERE nombre_usuario = ?", (usuario,))
    user_row = cursor.fetchone()

    if user_row:
        # El usuario existe, validar PIN
        user_id, db_pin = user_row
        if db_pin != pin:
            conn.close()
            return jsonify({"error": "PIN incorrecto para este usuario."}), 401
    else:
        # Registrar nuevo usuario
        edad_val = int(edad) if edad.isdigit() else None
        cursor.execute("INSERT INTO usuarios (nombre_usuario, pin, edad) VALUES (?, ?, ?)", (usuario, pin, edad_val))
        user_id = cursor.lastrowid
        conn.commit()

    conn.close()

    # Iniciar sesión web
    session["usuario_id"] = user_id
    session["nombre_usuario"] = usuario
    
    return jsonify({"status": "ok"})

# ══════════════════════════════════════════════════════════════
# Rutas de la Aplicación Principal
# ══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "usuario_id" not in session:
        return render_template("login.html")
        
    usuario_id = session["usuario_id"]
    nombre = session.get("nombre_usuario", "Anónimo")
    
    session_id = str(uuid.uuid4())
    session["sesion_id"] = session_id
    session["paso"] = 0
    session["respuestas"] = {}
    session["ultimo_sentimiento"] = "neutral"
    
    # Crear registro en BD
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('INSERT INTO sesiones (id, usuario_id, fecha_inicio, completado) VALUES (?, ?, ?, ?)', 
                   (session_id, usuario_id, datetime.now(), False))
    conn.commit()
    conn.close()
    
    return render_template("index.html", nombre_usuario=nombre)

@app.route("/reiniciar", methods=["POST"])
def reiniciar():
    # Mantener autenticación
    usuario_id = session.get("usuario_id")
    nombre_usuario = session.get("nombre_usuario")
    
    session.clear()
    
    if usuario_id:
        session["usuario_id"] = usuario_id
        session["nombre_usuario"] = nombre_usuario
        
    session_id = str(uuid.uuid4())
    session["sesion_id"] = session_id
    session["paso"] = 0
    session["respuestas"] = {}
    session["ultimo_sentimiento"] = "neutral"
    
    # Crear nueva sesión en BD
    if usuario_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sesiones (id, usuario_id, fecha_inicio, completado) VALUES (?, ?, ?, ?)', 
                       (session_id, usuario_id, datetime.now(), False))
        conn.commit()
        conn.close()
        
    return jsonify({"status": "ok"})

@app.route("/predecir", methods=["POST"])
def predecir():
    data = request.get_json()
    texto = data.get("texto", "").strip()

    if not texto or len(texto.split()) < 3:
        return jsonify({"error": "Texto demasiado corto. Escribe al menos 3 palabras."}), 400

    # Obtener estado de la conversación
    paso_actual = session.get("paso", 0)
    respuestas = session.get("respuestas", {})
    
    # ── Validación de relevancia (Tolerante a errores ortográficos) ──
    if paso_actual > 0:
        # Evaluar la respuesta actual usando el estimador de Likert
        likert_test = estimar_likert_completo({paso_actual: texto})
        
        es_valido = False
        
        # Validar si el estudiante está indicando explícitamente que no tiene síntomas
        texto_lower = texto.lower()
        palabras_escape = ["ningun", "ningún", "nada", "no tengo", "no he notado", "no he sentido", "sin cambio", "no hay", "cero", "no"]
        if len(texto.split()) <= 10 and any(p in texto_lower for p in palabras_escape):
            es_valido = True
            
        if not es_valido:
            if paso_actual == 1 and any(v != 0 for v in likert_test[0:9]): es_valido = True
            elif paso_actual == 2 and any(v != 0 for v in likert_test[9:15]): es_valido = True
            elif paso_actual == 3 and any(v != 0 for v in likert_test[15:22]): es_valido = True
            elif paso_actual == 4 and any(v != 0 for v in likert_test[22:27]): es_valido = True
            elif paso_actual == 5 and any(v != 0 for v in likert_test[27:35]): es_valido = True
        
        if not es_valido:
            # Revisar si ya había fallado antes en esta misma pregunta
            fallos_previos = session.get(f"fallos_{paso_actual}", 0)
            
            if fallos_previos == 0:
                # Primer intento fallido: pedir aclaración y repetir la pregunta
                session[f"fallos_{paso_actual}"] = 1
                
                # Rescatar la pregunta que se le había hecho
                sentimiento_previo = session.get("ultimo_sentimiento", "neutral")
                bloque_pregunta = PREGUNTAS[paso_actual]
                pregunta_repetida = bloque_pregunta.get(sentimiento_previo, bloque_pregunta["neutral"])
                
                mensaje_error = (
                    "No he detectado palabras clave exactas relacionadas con la pregunta (a veces ocurre por errores ortográficos). "
                    "Por favor, intenta ser más específico sobre tu situación.<br><br>"
                    f"<i>Te repito la pregunta:</i><br><b>{pregunta_repetida}</b>"
                )
                return jsonify({"error": mensaje_error}), 400
            else:
                # Segundo intento: dejarlo avanzar para no frustrar la experiencia
                # Reseteamos el contador para evitar problemas futuros
                session[f"fallos_{paso_actual}"] = 0
                pass

    # Guardar respuesta actual y su sentimiento
    respuestas[str(paso_actual)] = texto
    session["respuestas"] = respuestas
    
    sentimiento_actual = analizar_sentimiento_general(texto)
    session["ultimo_sentimiento"] = sentimiento_actual
    
    # Guardar respuesta en la Base de Datos
    sesion_id = session.get("sesion_id")
    if sesion_id:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO respuestas (sesion_id, paso, texto, sentimiento)
            VALUES (?, ?, ?, ?)
        ''', (sesion_id, paso_actual, texto, sentimiento_actual))
        conn.commit()
        conn.close()

    # Avanzar al siguiente paso
    paso_siguiente = paso_actual + 1

    if paso_siguiente <= 5:
        # Hay más preguntas
        session["paso"] = paso_siguiente
        
        # Enrutamiento dinámico según sentimiento
        bloque_pregunta = PREGUNTAS[paso_siguiente]
        dimension = bloque_pregunta["dimension"]
        
        if sentimiento_actual == "positivo":
            pregunta = bloque_pregunta["positivo"]
        elif sentimiento_actual == "negativo":
            pregunta = bloque_pregunta["negativo"]
        else:
            pregunta = bloque_pregunta["neutral"]

        # Insertar insight dinámico basado en lo que el usuario acaba de decir
        insight = generar_insight_dinamico(texto, paso_actual)
        pregunta = insight + pregunta

        return jsonify({
            "tipo": "pregunta",
            "pregunta": pregunta,
            "dimension": dimension,
            "paso": paso_siguiente,
            "total_pasos": 5,
            "sentimiento_detectado": sentimiento_actual
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

        # Actualizar la base de datos con el resultado final
        sesion_id = session.get("sesion_id")
        if sesion_id:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sesiones 
                SET nivel_final = ?, confianza = ?, completado = ?
                WHERE id = ?
            ''', (nivel, confianza, True, sesion_id))
            conn.commit()
            conn.close()

        # Reiniciar variables de paso (mantener sesion_id)
        session["paso"] = 0
        session["respuestas"] = {}

        return jsonify({
            "tipo": "resultado",
            "nivel": nivel,
            "confianza": round(confianza, 4),
            "probas": probas,
            "recomendacion": RECOMENDACIONES.get(nivel, ""),
        })




# ══════════════════════════════════════════════════════════════
# Ejecutar servidor
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SISCO - Chatbot de Estres Academico")
    print("  Modelo Hibrido BETO + Likert")
    print(f"  Accuracy entrenamiento: {model_config.get('accuracy_test', 'N/A')}")
    print("=" * 55)
    print("  Abriendo en http://localhost:7860")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", debug=False, port=7860)
