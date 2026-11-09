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
import contextlib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModel

# ══════════════════════════════════════════════════════════════
# Detección de entorno: producción (PostgreSQL) o local (SQLite)
# ══════════════════════════════════════════════════════════════
DATABASE_URL = os.environ.get("DATABASE_URL")  # Supabase/PostgreSQL en producción

USE_POSTGRES = False
PSYCOPG_VERSION = None
PH = "?"

if DATABASE_URL:
    try:
        import psycopg
        from psycopg.rows import dict_row
        USE_POSTGRES = True
        PSYCOPG_VERSION = 3
        PH = "%s"
        print("[SISCO] Modo PRODUCCIÓN: conectado a PostgreSQL (psycopg v3)")
    except Exception:
        try:
            import psycopg2
            import psycopg2.extras
            USE_POSTGRES = True
            PSYCOPG_VERSION = 2
            PH = "%s"
            print("[SISCO] Modo PRODUCCIÓN: conectado a PostgreSQL (psycopg2)")
        except Exception as err:
            print(f"[SISCO ALERTA] No se pudo cargar driver PostgreSQL ({err}). Usando SQLite de respaldo.")
            USE_POSTGRES = False
            PH = "?"
            BASE_DIR_DB = os.path.dirname(os.path.abspath(__file__))
            DB_PATH = os.path.join(BASE_DIR_DB, "sisco_chatbot_v2.db")

if not USE_POSTGRES:
    BASE_DIR_DB = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(BASE_DIR_DB, "sisco_chatbot_v2.db")
    print("[SISCO] Modo LOCAL: usando SQLite")


@contextlib.contextmanager
def get_db_connection():
    """Context manager que devuelve una conexión según el entorno activo."""
    if USE_POSTGRES:
        if PSYCOPG_VERSION == 3:
            conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        else:
            conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db():
    """Crea las tablas si no existen. Compatible con SQLite y PostgreSQL."""
    if USE_POSTGRES:
        sql_usuarios = """
            CREATE TABLE IF NOT EXISTS usuarios (
                id SERIAL PRIMARY KEY,
                nombre_usuario TEXT UNIQUE NOT NULL,
                pin TEXT NOT NULL,
                edad INTEGER
            )"""
        sql_sesiones = """
            CREATE TABLE IF NOT EXISTS sesiones (
                id TEXT PRIMARY KEY,
                usuario_id INTEGER REFERENCES usuarios(id),
                fecha_inicio TIMESTAMP DEFAULT NOW(),
                nivel_final TEXT,
                confianza REAL,
                completado BOOLEAN DEFAULT FALSE
            )"""
        sql_respuestas = """
            CREATE TABLE IF NOT EXISTS respuestas (
                id SERIAL PRIMARY KEY,
                sesion_id TEXT REFERENCES sesiones(id),
                paso INTEGER,
                texto TEXT,
                sentimiento TEXT
            )"""
    else:
        sql_usuarios = """
            CREATE TABLE IF NOT EXISTS usuarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nombre_usuario TEXT UNIQUE,
                pin TEXT,
                edad INTEGER
            )"""
        sql_sesiones = """
            CREATE TABLE IF NOT EXISTS sesiones (
                id TEXT PRIMARY KEY,
                usuario_id INTEGER,
                fecha_inicio TIMESTAMP,
                nivel_final TEXT,
                confianza REAL,
                completado BOOLEAN,
                FOREIGN KEY(usuario_id) REFERENCES usuarios(id)
            )"""
        sql_respuestas = """
            CREATE TABLE IF NOT EXISTS respuestas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sesion_id TEXT,
                paso INTEGER,
                texto TEXT,
                sentimiento TEXT,
                FOREIGN KEY(sesion_id) REFERENCES sesiones(id)
            )"""

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql_usuarios)
        cursor.execute(sql_sesiones)
        cursor.execute(sql_respuestas)

# Inicializar DB al arrancar
init_db()

# Configuración del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelo_hibrido_final")
MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "sisco_fup_2026_secretkey")


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
        "positivo": "Me alegra saber que te has sentido tranquilo/a. Aún así, la universidad siempre tiene sus retos. ¿Qué tareas o dinámicas suelen exigirte más esfuerzo o causarte algo de presión? (Por ejemplo: trabajos grupales, parciales, exponer, etc.).",
        "negativo": "Lamento que estés pasando por eso — es más común de lo que crees entre los estudiantes. Para poder entenderte mejor, cuéntame: ¿qué es específicamente lo que más te está generando esa presión en la universidad? (Por ejemplo: acumulación de trabajos, miedo a perder una materia, exposiciones, etc.).",
        "neutral": "Gracias por contarme. ¿Hay alguna tarea o dinámica de la universidad que actualmente te esté generando mayor presión o exigencia?"
    },
    2: {
        "dimension": "Sintomas fisicos",
        "positivo": "Qué bueno. Ahora, cambiando un poco el tema, me gustaría preguntarte sobre tu cuerpo — a veces el estrés se manifiesta físicamente sin que nos demos cuenta. ¿Has tenido problemas para dormir, dolores de cabeza, cansancio frecuente o malestares estomacales últimamente?",
        "negativo": "Es totalmente comprensible que todo eso te afecte. Ahora cuéntame, ¿has notado algún cambio en tu cuerpo? Por ejemplo: dificultad para dormir, cansancio inusual, dolores de cabeza o malestar en el estómago.",
        "neutral": "Gracias. Ahora me gustaría saber cómo ha respondido tu cuerpo a toda esta carga. ¿Has tenido problemas para dormir, cansancio frecuente, dolores de cabeza, malestar estomacal o sensación de somnolencia?"
    },
    3: {
        "dimension": "Sintomas psicologicos",
        "positivo": "Gracias por compartirlo. ¿Y a nivel emocional cómo te has sentido en general? ¿Has notado algo de ansiedad, dificultad para concentrarte o irritabilidad, aunque sea leve?",
        "negativo": "Sé que hablar de esto no siempre es fácil, y te lo agradezco. ¿Y emocionalmente cómo estás? ¿Has sentido ansiedad intensa, tristeza, dificultad para concentrarte, muchos nervios o ganas de abandonar todo?",
        "neutral": "Gracias por contarme. ¿Y emocionalmente cómo te has sentido? ¿Has experimentado ansiedad, nervios, tristeza, dificultad para concentrarte o irritabilidad?"
    },
    4: {
        "dimension": "Comportamiento",
        "positivo": "Entendido. ¿Has notado algún cambio en tu rutina o comportamiento diario? Por ejemplo: menos ganas de socializar, cambios en tu apetito o desgano para ir a clase.",
        "negativo": "Gracias por tu honestidad. Cuando cargamos tanto peso emocional, es natural que nuestra rutina también cambie. ¿Has notado algo diferente en tu comportamiento? Por ejemplo: aislarte, cambios fuertes en tu apetito, conflictos con personas cercanas o desgano total para ir a clase.",
        "neutral": "Gracias. ¿Has notado cambios en tu comportamiento o rutina diaria? Por ejemplo: aislarte de los demás, cambios en tu alimentación, desgano para ir a clase o conflictos con otras personas."
    },
    5: {
        "dimension": "Afrontamiento",
        "positivo": "Ya casi terminamos. ¿Qué estrategias usas normalmente para mantener el equilibrio y manejar el estrés? (ej. hacer ejercicio, organizar tu tiempo, salir con amigos, algún pasatiempo).",
        "negativo": "Ya casi terminamos, gracias por tu paciencia y confianza. Cuando te sientes así de agobiado/a, ¿qué sueles hacer para intentar manejarlo? Por ejemplo: hablar con alguien, escuchar música, buscar ayuda, rezar u organizar mejor tu tiempo.",
        "neutral": "Ya casi terminamos. Cuando sientes que el estrés aumenta, ¿qué haces para manejarlo? Por ejemplo: hablar con alguien, hacer ejercicio, escuchar música, buscar ayuda profesional o algún pasatiempo."
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
    insights = []   # Lista en lugar de un solo insight (soporta múltiples síntomas)
    consejo = ""

    # Detectar si el usuario está pidiendo ayuda explícitamente
    palabras_ayuda = ["recomend", "consejo", "que hago", "qué hago", "ayuda", "tip", "que me sugieres", "que puedo hacer", "qué puedo hacer"]
    pide_consejo = any(p in t for p in palabras_ayuda)

    # ── Paso 0 / 1: Estresores académicos ─────────────────────────
    if paso == 0 or paso == 1:
        if "tesis" in t or "sustentacion" in t or "sustentación" in t or "monografia" in t or "monografía" in t or "grado" in t or "trabajo de grado" in t:
            insights.append("Preparar una tesis o sustentación es uno de los procesos académicos más demandantes que existen — implica esfuerzo intelectual, emocional y de tiempo al mismo tiempo.")
            consejo = "Dividir el proceso en hitos semanales pequeños (hoy reviso esta sección, mañana esa otra) ayuda muchísimo a no sentir el peso de todo al mismo tiempo."
        if "correcciones" in t or "corrección" in t or "correcciones" in t:
            insights.append("Las correcciones y ajustes de un documento académico pueden sentirse interminables, especialmente cuando ya llevas mucho tiempo en el proceso.")
            consejo = "Tómate unos minutos al empezar cada sesión para reconocer cuánto has avanzado — eso te dará más energía para continuar."
        if "parciales" in t or "examen" in t or "evaluacion" in t or "evaluación" in t:
            insights.append("Los periodos de exámenes siempre traen una carga de presión muy alta.")
            consejo = "Dividir los temas por días y estudiar en bloques de 45 minutos ayuda mucho."
        if "grupo" in t or "equipo" in t or "compañeros" in t or "exponer" in t or "exposicion" in t or "exposición" in t:
            insights.append("Los trabajos en equipo o exposiciones requieren mucha coordinación interpersonal — es normal que generen un desgaste extra.")
            consejo = "Establece roles y fechas de entrega claras desde la primera reunión del grupo."
        if ("tiempo" in t or "trabajos" in t or "muchas cosas" in t or "muchos trabajos" in t or "muchas materias" in t) and not insights:
            insights.append("Se nota que estás lidiando con muchas responsabilidades a la vez — es normal sentirse abrumado/a cuando todo llega junto.")
            consejo = "Una lista de prioridades o una agenda sencilla puede darte más sensación de control."
        if ("profesor" in t or "docente" in t or "entender" in t) and not insights:
            insights.append("Es frustrante cuando la dinámica con las clases o los docentes no es la mejor.")
            consejo = "No dudes en pedir tutorías o apoyarte en tus compañeros de clase."

    # ── Paso 2: Síntomas físicos ───────────────────────────────────
    elif paso == 2:
        if "sueño" in t or "dormir" in t or "insomnio" in t or "desvelo" in t or "no duermo" in t:
            insights.append("Los problemas para dormir afectan muchísimo el rendimiento y el estado de ánimo al día siguiente.")
            consejo = "Intenta establecer una rutina relajante y desconectarte de pantallas al menos una hora antes de ir a la cama."
        if "cansad" in t or "fatiga" in t or "agotad" in t or "sin energía" in t or "sin energia" in t:
            insights.append("El cansancio constante es una señal de que tu cuerpo necesita descanso real, no solo dormir.")
            if not consejo:
                consejo = "Incluso 20 minutos de descanso sin pantalla durante el día pueden hacer una gran diferencia."
        if "cabeza" in t or "migraña" in t or "migrana" in t or "cefalea" in t:
            insights.append("Los dolores de cabeza son una señal clara de que tu cuerpo está acumulando mucha tensión.")
            if not consejo:
                consejo = "Intenta hacer pausas activas cada hora, alejándote por completo de las pantallas."
        if "estomago" in t or "estómago" in t or "digestion" in t or "digestión" in t or "nausea" in t or "náusea" in t:
            insights.append("El estrés muchas veces se manifiesta directamente en el estómago — es una reacción muy natural del cuerpo.")
            if not consejo:
                consejo = "Trata de no saltarte las comidas principales, incluso si sientes que no tienes mucho tiempo."

    # ── Paso 3: Síntomas psicológicos ─────────────────────────────
    elif paso == 3:
        if "ansiedad" in t or "nervios" in t or "nerviosismo" in t or "angustia" in t or "desesper" in t:
            insights.append("La ansiedad y los nervios pueden ser muy agotadores — lamento que estés pasando por eso.")
            consejo = "Hacer respiraciones lentas (inhalar 4 segundos, retener 4, exhalar 6) ayuda a calmar el sistema nervioso en momentos de tensión."
        if "triste" in t or "llorar" in t or "depre" in t or "desanim" in t:
            insights.append("Es completamente válido sentirse triste bajo tanta presión — tus emociones son importantes y merecen atención.")
            if not consejo:
                consejo = "A veces ayuda escribir lo que sientes, o desahogarte con alguien de confianza."
        if "concentrar" in t or "mente en blanco" in t or "concentracion" in t or "concentración" in t or "distrai" in t:
            insights.append("Cuando hay tanta saturación mental, es muy frustrante sentir que la mente no responde.")
            if not consejo:
                consejo = "Si te pasa, cierra los libros 15 minutos, camina un poco y luego retoma. No fuerces tu cerebro."
        if "dejar" in t or "abandonar" in t or "renunciar" in t or "ya no quiero" in t:
            insights.append("Sentir ganas de dejarlo todo es más común de lo que crees cuando la carga es demasiada — no significa que seas débil.")
            if not consejo:
                consejo = "Hablar con alguien de confianza sobre cómo te sientes puede aliviarte mucho en esos momentos."

    # ── Paso 4: Comportamiento ─────────────────────────────────────
    elif paso == 4:
        if "comer" in t or "apetito" in t or "alimentac" in t or "hambre" in t or "ganas de comer" in t or "no como" in t or "menos ganas" in t:
            insights.append("El apetito es de las primeras cosas que el estrés altera — que notes eso habla de que tu cuerpo está dando señales importantes.")
            consejo = "Aunque no tengas mucha hambre, intenta no saltarte las comidas principales para mantener tu energía."
        if "aislad" in t or "encerrad" in t or "no salgo" in t or "no quiero ver" in t or "sin ganas de salir" in t:
            insights.append("Aislarse es un reflejo protector muy común cuando nos sentimos sobrepasados/as.")
            if not consejo:
                consejo = "Intentar conversar aunque sea 10 minutos con un amigo puede aliviar mucho esa sensación."
        if "discut" in t or "pele" in t or "irritable" in t or "mal genio" in t or "conflicto" in t:
            insights.append("Es normal que el estrés acumulado a veces salga en forma de irritabilidad con las personas cercanas.")
            if not consejo:
                consejo = "Date espacios a solas para respirar hondo antes de reaccionar cuando sientes que vas a explotar."
        if ("desgano" in t or "flojera" in t or "pereza" in t or "no quiero ir" in t or "faltar" in t) and not insights:
            insights.append("El desgano para ir a clase o cumplir con las responsabilidades es una señal de agotamiento que vale la pena atender.")
            if not consejo:
                consejo = "A veces un pequeño descanso intencional (una tarde libre sin culpa) recarga mucho las pilas."

    # ── Construir el texto final ───────────────────────────────────
    if not insights:
        return ""

    texto_insights = " ".join(insights)
    resultado = texto_insights

    if pide_consejo and consejo:
        resultado += f"<br><br>💡 <b>Consejo:</b> {consejo}"

    return f"<em>{resultado}</em><br><br>"

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

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Buscar si el usuario existe
        cursor.execute(f"SELECT id, pin FROM usuarios WHERE nombre_usuario = {PH}", (usuario,))
        user_row = cursor.fetchone()

        if user_row:
            # El usuario existe, validar PIN
            user_id = user_row["id"] if USE_POSTGRES else user_row[0]
            db_pin  = user_row["pin"] if USE_POSTGRES else user_row[1]
            if db_pin != pin:
                return jsonify({"error": "PIN incorrecto para este usuario."}), 401
        else:
            # Registrar nuevo usuario
            edad_val = int(edad) if str(edad).isdigit() else None
            if USE_POSTGRES:
                cursor.execute(
                    f"INSERT INTO usuarios (nombre_usuario, pin, edad) VALUES ({PH}, {PH}, {PH}) RETURNING id",
                    (usuario, pin, edad_val)
                )
                user_id = cursor.fetchone()["id"]
            else:
                cursor.execute(
                    f"INSERT INTO usuarios (nombre_usuario, pin, edad) VALUES ({PH}, {PH}, {PH})",
                    (usuario, pin, edad_val)
                )
                user_id = cursor.lastrowid

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
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'INSERT INTO sesiones (id, usuario_id, fecha_inicio, completado) VALUES ({PH}, {PH}, {PH}, {PH})',
                (session_id, usuario_id, datetime.now(), False)
            )
    except Exception as err_db:
        print(f"[SISCO ALERTA DB] No se pudo crear sesion en BD: {err_db}")
    
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
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f'INSERT INTO sesiones (id, usuario_id, fecha_inicio, completado) VALUES ({PH}, {PH}, {PH}, {PH})',
                (session_id, usuario_id, datetime.now(), False)
            )
        
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
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f'INSERT INTO respuestas (sesion_id, paso, texto, sentimiento) VALUES ({PH}, {PH}, {PH}, {PH})',
                    (sesion_id, paso_actual, texto, sentimiento_actual)
                )
        except Exception as err_db:
            print(f"[SISCO ALERTA DB] No se pudo guardar respuesta en BD: {err_db}")

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

        # Si hay insight, eliminar el conector redundante al inicio de la pregunta
        # (ej: "Comprendo.", "Entiendo.", "Gracias por contarme.") para que la
        # transición sea más fluida y no suene a dos voces distintas pegadas.
        if insight:
            conectores_redundantes = [
                "entiendo,", "entiendo.", "entiendo ",
                "comprendo,", "comprendo.", "comprendo ",
                "gracias por contarme,", "gracias por contarme.",
                "gracias por compartirlo,", "gracias por compartirlo.",
                "gracias.", "gracias,",
                "entendido.", "entendido,",
                "sé que hablar de esto no siempre es fácil, y te lo agradezco.",
            ]
            pregunta_lower = pregunta.lower()
            for conector in conectores_redundantes:
                if pregunta_lower.startswith(conector):
                    # Recortar el conector y capitalizar la primera letra restante
                    pregunta = pregunta[len(conector):].lstrip()
                    pregunta = pregunta[0].upper() + pregunta[1:] if pregunta else pregunta
                    break

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
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(f'''
                        UPDATE sesiones 
                        SET nivel_final = {PH}, confianza = {PH}, completado = {PH}
                        WHERE id = {PH}
                    ''', (nivel, confianza, True, sesion_id))
            except Exception as err_db:
                print(f"[SISCO ALERTA DB] No se pudo actualizar sesion final: {err_db}")

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
