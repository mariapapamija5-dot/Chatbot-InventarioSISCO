# -*- coding: utf-8 -*-
"""
Diccionario de Datos Conversacional — SISCO
Módulo centralizado de normalización de texto para el chatbot.

Contiene:
- Jerga colombiana universitaria (~200+ entradas)
- Correcciones ortográficas comunes
- Abreviaciones de chat
- Frases hechas (multi-palabra)
- Intensificadores y negaciones
- Funciones de normalización dual (para BETO y para keywords)
"""
import re

# ══════════════════════════════════════════════════════════════
# FRASES HECHAS (multi-palabra) — se procesan PRIMERO
# Orden: más largas primero para evitar matcheos parciales
# ══════════════════════════════════════════════════════════════
FRASES_HECHAS = {
    # ── Expresiones de estrés alto ─────────────────────────────
    "no doy más":           "estoy agotado y no puedo más",
    "no doy mas":           "estoy agotado y no puedo más",
    "no puedo más":         "estoy agotado y no puedo más",
    "no puedo mas":         "estoy agotado y no puedo más",
    "me tiene loco":        "me tiene muy estresado",
    "me tiene loca":        "me tiene muy estresada",
    "me tiene mamado":      "me tiene muy cansado",
    "me tiene mamada":      "me tiene muy cansada",
    "estoy que no puedo":   "estoy agotado y no puedo más",
    "ya no puedo":          "estoy agotado y no puedo más",
    "me quiero morir":      "estoy desesperado con la situación",
    "quiero tirar la toalla": "quiero abandonar todo",
    "tirar la toalla":      "quiero abandonar todo",
    "me voy a enloquecer":  "estoy muy estresado",
    "voy a enloquecer":     "estoy muy estresado",
    "me voy a volar":       "estoy muy estresado",
    "no me da la vida":     "no tengo tiempo suficiente",
    "no me alcanza el tiempo": "no tengo tiempo suficiente",
    "estoy hasta la coronilla": "estoy harto y estresado",
    "estoy hasta el cuello": "estoy abrumado de trabajo",
    "me tiene los nervios de punta": "estoy muy ansioso",
    "nervios de punta":     "estoy muy ansioso",
    "me da cosa":           "me da miedo y ansiedad",
    "me da oso":            "me da vergüenza",
    "me da pena":           "me da vergüenza",
    "me siento mal":        "me siento mal emocionalmente",
    "me siento re mal":     "me siento muy mal emocionalmente",
    "la tengo difícil":     "la situación es muy difícil",
    "la veo difícil":       "la situación es muy difícil",
    "me la paso estudiando": "paso demasiado tiempo estudiando",
    "no he podido dormir":  "tengo problemas graves de sueño",
    "casi no duermo":       "tengo problemas graves de sueño",
    "no como bien":         "tengo cambios en la alimentación",
    "no he comido bien":    "tengo cambios en la alimentación",
    "me da igual todo":     "siento apatía y vacío emocional",
    "todo me vale":         "siento apatía y vacío emocional",
    "me vale todo":         "siento apatía y vacío emocional",
    "a cada rato":          "con mucha frecuencia",
    "todo el tiempo":       "constantemente",
    "ya no me importa":     "siento apatía y vacío",
    "qué pereza":           "siento mucho desgano",
    "que pereza":           "siento mucho desgano",
    "de malas":             "de mal humor y con mala suerte",
    "de buenas":            "de buen humor y con suerte",
    "por allá bien":        "estoy bien sin problemas",
    "pilas con":            "hay que tener cuidado con",

    # ── Expresiones detectadas de estudiantes reales ───────────
    "no he podido dormir":  "tengo problemas graves de sueño",
    "no puedo dormir":      "tengo problemas graves de sueño",
    "me sentido cansada":   "me he sentido cansada",
    "me sentido cansado":   "me he sentido cansado",
    "me he sentido cansada": "me siento agotada y fatigada",
    "me he sentido cansado": "me siento agotado y fatigado",
    "no tengo ganas":       "siento desgano y apatía",
    "no tengas de hablar":  "no tengo ganas de hablar",
    "no tengas de":         "no tengo ganas de",
    "sin ganas de nada":    "siento desgano total y apatía",
    "sin ganas de hablar":  "no quiero hablar con nadie",
    "sin ganas de ir":      "no quiero ir",
    "sin ganas de estudiar": "no quiero estudiar",
    "no he podido concentrarme": "tengo problemas graves de concentración",
    "no puedo concentrarme": "tengo problemas graves de concentración",
    "no me concentro":      "tengo problemas de concentración",
    "me sentido agrumada":  "me he sentido abrumada",
    "me sentido abrumada":  "me he sentido abrumada",
    "me siento abrumada":   "me siento agobiada y sobrecargada",
    "me siento abrumado":   "me siento agobiado y sobrecargado",
    "no entrar a clases":   "no quiero asistir a clases",
    "no entrar clases":     "no quiero asistir a clases",
    "hagas no entrar":      "ganas de no entrar",
    "sentido hagas":        "sentido ganas",
    "no he podido concentrarme en": "tengo problemas graves de concentración en",
    "dificultad para concentrarme": "tengo dificultad grave para concentrarme",
    "buenas notas":         "buenas calificaciones",
    "tener buenas notas":   "mantener buenas calificaciones",
    "tener un promedio":    "mantener un buen promedio académico",

    # ── Expresiones de nivel bajo ──────────────────────────────
    "todo bien":            "todo está bien y tranquilo",
    "todo normal":          "todo está bien y tranquilo",
    "todo tranquilo":       "todo está bien y tranquilo",
    "de lo más bien":       "estoy muy bien",
    "lo más de bien":       "estoy muy bien",
    "a toda":               "estoy muy bien y con energía",
    "a toda mecha":         "estoy muy bien y con energía",
    "sin problema":         "sin ningún problema",
    "sin problemas":        "sin ningún problema",
    "lo tengo controlado":  "tengo la situación bajo control",
    "todo bajo control":    "tengo la situación bajo control",
    "no me preocupa":       "no me genera preocupación",
    "no me preocupo":       "no me genera preocupación",
    "no me estreso":        "no siento estrés",
    "me siento bien":       "me siento bien emocionalmente",
    "me va bien":           "me va bien académicamente",

    # ── Expresiones de nivel moderado ──────────────────────────
    "más o menos":          "regular ni bien ni mal",
    "mas o menos":          "regular ni bien ni mal",
    "ahí vamos":            "regular vamos sobrellevándolo",
    "ahí voy":              "regular voy sobrellevándolo",
    "ahí más o menos":      "regular ni bien ni mal",
    "a veces sí a veces no": "es intermitente y variable",
    "depende del día":      "es variable según el día",
    "unos días sí otros no": "es intermitente y variable",
    "hay días":             "es variable y depende del día",
}

# ══════════════════════════════════════════════════════════════
# JERGA COLOMBIANA UNIVERSITARIA — Palabra → Español estándar
# Organizado por categorías para facilitar mantenimiento
# ══════════════════════════════════════════════════════════════
JERGA_COLOMBIANA = {
    # ── Abreviaciones de chat ──────────────────────────────────
    "xq":   "porque",
    "pq":   "porque",
    "xk":   "porque",
    "pk":   "porque",
    "porq": "porque",
    "x":    "por",
    "q":    "que",
    "d":    "de",
    "bn":   "bien",
    "ml":   "mal",
    "tb":   "también",
    "tmb":  "también",
    "tbn":  "también",
    "tbn":  "también",
    "tmbn": "también",
    "pa":   "para",
    "pal":  "para el",
    "dms":  "demasiado",
    "ns":   "no sé",
    "npi":  "no tengo idea",
    "nse":  "no sé",
    "ntc":  "no te creas",
    "np":   "no puedo",
    "nd":   "nada",
    "ps":   "pues",
    "pss":  "pues",
    "osea": "o sea",
    "osi":  "o sea",
    "msj":  "mensaje",
    "dsp":  "después",
    "desp": "después",
    "toy":  "estoy",
    "stoy": "estoy",
    "toi":  "estoy",
    "ta":   "está",
    "ando": "estoy",
    "tmpc": "tampoco",
    "noc":  "no sé",
    "aki":  "aquí",
    "aca":  "acá",
    "bno":  "bueno",
    "bna":  "buena",
    "wno":  "bueno",
    "vrdd": "verdad",
    "vdd":  "verdad",
    "grax": "gracias",
    "grs":  "gracias",
    "plz":  "por favor",
    "pls":  "por favor",
    "cont": "contigo",
    "alv":  "a la mala",
    "f":    "fracaso",
    "gg":   "perdí",
    "rip":  "acabé",

    # ── Jerga colombiana — Coloquial ───────────────────────────
    "parce":     "amigo",
    "parcero":   "amigo",
    "parcera":   "amiga",
    "llave":     "amigo cercano",
    "marica":    "amigo",
    "ñero":      "amigo",
    "ñera":      "amiga",
    "mijo":      "amigo",
    "mija":      "amiga",
    "hermano":   "amigo",
    "hermanito": "amigo",
    "mono":      "amigo",
    "viejo":     "amigo",
    "nea":       "amigo",
    "pana":      "amigo",
    "man":       "persona",
    "men":       "persona",

    # ── Jerga colombiana — Estados emocionales ─────────────────
    "mamado":    "cansado",
    "mamada":    "cansada",
    "embalado":  "estresado",
    "embalada":  "estresada",
    "encartado": "agobiado",
    "encartada": "agobiada",
    "jarto":     "harto",
    "jarta":     "harta",
    "jartera":   "aburrimiento y desgano",
    "aburrido":  "aburrido y desanimado",
    "achantado": "desanimado y triste",
    "achantada": "desanimada y triste",
    "bajoneado": "triste y desanimado",
    "bajoneada": "triste y desanimada",
    "azarado":   "nervioso y asustado",
    "azarada":   "nerviosa y asustada",
    "boleta":    "vergonzoso",
    "tragado":   "enamorado",
    "tragada":   "enamorada",
    "chiviado":  "nervioso",
    "chiviada":  "nerviosa",

    # ── Jerga colombiana — Adjetivos de situación ──────────────
    "berraco":   "difícil",
    "verraco":   "difícil",
    "berraca":   "difícil",
    "tenaz":     "difícil e intenso",
    "chimbo":    "malo y de mala calidad",
    "chimba":    "increíble",
    "bacano":    "bueno y agradable",
    "bacana":    "buena y agradable",
    "chévere":   "bueno y agradable",
    "chevere":   "bueno y agradable",
    "regio":     "excelente",
    "brutal":    "muy difícil",
    "teso":      "muy bueno y capaz",
    "tesa":      "muy buena y capaz",
    "gonorrea":  "muy difícil",
    "maluquera": "malestar",
    "maluco":    "malo y desagradable",
    "maluca":    "mala y desagradable",

    # ── Jerga colombiana — Verbos y acciones ───────────────────
    "camellar":  "trabajar mucho",
    "camello":   "trabajo pesado",
    "camellando": "trabajando mucho",
    "rumbear":   "salir de fiesta",
    "parchar":   "descansar y relajarse",
    "parchando": "descansando",
    "parchado":  "tranquilo y relajado",
    "parchada":  "tranquila y relajada",
    "relajao":   "relajado",
    "relajá":    "relajada",
    "enrumbarse": "salir de fiesta",
    "trasnochar": "quedarse despierto hasta tarde",
    "trasnochando": "quedándose despierto hasta tarde",

    # ── Jerga colombiana — Académicas ──────────────────────────
    "rajé":      "reprobé",
    "rajarse":   "reprobar",
    "raspé":     "reprobé",
    "raspar":    "reprobar",
    "colgué":    "me atrasé",
    "colgado":   "atrasado",
    "colgada":   "atrasada",
    "cancelar":  "retirar materia",
    "quemé":     "reprobé",
    "quemar":    "reprobar",
    "la materia": "la asignatura",
    "profe":     "profesor",
    "profes":    "profesores",
    "semestre":  "periodo académico",
    "parcial":   "examen parcial",
    "parciales": "exámenes parciales",
    "corte":     "periodo de evaluación",
    "nota":      "calificación",
    "notas":     "calificaciones",

    # ── Jerga colombiana — Intensificadores coloquiales ────────
    "re":        "muy",
    "super":     "muy",
    "súper":     "muy",
    "full":      "mucho",
    "harto":     "mucho",
    "caleta":    "mucho",
    "resto":     "mucho",
    "un resto":  "mucho",
    "hartos":    "muchos",
    "un poco":   "algo",
    "algo":      "un poco",
    "mero":      "muy",
    "bien":      "bastante",
    "mega":      "muy",
    "ultra":     "extremadamente",
}

# ══════════════════════════════════════════════════════════════
# CORRECCIONES ORTOGRÁFICAS — Errores comunes en chat informal
# ══════════════════════════════════════════════════════════════
CORRECCIONES_ORTOGRAFICAS = {
    # Tildes faltantes comunes
    "tambien":    "también",
    "asi":        "así",
    "mas":        "más",
    "dificil":    "difícil",
    "facil":      "fácil",
    "estres":     "estrés",
    "examenes":   "exámenes",
    "examen":     "examen",
    "animo":      "ánimo",
    "ultimo":     "último",
    "ultima":     "última",
    "calificacion": "calificación",
    "evaluacion": "evaluación",
    "exposicion": "exposición",
    "presion":    "presión",
    "depresion":  "depresión",
    "situacion":  "situación",
    "concentracion": "concentración",
    "motivacion": "motivación",
    "organizacion": "organización",
    "alimentacion": "alimentación",
    "comunicacion": "comunicación",

    # Errores ortográficos comunes
    "aser":       "hacer",
    "acer":       "hacer",
    "haver":      "haber",
    "haber":      "haber",
    "aber":       "haber",
    "lla":        "ya",
    "io":         "yo",
    "i":          "y",
    "ay":         "hay",
    "ahi":        "ahí",
    "ahy":        "ahí",
    "hai":        "hay",
    "haci":       "así",
    "ací":        "así",
    "derrepente": "de repente",
    "haya":       "haya",
    "alla":       "allá",
    "voi":        "voy",
    "oy":         "hoy",
    "aller":      "ayer",
    "ciendo":     "siendo",
    "estoi":      "estoy",
    "vien":       "bien",
    "mui":        "muy",
    "siempre":    "siempre",
    "deveria":    "debería",
    "deverdad":   "de verdad",
    "esque":      "es que",
    "osea":       "o sea",
    "talves":     "tal vez",
    "talvez":     "tal vez",
    "atraves":    "a través",
    "aparte":     "aparte",
    "apesar":     "a pesar",
    "nesesito":   "necesito",
    "nesecito":   "necesito",
    "necesito":   "necesito",
    "siento":     "siento",
    "ciento":     "siento",
    "cansa":      "cansa",
    "cansansio":  "cansancio",
    "cansancio":  "cansancio",
    "ansiedad":   "ansiedad",
    "anciedad":   "ansiedad",
    "ansieda":    "ansiedad",
    "sicólogo":   "psicólogo",
    "sicologo":   "psicólogo",

    # Errores detectados de estudiantes reales
    "agrumada":   "abrumada",
    "agrumado":   "abrumado",
    "abrumda":    "abrumada",
    "aburrmida":  "aburrida",
    "hagas":      "ganas",
    "tengas":     "tengo ganas",
    "sentido":    "sentido",
    "cansda":     "cansada",
    "cansdo":     "cansado",
    "estresda":   "estresada",
    "estresdo":   "estresado",
    "precion":    "presión",
    "presionado": "presionado",
    "presionada": "presionada",
    "promedio":   "promedio",
    "conscentrarme": "concentrarme",
    "concentrame":   "concentrarme",
    "concetrarme":   "concentrarme",
    "consentrarme":  "concentrarme",
    "dormido":    "dormido",
    "dormirme":   "dormirme",
    "claces":     "clases",
    "companeros": "compañeros",
    "companeras": "compañeras",
    "compañero":  "compañero",
    "musica":     "música",
    "angustida":  "angustiada",
    "agobiada":   "agobiada",
    "agobidada":  "agobiada",
    "nervosa":    "nerviosa",
    "nervoso":    "nervioso",
    "trankilo":   "tranquilo",
    "trankila":   "tranquila",
    "aburrda":    "aburrida",
    "irritble":   "irritable",
    "sueño":      "sueño",
    "somnolecia": "somnolencia",
}

# ══════════════════════════════════════════════════════════════
# INTENSIFICADORES Y NEGACIONES
# ══════════════════════════════════════════════════════════════
INTENSIFICADORES = [
    "mucho", "muchísimo", "demasiado", "bastante", "muy",
    "extremadamente", "totalmente", "completamente", "absolutamente",
    "súper", "super", "re", "mega", "ultra", "full",
    "horrible", "terrible", "fatal", "pésimo",
    "increíblemente", "excesivamente", "exageradamente",
    "un montón", "un resto", "caleta", "harto",
    "siempre", "constantemente", "todo el tiempo", "a cada rato",
]

NEGACIONES = [
    "no", "ni", "sin", "nunca", "jamás", "jamas",
    "tampoco", "nada", "ningún", "ningun", "ninguno",
    "ninguna", "nadie", "para nada",
]

# ══════════════════════════════════════════════════════════════
# FUNCIONES DE NORMALIZACIÓN
# ══════════════════════════════════════════════════════════════

def _aplicar_frases_hechas(texto):
    """Reemplaza frases hechas multi-palabra (las más largas primero)."""
    t = texto
    # Ordenar por longitud descendente para matchear frases más largas primero
    frases_ordenadas = sorted(FRASES_HECHAS.items(), key=lambda x: len(x[0]), reverse=True)
    for frase, reemplazo in frases_ordenadas:
        t = t.replace(frase, reemplazo)
    return t


def _aplicar_jerga(texto):
    """Reemplaza jerga colombiana por equivalentes en español estándar."""
    palabras = texto.split()
    resultado = []
    for palabra in palabras:
        # Buscar la palabra limpia (sin puntuación al final)
        palabra_limpia = re.sub(r'[.,;:!?]+$', '', palabra)
        sufijo = palabra[len(palabra_limpia):]

        if palabra_limpia in JERGA_COLOMBIANA:
            resultado.append(JERGA_COLOMBIANA[palabra_limpia] + sufijo)
        else:
            resultado.append(palabra)
    return " ".join(resultado)


def _aplicar_correcciones(texto):
    """Corrige errores ortográficos comunes."""
    palabras = texto.split()
    resultado = []
    for palabra in palabras:
        palabra_limpia = re.sub(r'[.,;:!?]+$', '', palabra)
        sufijo = palabra[len(palabra_limpia):]

        if palabra_limpia in CORRECCIONES_ORTOGRAFICAS:
            resultado.append(CORRECCIONES_ORTOGRAFICAS[palabra_limpia] + sufijo)
        else:
            resultado.append(palabra)
    return " ".join(resultado)


def normalizar_para_beto(texto):
    """
    Normalización SUAVE para el modelo BETO.
    Conserva estructura semántica pero traduce jerga a español estándar.
    BETO entiende mejor español formal que jerga.

    Pipeline:
    1. Minúsculas
    2. Frases hechas → español estándar
    3. Jerga → español estándar
    4. Correcciones ortográficas
    5. Limpieza de caracteres especiales (conserva letras con tilde)
    6. Normalizar espacios
    """
    t = str(texto).lower().strip()
    t = _aplicar_frases_hechas(t)
    t = _aplicar_jerga(t)
    t = _aplicar_correcciones(t)
    # Conservar solo letras (incluyendo tildes y ñ) y espacios
    t = re.sub(r"[^a-záéíóúüñ\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalizar_para_keywords(texto):
    """
    Normalización AGRESIVA para matching de keywords Likert.
    Igual que para BETO pero sin eliminar la estructura de negaciones.

    Pipeline:
    1. Todo lo de normalizar_para_beto
    2. Se conservan las negaciones intactas para detección posterior
    """
    # Usamos la misma normalización base
    return normalizar_para_beto(texto)


def detectar_intensidad_global(texto):
    """
    Detecta la intensidad emocional global del texto.

    Returns:
        float: multiplicador de intensidad (1.0 = normal, hasta 2.0 = muy intenso)
    """
    texto_lower = texto.lower()
    conteo = 0
    for intensificador in INTENSIFICADORES:
        if intensificador in texto_lower:
            conteo += 1

    if conteo >= 5:
        return 1.8
    elif conteo >= 3:
        return 1.5
    elif conteo >= 1:
        return 1.2
    return 1.0


def tiene_negacion_antes(texto, posicion_keyword):
    """
    Verifica si hay una negación en las 3 palabras anteriores a la posición del keyword.

    Args:
        texto: texto completo normalizado
        posicion_keyword: posición (índice de carácter) donde empieza el keyword

    Returns:
        bool: True si hay una negación antes del keyword
    """
    # Extraer las palabras antes del keyword
    texto_antes = texto[:posicion_keyword].strip()
    palabras_antes = texto_antes.split()

    # Revisar las últimas 3 palabras
    ultimas = palabras_antes[-3:] if len(palabras_antes) >= 3 else palabras_antes

    for palabra in ultimas:
        if palabra in NEGACIONES:
            return True
    return False
