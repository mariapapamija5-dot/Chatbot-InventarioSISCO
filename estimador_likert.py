# -*- coding: utf-8 -*-
"""
Estimador de valores Likert (-3 a 3) desde texto conversacional.
Analiza las respuestas del estudiante por dimensión SISCO y genera
los 35 valores Likert que alimentan la rama numérica del modelo híbrido.

MEJORAS v2:
- Matching por palabras completas (regex \\b) en vez de subcadenas
- Detección de negaciones que invierten el puntaje
- Scoring continuo proporcional en vez de escalones bruscos
- Bonus de intensificadores
- Keywords ampliados con jerga colombiana universitaria
"""
import re
from diccionario_datos import (
    normalizar_para_keywords,
    detectar_intensidad_global,
    NEGACIONES,
)

# ══════════════════════════════════════════════════════════════
# Palabras clave por columna Likert — AMPLIADO v2
# Cada columna tiene listas de indicadores "alto" (+) y "bajo" (-)
# Más matches alto → valor positivo → más estrés (excepto afrontamiento)
# ══════════════════════════════════════════════════════════════

DIMENSION_KEYWORDS = {
    # ── ESTRESORES (e_3_1 a e_3_9) ────────────────────────────
    "e_3_1": {
        "tema": "sobrecarga de tareas",
        "alto": ["tareas", "trabajos", "acumulan", "demasiado", "mucho", "carga",
                 "exceso", "cantidad", "saturado", "abrumado", "sobrecarga", "montón",
                 "acumulados", "encima", "miles", "un poco de todo", "todo junto",
                 "trabajo pesado", "mucho trabajo", "demasiados trabajos",
                 "no termino", "no acabo", "sin parar", "uno tras otro",
                 "agobiado", "agobiada", "estresado", "estresada", "saturada"],
        "bajo": ["poco", "tranquilo", "organizado", "manejable", "pocas", "al día",
                 "controlado", "controlada", "llevadero", "normal", "poquito",
                 "sin problemas", "nada grave", "todo bien", "bien"]
    },
    "e_3_2": {
        "tema": "evaluaciones y exámenes",
        "alto": ["examen", "parcial", "quiz", "evaluación", "nota", "calificación",
                 "reprobar", "perder", "reprobé", "suspenso", "prueba",
                 "exámenes", "parciales", "notas", "calificaciones",
                 "pérdida", "perdí", "reprobado", "reprobada",
                 "mal en el examen", "me fue mal", "saqué mal",
                 "baja nota", "mala nota", "malas notas",
                 "voy a perder", "miedo a perder"],
        "bajo": ["preparado", "fácil", "confianza", "estudié", "aprobé", "pasé",
                 "preparada", "bien preparado", "bien preparada",
                 "me fue bien", "buena nota", "buenas notas",
                 "aprobado", "aprobada", "confío"]
    },
    "e_3_3": {
        "tema": "tipo de trabajos",
        "alto": ["exposición", "proyecto", "grupal", "difícil", "complicado",
                 "presentación", "sustentación", "oral", "grupo",
                 "exposiciones", "proyectos", "presentaciones",
                 "trabajo en grupo", "trabajo grupal",
                 "frente a todos", "delante de todos",
                 "me cuesta", "no sé cómo hacerlo", "no entiendo"],
        "bajo": ["sencillo", "manejable", "simple", "fácil",
                 "ya lo hice", "terminé", "acabé", "fáciles"]
    },
    "e_3_4": {
        "tema": "profesores",
        "alto": ["profesor", "profe", "injusto", "exigente", "estricto", "duro",
                 "regaña", "humilla", "difícil", "malo", "grosero",
                 "profesores", "profes", "docente", "docentes",
                 "mala gente", "no explica", "no entiende",
                 "no enseña", "regañón", "regañona",
                 "injusta", "arbitrario", "severo", "severa",
                 "nos trata mal", "nos grita", "nos humilla"],
        "bajo": ["bueno", "comprensivo", "flexible", "amable", "ayuda",
                 "buen profesor", "buena profesora", "explica bien",
                 "comprensiva", "paciente", "buena gente"]
    },
    "e_3_5": {
        "tema": "tiempo limitado",
        "alto": ["tiempo", "no alcanzo", "corto", "rápido", "urgente", "apurado",
                 "plazo", "fecha", "límite", "corriendo", "no llego", "alcanza",
                 "apurada", "contra el tiempo", "a último momento",
                 "última hora", "no alcancé", "no terminé a tiempo",
                 "falta tiempo", "poco tiempo", "se me acabó el tiempo",
                 "fecha de entrega", "fecha límite", "muy rápido",
                 "no suficiente tiempo", "sin tiempo"],
        "bajo": ["suficiente", "calma", "holgura", "tranquilo", "alcanzo",
                 "con tiempo", "a tiempo", "organizado", "planificado",
                 "sin afán", "sin prisa", "alcancé"]
    },
    "e_3_6": {
        "tema": "competencia con compañeros",
        "alto": ["compañeros", "comparar", "competir", "presión", "compiten",
                 "mejor que", "peor que", "competencia", "superan",
                 "me comparo", "todos saben más", "soy el peor",
                 "soy la peor", "los demás van adelante",
                 "me quedo atrás", "no doy la talla",
                 "no estoy al nivel", "compañeras"],
        "bajo": ["apoyo", "equipo", "ayudan", "compañerismo", "juntos",
                 "nos ayudamos", "trabajo en equipo", "cooperamos",
                 "buena relación", "nos llevamos bien"]
    },
    "e_3_7": {
        "tema": "participar en clase",
        "alto": ["participar", "hablar", "pena", "vergüenza", "miedo",
                 "nervios", "opinar", "preguntar", "frente", "público",
                 "me da miedo hablar", "no me atrevo", "me da oso",
                 "pánico escénico", "me bloqueo", "me quedo callado",
                 "me quedo callada", "tartamudeo", "me tiembla la voz",
                 "me pongo rojo", "me pongo roja", "me da cosa"],
        "bajo": ["cómodo", "participo", "pregunto", "hablo", "confianza",
                 "me siento cómodo", "me siento cómoda", "tranquilo",
                 "levanto la mano", "participo activamente", "sin pena"]
    },
    "e_3_8": {
        "tema": "carga horaria",
        "alto": ["horario", "clases", "largo", "jornada", "pesado", "muchas horas",
                 "todo el día", "cansado", "madrugada", "temprano",
                 "cansada", "desde temprano", "jornada larga",
                 "muchas materias", "horario pesado", "demasiadas clases",
                 "de mañana a noche", "largas jornadas", "largo día"],
        "bajo": ["flexible", "leve", "corto", "pocos", "tiempo libre",
                 "horario cómodo", "pocas clases", "tranquilo",
                 "tengo libre", "descanso"]
    },
    "e_3_9": {
        "tema": "no entender temas",
        "alto": ["entender", "perdido", "confundido", "difícil", "no entiendo",
                 "complicado", "confuso", "enredo", "no comprendo", "temas",
                 "perdida", "confundida", "no le entiendo",
                 "no le pillo", "me pierdo", "me perdí",
                 "no comprendo nada", "todo confuso",
                 "materias difíciles", "no me entra", "no retengo"],
        "bajo": ["comprendo", "claro", "entiendo", "fácil", "sencillo",
                 "le entiendo", "todo claro", "comprensible",
                 "me queda claro", "logro entender"]
    },

    # ── SÍNTOMAS FÍSICOS (sf_4_1 a sf_4_6) ────────────────────
    "sf_4_1": {
        "tema": "problemas de sueño",
        "alto": ["dormir", "insomnio", "desvelado", "noche", "no duermo",
                 "despierto", "trasnochando", "trasnocho", "madrugada",
                 "desvelo", "mal dormir", "poco sueño",
                 "desvelada", "no puedo dormir", "no concilio el sueño",
                 "me despierto", "interrumpido", "pesadillas",
                 "duermo poco", "duermo mal", "noches en vela",
                 "no logro dormir", "acostado sin dormir",
                 "me acuesto tarde", "me desvelo"],
        "bajo": ["duermo bien", "descanso", "bien dormido", "sueño reparador",
                 "duermo tranquilo", "duermo tranquila", "ocho horas",
                 "buen sueño", "descanso bien", "sin problemas para dormir"]
    },
    "sf_4_2": {
        "tema": "fatiga y cansancio",
        "alto": ["cansado", "agotado", "fatigado", "sin energía", "mamado",
                 "exhausto", "rendido", "drenado", "muerto", "arrastrando",
                 "cansada", "agotada", "fatigada", "mamada",
                 "exhausta", "rendida", "drenada",
                 "no tengo fuerzas", "sin fuerzas", "sin ánimo",
                 "destrozado", "destrozada", "acabado", "acabada",
                 "hecho pedazos", "me arrastro"],
        "bajo": ["energía", "activo", "descansado", "vital", "bien",
                 "activa", "descansada", "con energía",
                 "lleno de energía", "renovado", "renovada", "fresco"]
    },
    "sf_4_3": {
        "tema": "dolor de cabeza",
        "alto": ["cabeza", "dolor", "migraña", "jaqueca", "punzada",
                 "cefalea", "duele la cabeza", "dolor de cabeza",
                 "me duele", "me punza", "presión en la cabeza",
                 "tengo dolor", "dolores frecuentes"],
        "bajo": ["sin dolor", "bien", "normal", "sin molestias",
                 "no me duele", "cero dolor"]
    },
    "sf_4_4": {
        "tema": "problemas digestivos",
        "alto": ["estómago", "nauseas", "vómito", "digestión", "gastritis",
                 "diarrea", "colitis", "asco", "acidez", "barriga",
                 "náuseas", "malestar estomacal", "dolor de estómago",
                 "se me revuelve", "me da asco", "no me cae bien",
                 "reflujo", "indigestión", "retortijones"],
        "bajo": ["bien", "normal", "sin problemas",
                 "como normal", "sin malestar", "digiero bien"]
    },
    "sf_4_5": {
        "tema": "morderse uñas o rascarse",
        "alto": ["uñas", "rascarse", "nervioso", "tic", "morderme",
                 "morder", "rasguñar", "pellizcar", "manos",
                 "me muerdo las uñas", "me rasco",
                 "no puedo quedarme quieto", "inquieto", "inquieta",
                 "me arranco", "me jalo el pelo", "me toco la cara"],
        "bajo": ["tranquilo", "relajado", "relajada",
                 "sin tics", "quieto", "calmado", "calmada"]
    },
    "sf_4_6": {
        "tema": "somnolencia",
        "alto": ["sueño", "somnolencia", "dormirme", "bostezar", "cabeceando",
                 "dormido en clase", "pesados los ojos",
                 "dormida en clase", "me da sueño", "me duermo en clase",
                 "me quedo dormido", "me quedo dormida",
                 "ojos pesados", "somnoliento", "somnolienta",
                 "cabeceo", "bostezando"],
        "bajo": ["despierto", "alerta", "activo", "atento",
                 "despierta", "atenta", "activa",
                 "concentrado", "concentrada", "despabilado"]
    },

    # ── SÍNTOMAS PSICOLÓGICOS (sp_4_7 a sp_4_13) ──────────────
    "sp_4_7": {
        "tema": "inquietud",
        "alto": ["inquieto", "intranquilo", "nervios", "ansiedad", "ansioso",
                 "impaciente", "desesperado", "acelerado", "alterado",
                 "inquieta", "intranquila", "ansiosa", "impaciente",
                 "desesperada", "acelerada", "alterada",
                 "no puedo estar quieto", "no puedo estar quieta",
                 "temblando", "tembloroso", "agitado", "agitada",
                 "nervioso", "nerviosa", "taquicardia",
                 "el corazón se me sale", "se me acelera"],
        "bajo": ["calma", "paz", "sereno", "tranquilo", "relajado",
                 "serena", "tranquila", "relajada",
                 "en paz", "calmado", "calmada", "sosegado"]
    },
    "sp_4_8": {
        "tema": "tristeza o depresión",
        "alto": ["triste", "deprimido", "llorar", "solo", "vacío", "lloré",
                 "decaído", "melancólico", "abatido", "desanimado", "ganas de llorar",
                 "deprimida", "decaída", "melancólica", "abatida", "desanimada",
                 "lloro", "me pongo a llorar", "me dan ganas de llorar",
                 "tristeza", "soledad", "sin ganas de nada",
                 "desanimado", "bajoneado", "bajoneada",
                 "achantado", "achantada", "sin motivación"],
        "bajo": ["contento", "feliz", "alegre", "animado", "bien",
                 "contenta", "animada", "alegría", "de buen ánimo",
                 "motivado", "motivada", "positivo", "positiva",
                 "optimista", "con ganas"]
    },
    "sp_4_9": {
        "tema": "ansiedad y angustia",
        "alto": ["ansiedad", "angustia", "miedo", "pánico", "terror",
                 "angustiado", "aterrado", "asustado", "preocupado", "temor",
                 "angustiada", "aterrada", "asustada", "preocupada",
                 "ataque de ansiedad", "ataque de pánico",
                 "me paralizo", "no puedo respirar", "me falta el aire",
                 "me ahogo", "siento presión", "presión en el pecho",
                 "nudo en la garganta", "nerviosismo"],
        "bajo": ["tranquilo", "seguro", "confiado", "calmado",
                 "tranquila", "segura", "confiada", "calmada",
                 "en paz", "sin preocupaciones", "relajado", "relajada"]
    },
    "sp_4_10": {
        "tema": "problemas de concentración",
        "alto": ["concentrar", "distraigo", "olvido", "atención", "pierdo",
                 "desconcentro", "distraído", "no retengo", "memoria", "olvidadizo",
                 "distraída", "olvidadiza", "desconcentrada",
                 "no me concentro", "pierdo el hilo",
                 "se me olvida todo", "no puedo enfocarme",
                 "me distraigo fácil", "se me va la mente",
                 "cabeza en otro lado", "divagando"],
        "bajo": ["enfocado", "concentrado", "atento", "retengo",
                 "enfocada", "concentrada", "atenta",
                 "buena memoria", "presto atención", "me enfoco"]
    },
    "sp_4_11": {
        "tema": "irritabilidad o agresividad",
        "alto": ["irritable", "enojado", "agresivo", "mal genio", "rabia",
                 "ira", "molesto", "furioso", "exploto", "peleo",
                 "irritada", "enojada", "agresiva", "molesta", "furiosa",
                 "me enojo fácil", "mal humor", "de mal humor",
                 "cualquier cosa me molesta", "exploto por todo",
                 "me altero", "de malas", "amargado", "amargada",
                 "intolerante", "pierdo la paciencia"],
        "bajo": ["paciente", "calmado", "tolerante", "tranquilo",
                 "calmada", "tolerante", "tranquila",
                 "de buen genio", "de buen humor", "pacífico"]
    },
    "sp_4_12": {
        "tema": "sentimiento de vacío",
        "alto": ["vacío", "sin sentido", "propósito", "perdido", "nada importa",
                 "sin rumbo", "desorientado", "inútil", "no sirvo",
                 "perdida", "desorientada", "me siento inútil",
                 "para qué", "no tiene sentido", "no vale la pena",
                 "me siento vacío", "me siento vacía",
                 "sin propósito", "no sé qué hago aquí",
                 "apatía", "apático", "apática",
                 "me da igual", "todo me vale"],
        "bajo": ["motivado", "claro", "con propósito", "tiene sentido", "con sentido", "enfocado",
                 "motivada", "clara", "enfocada",
                 "sé lo que quiero", "tengo metas", "con rumbo",
                 "con objetivos", "sé para dónde voy"]
    },
    "sp_4_13": {
        "tema": "ganas de huir o abandonar",
        "alto": ["huir", "escapar", "dejar", "abandonar", "salir", "retirar",
                 "no quiero", "renunciar", "tirar la toalla", "cancelar",
                 "quiero irme", "quiero dejar todo", "no quiero seguir",
                 "quiero salirme", "dejar la carrera", "dejar de estudiar",
                 "retirarme", "desertar", "ya no quiero",
                 "quiero desaparecer", "quiero que se acabe"],
        "bajo": ["compromiso", "continuar", "seguir", "terminar", "meta",
                 "voy a seguir", "quiero terminar", "no me rindo",
                 "tengo que seguir", "comprometido", "comprometida",
                 "perseverar", "echarle ganas", "seguir adelante"]
    },

    # ── SÍNTOMAS COMPORTAMENTALES (sc_4_14 a sc_4_18) ──────────
    "sc_4_14": {
        "tema": "conflictos o discusiones",
        "alto": ["pelea", "discutir", "conflicto", "gritar", "peleo",
                 "discusión", "problemas con", "mal con",
                 "peleas", "discusiones", "conflictos",
                 "me peleo", "discuto mucho", "peleas con",
                 "problemas con mi familia", "problemas con mis amigos",
                 "me grita", "le grito", "nos peleamos",
                 "roces", "tensión con", "mal ambiente"],
        "bajo": ["armonía", "bien con", "llevamos bien",
                 "buena relación", "en paz", "sin conflictos",
                 "nos entendemos", "buena convivencia",
                 "no tengo problemas con nadie"]
    },
    "sc_4_15": {
        "tema": "aislamiento social",
        "alto": ["solo", "aislado", "encerrado", "evitar", "no salgo",
                 "evito", "alejé", "apartado", "no quiero ver",
                 "sola", "aislada", "encerrada", "apartada",
                 "me aíslo", "me encierro", "no quiero salir",
                 "no quiero ver a nadie", "prefiero estar solo",
                 "prefiero estar sola", "no hablo con nadie",
                 "me alejé de todos", "no salgo de mi cuarto"],
        "bajo": ["social", "amigos", "compañía", "salgo", "gente",
                 "amigas", "compañeros", "compañeras",
                 "salgo con amigos", "me junto con",
                 "me gusta socializar", "buena vida social"]
    },
    "sc_4_16": {
        "tema": "desgano escolar",
        "alto": ["desgano", "aburrido", "flojera", "no quiero", "pereza",
                 "sin ganas", "desmotivado", "harto", "fastidio", "jarto",
                 "aburrida", "desmotivada", "harta",
                 "me aburre", "no me dan ganas", "qué pereza",
                 "no tengo ganas de nada", "me da igual la clase",
                 "no quiero ir a clase", "sin ánimo de estudiar",
                 "desgana", "desinterés", "desinteresado"],
        "bajo": ["motivado", "con ganas", "ganas de estudiar", "interés", "entusiasmado",
                 "motivada", "entusiasmada", "interesado", "interesada",
                 "con ánimo", "emocionado", "emocionada",
                 "me gusta", "me apasiona"]
    },
    "sc_4_17": {
        "tema": "cambios en alimentación",
        "alto": ["comer", "apetito", "no como", "atracón", "como mucho",
                 "como poco", "hambre", "sin comer", "no como bien",
                 "perdí el apetito", "sin apetito", "no tengo hambre",
                 "como por ansiedad", "como de más", "como de menos",
                 "no desayuno", "salto comidas", "no almuerzo",
                 "como cualquier cosa", "no me da hambre",
                 "cambios en lo que como", "como pura chatarra"],
        "bajo": ["normal", "saludable", "bien", "equilibrado",
                 "como bien", "como saludable", "como a mis horas",
                 "alimentación normal", "como regular",
                 "buena alimentación", "como tres veces al día"]
    },
    "sc_4_18": {
        "tema": "faltar a clases",
        "alto": ["faltar", "falté", "no fui", "ausentismo", "no asisto",
                 "no voy", "pierdo clase", "falta",
                 "faltas", "no he ido", "dejé de ir",
                 "llego tarde", "no entro a clase",
                 "me salgo de clase", "no asistí",
                 "no me presento", "a veces no voy",
                 "falto mucho", "llego tarde siempre"],
        "bajo": ["asisto", "cumplo", "voy", "puntual", "todas las clases",
                 "nunca falto", "siempre voy", "asisto a todo",
                 "no he faltado", "responsable", "cumplido"]
    },

    # ── ESTRATEGIAS DE AFRONTAMIENTO (af_5_1 a af_5_8) ─────────
    # NOTA: Aquí "alto" = USA la estrategia (positivo/protector)
    "af_5_1": {
        "tema": "defender derechos y opiniones",
        "alto": ["defiendo", "hablo", "comunico", "expreso", "opino",
                 "reclamo", "digo lo que pienso",
                 "me expreso", "hago valer", "me hago escuchar",
                 "doy mi opinión", "me quejo", "exijo",
                 "pido que me respeten", "hablo claro"],
        "bajo": ["callo", "aguanto", "no digo nada", "me quedo callado",
                 "me quedo callada", "me aguanto",
                 "me trago las cosas", "no digo lo que siento",
                 "no reclamo", "mejor no digo nada"]
    },
    "af_5_2": {
        "tema": "planificación y organización",
        "alto": ["organizo", "planifico", "agenda", "horario", "programo",
                 "calendario", "listas", "priorizo", "ordeno",
                 "me organizo", "hago un plan", "organización",
                 "planifico mi tiempo", "llevo agenda",
                 "prioridades", "cronograma", "me programo"],
        "bajo": ["desorden", "improviso", "sin plan", "desordenado",
                 "desordenada", "no planifico", "a última hora",
                 "al azar", "sin organización", "no me organizo"]
    },
    "af_5_3": {
        "tema": "religiosidad o espiritualidad",
        "alto": ["dios", "rezo", "fe", "iglesia", "oración", "medito",
                 "espiritual", "creo", "confío en dios",
                 "pido a dios", "le pido a dios", "meditación",
                 "reflexiono", "espiritualidad", "misa",
                 "bendición", "creyente", "mi fe"],
        "bajo": ["no creo", "nada", "no rezo",
                 "no soy creyente", "no practico", "no voy a iglesia"]
    },
    "af_5_4": {
        "tema": "búsqueda de información",
        "alto": ["investigo", "busco", "pregunto", "leo", "informo",
                 "averiguo", "consulto", "estudio", "repaso",
                 "busco información", "investigo por mi cuenta",
                 "pregunto al profesor", "busco en internet",
                 "tutoriales", "videos", "busco ayuda académica",
                 "me preparo", "profundizo"],
        "bajo": ["no busco", "no pregunto", "no investigo",
                 "no repaso", "no estudio", "no consulto",
                 "no leo", "me quedo con la duda"]
    },
    "af_5_5": {
        "tema": "desahogarse y hablar",
        "alto": ["hablo", "cuento", "desahogo", "comparto", "confío",
                 "amigos", "familia", "converso", "apoyo",
                 "le cuento a alguien", "hablo con mi mamá",
                 "hablo con mi papá", "hablo con mis amigos",
                 "me desahogo", "busco apoyo", "pido consejo",
                 "necesito hablar", "me siento apoyado",
                 "me siento apoyada", "red de apoyo"],
        "bajo": ["guardo", "callo", "no cuento", "nadie", "solo",
                 "sola", "me lo guardo", "no hablo con nadie",
                 "no le cuento a nadie", "me lo trago",
                 "nadie me entiende", "no tengo a quién contarle"]
    },
    "af_5_6": {
        "tema": "actividad física",
        "alto": ["ejercicio", "deporte", "camino", "gimnasio", "correr",
                 "fútbol", "entreno", "bicicleta", "nadar", "bailar",
                 "entrené", "voy al gimnasio", "hago deporte",
                 "hago ejercicio", "trotar", "troto",
                 "salgo a caminar", "yoga", "natación",
                 "basquetbol", "voleibol", "atletismo"],
        "bajo": ["sedentario", "nada", "no hago ejercicio", "no hago deporte",
                 "sedentaria", "no me muevo", "acostado",
                 "acostada", "no salgo", "encerrado", "encerrada"]
    },
    "af_5_7": {
        "tema": "apoyo profesional",
        "alto": ["psicólogo", "terapia", "bienestar", "ayuda profesional",
                 "orientador", "consejero", "salud mental",
                 "psicóloga", "terapeuta", "orientación",
                 "bienestar universitario", "busqué ayuda",
                 "fui al psicólogo", "voy a terapia",
                 "necesito ayuda profesional", "salud emocional"],
        "bajo": ["no busco ayuda", "solo", "no voy",
                 "sola", "no he buscado ayuda", "no voy al psicólogo",
                 "no creo en eso", "no necesito ayuda",
                 "me las arreglo solo", "me las arreglo sola"]
    },
    "af_5_8": {
        "tema": "entretenimiento y ocio",
        "alto": ["música", "serie", "juego", "pasatiempo", "película",
                 "videojuego", "hobby", "leer", "dibujar", "salir",
                 "escucho música", "veo series", "juego videojuegos",
                 "netflix", "redes sociales", "instagram", "tiktok",
                 "pinto", "canto", "cocino", "salgo con amigos",
                 "pasear", "viajar", "descansar", "relajarme"],
        "bajo": ["nada", "aburrido", "no hago nada", "encerrado",
                 "aburrida", "encerrada", "no tengo tiempo para nada",
                 "no hago nada divertido", "puro estudio",
                 "solo estudio", "no salgo para nada"]
    },
}

# Mapeo dimensión → columnas
DIMENSIONES = {
    "estresores":       ["e_3_1", "e_3_2", "e_3_3", "e_3_4", "e_3_5",
                         "e_3_6", "e_3_7", "e_3_8", "e_3_9"],
    "sintomas_fisicos": ["sf_4_1", "sf_4_2", "sf_4_3", "sf_4_4",
                         "sf_4_5", "sf_4_6"],
    "sintomas_psico":   ["sp_4_7", "sp_4_8", "sp_4_9", "sp_4_10",
                         "sp_4_11", "sp_4_12", "sp_4_13"],
    "comportamiento":   ["sc_4_14", "sc_4_15", "sc_4_16", "sc_4_17",
                         "sc_4_18"],
    "afrontamiento":    ["af_5_1", "af_5_2", "af_5_3", "af_5_4",
                         "af_5_5", "af_5_6", "af_5_7", "af_5_8"],
}

# Orden de los pasos conversacionales (1-indexed)
PASO_A_DIMENSION = {
    1: "estresores",
    2: "sintomas_fisicos",
    3: "sintomas_psico",
    4: "comportamiento",
    5: "afrontamiento",
}

# Lista ordenada completa de columnas (mismo orden que el modelo)
COLS_LIKERT_ORDEN = [
    "e_3_1", "e_3_2", "e_3_3", "e_3_4", "e_3_5", "e_3_6", "e_3_7", "e_3_8", "e_3_9",
    "sf_4_1", "sf_4_2", "sf_4_3", "sf_4_4", "sf_4_5", "sf_4_6",
    "sp_4_7", "sp_4_8", "sp_4_9", "sp_4_10", "sp_4_11", "sp_4_12", "sp_4_13",
    "sc_4_14", "sc_4_15", "sc_4_16", "sc_4_17", "sc_4_18",
    "af_5_1", "af_5_2", "af_5_3", "af_5_4", "af_5_5", "af_5_6", "af_5_7", "af_5_8",
]


# ══════════════════════════════════════════════════════════════
# FUNCIONES DE MATCHING MEJORADO
# ══════════════════════════════════════════════════════════════

def _buscar_keyword_con_contexto(texto_limpio, keyword):
    """
    Busca un keyword en el texto con contexto (word boundaries).
    Maneja keywords multi-palabra y de una sola palabra.

    Returns:
        list of (posición, negado): cada match encontrado con su posición
              y si está negado (True/False)
    """
    matches = []

    # Para keywords multi-palabra, buscar como subcadena exacta
    if " " in keyword:
        patron = re.escape(keyword)
    else:
        # Para palabras individuales, usar word boundaries
        patron = r'\b' + re.escape(keyword) + r'\b'

    for m in re.finditer(patron, texto_limpio):
        pos = m.start()
        # Verificar si hay negación antes
        texto_antes = texto_limpio[:pos].strip()
        palabras_antes = texto_antes.split()
        ultimas_n = palabras_antes[-5:] if len(palabras_antes) >= 5 else palabras_antes

        negado = False
        for palabra in ultimas_n:
            if palabra in NEGACIONES:
                negado = True
                break

        matches.append((pos, negado))

    return matches


def _estimar_columna(texto_limpio, col_key, intensidad_global=1.0):
    """
    Estima un valor Likert (-3 a 3) para una columna específica
    usando matching por palabras completas, detección de negaciones,
    y scoring continuo proporcional.

    Args:
        texto_limpio: texto ya normalizado
        col_key: clave de la columna (ej: "e_3_1")
        intensidad_global: multiplicador de intensidad del texto

    Returns:
        float: valor estimado en rango [-3.0, 3.0]
    """
    kw = DIMENSION_KEYWORDS[col_key]
    palabras_alto = kw["alto"]
    palabras_bajo = kw["bajo"]

    # Contar matches para "alto" (con detección de negación)
    score_alto = 0
    score_alto_negado = 0
    for keyword in palabras_alto:
        matches = _buscar_keyword_con_contexto(texto_limpio, keyword)
        for _pos, negado in matches:
            if negado:
                score_alto_negado += 1  # "no estoy estresado" → cuenta como bajo
            else:
                score_alto += 1

    # Contar matches para "bajo" (con detección de negación)
    score_bajo = 0
    score_bajo_negado = 0
    for keyword in palabras_bajo:
        matches = _buscar_keyword_con_contexto(texto_limpio, keyword)
        for _pos, negado in matches:
            if negado:
                score_bajo_negado += 1  # "no estoy bien" → cuenta como alto
            else:
                score_bajo += 1

    # Los negados se suman al lado opuesto
    total_alto = score_alto + score_bajo_negado
    total_bajo = score_bajo + score_alto_negado

    if total_alto == 0 and total_bajo == 0:
        return 0.0  # No se mencionó este tema

    # Scoring continuo proporcional: (alto - bajo) / (alto + bajo) * 3
    total = total_alto + total_bajo
    diferencia = total_alto - total_bajo
    valor_base = (diferencia / total) * 3.0

    # Aplicar bonus de intensidad global
    valor_final = valor_base * min(intensidad_global, 1.5)

    # Clampear a rango [-3, 3]
    valor_final = max(-3.0, min(3.0, valor_final))

    # Redondear a 1 decimal para mejor granularidad
    return round(valor_final, 1)


def estimar_likert_por_dimension(texto_respuesta, dimension, intensidad_global=1.0):
    """
    Estima valores Likert para todas las columnas de una dimensión SISCO.

    Args:
        texto_respuesta: Texto del estudiante para esa dimensión
        dimension: Nombre de la dimensión (ej: "estresores")
        intensidad_global: multiplicador de intensidad del texto

    Returns:
        dict con {columna: valor_likert} para esa dimensión
    """
    texto_limpio = normalizar_para_keywords(texto_respuesta)
    columnas = DIMENSIONES[dimension]
    return {
        col: _estimar_columna(texto_limpio, col, intensidad_global)
        for col in columnas
    }


def estimar_likert_completo(respuestas_por_paso):
    """
    Genera los 35 valores Likert a partir de las respuestas
    del estudiante en cada paso conversacional.

    MEJORA v3: Detecta repetición cross-dimensional.
    Si un síntoma se menciona en múltiples respuestas (ej: "no puedo dormir"
    en paso 0, 2 y 3), se amplifica el valor de las columnas relacionadas.

    Args:
        respuestas_por_paso: dict {paso_num: texto_respuesta}
            - paso 0: respuesta inicial (se usa para todas las dimensiones como contexto)
            - paso 1-5: respuestas específicas por dimensión

    Returns:
        lista de 35 valores float en el orden correcto (COLS_LIKERT_ORDEN)
    """
    valores = {}

    # Texto inicial (paso 0) aporta contexto a todas las dimensiones
    texto_inicial = respuestas_por_paso.get(0, "")

    # Calcular intensidad global desde todo el texto combinado
    texto_total = " ".join(str(v) for v in respuestas_por_paso.values())
    intensidad = detectar_intensidad_global(texto_total)

    # ── Paso 1: Estimación base por dimensión ──────────────────
    for paso, dimension in PASO_A_DIMENSION.items():
        texto_especifico = respuestas_por_paso.get(paso, "")
        texto_combinado = f"{texto_especifico} {texto_inicial}"

        estimaciones = estimar_likert_por_dimension(
            texto_combinado, dimension, intensidad
        )
        valores.update(estimaciones)

    # ── Paso 2: Detectar repetición cross-dimensional ──────────
    # Analizar TODAS las respuestas juntas para encontrar temas
    # que el estudiante repite a lo largo de la conversación
    texto_total_normalizado = normalizar_para_keywords(texto_total)

    # Temas clave que indican estrés alto cuando se repiten
    TEMAS_REPETICION = {
        "sueño":         {"keywords": ["dormir", "sueño", "insomnio", "desvelado", "trasnocho", "desvelo"],
                          "columnas": ["sf_4_1", "sf_4_6"]},
        "cansancio":     {"keywords": ["cansado", "cansada", "agotado", "agotada", "fatigado", "fatigada", "mamado", "mamada"],
                          "columnas": ["sf_4_2"]},
        "cabeza":        {"keywords": ["cabeza", "dolor", "migraña", "jaqueca"],
                          "columnas": ["sf_4_3"]},
        "concentracion": {"keywords": ["concentrar", "concentrarme", "distraigo", "concentración"],
                          "columnas": ["sp_4_10"]},
        "ansiedad":      {"keywords": ["ansiedad", "ansioso", "ansiosa", "angustia", "nervios", "nervioso", "nerviosa"],
                          "columnas": ["sp_4_7", "sp_4_9"]},
        "tristeza":      {"keywords": ["triste", "llorar", "deprimido", "deprimida", "desanimado", "desanimada"],
                          "columnas": ["sp_4_8"]},
        "aislamiento":   {"keywords": ["solo", "sola", "aislado", "aislada", "encerrado", "encerrada", "no hablar"],
                          "columnas": ["sc_4_15"]},
        "desgano":       {"keywords": ["ganas", "desgano", "pereza", "aburrido", "aburrida", "no quiero"],
                          "columnas": ["sc_4_16"]},
        "faltar":        {"keywords": ["faltar", "no entrar", "no asistir", "no ir a clase"],
                          "columnas": ["sc_4_18"]},
    }

    # Contar en cuántas respuestas DIFERENTES aparece cada tema
    for tema, config in TEMAS_REPETICION.items():
        menciones_por_paso = 0
        for paso_num, texto_resp in respuestas_por_paso.items():
            texto_norm = normalizar_para_keywords(str(texto_resp))
            for kw in config["keywords"]:
                if " " in kw:
                    if kw in texto_norm:
                        menciones_por_paso += 1
                        break
                else:
                    if re.search(r'\b' + re.escape(kw) + r'\b', texto_norm):
                        menciones_por_paso += 1
                        break

        # Si se menciona en 2+ respuestas diferentes → amplificar
        if menciones_por_paso >= 3:
            boost = 1.5  # Muy repetido → boost fuerte
        elif menciones_por_paso >= 2:
            boost = 1.25  # Repetido → boost moderado
        else:
            boost = 1.0  # Sin repetición

        if boost > 1.0:
            for col in config["columnas"]:
                if col in valores and valores[col] != 0:
                    nuevo_valor = valores[col] * boost
                    valores[col] = round(max(-3.0, min(3.0, nuevo_valor)), 1)

    # Construir lista en el orden correcto
    resultado = [float(valores.get(col, 0)) for col in COLS_LIKERT_ORDEN]
    return resultado

