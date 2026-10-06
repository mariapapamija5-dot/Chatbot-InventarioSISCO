# -*- coding: utf-8 -*-
import re

# Diccionario de palabras y expresiones con polaridad para contexto académico
PALABRAS_POSITIVAS = [
    "bien", "tranquilo", "tranquila", "relajado", "relajada", "feliz", 
    "excelente", "genial", "super", "súper", "animado", "animada", "motivado", 
    "motivada", "calmado", "calmada", "alegre", "optimista", "bueno", "buena",
    "fresco", "fresca", "suave", "sin problema", "todo en orden", "normal"
]

PALABRAS_NEGATIVAS = [
    "mal", "estresado", "estresada", "ansioso", "ansiosa", "triste", "cansado", 
    "cansada", "agotado", "agotada", "abrumado", "abrumada", "presionado", 
    "presionada", "angustiado", "angustiada", "preocupado", "preocupada", 
    "nervioso", "nerviosa", "pésimo", "pesimo", "horrible", "terrible", 
    "deprimido", "deprimida", "malísimo", "malisimo", "llorar", "dolor", 
    "enfermo", "enferma", "frustrado", "frustrada", "desesperado", "desesperada",
    "pesado", "duro", "difícil", "dificil", "complejo", "complicado",
    "nada bien", "muy mal", "fatal", "grave"
]

NEGADORES = ["no", "ni", "nunca", "jamas", "jamás", "tampoco", "cero", "nada"]

def analizar_sentimiento_frase(frase):
    """
    Analiza el sentimiento de una sola frase o segmento corto.
    Retorna "positivo", "negativo" o "neutral".
    """
    frase = frase.lower()
    
    # Contar palabras positivas y negativas
    puntaje = 0
    
    palabras = re.findall(r'\b\w+\b', frase)
    
    # Revisar bigramas para frases hechas (ej. "nada bien")
    for neg in PALABRAS_NEGATIVAS:
        if neg in frase:
            puntaje -= 1
            
    for pos in PALABRAS_POSITIVAS:
        if pos in frase:
            puntaje += 1
            
    # Manejo básico de negaciones (si hay un negador, invierte el sentimiento de las siguientes palabras)
    hay_negacion = False
    for i, word in enumerate(palabras):
        if word in NEGADORES:
            hay_negacion = True
            # Limitar el alcance de la negación a las siguientes 3 palabras
            rango_evaluacion = palabras[i+1:i+4]
            
            for w in rango_evaluacion:
                if w in PALABRAS_POSITIVAS:
                    puntaje -= 2 # Era positivo, pero está negado ("no estoy bien")
                    hay_negacion = False
                    break
                elif w in PALABRAS_NEGATIVAS:
                    puntaje += 2 # Era negativo, pero está negado ("no estoy mal")
                    hay_negacion = False
                    break

    if puntaje > 0:
        return "positivo"
    elif puntaje < 0:
        return "negativo"
    else:
        return "neutral"

def analizar_sentimiento_general(texto):
    """
    Divide el texto en frases (por comas o puntos) y retorna el sentimiento global.
    """
    # Dividir por signos de puntuación básicos o conjunciones como 'pero'
    frases = re.split(r'[.,;!¡?¿]| pero | aunque ', texto)
    
    positivos = 0
    negativos = 0
    
    for f in frases:
        if len(f.strip()) > 0:
            sent = analizar_sentimiento_frase(f)
            if sent == "positivo":
                positivos += 1
            elif sent == "negativo":
                negativos += 1
                
    # Si hay ambigüedad o empate, el contexto negativo/estrés pesa más en SISCO
    if negativos > 0 and negativos >= positivos:
        return "negativo"
    elif positivos > negativos:
        return "positivo"
    else:
        return "neutral"
