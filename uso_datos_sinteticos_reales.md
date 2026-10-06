# Documento Explicativo: Datos Sintéticos vs. Datos Reales en el Chatbot SISCO

Este documento detalla la estrategia de manejo de datos utilizada para el entrenamiento y funcionamiento del Chatbot de Estrés Académico (basado en el inventario SISCO), explicando la necesidad de utilizar datos sintéticos, el rol de los datos reales y cómo ambos convergen en la aplicación final.

---

## 1. El Desafío de los Datos en Salud Mental

Entrenar un modelo de Inteligencia Artificial (como **BETO**, el modelo base utilizado en este proyecto) requiere miles de ejemplos clasificados correctamente para que la red neuronal aprenda a identificar patrones lingüísticos. 

En el contexto psicológico y del estrés académico, recolectar **datos reales** presenta dos grandes obstáculos:
1. **Privacidad y Ética:** Obtener historiales reales de estudiantes requiere consentimientos informados complejos.
2. **Volumen y Balanceo:** Es difícil conseguir suficientes ejemplos que cubran equitativamente todos los niveles de estrés (Leve, Moderado, Profundo). Generalmente, los datos reales tienden a concentrarse en los casos moderados, lo que "sesga" a la Inteligencia Artificial.

Para solucionar esto, el proyecto utiliza una estrategia híbrida que combina la generación de **datos sintéticos** con la validación de **datos reales**.

---

## 2. Datos Sintéticos: El Motor de Entrenamiento

### ¿Qué son?
Son textos y respuestas generadas artificialmente mediante algoritmos o Modelos de Lenguaje Grandes (LLMs, como GPT-4) a los que se les dio instrucciones estrictas para comportarse como estudiantes universitarios. Estos perfiles artificiales fueron diseñados basándose rigurosamente en las 35 dimensiones del inventario SISCO.

### ¿Para qué se usaron en el proyecto?
- **Entrenamiento Masivo (Bootstrapping):** Permitieron crear los datasets (`dataset_final_balanceado.csv` y `dataset_recalibrado.csv`) con miles de ejemplos en cuestión de horas.
- **Balanceo Perfecto:** Se aseguró matemáticamente que el modelo BETO viera la misma cantidad de estudiantes con estrés leve, moderado y profundo, evitando que el modelo se incline por adivinar siempre una sola categoría.
- **Casos Extremos (Edge Cases):** Se simularon escenarios atípicos (estudiantes que niegan tener estrés pero muestran síntomas físicos, o estudiantes usando lenguaje muy informal) para que el modelo sea robusto.

---

## 3. Datos Reales: La Ejecución en Vivo

### ¿Qué son?
Son los mensajes (inputs) que los estudiantes de carne y hueso ingresan en la interfaz web del chat (`index.html`) durante su uso cotidiano. 

### ¿Para qué sirven?
- **Inferencia (Uso del Chat):** Es el momento de la verdad. El modelo pre-entrenado con datos sintéticos se expone a la variabilidad del mundo real: errores ortográficos, modismos colombianos/locales, frases incompletas y emociones genuinas.
- **Validación del Modelo Híbrido:** Aquí es donde entra en juego la **arquitectura híbrida**. Como los datos sintéticos a veces pueden ser demasiado "perfectos" o estructurados, el sistema no confía 100% en la red neuronal (BETO). 
  - Se implementó un algoritmo basado en reglas (`estimador_likert.py`) que escanea los **datos reales** en busca de palabras clave específicas de los síntomas y estresores (por ejemplo, "dolor de cabeza", "insomnio", "exámenes").
  - El resultado final es una combinación ponderada (Ensemble) donde la IA aporta el 70% de la decisión (entendimiento semántico) y el diccionario de reglas exactas aporta el 30%.

---

## 4. Flujo de Datos en el Chatbot

Cuando un estudiante usa la aplicación, la interacción entre datos sintéticos (conocimiento previo) y datos reales (input actual) ocurre de la siguiente manera:

1. **Recolección (Datos Reales):** El estudiante chatea respondiendo las 5 preguntas (Estresores, Síntomas Físicos, Psicológicos, Comportamentales, Estrategias de Afrontamiento).
2. **Preprocesamiento:** El texto real se limpia y normaliza (`diccionario_datos.py`).
3. **Análisis de Reglas (Datos Reales):** `estimador_likert.py` analiza el texto crudo del estudiante y mapea sus palabras directamente a las 35 variables del test SISCO.
4. **Análisis Neuronal (Memoria Sintética):** El texto se pasa al modelo BETO (cargado con `pesos.pt`), el cual utiliza su **entrenamiento previo con datos sintéticos** para inferir patrones profundos de estrés.
5. **Decisión Final:** El `app.py` fusiona ambas predicciones y devuelve el diagnóstico final (Leve, Moderado o Profundo) sin guardar permanentemente los datos reales por motivos de privacidad.

---

## 5. Conclusión

El uso de **datos sintéticos** fue vital para darle al chatbot una base de conocimiento sólida, amplia y estadísticamente balanceada sin comprometer la privacidad de ningún estudiante real durante la fase de desarrollo. 

Por otro lado, la ingesta de **datos reales** durante el uso de la aplicación valida el sistema, apoyándose en un enfoque híbrido que garantiza que la IA no alucine y mantenga sus predicciones ancladas a las reglas clínicas reales del inventario SISCO.
