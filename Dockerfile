FROM python:3.9-slim

# Instalar dependencias del sistema (útil para PyTorch/Scikit-learn si hiciera falta)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo en el contenedor
WORKDIR /app

# Copiar primero requirements para aprovechar el caché de Docker
COPY requirements.txt .

# Instalar librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto 7860 (requerido por Hugging Face Spaces)
EXPOSE 7860

# Comando para ejecutar la app
CMD ["python", "app.py"]
