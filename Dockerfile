# Imagen base estable y ligera compatible con Dlib y OpenCV
FROM python:3.10.13-slim

# Instalar dependencias del sistema necesarias para compilar Dlib y OpenCV
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgtk2.0-dev \
    libboost-all-dev \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear carpeta de la app
WORKDIR /app

# Copiar todos los archivos del proyecto
COPY . .

# Actualizar pip y luego instalar dependencias del proyecto
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exponer puerto Flask
EXPOSE 5000

# Comando de inicio del servidor
CMD ["python", "app.py"]
