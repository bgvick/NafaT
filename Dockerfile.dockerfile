FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier les fichiers nécessaires
COPY requirements.txt .
COPY NafaT.py .
COPY model_2.h5 .

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Exposer le port utilisé par Streamlit
EXPOSE 8080

# Définir la variable d'environnement pour le port
ENV PORT 8080

# Commande pour démarrer l'application
CMD streamlit run --server.port $PORT --server.address 0.0.0.0 NafaT.py