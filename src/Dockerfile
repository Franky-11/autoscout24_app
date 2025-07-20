# Basierend auf dem offiziellen Python-Image
FROM python:3.13

# Arbeitsverzeichnis im Container setzen
WORKDIR /app


COPY requirements.txt .

# Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# kopiert Inhalte aus src/ nach /app
# holt die Theme-Konfig aus dem Projekt-Root
COPY src/ /app/
COPY .streamlit/ /app/.streamlit/

# Den Port freigeben

EXPOSE 8501

# Starte die Streamlit App direkt mit dem Python-Interpreter
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]
