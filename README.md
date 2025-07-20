# Autoscout24-Web-App

Diese Streamlit-Webanwendung bietet eine interaktive Analyse von Fahrzeugdaten 
aus dem Datensatz:   
**Germany Cars Dataset from kaggle** 
* Datensatz-Analyse (NaN, Duplikate, Ausreißer)
* Dashboard (Interaktive Visualisierungen) 
* Machine Learning zur Autopreisevorhersage (Feature Auswahl, ML-Pipeline, Modell-Performance)


---

##  Features

- Datenvisualisierung nach Preis, Marke und Modell
- Vorhersagemodul mit XGBoost, RandomForest & Linear Regression
- Residuenanalyse inkl. QQ-Plot, Fehler nach Preissegmenten
- Downloadbereich für Pipelines & Vorhersagen
- Docker-ready mit Dockerfile, requirements.txt und .dockerignore


---

##  Lokale Ausführung

```bash
# Projekt klonen
git clone https://github.com/Franky-11/autoscout24-app.git
cd autoscout24-app/src

# Virtuelle Umgebung erstellen (optional)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate      # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt

# Streamlit starten
streamlit run home.py


```
---

##  Docker-Ausführung

```bash
# Docker-Image bauen, Dockerfile liegt in src/
docker build -f src/Dockerfile -t autoscout-app .

# Container starten
docker run --name autoscout-container -p 8501:8501 autoscout-app
```
Die Anwendung ist dann erreichbar unter:  http://localhost:8501

---

##  Projektstruktur

```bash
autoscout24/             ← Projekt-Root 
├── .gitignore           ← Git-Ausschlussregeln
├── requirements.txt     ← Python-Abhängigkeiten
├── README.md            ← Dokumentation
└── src/                 ← Docker-Build-Kontext & App-Logik
    ├── home.py          ← Entry point Streamlit-App
    ├── autoscout24.csv  ← Datendatei
    ├── Dockerfile       ← Container-Definition
    ├── .dockerignore    ← Docker-Ausschlussregeln (nur für Build aus src/)
    └── ...              ← Weitere .py Dateien (dashboard.py,ml_modell.py etc...)
```

---
✍️ Entwickler

Frank Schulnies 