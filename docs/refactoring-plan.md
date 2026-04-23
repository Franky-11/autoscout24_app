# Refaktorierungsplan: AutoScout24 Preisvorhersage

## Ziel

Das Projekt soll von einer einzelnen, stark UI-getriebenen Streamlit-Codebasis zu einem klar strukturierten Data-Science-Produkt weiterentwickelt werden:

- saubere Trennung von UI, Datenlogik, Feature Engineering, Training und Evaluation
- reproduzierbare Modellierung statt ad-hoc Training im Frontend
- robustere fachliche Preisvorhersage mit nachvollziehbarer Modellbewertung
- einfachere Wartbarkeit, Tests und spätere Erweiterbarkeit

## Kurzbefund zum Ist-Zustand

### Repository und Struktur

- Die App-Logik liegt fast vollständig direkt unter `src/`.
- Daten, Bildassets, Docker-Kontext und Python-Code sind vermischt.
- Es gibt keine Paketstruktur, keine klaren Module pro Verantwortungsbereich und kein `docs/`- oder `tests/`-Verzeichnis.
- Das Projekt enthält derzeit eine CSV-Datei im Anwendungsordner (`src/autoscout24.csv`, ca. 2.5 MB) und ein großes Bildasset (`src/car_dealer.png`, ca. 4.7 MB).

### Code und Architektur

- `src/ml_modell.py` mischt UI, Session-State, Modellkonfiguration, Training, Evaluation, Persistenz und Inferenz in einer Datei.
- `src/functions.py` und `src/ml_functions.py` enthalten gemischte Zuständigkeiten: Laden, Bereinigen, Visualisierung, Modellierung und Vorhersagehilfen.
- Datenlogik ist teilweise doppelt vorhanden:
  - `read_csv` / `read_csv_with_nan_duplicates` in [src/functions.py](/home/frank/workspace/code/apps/autoscout24/src/functions.py)
  - ähnliche Preprocessing-Logik erneut in [src/ml_functions.py](/home/frank/workspace/code/apps/autoscout24/src/ml_functions.py)
- Es werden `import *`-Imports verwendet, was die Abhängigkeiten unklar macht.
- Es gibt keine automatisierten Tests, keine Linting-/Formatierungs-Checks und keine reproduzierbare Experimentlogik außerhalb der Streamlit-Session.

### Daten und fachlicher Status

Datensatzprofil aus `src/autoscout24.csv`:

- 46.405 Zeilen, 9 Spalten
- 2.140 Duplikate
- Fehlwerte vor allem in `gear` (182), `model` (143), `hp` (29)
- `price` mit extremen Ausreißern bis 1.199.900
- `mileage` bis 1.111.111
- hohe Kardinalität bei `model` (842 Werte), `make` hat 77 Werte
- `offerType` hat 5 Ausprägungen, wird im ML-Preprocessing aktuell entfernt

### Methodische Schwächen im ML-Teil

- Outlier-Filterung erfolgt vor dem Split und nutzt dabei auch die Zielvariable `price`.
  Das erzeugt Leakage in die Evaluierung.
- One-Hot-Encoding wird vor dem Modelltraining über `pd.get_dummies` vorbereitet, statt sauber im train-only Preprocessing zu liegen.
- Es gibt nur einen einfachen `train_test_split`, keine Cross-Validation und keinen stabilen Vergleich über mehrere Runs.
- Es existiert kein echter Baseline-Vergleich gegen einfache Benchmarks.
- Hyperparameter sind fest verdrahtet; Tuning oder strukturierte Experimente fehlen.
- Metriken sind zu schmal:
  - RMSE, MAE, R² sind vorhanden
  - aber keine MAPE/sMAPE, keine segmentierte Evaluierung nach Marke, Jahr, Fuel oder seltenen Kategorien
- Modellartefakte leben im Session-State und sind nicht dauerhaft versioniert.

## Zielbild für die Zielstruktur

Empfohlene Zielstruktur:

```text
autoscout24/
├── docs/
│   ├── refactoring-plan.md
│   ├── architecture.md
│   └── modeling-decisions.md
├── data/
│   ├── raw/
│   │   └── autoscout24.csv
│   ├── interim/
│   └── processed/
├── models/
│   ├── trained/
│   └── reports/
├── notebooks/
│   └── exploration/
├── src/
│   └── autoscout24/
│       ├── app/
│       │   ├── main.py
│       │   ├── pages/
│       │   │   ├── welcome.py
│       │   │   ├── dataset.py
│       │   │   ├── dashboard.py
│       │   │   └── modeling.py
│       │   └── state.py
│       ├── data/
│       │   ├── io.py
│       │   ├── schema.py
│       │   ├── cleaning.py
│       │   └── validation.py
│       ├── features/
│       │   ├── engineering.py
│       │   ├── preprocessing.py
│       │   └── encoders.py
│       ├── modeling/
│       │   ├── baselines.py
│       │   ├── train.py
│       │   ├── predict.py
│       │   ├── evaluate.py
│       │   ├── registry.py
│       │   └── tuning.py
│       ├── visualization/
│       │   ├── dashboard.py
│       │   └── diagnostics.py
│       └── config.py
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_training.py
│   └── test_prediction.py
├── requirements.txt
├── README.md
└── pyproject.toml
```

## Refaktorierungsfahrplan

## Phase 0: Stabilisierung und technische Hygiene

Ziel: Das Projekt zuerst zuverlässig und bearbeitbar machen.

Maßnahmen:

- Paketstruktur einführen (`src/autoscout24/...`) statt lose Python-Dateien.
- `home.py` in einen echten App-Entry-Point verschieben.
- `functions.py` in fachlich getrennte Module zerlegen.
- `import *` vollständig entfernen.
- `pyproject.toml` ergänzen für Tooling und lokale Entwicklung.
- `ruff` und `pytest` einführen.
- Basis-Checks definieren:
  - `python -m pytest`
  - `ruff check .`
  - `ruff format --check .`

Ergebnis:

- klarer Importgraph
- geringere Kopplung
- bessere Grundlage für jede weitere Änderung

## Phase 1: Datenebene sauber aufbauen

Ziel: Ein eindeutiger, reproduzierbarer Datenpfad.

Maßnahmen:

- Daten aus `src/` nach `data/raw/` verschieben.
- Ein zentrales Lade-Modul schaffen:
  - CSV laden
  - Schema validieren
  - Datentypen setzen
  - optionale Nullwert- und Duplikatbehandlung
- Data Contracts definieren:
  - erwartete Spalten
  - erlaubte Werte
  - Typen
  - Bereichsprüfungen
- Bereinigungsschritte explizit und konfigurierbar machen:
  - Duplikate entfernen
  - Missing-Value-Strategie
  - seltene Kategorien bündeln
  - optionale Ausreißerbehandlung

Wichtige fachliche Korrektur:

- Outlier-Handling darf nicht global vor dem Split auf Basis des gesamten Datensatzes laufen.
- Stattdessen:
  - Outlier-Regeln nur aus Trainingsdaten ableiten
  - oder robuste Modelle/Target-Transformation nutzen und ganz auf aggressives Filtern verzichten

## Phase 2: Feature Engineering professionalisieren

Ziel: Features nachvollziehbar, testbar und wiederverwendbar machen.

Aktuell vorhandene Features:

- `car_age`
- `mileage_per_year`
- `log_mileage`
- `log_hp`
- `make_model`
- `fuel_gear_combo`

Probleme heute:

- Feature-Bildung liegt verstreut im ML-Helfermodul.
- Inferenzlogik für Einzelvorhersagen bildet die Features erneut manuell nach.
- Trainings- und Inferenzpfad können auseinanderlaufen.

Maßnahmen:

- Feature Engineering in eine dedizierte Transformationsschicht verschieben.
- Dieselbe Transformationslogik für:
  - Training
  - Evaluation
  - Einzelvorhersage
  - Batch-Prediction
- Feature-Namen und Feature-Sets versionieren.

Fachliche Erweiterungen:

- `log_price` als optionales Ziel für stabilere Modellierung testen.
- robustere Nutzung von `offerType` prüfen:
  - seltene Klassen bündeln
  - z. B. `used`, `nearly_new`, `commercial/other`
- zusätzliche sinnvolle Features testen:
  - `hp_per_year`
  - `price_band` nur für Analyse, nicht als Feature
  - Interaktionen `make x fuel`, `make x gear`
  - quantisierte `mileage`-Bänder
  - Premium-/Volumenmarkencluster
- High-Cardinality-Features strategisch behandeln:
  - Target Encoding nur mit sauberer CV
  - alternativ Frequency Encoding
  - alternativ CatBoost als Default-Modell für Kategorie-lastige Daten

## Phase 3: Training und Evaluation neu aufsetzen

Ziel: valide Modellbewertung statt UI-gesteuertem Einmaltraining.

Maßnahmen:

- Trainingslogik aus Streamlit herauslösen.
- Eine Trainingsschnittstelle definieren:
  - Eingabe: Konfiguration
  - Ausgabe: Modellartefakt, Metriken, Diagnostik, Feature-Metadaten
- Einheitlichen Preprocessing-Stack mit `ColumnTransformer` für alle Modelle verwenden, soweit möglich.
- Baselines definieren:
  - Medianpreis global
  - Medianpreis je `make`
  - lineares Modell mit Basisfeatures
- Cross-Validation einführen:
  - mindestens `KFold`
  - besser `RepeatedKFold`
- Eine separate Holdout-Evaluierung für den finalen Modellvergleich reservieren.

Fachliche Verbesserungen:

- Zielmetrik priorisieren:
  - `MAE` für geschäftlich gut interpretierbaren Fehler in Euro
  - `RMSE` als Zusatzmetrik für starke Ausreißerstrafe
  - `MAPE` oder `sMAPE` für relative Fehler
- Segmentmetriken verpflichtend ausweisen:
  - nach Preisband
  - nach Marke
  - nach Fuel
  - nach Fahrzeugalter
- Modellkalibrierung prüfen:
  - systematische Über-/Unterbewertung in Segmenten
- Residualanalyse ausbauen:
  - Fehler nach `make`, `model`, `offerType`, `year`
  - Fehler gegen `mileage`, `hp`, `car_age`

Empfohlene Modellstrategie:

1. Baseline: Linear Regression oder Elastic Net auf sauberem Pipeline-Stack
2. Starker Standardkandidat: CatBoost
3. Alternativen: LightGBM und XGBoost
4. RandomForest nur noch als Referenz, nicht als primärer Favorit

Begründung:

- Das Problem ist tabellarisch und enthält relevante kategoriale Variablen mit hoher Kardinalität.
- CatBoost ist dafür in vielen Fällen robuster als manuelles One-Hot-Encoding.
- Ein lineares Basismodell bleibt wichtig für Interpretierbarkeit und als Sanity Check.

## Phase 4: Experimentmanagement und Reproduzierbarkeit

Ziel: Ergebnisse nachvollziehbar speichern und vergleichen.

Maßnahmen:

- Session-State-abhängige Speicherung ablösen.
- Experimente als strukturierte Artefakte speichern:
  - Konfiguration
  - Datensatzversion
  - Feature-Set
  - Metriken
  - Modellpfad
  - Trainingszeit
- Optionales Experimenttracking einführen:
  - leichtgewichtig lokal per JSON/CSV
  - später MLflow oder Weights & Biases
- Modellregister unter `models/` einführen.

Ergebnis:

- Ergebnisse sind nach App-Neustart nicht verloren.
- Beste Modelle können nachvollziehbar wiederverwendet werden.

## Phase 5: Streamlit-App neu schneiden

Ziel: Die App soll Orchestrierung und Visualisierung machen, nicht die Fachlogik tragen.

Maßnahmen:

- Seiten in `src/autoscout24/app/pages/` verschieben.
- State-Handling zentralisieren.
- Jede Seite nur noch:
  - User Input erfassen
  - Service-Funktionen aufrufen
  - Ergebnisse rendern
- Trainingsläufe optional asynchron oder als definierte Aktion kapseln.
- Vorhersage-UI auf ein stabiles Inferenz-Interface umstellen.

Empfohlene Service-Schnittstellen:

- `load_dataset()`
- `build_feature_frame(config)`
- `train_model(config)`
- `evaluate_model(run_id)`
- `predict_single(vehicle_input, model_id)`
- `list_models()`

## Phase 6: Tests und Qualitätssicherung

Ziel: Regressionen verhindern und Refaktorierung absichern.

Pflichttests:

- Datenlade- und Schema-Tests
- Tests für Missing-Value- und Duplikatbehandlung
- Tests für Feature Engineering
- Tests für Trainingspipeline
- Tests für Einzelvorhersage mit bekannten Inputs
- Snapshot-Tests für zentrale Konfigurationsobjekte

Besonders wichtig:

- Sicherstellen, dass Trainings- und Inferenzfeatures identisch erzeugt werden.
- Sicherstellen, dass unbekannte Kategorien sauber behandelt werden.
- Sicherstellen, dass Modellartefakte reloadbar bleiben.

## Phase 7: Deployment, Packaging und Betrieb

Ziel: Das Projekt professioneller deploybar machen.

Maßnahmen:

- `pyproject.toml` statt reinem `requirements.txt`
- Python-Version festziehen, z. B. `3.11` oder `3.12`
- Dockerfile auf Paketstruktur und sauberen Build anpassen
- `Makefile` oder einfache Task-Kommandos ergänzen:
  - `make install`
  - `make test`
  - `make train`
  - `make app`
- Optional CLI ergänzen:
  - `python -m autoscout24.modeling.train`
  - `python -m autoscout24.app.main`

Hinweis:

- `python:3.13` im Dockerfile ist für manche ML-Abhängigkeiten unnötig riskant.
- Für dieses Projekt ist eine konservativere Version sinnvoller.

## Fachliche Verbesserungen an der Preisvorhersage

## 1. Datenverständnis verbessern

Empfohlene zusätzliche Analysen:

- Preisverteilung pro `make` und `model`
- Ausreißerprofile je Segment statt global
- fehlende Werte segmentiert untersuchen
- sehr seltene Modelle und Marken identifizieren
- Einfluss von `offerType` auf Preisniveau explizit prüfen

## 2. Robusteres Zielbild für die Modellierung

Empfehlung:

- Primärmodell auf `log(price)` trainieren und Rücktransformation evaluieren
- Parallel ein direktes `price`-Modell als Vergleich behalten

Vorteile:

- stabilere Fehler bei teuren Fahrzeugen
- weniger Dominanz extremer Ausreißer
- oft bessere relative Fehlerverteilung

## 3. Bessere Evaluierung entlang echter Business-Fragen

Die zentrale Frage ist nicht nur "Wie gut ist das Modell insgesamt?", sondern:

- Wie gut schätzt es Volumenmarken vs. Premium?
- Wie gut funktioniert es bei alten Fahrzeugen vs. jungen Fahrzeugen?
- Wie robust ist es bei seltenen Modellen?
- Wie groß ist der Fehler in Euro für typische Privatkundenfälle?

Dafür sollte das Reporting standardmäßig enthalten:

- Gesamtmetriken
- Fehler je Preisband
- Fehler je Marke
- Fehler je `offerType`
- Fehler je Jahrgang
- Top-Underpredictions / Top-Overpredictions

## 4. Erweiterungen mit hohem fachlichem Nutzen

Sinnvolle nächste Produktfunktionen:

- Unsicherheitsintervall zur Vorhersage
  - z. B. Quantile Regression oder Residualband
- Vergleichbare Fahrzeuge anzeigen
  - nearest-neighbor ähnliche Listings
- Feature-Impact pro Vorhersage
  - SHAP oder modellabhängige Beiträge
- Szenarioanalyse
  - "Wie verändert sich der Preis bei 20.000 km mehr?"
- Modellvergleich mit gespeicherten Runs außerhalb der Session

## Priorisierte Roadmap

## Sprint 1: Fundament

- Paketstruktur einführen
- `docs/`, `tests/`, `data/`, `models/` anlegen
- Datenlade- und Preprocessing-Module extrahieren
- `functions.py` auflösen
- erste Unit-Tests für Daten und Features

## Sprint 2: ML-Kern sanieren

- saubere Trainingspipeline bauen
- Leakage im Outlier-Handling entfernen
- Baselines und Cross-Validation einführen
- CatBoost als Referenzmodell sauber evaluieren
- Experimentartefakte persistent speichern

## Sprint 3: App entkoppeln

- Streamlit-Seiten auf Service-Layer umstellen
- Session-State vereinfachen
- Modellliste und gespeicherte Runs aus Artefakten laden
- Vorhersage-Workflow stabilisieren

## Sprint 4: Fachliche Ausbaustufe

- `offerType` und seltene Kategorien strategisch einbauen
- log-Target vergleichen
- Unsicherheitsabschätzung ergänzen
- segmentiertes Performance-Reporting erweitern

## Konkrete erste Aufgabenliste

1. Paketstruktur anlegen und `src/` neu schneiden.
2. Datenfile aus `src/` nach `data/raw/` verschieben.
3. Ein zentrales Data-Loading-Modul mit Schema-Validation einführen.
4. Feature Engineering in eine gemeinsame Transformationsschicht auslagern.
5. Training und Evaluation aus `ml_modell.py` entfernen.
6. Cross-Validation und Baselines ergänzen.
7. Persistente Artefakte für Modelle und Runs einführen.
8. Streamlit nur noch als UI-Orchestrierung nutzen.
9. Tests und Linting etablieren.
10. Danach erst modellfachliche Optimierung und neue Produktfunktionen angehen.

## Erfolgskriterien

Der Refaktor ist erfolgreich, wenn:

- die App weiterhin läuft, aber deutlich weniger Logik in den Seiten trägt
- Training, Evaluation und Inferenz ohne Streamlit ausführbar sind
- Modelle reproduzierbar trainiert und gespeichert werden können
- Leakage reduziert und die Evaluierung belastbarer ist
- neue Features oder Modelle ohne Umbau der UI ergänzt werden können
- das Projekt für weitere ML-Iterationen deutlich schneller bearbeitbar ist
