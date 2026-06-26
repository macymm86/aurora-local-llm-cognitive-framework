# Aurora Complete System (v2.3)

### Enhanced Memory & Learning Standalone Ableger

## 📖 Einleitung & Kontext
Bei diesem monolithischen **Aurora Complete System (v2.3)** handelt es sich um einen erweiterten, voll funktionsfähigen Ableger der Aurora-Architektur. Das primäre, eigentliche Aurora-Framework ist ein monumentales Meta-Kognitionsframework, das sich über **93 Module**, **6 separate Gedächtnisspeicher** und eine spezialisierte **Neural Forge** erstreckt.

Dieser Ableger wurde ausgekoppelt, um der Community ein mächtiges, in sich geschlossenes Werkzeug zur Verfügung zu stellen. Er demonstriert, wie ein lokales KI-System über reine Chat-Historien hinauswächst, indem es eigenständig Wissen strukturiert, Muster erkennt und ein persistentes, relationales Gedächtnis aufbaut.

---

## 🧠 Kern-Features dieser Version
* **7-Dimensionale Verarbeitung (D0–D6):** Voll integrierte kognitive Pipeline zur Echtzeit-Analyse und emotional-strukturellen Steuerung von Prompts.
* **Persistenter SQL-Speicher (SQLAlchemy):** Interaktionsverläufe, Reflexionsergebnisse und Systemzustände werden dauerhaft in einer relationalen SQLite-Datenbank strukturiert.
* **Semantische Suche & Embeddings:** Nutzung von lokalen Vektor-Repräsentationen zur schnellen, inhaltsbasierten Erinnerungsabfrage.
* **Wissensgraphen-Konstruktion (Knowledge Graph):** Das System extrahiert autonom Entitäten und Beziehungen aus den Interaktionen und verknüpft sie logisch.
* **Autonome Meta-Reflexion:** Mechanismen zur kontinuierlichen Selbstentwicklung und Systembeobachtung.
* **Volle Ollama-Integration:** Direkte Kommunikation mit lokal laufenden Sprachmodellen im Hintergrund.

---

## ⚙️ Technische Architektur & Konfiguration

### 1. Autonome Pfadverwaltung
Das System verfügt über eine robuste, plattformunabhängige Pfadkonfiguration mittels `pathlib`. Es ist kein manuelles Hardcodieren von `C:\`-Pfaden notwendig. Beim ersten Start ermittelt die Anwendung ihr eigenes Ausführungsverzeichnis und erstellt autonom einen lokalen Unterordner namens `aurora_data` für alle Datenbanken, Embeddings und Logs. Das sorgt für maximale Portabilität (Windows, Linux und macOS).

### 2. Anpassung der Graphen-Erstellung (Knowledge Graph)
Die visuelle und strukturelle Generierung des Wissensgraphen ist im Code so aufgebaut, dass sie individuell konfiguriert werden kann. Entwickler können die mathematischen Gewichtungen und die Regeln zur Extraktion von Knoten (Entities) und Kanten (Relations) direkt in den entsprechenden Graphen-Klassen im Code anpassen, um die topologische Darstellung des KI-Gedächtnisses zu skalieren.

---

## 🚀 Installation & Schnellstart

### 1. Repository klonen
Klone das Projekt auf deinen lokalen Rechner:

```bash
git clone [https://github.com/macymm86/aurora-complete-system.git](https://github.com/macymm86/aurora-complete-system.git)
cd aurora-complete-system
2. Virtuelle Umgebung einrichten (Empfohlen)
Erstelle eine isolierte Python-Umgebung, um Paketkonflikte zu vermeiden:

Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/MacOS
python3 -m venv .venv
source .venv/bin/activate
3. Abhängigkeiten installieren
Installiere alle benötigten Core-Bibliotheken (PyQt6, SQLAlchemy, Machine Learning Utilities):

Bash
pip install -r requirements.txt
4. Anwendung starten
Stellen Sie sicher, dass Ihr lokaler Ollama-Server im Hintergrund erreichbar ist (Standard-Port 11434), und starten Sie das System:

Bash
python aurora-complete-extended.py
🛡️ Stabilität & Bugfixes
Das System enthält einen automatischen Runtime-Patch für neuere Versionen des huggingface_hub, welcher das Entfernen der veralteten cached_download-Methode in älteren sentence-transformers-Bibliotheken abfängt. Das System ist auf deterministische Stabilität ausgelegt und sichert den Betrieb der Benutzeroberfläche und der lokalen Inferenz auch bei variierenden Modellumgebungen.
