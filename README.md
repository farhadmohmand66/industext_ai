# Industext AI

Intelligence for industrial maintenance logs. This project uses a BERT-based Named Entity Recognition (NER) model to extract key entities — Faults, Components, Actions, and Equipment — from unstructured text. It includes:

- A Streamlit web UI for single and batch analysis
- A rule-based/optional spaCy pipeline to bootstrap annotations
- A BERT training pipeline to produce a deployable NER model
- Dockerized deployment

## Table of Contents

- Features
- Quick Start (Local)
- Docker
- Using the Streamlit App
- Batch File Formats
- Training a NER Model
- Rule-based NER Extraction (Bootstrapping)
- Project Structure
- Configuration and Environment
- Tech Stack
- License

---

## Features

- BERT-based NER for FAULT, COMPONENT, ACTION, EQUIPMENT
- Real-time single-log analysis and batch processing in Streamlit
- Highlighted entities and simple event extraction
- Docker image for easy deployment

---

## Quick Start (Local)

Prerequisites:
- Python 3.10+
- pip

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Ensure a trained model is available at output/ner_model (see “Training a NER Model” below).

3) Run the app:
```bash
streamlit run app.py
```

4) Open the app in your browser:
- http://localhost:8501

Notes:
- The app automatically uses GPU if PyTorch detects CUDA.

---

## Docker

Build:
```bash
docker build -t industext_ai .
```

Run:
```bash
docker run -p 8501:8501 industext_ai
```

Then open http://localhost:8501

---

## Using the Streamlit App

- Single Log Analysis: Paste a maintenance log sentence/paragraph and click “Analyze”.
- Batch Log Upload:
  - TXT: one log per line.
  - JSON: see “Batch File Formats” below.
- The app highlights recognized entities and shows simple event tuples:
  ```json
  {
    "event_type": "Maintenance_Event",
    "fault": "abnormal vibration",
    "component": "bearing",
    "confidence": 0.874
  }
  ```

Language:
- UI supports English and 中文 via a sidebar selector.

Model location:
- The app looks for a model in output/ner_model relative to the repository (or BASE_DIR, see Configuration).
- Expected files include tokenizer and model artifacts (e.g., config.json, tokenizer.json, vocab files, pytorch_model.bin).

---

## Batch File Formats

TXT format:
- One log entry per line.

JSON formats accepted:
1) Array of objects with `text` or `log_text`:
```json
[
  {"id": 1, "text": "Abnormal vibration observed in the pump; bearing replaced."},
  {"id": 2, "log_text": "Oil leakage detected on seal. Cleaned thoroughly and tightened."}
]
```

2) Object with a `logs` array:
```json
{
  "logs": [
    {"id": "A-1001", "text": "Pump shows overheating; lubrication applied."},
    {"id": "A-1002", "log_text": "Pressure drop recorded; valve checked and replaced."}
  ]
}
```

---

## Training a NER Model

Use the BERT pipeline in train_bert.py to train and export a model.

Data expectations (see schema.md for details):
- data/logs.json: raw logs
- data/ner.json: weak/auto-labeled entities (can be produced via extract_ner.py)
- annotations/ner_gold.jsonl: optional gold-standard annotations for evaluation

Typical steps:
1) Prepare data as per schema.md.
2) (Optional) Generate weak labels:
   ```bash
   python extract_ner.py --input data/logs.json --output data/ner.json
   ```
3) Train:
   ```bash
   python train_bert.py
   ```
   - The script creates model artifacts and results. Review the path constants at the top of train_bert.py and adjust to your environment so that the final model is saved under output/ner_model.

After training:
- Ensure output/ner_model contains both tokenizer and model files. The Streamlit app loads from this directory.

---

## Rule-based NER Extraction (Bootstrapping)

Use extract_ner.py to quickly bootstrap NER labels from logs:
```bash
python extract_ner.py --input data/logs.json --output data/ner.json
```

Highlights:
- Uses spaCy if available; otherwise falls back to a robust rule-based approach (regex + gazetteer + optional fuzzy matching with RapidFuzz).
- Preserves key overlapping spans where applicable.
- You can extend seed vocabularies for components, faults, actions, and equipment within the script.

Recommended:
- Manually label a gold subset (50–80 records) with tools like Doccano or Label Studio to validate and improve model quality.

---

## Project Structure

```
.
├─ app.py                   # Streamlit UI and inference
├─ extract_ner.py           # Rule-based/spaCy extraction to bootstrap ner.json
├─ train_bert.py            # BERT NER training pipeline
├─ generate_logs.py         # Utility to generate or process logs (optional helper)
├─ data/                    # Datasets (see schema.md)
│  └─ logs.json             # Example input for extraction/training (user-provided)
├─ output/
│  └─ ner_model/            # Trained model artifacts loaded by app.py
├─ test_batch_logs.json     # Example batch JSON for the UI
├─ requirements.txt         # Python dependencies
├─ Dockerfile               # Dockerized deployment
├─ schema.md                # Dataset schema and notes
├─ .env                     # Environment variables (do not commit secrets)
├─ .dockerignore
└─ .gitattributes
```

---

## Configuration and Environment

- BASE_DIR: app.py reads BASE_DIR from environment; defaults to the repository directory. The model path is resolved as $BASE_DIR/output/ner_model.
  - Set explicitly if needed:
    ```bash
    export BASE_DIR=/path/to/industext_ai
    ```
- .env: Contains placeholders for external API keys (e.g., DEEPSEEK_API_KEY). Not required for running the Streamlit app and BERT model as provided. Do not commit real secrets.

Troubleshooting:
- Model not found or tokenizer error: Ensure output/ner_model contains a valid Hugging Face token classification checkpoint (config.json, tokenizer files, model weights).
- CPU/GPU: PyTorch will use CUDA automatically if available; otherwise falls back to CPU.

---

## Tech Stack

- Python, PyTorch, Hugging Face Transformers
- Streamlit
- Optional: spaCy, RapidFuzz (for rule-based bootstrapping)
- Docker

---

## License

MIT

---

Developer: Farhad Khan  
Project: Industrial Maintenance Log Intelligence System  
Technology: BERT-based NER · Event Extraction  
Competition: Shijiazhuang Innovation & Entrepreneurship Competition (International Track)  
GitHub: https://github.com/farhadmohmand66
