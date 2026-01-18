# Industext AI

Industrial maintenance log intelligence using BERT-based NER to extract faults, components, actions, and equipment from unstructured logs. Includes a Streamlit UI for single and batch analysis.

## Features
- BERT-based named entity recognition
- Single log and batch log analysis
- Streamlit web UI
- Dockerized deployment

## Local Run
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the app:
   - `streamlit run app.py`
3. Open: http://localhost:8501

## Docker
Build:
- `docker build -t industext_ai .`

Run:
- `docker run -p 8501:8501 industext_ai`

## Project Structure
- `app.py` Streamlit UI and inference
- `output/ner_model` Trained model artifacts
- `data/` Sample datasets

## License
MIT
