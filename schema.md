
# Dataset schema

Generated on: 2026-01-16T08:02:07.824380

Files:
- raw/logs.json : 391 records. Each record fields:
  - log_id, date (YYYY-MM-DD), factory, equipment_id, equipment_type, shift, technician_level, log_text

- annotations/ner.json : 391 records (character-span entities)
  - entities format: {'start': int, 'end': int, 'label': str}

- annotations/ner_gold.jsonl : 51 records (gold standard annotations)
  - label format: [[start, end, label], ...]


Vocabulary (closed world):
Faults: abnormal vibration, overheating, oil leakage, pressure drop, unusual noise, bearing wear, excessive bearing wear
Components: bearing, shaft, seal, impeller, coupling, pump
Actions: bearing replaced, lubrication applied, seal tightened, component cleaned, pump shut down, seal replaced, cleaned and reassembled
Equipment: system, fan, pump, motor, valve, turbine
Notes:
- NER was auto-generated using exact-matching span rules. Consider manually annotating a gold subset (50-80 records) using Doccano or Label Studio for highest quality.
