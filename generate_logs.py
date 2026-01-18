"""
generate_logs.py

Purpose:
Generate synthetic industrial maintenance logs EXCLUSIVELY using the DeepSeek API
and save them to:

  data/logs.json

Requirements:
- Environment variable: DEEPSEEK_API_KEY
- Optional: DEEPSEEK_API_URL (default provided)

No annotations, no fallback generation, no extra outputs.
"""

import os
import json
import random
import time
import dotenv
import dotenv
import requests
from datetime import datetime, timedelta
from dateutil import parser as dateparser
from dotenv import load_dotenv

# ============================================================
# Configuration
# ============================================================

load_dotenv()
API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY environment variable is required.")

API_URL = os.getenv(
    "DEEPSEEK_API_URL",
    "https://api.deepseek.com/v1/chat/completions"
)
print("API_KEY:", API_KEY)
MODEL_NAME = "deepseek-chat"
TEMPERATURE = 0.25
MAX_TOKENS = 900

TOTAL_LOGS = 400
BATCH_SIZE = 5
MAX_RETRIES = 3
SEED = 42

OUTPUT_DIR = "data"
OUTPUT_FILE = "logs.json"

random.seed(SEED)

# ============================================================
# Closed-world vocabulary
# ============================================================

VOCAB = {
    "equipment_type": "Hydraulic Pump",
    "equipment_ids": [f"Pump-A{100+i}" for i in range(20)],
    "faults": [
        "abnormal vibration",
        "overheating",
        "oil leakage",
        "pressure drop",
        "unusual noise",
        "bearing wear"
    ],
    "components": ["bearing", "shaft", "seal", "impeller", "coupling"],
    "actions": [
        "bearing replaced",
        "lubrication applied",
        "seal tightened",
        "component cleaned",
        "pump shut down"
    ],
    "parameters": ["temperature", "pressure", "vibration level"],
    "shifts": ["Morning", "Evening", "Night"],
    "technician_levels": ["Junior", "Senior"],
    "factories": ["Factory-01", "Factory-02"]
}

# ============================================================
# Utility Functions
# ============================================================

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def random_dates(start_date, end_date, n):
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)
    days = (end - start).days
    return [
        (start + timedelta(days=random.randint(0, days))).date().isoformat()
        for _ in range(n)
    ]


# ============================================================
# Prompt Construction
# ============================================================

def build_system_prompt():
    return (
        "You are an industrial maintenance log generator.\n"
        "Rules:\n"
        "- Output ONLY valid JSON (no explanations, no markdown).\n"
        "- Use ONLY the provided vocabulary.\n"
        "- Each log_text must be 3–6 sentences.\n"
        "- Follow this order in log_text: fault → parameter → component → action.\n"
        "- Do NOT invent new fields or terms.\n"
    )


def build_user_prompt(num_records, start_date, end_date):
    schema = {
        "log_id": "string",
        "date": "YYYY-MM-DD",
        "factory": "Factory-01 | Factory-02",
        "equipment_id": "string",
        "equipment_type": VOCAB["equipment_type"],
        "shift": "Morning | Evening | Night",
        "technician_level": "Junior | Senior",
        "log_text": "string"
    }

    vocab_text = (
        f"FAULTS:\n- " + "\n- ".join(VOCAB["faults"]) + "\n\n"
        f"COMPONENTS:\n- " + "\n- ".join(VOCAB["components"]) + "\n\n"
        f"ACTIONS:\n- " + "\n- ".join(VOCAB["actions"]) + "\n\n"
        f"PARAMETERS:\n- " + "\n- ".join(VOCAB["parameters"]) + "\n\n"
        f"EQUIPMENT IDS:\n- " + "\n- ".join(VOCAB["equipment_ids"])
    )

    return (
        f"Generate {num_records} maintenance logs as a JSON array.\n\n"
        f"Schema:\n{json.dumps(schema, indent=2)}\n\n"
        f"Date range: {start_date} to {end_date}\n\n"
        f"{vocab_text}\n\n"
        "Rules:\n"
        "- Each record must contain exactly one fault and one action.\n"
        "- Rotate shifts and technician levels naturally.\n"
        "- Use unique log_id values.\n"
        "Return ONLY the JSON array."
    )


# ============================================================
# DeepSeek API Call
# ============================================================
def call_deepseek(system_prompt, user_prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"DeepSeek API error {response.status_code}: {response.text}"
        )

    data = response.json()

    # ---- Robust content extraction ----
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected DeepSeek response format:\n{json.dumps(data, indent=2)}")

    if not content or not content.strip():
        raise RuntimeError("DeepSeek returned empty content")

    return content.strip()


# ============================================================
# Main Pipeline
# ============================================================
def parse_json_from_model(text: str):
    """
    Extract JSON array from DeepSeek output.
    Handles:
    - ```json ... ```
    - extra explanations
    - whitespace
    """
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = text.split("```", 2)[1]

    # Extract JSON array
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in model output:\n{text[:500]}")

    return json.loads(text[start:end + 1])

def generate_logs(
    total_logs=TOTAL_LOGS,
    batch_size=BATCH_SIZE,
    start_date="2024-02-01",
    end_date="2024-06-30"
):
    ensure_output_dir()

    logs = []
    next_id = 1

    system_prompt = build_system_prompt()

    while len(logs) < total_logs:
        current_batch = min(batch_size, total_logs - len(logs))
        user_prompt = build_user_prompt(current_batch, start_date, end_date)

        for attempt in range(MAX_RETRIES):
            try:
                print(f"[INFO] Requesting {current_batch} logs (attempt {attempt+1})")
                #--------------
                raw = call_deepseek(system_prompt, user_prompt)

                try:
                    parsed = parse_json_from_model(raw)
                except Exception as e:
                    print("\n[ERROR] Failed to parse DeepSeek response")
                    print("---------- RAW RESPONSE ----------")
                    print(raw[:1500])
                    print("---------- END RESPONSE ----------\n")
                    raise e
                #-----------------

                if not isinstance(parsed, list):
                    raise ValueError("Response is not a JSON array")

                for record in parsed:
                    record["log_id"] = f"{next_id:06d}"
                    next_id += 1
                    logs.append(record)

                break

            except Exception as e:
                print(f"[WARN] Batch failed: {e}")
                time.sleep(2)

        else:
            raise RuntimeError("DeepSeek API failed repeatedly. Aborting.")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Generated {len(logs)} logs → {output_path}")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    generate_logs()
