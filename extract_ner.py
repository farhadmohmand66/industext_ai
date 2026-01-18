
"""
extract_ner.py

Usage:
    python extract_ner.py --input data/logs.json --output data/ner.json

Requirements (recommended):
    pip install spacy rapidfuzz
    python -m spacy download en_core_web_trf   

Notes:
- The script uses spaCy if available. If spaCy is missing, the script will fall back to
    a robust rule-based extractor (regex + gazetteer + fuzzy matching).
- The script preserves overlapping spans (some entities intentionally overlap,
    e.g., "excessive bearing wear" -> FAULT and "bearing" -> COMPONENT).
"""

import json
import argparse
import re
from collections import defaultdict

# !pip install rapidfuzz
# !python -m spacy download en_core_web_trf
#

import json
import os

logs_file_path = os.path.join('data', 'logs.json')

with open(logs_file_path, "r", encoding="utf8") as f:
    logs_data = json.load(f)

print(f"Loaded {len(logs_data)} records from {logs_file_path}")

# Optional imports
try:
    import spacy
    from spacy.pipeline import EntityRuler
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    from rapidfuzz import process as rf_process
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# ---- Seed lexicons you can extend ----
SEED_COMPONENTS = [
    "bearing", "impeller", "seal", "shaft", "pump housing", "valve", "gasket", "motor", "coupling"
]
SEED_FAULTS = [
    "vibration", "abnormal vibration", "overheating", "pressure drop", "leakage", "oil leakage",
    "fouled", "damaged", "worn", "bearing wear", "failing seal"
]
SEED_ACTIONS = [
    "replaced", "bearing replaced", "cleaned", "cleaned thoroughly", "tightened", "lubrication applied",
    "shut down", "shut down for safety", "resolved", "applied lubrication", "repaired"
]
# Generic equipment tokens to improve recall (you can extend)
SEED_EQUIPMENT = [
    "pump", "hydraulic pump", "motor", "compressor", "generator", "pump housing"
]

# ---- Utilities ----
def smart_find_spans_ci(text, phrase):
    """
    Find case-insensitive spans. Use word boundaries for pure-word phrases, otherwise
    match the raw escaped phrase (for IDs like 'Pump-A103').
    """
    # If phrase contains non-word characters (like '-' or spaces), avoid strict \b boundaries for better matching
    if re.search(r"[^\w\s]", phrase):  # non-word but not whitespace? include special chars
        pattern = re.compile(re.escape(phrase), flags=re.IGNORECASE)
    else:
        # add simple word boundaries for clean token matching
        pattern = re.compile(r"\b" + re.escape(phrase) + r"\b", flags=re.IGNORECASE)
    return [(m.start(), m.end()) for m in pattern.finditer(text)]

def add_spans_for_terms(text, terms, label, collector, longest_first=True):
    """
    Add matched spans to collector list for each term found in text.
    """
    terms_sorted = sorted(set(terms), key=lambda x: -len(x)) if longest_first else sorted(set(terms))
    for term in terms_sorted:
        for (s,e) in smart_find_spans_ci(text, term):
            collector.append({'start': s, 'end': e, 'label': label})

def unique_spans(spans):
    """
    Deduplicate identical (start,end,label) entries while preserving order.
    """
    seen = set()
    out = []
    for s in spans:
        key = (s['start'], s['end'], s['label'])
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out

# ---- Action phrase heuristic (shrunk window, more precise) ----
ACTION_VERBS = ["replaced", "cleaned", "tightened", "applied", "shut down", "resolved", "repaired", "replaced to"]

def find_action_phrases(text, components_list=None, max_right_chars=40):
    """
    Heuristic finder for actions:
    - looks for known verb forms
    - includes a preceding component token if immediately adjacent
    - keeps the span compact to reduce accidental overlap with other entities
    """
    spans = []
    lowered = text.lower()
    for verb in ACTION_VERBS:
        for m in re.finditer(re.escape(verb), text, flags=re.IGNORECASE):
            vstart, vend = m.start(), m.end()

            # Try to include one preceding token if it's a known component or equipment (and is adjacent/punct-free)
            left = vstart
            prev_window = text[max(0, vstart-30):vstart]
            prev_word_match = re.search(r"(\b[\w'-]{2,}\b)\s*$", prev_window)
            if prev_word_match:
                candidate_word = prev_word_match.group(1)
                # include preceding token if it's a known component/equipment
                if components_list and any(re.fullmatch(re.escape(c), candidate_word, flags=re.IGNORECASE) for c in components_list):
                    left = max(0, vstart - (len(prev_window) - prev_word_match.start(1)))

            # Right boundary: stop at punctuation or limit to max_right_chars
            post = text[vend:]
            punc = re.search(r"[.;,]", post)
            right = vend + (punc.start() if punc else min(len(post), max_right_chars))

            # Ensure span is not empty and reasonably sized
            if right - left > 1 and right - left <= 200:
                spans.append({'start': left, 'end': right, 'label': 'ACTION'})
    return spans

# ---- Overlap resolution ----
LABEL_PRIORITY = {
    "EQUIPMENT": 40,
    "COMPONENT": 30,
    "FAULT": 20,
    "ACTION": 10
}

def score_span(span):
    """Score = priority * 1000 + length (favor label priority, then longer spans)."""
    prio = LABEL_PRIORITY.get(span.get('label', ''), 0)
    length = span['end'] - span['start']
    return prio * 1000 + length

def overlaps(a, b):
    return not (a['end'] <= b['start'] or a['start'] >= b['end'])

def resolve_overlaps(candidates):
    """
    Label-aware overlap resolution:
    - ACTION is allowed to overlap with COMPONENT and EQUIPMENT
    - FAULT overlaps are resolved normally
    - Prefer higher priority + longer spans
    """
    candidates = unique_spans(candidates)

    for c in candidates:
        c['_score'] = score_span(c)

    candidates_sorted = sorted(
        candidates, key=lambda x: (-x['_score'], x['start'])
    )

    selected = []

    for c in candidates_sorted:
        conflict = False
        for s in selected:
            if overlaps(c, s):
                # Allow ACTION to overlap with COMPONENT / EQUIPMENT
                if c['label'] == "ACTION" and s['label'] in {"COMPONENT", "EQUIPMENT"}:
                    continue
                if s['label'] == "ACTION" and c['label'] in {"COMPONENT", "EQUIPMENT"}:
                    continue
                conflict = True
                break
        if not conflict:
            selected.append(c)

    return sorted(
        [{k:v for k,v in s.items() if k != "_score"} for s in selected],
        key=lambda x: x['start']
    )

# ---- Main pipeline ----
def pipeline_extract(logs, use_spacy=True, fuzzy_threshold=80):
    """
    logs: list of dicts with fields at least: log_id, log_text (or text)
    Returns list of {'log_id','text','entities':[{'start','end','label'}, ...]}
    """
    results = []

    # Build a combined global equipment lexicon (from records + seed)
    # We'll also create per-record equipment list to capture IDs/types present in that record.
    global_equipment = set(SEED_EQUIPMENT)

    for r in logs:
        if 'equipment_id' in r and r['equipment_id']:
            global_equipment.add(r['equipment_id'])
        if 'equipment_type' in r and r['equipment_type']:
            global_equipment.add(r['equipment_type'])
    global_equipment = list(global_equipment)

    # Setup spaCy (if requested and available)
    nlp = None
    if use_spacy and SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_trf")
        except Exception:
            try:
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                nlp = None

        if nlp is not None:
            # build an EntityRuler with equipment + components + faults + actions patterns
            # attach before ner so rulings are effective
            try:
                # remove existing ruler if re-running in notebook
                if 'entity_ruler' in nlp.pipe_names:
                    # fragile: only safe if name known; otherwise create new with unique name
                    pass
                ruler = nlp.add_pipe("entity_ruler", before="ner")
                patterns = []
                for t in global_equipment:
                    patterns.append({"label": "EQUIPMENT", "pattern": t})
                for t in SEED_COMPONENTS:
                    patterns.append({"label": "COMPONENT", "pattern": t})
                for t in SEED_FAULTS:
                    patterns.append({"label": "FAULT", "pattern": t})
                for t in SEED_ACTIONS:
                    patterns.append({"label": "ACTION", "pattern": t})
                ruler.add_patterns(patterns)
            except Exception:
                # if cannot add ruler (multiple runs), ignore
                pass

    # iterate logs
    for rec in logs:
        text = rec.get("log_text") or rec.get("text") or ""
        log_id = rec.get("log_id", "")
        candidates = []

        # per-record equipment lexicon (IDs/types present)
        equipment_terms = set(global_equipment)
        if 'equipment_id' in rec and rec['equipment_id']:
            equipment_terms.add(rec['equipment_id'])
        if 'equipment_type' in rec and rec['equipment_type']:
            equipment_terms.add(rec['equipment_type'])
        equipment_terms = list(equipment_terms)

        # 1) spaCy NER + mapping to our labels (if available)
        if nlp is not None:
            doc = nlp(text)
            # noun chunk-based equipment detection: if noun chunk contains equipment word, map it
            # collect noun chunks that contain equipment tokens
            for nc in doc.noun_chunks:
                nc_text = nc.text
                for eq in equipment_terms:
                    if re.search(r'\b' + re.escape(eq) + r'\b', nc_text, flags=re.IGNORECASE) or re.search(re.escape(eq), nc_text, flags=re.IGNORECASE):
                        candidates.append({'start': nc.start_char, 'end': nc.end_char, 'label': 'EQUIPMENT'})
                        break
            # add spaCy entities mapped to our labels (and map PRODUCT/ORG/FAC -> EQUIPMENT)
            for ent in doc.ents:
                lbl = ent.label_.upper()
                if lbl in {"EQUIPMENT", "COMPONENT", "FAULT", "ACTION"}:
                    candidates.append({'start': ent.start_char, 'end': ent.end_char, 'label': lbl})
                else:
                    if lbl in {"PRODUCT", "ORG", "WORK_OF_ART", "FAC", "NORP"}:
                        candidates.append({'start': ent.start_char, 'end': ent.end_char, 'label': 'EQUIPMENT'})

        # 2) Gazetteer exact matches (equipment/components/faults)
        add_spans_for_terms(text, equipment_terms, 'EQUIPMENT', candidates)
        add_spans_for_terms(text, SEED_COMPONENTS, 'COMPONENT', candidates)
        add_spans_for_terms(text, SEED_FAULTS, 'FAULT', candidates)

        # 3) Action heuristic (use components as guide to include preceding token)
        action_spans = find_action_phrases(text, components_list=SEED_COMPONENTS)
        for sp in action_spans:
            candidates.append(sp)

        # 4) RapidFuzz fuzzy matching (optional)
        if RAPIDFUZZ_AVAILABLE:
            # tokens (words) from text
            tokens = re.findall(r"\b[\w'-]{3,}\b", text.lower())
            for token in set(tokens):
                # fuzzy for equipment
                best_eq = rf_process.extractOne(token, equipment_terms, score_cutoff=fuzzy_threshold)
                if best_eq:
                    for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE):
                        candidates.append({'start': m.start(), 'end': m.end(), 'label': 'EQUIPMENT'})
                # fuzzy to components and faults
                best_comp = rf_process.extractOne(token, SEED_COMPONENTS, score_cutoff=fuzzy_threshold)
                if best_comp:
                    for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE):
                        candidates.append({'start': m.start(), 'end': m.end(), 'label': 'COMPONENT'})
                best_fault = rf_process.extractOne(token, SEED_FAULTS, score_cutoff=fuzzy_threshold)
                if best_fault:
                    for m in re.finditer(re.escape(token), text, flags=re.IGNORECASE):
                        candidates.append({'start': m.start(), 'end': m.end(), 'label': 'FAULT'})

        # 5) final dedupe & resolve overlaps (this enforces NO overlaps)
        # normalize label text and dedupe exact duplicates
        for c in candidates:
            c['label'] = c['label'].upper()
        candidates = unique_spans(candidates)

        # Resolve overlaps: choose best-scoring non-overlapping set
        final_entities = resolve_overlaps(candidates)

        results.append({
            "log_id": log_id,
            "text": text,
            "entities": [{"start": e['start'], "end": e['end'], "label": e['label']} for e in final_entities]
        })

    return results

# ---- CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=os.path.join('data', 'logs.json'))
    parser.add_argument("--output", default=os.path.join('data', 'ner.json'))
    parser.add_argument("--no-spacy", action="store_true", help="Disable spaCy usage even if installed")

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf8") as f:
        logs = json.load(f)

    use_spacy = not args.no_spacy
    results = pipeline_extract(logs, use_spacy=use_spacy)

    with open(args.output, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(results)} records to {args.output}")

if __name__ == "__main__":
    main()


