# -*- coding: utf-8 -*-
"""BERT NER Training Pipeline for Industrial Text"""

import warnings
warnings.filterwarnings('ignore')

import os
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from transformers import (
    BertTokenizerFast, BertForTokenClassification,
    TrainingArguments, Trainer,
    DataCollatorForTokenClassification, set_seed
)
from seqeval.metrics import precision_score, recall_score, f1_score

# --------- DEVICE & CONFIG ----------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# Paths - updated for local machine
BASE_DIR = Path("d:/coding/idustex")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "ner_model"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create output directories
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_OUTPUT_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model settings
MAX_LENGTH = 128
MODEL_NAME = "bert-base-uncased"
TARGET_ENTITY_LABELS = {"FAULT", "COMPONENT", "ACTION", "EQUIPMENT"}

# --------- UTIL: FILE HELPERS ----------
def load_json(path):
    """Load JSON file"""
    with open(path, 'r', encoding='utf8') as f:
        return json.load(f)

def load_jsonl(path):
    """Load JSONL file"""
    out = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def save_json(path, obj):
    """Save JSON file"""
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def save_jsonl(path, list_of_objs):
    """Save JSONL file"""
    with open(path, 'w', encoding='utf8') as f:
        for o in list_of_objs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

# --------- DATA LOADING ----------
def load_data():
    """Load all required data files"""
    files = {
        "ner": DATA_DIR / 'ner.json',
        "logs": DATA_DIR / 'logs.json',
        "ner_gold": DATA_DIR / 'ner_gold.jsonl',
        "schema": BASE_DIR / 'schema.md'
    }
    
    # Check for missing files
    for name, path in files.items():
        if not path.exists():
            print(f"⚠️ Warning: expected file missing: {path}")
    
    ner_data = load_json(files["ner"]) if files["ner"].exists() else []
    logs_data = load_json(files["logs"]) if files["logs"].exists() else []
    ner_gold_data = load_jsonl(files["ner_gold"]) if files["ner_gold"].exists() else []
    
    schema_content = ""
    if files["schema"].exists():
        with open(files["schema"], 'r', encoding='utf8') as f:
            schema_content = f.read()
    
    print(f"Data loaded - NER: {len(ner_data)}, Logs: {len(logs_data)}, Gold: {len(ner_gold_data)}")
    print(f"Schema:\n{schema_content}\n")
    
    return ner_data, logs_data, ner_gold_data

# --------- TOKENIZATION & BIO CONVERSION ----------
def tokenize_with_offsets(text):
    """Return list of (token, start, end) using whitespace"""
    tokens = []
    for match in re.finditer(r'\S+', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

def get_text(record):
    """Extract text from record"""
    if "log_text" in record:
        return record["log_text"]
    if "text" in record:
        return record["text"]
    return ""

def record_to_bio(record, target_labels=TARGET_ENTITY_LABELS):
    """
    Convert record to BIO format.
    Works with both ner.json (entities dicts) and gold jsonl ([start,end,label])
    """
    text = get_text(record)
    tokens = tokenize_with_offsets(text)
    
    bio_labels = ["O"] * len(tokens)
    spans = []
    
    # Parse entities
    if "entities" in record and isinstance(record["entities"], list):
        for ent in record["entities"]:
            if not isinstance(ent, dict):
                continue
            s, e, lab = ent.get("start"), ent.get("end"), ent.get("label")
            if s is None or e is None or lab not in target_labels:
                continue
            spans.append((int(s), int(e), lab))
    
    elif "label" in record and isinstance(record["label"], list):
        for ent in record["label"]:
            if len(ent) < 3:
                continue
            s, e, lab = ent[0], ent[1], ent[2]
            if lab not in target_labels:
                continue
            spans.append((int(s), int(e), lab))
    
    spans.sort(key=lambda x: (x[0], x[1]))
    
    # Assign BIO labels
    for ent_start, ent_end, ent_label in spans:
        inside = False
        for i, (_, tok_start, tok_end) in enumerate(tokens):
            overlap = tok_start < ent_end and tok_end > ent_start
            if not overlap or bio_labels[i] != "O":
                continue
            
            if not inside:
                bio_labels[i] = f"B-{ent_label}"
                inside = True
            else:
                bio_labels[i] = f"I-{ent_label}"
    
    return [(tok, lab) for (tok, _, _), lab in zip(tokens, bio_labels)]

def convert_dataset_to_bio(records, converter=record_to_bio):
    """Convert records to BIO format"""
    out = []
    for rec in records:
        out.append({
            "log_id": rec.get("log_id"),
            "bio": converter(rec)
        })
    return out

# --------- TORCH DATASET ----------
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        item = self.encodings[idx].copy()
        return {k: v for k, v in item.items()}

# --------- TOKENIZATION & ALIGNMENT ----------
def setup_labels():
    """Setup label mappings"""
    label_list = ["O"]
    for lab in sorted(TARGET_ENTITY_LABELS):
        label_list.append(f"B-{lab}")
        label_list.append(f"I-{lab}")
    
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}
    
    print(f"NER labels: {label_list}\n")
    return label_list, label2id, id2label

def tokenize_and_align_labels(bio_record, tokenizer, label2id, max_length=MAX_LENGTH):
    """Tokenize and align labels with subword tokens"""
    tokens = [t for t, _ in bio_record["bio"]]
    labels = [l for _, l in bio_record["bio"]]
    
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False
    )
    
    word_ids = encoding.word_ids()
    aligned_labels = []
    prev_word_id = None
    
    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word_id:
            lab = labels[word_id]
            aligned_labels.append(label2id.get(lab, 0))
        else:
            lab = labels[word_id]
            if lab.startswith("B-"):
                lab = lab.replace("B-", "I-")
            aligned_labels.append(label2id.get(lab, 0))
        prev_word_id = word_id
    
    encoding["labels"] = aligned_labels
    return encoding

# --------- METRICS ----------
def compute_metrics(eval_pred, id2label):
    """Compute NER metrics"""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    true_labels = []
    true_preds = []
    
    for pred_row, label_row in zip(preds, labels):
        sent_true = []
        sent_pred = []
        
        for p_i, l_i in zip(pred_row, label_row):
            if l_i == -100:
                continue
            sent_true.append(id2label[l_i])
            sent_pred.append(id2label[p_i])
        
        if len(sent_true) > 0:
            true_labels.append(sent_true)
            true_preds.append(sent_pred)
    
    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

# --------- ANALYSIS ----------
def analyze_co_occurrence(train_data):
    """Analyze entity co-occurrence patterns"""
    co_occurrence = defaultdict(int)
    
    for rec in train_data:
        ents = rec.get("entities", [])
        text = rec.get("text", "")
        
        parsed = []
        for e in ents:
            if "start" in e and "end" in e:
                parsed.append({
                    "text": text[e["start"]:e["end"]],
                    "label": e["label"]
                })
        
        faults = [e["text"] for e in parsed if e["label"] == "FAULT"]
        components = [e["text"] for e in parsed if e["label"] == "COMPONENT"]
        actions = [e["text"] for e in parsed if e["label"] == "ACTION"]
        
        for f in faults:
            for c in components:
                for a in actions:
                    co_occurrence[(f, c, a)] += 1
    
    print("Top FAULT–COMPONENT–ACTION patterns:")
    for k, v in sorted(co_occurrence.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {k} → {v}")
    print()

def plot_confusion_matrix(trainer, test_dataset, id2label, label_list):
    """Plot confusion matrix"""
    pred_output = trainer.predict(test_dataset)
    preds = np.argmax(pred_output.predictions, axis=-1)
    labels = pred_output.label_ids
    
    true_flat, pred_flat = [], []
    
    for p_row, l_row in zip(preds, labels):
        for p, l in zip(p_row, l_row):
            if l != -100 and id2label[l] != "O":
                true_flat.append(id2label[l])
                pred_flat.append(id2label[p])
    
    entity_labels = [l for l in label_list if l != "O"]
    
    print("\n=== Classification Report ===")
    print(classification_report(true_flat, pred_flat, labels=entity_labels))
    
    cm = confusion_matrix(true_flat, pred_flat, labels=entity_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=entity_labels,
        yticklabels=entity_labels,
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("NER Confusion Matrix (Entity-Level)")
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=100, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {RESULTS_DIR / 'confusion_matrix.png'}")
    plt.close()

# --------- MAIN PIPELINE ----------
def main():
    """Main training pipeline"""
    print("=" * 60)
    print("BERT NER TRAINING PIPELINE")
    print("=" * 60 + "\n")
    
    # Load data
    ner_data, logs_data, ner_gold_data = load_data()
    
    # Check for mismatches
    ner_ids = {d.get("log_id") for d in ner_data if isinstance(d, dict) and d.get("log_id")}
    logs_ids = {d.get("log_id") for d in logs_data if isinstance(d, dict) and d.get("log_id")}
    gold_ids = {d.get("log_id") for d in ner_gold_data if d.get("log_id")}
    
    missing_logs = ner_ids - logs_ids
    missing_ner = logs_ids - ner_ids
    
    if missing_logs:
        print(f"⚠️ {len(missing_logs)} NER records have no matching log")
    if missing_ner:
        print(f"⚠️ {len(missing_ner)} logs have no NER annotations\n")
    
    # Create train/val/test split
    valid_gold = [g for g in ner_gold_data if g.get("log_id") is not None]
    if len(valid_gold) != len(ner_gold_data):
        print(f"⚠️ {len(ner_gold_data)-len(valid_gold)} gold records missing log_id\n")
    
    val, test = train_test_split(valid_gold, test_size=0.5, random_state=42)
    train_data = [r for r in ner_data if r.get("log_id") not in gold_ids]
    
    train_ids = {r.get("log_id") for r in train_data if r.get("log_id")}
    if not train_ids.isdisjoint(gold_ids):
        print("❌ Overlap detected between training and gold data!")
    else:
        print("✅ No gold data in training set")
    
    print(f"Data split - Train: {len(train_data)}, Val: {len(val)}, Test: {len(test)}\n")
    
    # Convert to BIO
    train_bio = convert_dataset_to_bio(train_data)
    val_bio = convert_dataset_to_bio(val)
    test_bio = convert_dataset_to_bio(test)
    
    if train_bio:
        print(f"Sample BIO (train): {train_bio[0]['bio'][:5]}\n")
    
    # Setup labels and tokenizer
    label_list, label2id, id2label = setup_labels()
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Tokenize and align
    train_encodings = [tokenize_and_align_labels(x, tokenizer, label2id) for x in train_bio]
    val_encodings = [tokenize_and_align_labels(x, tokenizer, label2id) for x in val_bio]
    test_encodings = [tokenize_and_align_labels(x, tokenizer, label2id) for x in test_bio]
    
    print(f"Encodings created - Train: {len(train_encodings)}, Val: {len(val_encodings)}, Test: {len(test_encodings)}\n")
    
    # Create datasets
    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)
    test_dataset = NERDataset(test_encodings)
    
    # Load model
    model = BertForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    print(f"✅ Model loaded: {MODEL_NAME}\n")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, id2label),
    )
    
    # Train
    print("=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)
    trainer.train()
    
    # Evaluate
    print("\n=== Validation Results ===")
    val_metrics = trainer.evaluate()
    print(val_metrics)
    
    print("\n=== Test Results ===")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    print(test_metrics)
    
    print(f"\nBest F1: {trainer.state.best_metric}")
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    
    # Save model
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    print(f"\n✅ Model saved to {MODEL_OUTPUT_DIR}\n")
    
    # Analysis
    analyze_co_occurrence(train_data)
    plot_confusion_matrix(trainer, test_dataset, id2label, label_list)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()