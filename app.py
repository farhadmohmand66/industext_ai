import streamlit as st
import torch
import json
import re
import os
from transformers import BertTokenizerFast, BertForTokenClassification

# ================== CONFIG ==================
BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "output", "ner_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_COLORS = {
    "FAULT": "#ff4d4f",
    "COMPONENT": "#1890ff",
    "ACTION": "#52c41a",
    "EQUIPMENT": "#faad14"
}

# ================== LANGUAGE ==================
LANG = st.sidebar.selectbox("🌐 Language / 语言", ["English", "中文"])

TEXT = {
    "English": {
        "title": "🔧 Industrial Maintenance Log Intelligence",
        "desc": "Automatically extract **Faults, Components, Actions, and Equipment** from unstructured maintenance logs.",
        "single": "📝 Single Log Analysis",
        "batch": "📂 Batch Log Upload",
        "input": "Enter maintenance log text:",
        "btn": "🚀 Analyze",
        "highlight": "🧠 Extracted Entities",
        "events": "⚙️ Extracted Maintenance Events",
        "upload": "Upload TXT (one log per line) or JSON file",
        "footer": """
        **Developer:** Farhad Khan  
        🎓 PhD Researcher (AI & NLP)  

        **Project:** Industrial Maintenance Log Intelligence System  
        **Technology:** BERT-based NER · Event Extraction  

        🏆 Prepared for **Shijiazhuang Innovation & Entrepreneurship Competition (International Track)**  
        🔗 GitHub: https://github.com/farhadmohmand66
        """
    },
    "中文": {
        "title": "🔧 工业维修日志智能分析系统",
        "desc": "自动从**非结构化维修日志**中提取故障、部件、维修动作和设备信息。",
        "single": "📝 单条日志分析",
        "batch": "📂 批量日志上传",
        "input": "请输入维修日志文本：",
        "btn": "🚀 开始分析",
        "highlight": "🧠 实体识别结果",
        "events": "⚙️ 结构化维修事件",
        "upload": "上传 TXT（每行一条）或 JSON 文件",
        "footer": """
        **开发者：Farhad Khan**  
        🎓 人工智能与自然语言处理方向博士研究生  

        **项目名称：工业维修日志智能分析系统**  
        **核心技术：BERT · 命名实体识别 · 事件抽取**

        🏆 石家庄创新创业大赛（海外赛道）参赛项目  
        🔗 GitHub：https://github.com/farhadmohmand66
        """
    }
}

T = TEXT[LANG]

# ================== MODEL ==================
@st.cache_resource
def load_model():
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
    model = BertForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ================== NER ==================
def decode_bio(tokens, labels, scores, offsets):
    entities, current = [], None
    for tok, lab, sc, (s, e) in zip(tokens, labels, scores, offsets):
        if lab.startswith("B-"):
            if current: entities.append(current)
            current = {"text": tok, "label": lab[2:], "start": s, "end": e, "confidence": sc}
        elif lab.startswith("I-") and current:
            current["text"] += " " + tok
            current["end"] = e
            current["confidence"] = min(current["confidence"], sc)
        else:
            if current:
                entities.append(current)
                current = None
    if current: entities.append(current)
    return entities


def predict_entities(text):
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=256)
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    probs = torch.softmax(out.logits[0], dim=-1)
    preds = torch.argmax(probs, dim=-1)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    labels = [model.config.id2label[p.item()] for p in preds]
    scores = [probs[i, p].item() for i, p in enumerate(preds)]
    clean = [(t.replace("##", ""), l, s, o) for t, l, s, o in zip(tokens, labels, scores, offsets) if t not in ["[CLS]", "[SEP]"]]
    return decode_bio(*zip(*clean))


def extract_events(entities):
    faults = [e for e in entities if e["label"] == "FAULT"]
    comps = [e for e in entities if e["label"] == "COMPONENT"]
    return [
        {
            "event_type": "Maintenance_Event",
            "fault": f["text"],
            "component": c["text"],
            "confidence": round((f["confidence"] + c["confidence"]) / 2, 3)
        }
        for f in faults for c in comps
    ]

# ================== UI STYLE ==================
st.set_page_config(layout="wide")

st.markdown("""
<style>
.main { max-width: 900px; margin: auto; text-align: center; }
.card {
    background-color: #f9f9f9;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 25px;
    text-align: center;
}
.desc { color: #555; font-size: 1.05rem; }
.footer {
    background: #f1f5f9;
    color: #334155;
    padding: 14px 18px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

def md_to_html(md: str) -> str:
    html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", md.strip())
    return "<br>".join([line.strip() for line in html.splitlines() if line.strip()])

# ================== UI ==================
sample_text = (
    "During inspection, abnormal vibration was detected in the pump. "
    "The bearing showed signs of wear and lubrication was applied."
)
st.markdown(f"<div class='main'><h1>{T['title']}</h1></div>", unsafe_allow_html=True)
st.markdown(f"<div class='main desc'>{md_to_html(T['desc'])}</div>", unsafe_allow_html=True)

# -------- Single --------
st.markdown(f"<div class='main card'><h3>{T['single']}</h3>", unsafe_allow_html=True)
text = st.text_area(T["input"], value=sample_text, height=140)
if st.button(T["btn"]) and text.strip():
    entities = predict_entities(text)
    events = extract_events(entities)

    highlighted, last = "", 0
    for e in sorted(entities, key=lambda x: x["start"]):
        highlighted += text[last:e["start"]]
        highlighted += f"<span style='background:{LABEL_COLORS[e['label']]};padding:3px;border-radius:4px'>{text[e['start']:e['end']]}</span>"
        last = e["end"]
    highlighted += text[last:]

    st.markdown(f"<b>{T['highlight']}</b>", unsafe_allow_html=True)
    st.markdown(highlighted, unsafe_allow_html=True)
    st.markdown(f"<b>{T['events']}</b>", unsafe_allow_html=True)
    st.json(events)
st.markdown("</div>", unsafe_allow_html=True)

# -------- Batch --------
st.markdown(f"<div class='main card'><h3>{T['batch']}</h3>", unsafe_allow_html=True)
file = st.file_uploader(T["upload"], type=["txt", "json"])
if file:
    logs = []
    if file.name.endswith(".txt"):
        content = file.getvalue().decode("utf-8", errors="ignore")
        logs = [{"id": i, "text": l} for i, l in enumerate(content.splitlines()) if l.strip()]
    else:
        content = file.getvalue().decode("utf-8", errors="ignore")
        raw = json.loads(content)
        if isinstance(raw, dict):
            raw = raw.get("logs", [])
        logs = [
            {"id": r.get("id", i), "text": r.get("text") or r.get("log_text") or ""}
            for i, r in enumerate(raw)
            if isinstance(r, dict)
        ]

    st.write(f"Loaded {len(logs)} logs")
    st.dataframe(logs[:10])

    if st.button("Analyze Batch", key="batch_analyze") and logs:
        results = []
        for r in logs:
            if not r["text"].strip():
                continue
            ents = predict_entities(r["text"])
            results.append({"id": r.get("id"), "events": extract_events(ents)})

        st.success(f"Processed {len(results)} logs")
        st.json(results[:10])
        st.download_button("⬇️ Download JSON", json.dumps(results, indent=2), "results.json")

st.markdown("</div>", unsafe_allow_html=True)

# -------- Footer --------
st.markdown("---")
st.markdown(f"<div class='footer'>{md_to_html(T['footer'])}</div>", unsafe_allow_html=True)
