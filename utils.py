
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

LABELS = {0: "FAKE", 1: "REAL"}


_loaded_models = {}

def load_model(model_path):
    if model_path not in _loaded_models:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        _loaded_models[model_path] = (tokenizer, model)
    return _loaded_models[model_path]

def clean_title(s: str) -> str:
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def predict_news(title: str, model_path: str):
    title = clean_title(title)
    tokenizer, model = load_model(model_path)

    inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()

    pred_class = torch.argmax(probs).item()
    return {
        "label": LABELS[pred_class],
        "confidence": float(probs[pred_class])
    }
