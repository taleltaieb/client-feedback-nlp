import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
from deep_translator import GoogleTranslator
from langdetect import detect
import os
from pathlib import Path

st.set_page_config(page_title="🌍 Multilingual BERT Sentiment Classifier", layout="centered")
# Load model and tokenizer
@st.cache_resource
def load_model():
    # Force clean absolute path with forward slashes
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(base_dir, "dashboard", "bert_sentiment_model")
    model_path = model_path.replace("\\", "/")  # 👈 ensure POSIX-style path

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"❌ Model folder not found at: {model_path}")

    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model


tokenizer, model = load_model()
model.eval()

# Translation function
def translate_to_english(text):
    try:
        lang = detect(text)
        if lang != "en":
            translated = GoogleTranslator(source='auto', target='en').translate(text)
            return translated, lang
        return text, 'en'
    except:
        return text, 'unknown'

# Sentiment prediction
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    return pred_class, confidence

# Streamlit UI

st.title("🤖 Multilingual Review Sentiment Classifier (BERT)")
st.markdown("Paste a review in **any language** and get a sentiment prediction:")

user_input = st.text_area("💬 Review Text", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        translated_text, detected_lang = translate_to_english(user_input)
        label, confidence = predict_sentiment(translated_text)

        if detected_lang != "en":
            st.markdown(f"🔁 **Translated from {detected_lang.upper()} to English:**\n> {translated_text}")

        if label == 1:
            st.success(f"✅ Positive Review ({confidence*100:.1f}% confidence)")
        else:
            st.error(f"⚠️ Negative Review ({confidence*100:.1f}% confidence)")
            