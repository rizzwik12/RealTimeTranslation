import os
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

# Get Hugging Face token from environment variable
token = os.getenv("HF_TOKEN")

# Function to load the model
def get_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name, use_auth_token=token)
    model = MarianMTModel.from_pretrained(model_name, use_auth_token=token)
    return tokenizer, model

# Function to translate text
def translate_text(text, src_lang, tgt_lang):
    tokenizer, model = get_model(src_lang, tgt_lang)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Streamlit UI
st.title("Real-Time Language Translation")
st.write("Translate text between multiple languages.")

# Supported languages
languages = {
    "English": "en", "Spanish": "es", "French": "fr", "Italian": "it", "Portuguese": "pt", 
    "Romanian": "ro", "German": "de", "Dutch": "nl", "Russian": "ru", "Chinese": "zh", 
    "Japanese": "ja", "Korean": "ko", "Hindi": "hi", "Arabic": "ar"
}

# Language selection
src_lang = st.selectbox("Select source language:", list(languages.keys()))
tgt_lang = st.selectbox("Select target language:", list(languages.keys()))

src_lang_code = languages[src_lang]
tgt_lang_code = languages[tgt_lang]

# Input text
user_input = st.text_area("Enter text to translate:")

if st.button("Translate"):
    translated_text = translate_text(user_input, src_lang_code, tgt_lang_code)
    st.write("### Translated Text:")
    st.write(translated_text)