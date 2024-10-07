# app.py

import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os

# Configuración de la página
st.set_page_config(
    page_title="Detector de Texto Humano vs IA",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Título de la aplicación
st.title("Detector de Texto: Humano vs IA")
st.write("Introduce un texto y descubre si fue escrito por un humano o generado por una IA.")

# Definir la clase de entrada
class Post:
    def __init__(self, text):
        self.text = text

# Función de carga del modelo
@st.cache_resource
def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return tokenizer, model

# Función de predicción
def predict(text, tokenizer, model):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)

    label = 'IA' if predicted_class.item() == 1 else 'Humano'
    probability = confidence.item()
    return label, probability

# Cargar el modelo y el tokenizador
model_dir = "trained_model"  # Asegúrate de que esta ruta es correcta
tokenizer, model = load_model(model_dir)

# Entrada de texto del usuario
user_input = st.text_area("Introduce el texto aquí:", height=200)

# Botón de predicción
if st.button("Detectar"):
    if user_input.strip() == "":
        st.warning("Por favor, introduce un texto para analizar.")
    else:
        label, prob = predict(user_input, tokenizer, model)
        st.success(f"**Clasificación:** {label}")
        st.info(f"**Probabilidad:** {prob:.2f}")
