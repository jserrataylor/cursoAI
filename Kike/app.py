import streamlit as st
import joblib
import numpy as np

# Título y descripción
st.title("Predicción de Compra de Cliente")
st.write("""
Este servicio despliega un modelo de clasificación entrenado sobre datos sintéticos
y predice la probabilidad de que un cliente compre un producto en función de:
- Edad
- Ingresos anuales
- Número de visitas al sitio
""")

# Función para cargar y cachear el modelo
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load("best_model.pkl")

model = load_model()

# Panel lateral para los parámetros de entrada
st.sidebar.header("Parámetros de entrada")
age = st.sidebar.slider("Edad", min_value=18, max_value=90, value=30, step=1)
income = st.sidebar.number_input("Ingresos anuales (USD)", min_value=0.0, value=50000.0, step=1000.0)
visits = st.sidebar.slider("Número de visitas al sitio", min_value=0, max_value=50, value=5, step=1)

# Preparar los datos de entrada y hacer la predicción
X_new = np.array([[age, income, visits]])
prob = model.predict_proba(X_new)[0, 1]
pred = model.predict(X_new)[0]

# Mostrar resultados
st.subheader("Resultados de la predicción")
st.markdown(f"- **Probabilidad de compra:** {prob:.2%}")
st.markdown(f"- **Predicción:** {'🛒 Comprar' if pred == 1 else '🚫 No comprar'}")
