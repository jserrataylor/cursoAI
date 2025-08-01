import streamlit as st
import joblib
import numpy as np
import os

st.title("Predicción de Compra de Cliente")

st.write("""
App que carga un modelo previamente entrenado (`best_model.pkl`)  
y predice si un cliente comprará un producto según sus datos.
""")

model = joblib.load("best_model.pkl")

# 1. Localizar la carpeta del script
# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Ruta al modelo en la misma carpeta
# MODEL_PATH = os.path.join(CURRENT_DIR, "best_model.pkl")

# 3. Mostrar lista de archivos para verificar que está donde esperamos
# st.write("**Archivos en la carpeta de la app:**")
# try:
#     st.write(os.listdir(CURRENT_DIR))
# except Exception as e:
#     st.write(f"Error listando directorio: {e}")

# 4. Función para cargar el modelo, cacheando como recurso
@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        st.error(f"❌ No se encontró `best_model.pkl` en:\n```\n{path}\n```")
        return None
    return joblib.load(path)

# 5. Cargar modelo
model = load_model(MODEL_PATH)
if model is None:
    st.stop()

# 6. Panel lateral: inputs del usuario
st.sidebar.header("Parámetros de entrada")
age     = st.sidebar.slider("Edad",  18, 90, 30)
income  = st.sidebar.number_input("Ingresos anuales (USD)", 0.0, 1e6, 50000.0, step=1000.0)
visits  = st.sidebar.slider("Número de visitas al sitio", 0, 100, 5)

# 7. Preparar y predecir
X_new = np.array([[age, income, visits]])
prob   = model.predict_proba(X_new)[0, 1]
pred   = model.predict(X_new)[0]

# 8. Mostrar resultados
st.subheader("Resultados de la predicción")
st.markdown(f"- **Probabilidad de compra:** {prob:.2%}")
st.markdown(f"- **Predicción:** {'🛒 Comprar' if pred == 1 else '🚫 No comprar'}")
