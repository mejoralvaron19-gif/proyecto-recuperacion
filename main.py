
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de Potencia de Bombeo",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'modelo_xgboost_final.joblib'.
model = load_model('modelo.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos de la bomba.
    """)

    # Slider para la presión
    Presion = st.slider(
        label='Presion (psi)',
        min_value=10.50,
        max_value=63.88,
        value=11.11, # Valor inicial
        step=1.00
    )
    st.caption("Representa la presión en el equipo. A mayor presión, mayor potencia es requerida")

    # Slider para la temperatura
    Temperatura = st.slider(
        label='Temperatura (°C)',
        min_value=67.04,
        max_value=198.13,
        value=70.12,
        step=1.00
    )
    st.caption("Influye en la viscosidad del l{iquido, lo cual, incide en la Carga neta positiva de succión y la potencia desarrollada")

    # Slider para el flujo
    Flujo = st.slider(
        label='Flujo (L/min)',
        min_value=142.09,
        max_value=249.88,
        value=143.20,
        step=1.00
    )
    st.caption("Variable proporcional a la potencia desarrollada. A mayor flujo, mayor potencia")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de Potencia de Bombeo")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir la Potencia de Bombeo en un proceso industrial.
**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para controlar los valores asociados a la potencia de bombeo empleada.
- **Predecir** el impacto de potencias muy altas o bajas en el coste operacional del proceso.
- **Solucionar** problemas potenciales simulando diferentes escenarios que ayuden a la optimización del proceso.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir Potencia Bombeo', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        df_input = pd.DataFrame({
            'Presion': [Presion],
            'Temperatura': [Temperatura],
            'Flujo': [Flujo]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Potencia de Bombeo Predicha:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el valor estimado de la potencia de bombeo que se desarrollará en la operación.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra la potencia de bombeo.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo de la potencia de bomveo en un proceso industrial.
    * **Características Usadas:** Presión, Temepratura y Flujo.
    """)
