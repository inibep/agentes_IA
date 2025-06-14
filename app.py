
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# CONFIGURACIÓN GENERAL DE LA APP
st.set_page_config(page_title="Curso Básico de ML - INIBEP", layout="wide")

# ESTILO PERSONALIZADO
st.markdown("""
    <style>
    .main {
        background-color: #f4f6f8;
        padding: 2rem;
    }
    .title {
        font-size: 36px;
        color: #005b94;
    }
    .subtitle {
        font-size: 24px;
        color: #0078b0;
    }
    .footer {
        font-size: 14px;
        color: gray;
        text-align: center;
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ENCABEZADO CON LOGO Y VIDEO
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://inibepsac.com/wp-content/uploads/2024/03/INIBEP_horizontal.png", width=180)
with col2:
    st.markdown("""
        <div class='title'>📊 Curso Básico de Machine Learning</div>
        <div class='subtitle'>Una iniciativa de INIBEP S.A.C. para transformar el aprendizaje práctico</div>
    """, unsafe_allow_html=True)

st.video("https://www.youtube.com/watch?v=Gv9_4yMHFhI")

st.markdown("""
Bienvenido a la app interactiva donde aprenderás los fundamentos del Machine Learning de forma **práctica**, usando datasets reales y visualizaciones dinámicas. Usa el menú lateral para navegar entre los módulos.
""")

# MENÚ LATERAL
modulo = st.sidebar.selectbox("Selecciona un módulo:", [
    "1. Introducción a Machine Learning",
    "2. Preparación de Datos",
    "3. Clasificación (Logistic Regression)",
    "4. Regresión (Linear Regression)",
    "5. Clustering (Próximamente)",
    "6. Proyecto Final"
])

if modulo == "1. Introducción a Machine Learning":
    st.header("¿Qué es Machine Learning?")
    st.markdown("""
    Machine Learning es una disciplina de la inteligencia artificial que permite que los sistemas aprendan de datos para hacer predicciones o decisiones.

    **Tipos:**
    - Aprendizaje Supervisado
    - Aprendizaje No Supervisado
    - Aprendizaje por Refuerzo

    ¿Listo para empezar? ¡Explora los siguientes módulos!
    """)

elif modulo == "2. Preparación de Datos":
    st.header("Explora un dataset real")
    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Vista previa del dataset:", df.head())

        if st.checkbox("Mostrar estadísticas"):
            st.write(df.describe())

        col = st.selectbox("Selecciona una columna para visualizar:", df.columns)
        fig, ax = plt.subplots()
        df[col].hist(ax=ax)
        st.pyplot(fig)

elif modulo == "3. Clasificación (Logistic Regression)":
    st.header("Clasificación con Logistic Regression")
    st.markdown("(Próximamente disponible: versión funcional con scikit-learn)")

elif modulo == "4. Regresión (Linear Regression)":
    st.header("Regresión Lineal")
    st.markdown("Utilizaremos un ejemplo simple para predecir valores.")

    uploaded_file = st.file_uploader("Sube un dataset CSV con una variable dependiente y una independiente", type="csv")
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Datos cargados:", data.head())

        columns = list(data.columns)
        x_col = st.selectbox("Variable independiente:", columns)
        y_col = st.selectbox("Variable dependiente:", columns)

        X = data[[x_col]]
        y = data[y_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        st.write(f"Error absoluto medio: {mae:.2f}")

        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test, label="Datos reales")
        ax.plot(X_test, predictions, color="red", label="Predicción")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)

elif modulo == "5. Clustering (Próximamente)":
    st.header("Clustering")
    st.markdown("Este módulo estará disponible en la siguiente versión.")

elif modulo == "6. Proyecto Final":
    st.header("Proyecto Final")
    st.markdown("""
    En este módulo podrás subir tu proyecto, presentar tu análisis y compartir tus resultados con otros estudiantes.

    🎯 Requisitos:
    - Explicación del problema
    - Dataset limpio y justificado
    - Modelo aplicado
    - Evaluación de resultados
    - Conclusiones

    🚀 ¡Anímate a aplicar todo lo aprendido!
    """)

# PIE DE PÁGINA
st.markdown("""
<div class='footer'>Desarrollado por INIBEP S.A.C. - www.inibepsac.com</div>
""", unsafe_allow_html=True)
