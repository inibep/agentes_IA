
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)

st.set_page_config(page_title="Curso Básico de ML - INIBEP", layout="wide")

# ENCABEZADO Y ESTILO
st.markdown("""
    <style>
    .main { background-color: #f4f6f8; padding: 2rem; }
    .title { font-size: 36px; color: #005b94; }
    .subtitle { font-size: 24px; color: #0078b0; }
    .footer { font-size: 14px; color: gray; text-align: center; padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://inibepsac.com/wp-content/uploads/2024/03/INIBEP_horizontal.png", width=180)
with col2:
    st.markdown("""<div class='title'>📊 Curso Básico de Machine Learning</div>
    <div class='subtitle'>Una iniciativa de INIBEP S.A.C. para transformar el aprendizaje práctico</div>""", unsafe_allow_html=True)

st.video("https://www.youtube.com/watch?v=Gv9_4yMHFhI")

# MÓDULOS COMO FUNCIONES
def modulo1():
    st.header("¿Qué es Machine Learning?")
    st.subheader("Aprendizaje Supervisado")
    st.write("Imagina que le muestras a una computadora miles de fotos de gatos y perros...")
    st.subheader("Aprendizaje No Supervisado")
    st.write("Piensa en agrupar clientes con comportamientos de compra similares...")
    st.subheader("Aprendizaje por Refuerzo")
    st.write("Un programa que aprende a jugar ajedrez, mejorando con cada partida...")
    st.markdown("**Recursos adicionales:**")
    st.markdown("- [Coursera - Intro to ML](https://www.coursera.org/learn/machine-learning)")
    st.markdown("- [Khan Academy - Conceptos de IA](https://es.khanacademy.org/computing/computer-science/ai)")

def modulo2():
    st.header("Explora y limpia tu dataset")
    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Vista previa del dataset:", df.head())
        st.subheader("Visualización")
        col = st.selectbox("Selecciona una columna para visualizar:", df.columns)
        chart_type = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot", "Dispersión con otra variable"])
        if chart_type == "Histograma":
            fig, ax = plt.subplots()
            df[col].hist(ax=ax)
            st.pyplot(fig)
        elif chart_type == "Boxplot":
            fig, ax = plt.subplots()
            sns.boxplot(data=df[col], ax=ax)
            st.pyplot(fig)
        elif chart_type == "Dispersión con otra variable":
            col2 = st.selectbox("Otra columna numérica:", df.select_dtypes(include='number').columns)
            fig, ax = plt.subplots()
            ax.scatter(df[col], df[col2])
            ax.set_xlabel(col)
            ax.set_ylabel(col2)
            st.pyplot(fig)
        if st.button("Eliminar filas con valores nulos"):
            df = df.dropna()
            st.success("Filas con valores nulos eliminadas.")
            st.write(df.head())

def modulo3():
    st.header("Clasificación con Regresión Logística")
    use_example = st.checkbox("Usar dataset Iris pre-cargado")
    if use_example:
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        df['target'] = (df['target'] == 0).astype(int)
    else:
        file = st.file_uploader("Sube un dataset CSV para clasificación", type="csv")
        if file:
            df = pd.read_csv(file)
    if 'df' in locals():
        st.write(df.head())
        cols = df.select_dtypes(include='number').columns
        X_cols = st.multiselect("Variables independientes:", cols)
        y_col = st.selectbox("Variable objetivo:", cols)
        if X_cols and y_col:
            X = df[X_cols]
            y = df[y_col]
            if y.nunique() > 2:
                st.error("⚠️ La variable objetivo debe ser binaria para aplicar regresión logística. Actualmente tiene más de dos clases o valores continuos.")
                return
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Precision:", precision_score(y_test, y_pred))
            st.write("Recall:", recall_score(y_test, y_pred))
            st.write("F1-score:", f1_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            st.pyplot()

def modulo4():
    st.header("Regresión Lineal")
    file = st.file_uploader("Sube un dataset CSV para regresión", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write(df.head())
        cols = df.select_dtypes(include='number').columns
        x_col = st.selectbox("Variable independiente:", cols)
        y_col = st.selectbox("Variable dependiente:", cols)
        X = df[[x_col]]
        y = df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("MAE:", mean_absolute_error(y_test, y_pred))
        st.write("MSE:", mean_squared_error(y_test, y_pred))
        st.write("R²:", r2_score(y_test, y_pred))
        st.write(f"Coeficiente: {model.coef_[0]:.2f}")
        st.write(f"Intercepto: {model.intercept_:.2f}")
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test)
        ax.plot(X_test, y_pred, color='red')
        st.pyplot(fig)

def modulo5():
    st.header("Clustering")
    st.markdown("Este módulo estará disponible en una próxima versión.")

def modulo6():
    st.header("Proyecto Final")
    st.markdown("""Proyectos sugeridos:
- Clasificador de reseñas
- Predicción de precios
- Segmentación de usuarios

**Estructura de entrega**:
1. Título y problema
2. Dataset y preparación
3. Modelo aplicado
4. Evaluación
5. Conclusiones
""")

# MENÚ PRINCIPAL
modulo = st.sidebar.selectbox("Selecciona un módulo:", [
    "1. Introducción a Machine Learning",
    "2. Preparación de Datos",
    "3. Clasificación (Logistic Regression)",
    "4. Regresión (Linear Regression)",
    "5. Clustering (Próximamente)",
    "6. Proyecto Final"
])

# EJECUCIÓN MODULAR
if modulo.startswith("1"): modulo1()
elif modulo.startswith("2"): modulo2()
elif modulo.startswith("3"): modulo3()
elif modulo.startswith("4"): modulo4()
elif modulo.startswith("5"): modulo5()
elif modulo.startswith("6"): modulo6()

# PIE DE PÁGINA
st.markdown("""<div class='footer'>Desarrollado por INIBEP S.A.C. - www.inibepsac.com</div>""", unsafe_allow_html=True)
