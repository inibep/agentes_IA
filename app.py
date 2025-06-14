
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="Curso ML - INIBEP", layout="wide")
st.image("https://inibepsac.com/wp-content/uploads/2024/03/INIBEP_horizontal.png", width=300)
st.title("📊 Curso Básico de Machine Learning")
st.markdown("Bienvenido a la versión completa del curso interactivo desarrollado por **INIBEP S.A.C.**")

modulo = st.sidebar.selectbox("Selecciona un módulo:", [
    "1. ¿Qué es Machine Learning?",
    "2. Preparación de Datos",
    "3. Clasificación (Logistic Regression)",
    "4. Regresión (Linear Regression)",
    "5. Clustering (KMeans)",
    "6. Proyecto Final"
])

if modulo == "1. ¿Qué es Machine Learning?":
    st.header("Introducción a Machine Learning")
    st.markdown("""
    Machine Learning permite que los sistemas aprendan de datos para hacer predicciones o tomar decisiones.

    **Tipos de Aprendizaje:**
    - **Supervisado**: con datos etiquetados (clasificación, regresión)
    - **No Supervisado**: sin etiquetas (clustering)
    - **Por Refuerzo**: aprendizaje basado en recompensas

    A continuación, pon a prueba lo aprendido.
    """)
    pregunta = st.radio("¿Cuál es un ejemplo de aprendizaje supervisado?",
                        ("PCA", "KMeans", "Regresión Lineal"))
    if pregunta == "Regresión Lineal":
        st.success("✅ Correcto")
    else:
        st.error("❌ Incorrecto")

elif modulo == "2. Preparación de Datos":
    st.header("Exploración y limpieza de datos")
    file = st.file_uploader("Sube un archivo CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write("Vista previa:", df.head())
        st.write("Tipos de datos:", df.dtypes)
        st.write("Valores nulos:", df.isnull().sum())
        if st.checkbox("Aplicar escalado (normalización)"):
            scaler = StandardScaler()
            df_scaled = pd.DataFrame(scaler.fit_transform(df.select_dtypes(include='number')), columns=df.select_dtypes(include='number').columns)
            st.write("Datos normalizados:", df_scaled.head())
        if st.checkbox("Mostrar correlación"):
            st.write(df.corr())

elif modulo == "3. Clasificación (Logistic Regression)":
    st.header("Clasificación Binaria con Regresión Logística")
    file = st.file_uploader("Sube un dataset para clasificación (target binario)", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.write(df.head())
        columnas = df.columns.tolist()
        X_cols = st.multiselect("Selecciona variables independientes:", columnas)
        y_col = st.selectbox("Selecciona la variable objetivo:", columnas)
        if X_cols and y_col:
            X = df[X_cols]
            y = df[y_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Precisión:", accuracy_score(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot()
            st.pyplot()

elif modulo == "4. Regresión (Linear Regression)":
    st.header("Regresión Lineal")
    file = st.file_uploader("Sube dataset para regresión", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write(df.head())
        columnas = df.columns.tolist()
        x_col = st.selectbox("Variable independiente:", columnas)
        y_col = st.selectbox("Variable dependiente:", columnas)
        X = df[[x_col]]
        y = df[y_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Error absoluto medio:", mean_absolute_error(y_test, y_pred))
        fig, ax = plt.subplots()
        ax.scatter(X_test, y_test)
        ax.plot(X_test, y_pred, color='red')
        st.pyplot(fig)

elif modulo == "5. Clustering (KMeans)":
    st.header("Clustering con KMeans")
    file = st.file_uploader("Sube dataset para agrupar", type="csv")
    if file:
        df = pd.read_csv(file)
        st.write(df.head())
        n_clusters = st.slider("Número de clusters:", 2, 10, 3)
        df_num = df.select_dtypes(include='number')
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(df_num)
        df['cluster'] = clusters
        st.write("Clusters asignados:", df[['cluster']].value_counts())
        if df_num.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df_num)
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
            st.pyplot(fig)

elif modulo == "6. Proyecto Final":
    st.header("Proyecto Final")
    st.markdown("""
    En este módulo podrás desarrollar un mini proyecto práctico:

    - Selecciona un problema real.
    - Prepara y analiza tu dataset.
    - Aplica un modelo.
    - Explica tus conclusiones.

    Puedes usar Google Colab, Jupyter o esta app como entorno.
    """)
