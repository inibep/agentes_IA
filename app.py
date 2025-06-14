import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder # Para manejar la codificación de etiquetas en clasificación

# Configuración de la página de Streamlit
st.set_page_config(page_title="Curso Básico de ML - INIBEP", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
    <style>
    /* Estilos generales del contenedor principal */
    .main {
        background-color: #f4f6f8; /* Un gris claro para el fondo */
        padding: 2rem;
        border-radius: 10px; /* Bordes redondeados */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra suave */
    }

    /* Estilo para el título principal */
    .title {
        font-size: 36px;
        color: #005b94; /* Azul oscuro */
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Estilo para el subtítulo */
    .subtitle {
        font-size: 24px;
        color: #0078b0; /* Azul medio */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Estilo para el pie de página */
    .footer {
        font-size: 14px;
        color: gray;
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0; /* Línea divisoria */
        margin-top: 2rem;
    }

    /* Ajustes para el sidebar de Streamlit */
    .stSidebar > div:first-child {
        background-color: #e6f2ff; /* Un azul muy claro para el sidebar */
        border-right: 2px solid #005b94;
    }

    /* Estilos para los selectbox y botones para mejor UX */
    .stSelectbox > div > div > div > div {
        border-radius: 5px;
        border: 1px solid #0078b0;
    }

    .stButton > button {
        background-color: #005b94;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #0078b0;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

# --- ENCABEZADO DE LA APLICACIÓN ---
col1, col2 = st.columns([1, 3])
with col1:
    st.image("https://inibepsac.com/wp-content/uploads/2024/03/INIBEP_horizontal.png", width=180)
with col2:
    st.markdown("""
        <div class='title'>📊 Curso Básico de Machine Learning</div>
        <div class='subtitle'>Una iniciativa de INIBEP S.A.C. para transformar el aprendizaje práctico</div>
    """, unsafe_allow_html=True)

# Video de introducción
st.video("https://www.youtube.com/watch?v=Gv9_4yMHFhI")

# --- FUNCIONES PARA CADA MÓDULO ---

def modulo1():
    """
    Contenido del Módulo 1: Introducción a Machine Learning.
    Explica los tipos de aprendizaje en ML.
    """
    st.header("1. ¿Qué es Machine Learning?")
    st.markdown("""
        El **Machine Learning (Aprendizaje Automático)** es una rama de la Inteligencia Artificial
        que permite a las máquinas "aprender" de los datos sin ser programadas explícitamente.
        Se basa en el desarrollo de algoritmos que pueden analizar datos, aprender de ellos
        y luego hacer predicciones o tomar decisiones.
    """)

    st.subheader("Aprendizaje Supervisado")
    st.write("""
        Imagina que le muestras a una computadora miles de fotos de gatos y perros,
        y le dices cuál es cuál en cada foto. Después de ver suficientes ejemplos,
        la computadora puede identificar si una nueva foto contiene un gato o un perro.
        Aquí, el "supervisor" (tú) proporciona las respuestas correctas (etiquetas)
        para que el modelo aprenda a mapear entradas a salidas.
        **Ejemplos:** Clasificación (spam/no spam), Regresión (predicción de precios).
    """)

    st.subheader("Aprendizaje No Supervisado")
    st.write("""
        Piensa en agrupar clientes con comportamientos de compra similares sin
        saber de antemano qué grupos existen. La computadora busca patrones y estructuras
        ocultas en los datos por sí misma. No hay etiquetas predefinidas.
        **Ejemplos:** Clustering (segmentación de clientes), Reducción de Dimensionalidad.
    """)

    st.subheader("Aprendizaje por Refuerzo")
    st.write("""
        Un programa que aprende a jugar ajedrez, mejorando con cada partida
        al recibir "recompensas" por movimientos exitosos y "penalizaciones"
        por los fallidos. El agente aprende a través de la interacción con un entorno
        para maximizar una señal de recompensa.
        **Ejemplos:** Robots autónomos, sistemas de recomendación, juegos.
    """)
    st.markdown("---")
    st.markdown("**Recursos adicionales:**")
    st.markdown("- [Coursera - Intro to ML](https://www.coursera.org/learn/machine-learning)")
    st.markdown("- [Khan Academy - Conceptos de IA](https://es.khanacademy.org/computing/computer-science/ai)")

@st.cache_data # Cacha los datos para que no se recarguen si el archivo no cambia
def load_data(uploaded_file):
    """Carga un archivo CSV en un DataFrame de pandas."""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}. Asegúrate de que sea un archivo CSV válido.")
        return None

def modulo2():
    """
    Contenido del Módulo 2: Preparación de Datos.
    Permite al usuario subir un CSV, visualizarlo y realizar limpieza básica.
    """
    st.header("2. Preparación de Datos")
    st.markdown("La calidad de los datos es crucial para el éxito de cualquier modelo de ML.")

    file = st.file_uploader("Sube un archivo CSV", type=["csv"], help="Sube tu dataset en formato CSV para comenzar a explorarlo.")

    if file:
        df = load_data(file)
        if df is not None:
            st.success("Archivo cargado exitosamente.")
            st.subheader("Vista Previa del Dataset")
            st.write(df.head())

            st.subheader("Estadísticas Descriptivas")
            st.write(df.describe())

            st.subheader("Información General del Dataset")
            st.write(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
            st.write("Tipos de datos:")
            st.write(df.dtypes)

            st.subheader("Valores Nulos")
            null_counts = df.isnull().sum()
            if null_counts.sum() == 0:
                st.info("¡No hay valores nulos en este dataset!")
            else:
                st.warning("Valores nulos por columna:")
                st.write(null_counts[null_counts > 0])
                if st.button("Eliminar filas con valores nulos"):
                    initial_rows = df.shape[0]
                    df = df.dropna().reset_index(drop=True)
                    st.success(f"Se eliminaron {initial_rows - df.shape[0]} filas con valores nulos.")
                    st.write("Vista previa del dataset después de la limpieza:")
                    st.write(df.head())
                    st.session_state['df_cleaned'] = df # Guardar el DF limpio en el estado de la sesión

            st.subheader("Visualización de Datos")
            # Filtrar solo columnas numéricas para visualización por simplicidad
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            if not numeric_cols:
                st.warning("No hay columnas numéricas para visualizar en este dataset.")
            else:
                col = st.selectbox("Selecciona una columna para visualizar:", numeric_cols)
                if col:
                    chart_type = st.radio("Tipo de gráfico:", ["Histograma", "Boxplot", "Dispersión con otra variable (si es aplicable)"])

                    if chart_type == "Histograma":
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.histplot(df[col], kde=True, ax=ax, color='skyblue')
                        ax.set_title(f'Histograma de {col}')
                        ax.set_xlabel(col)
                        ax.set_ylabel('Frecuencia')
                        st.pyplot(fig)
                    elif chart_type == "Boxplot":
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.boxplot(x=df[col], ax=ax, color='lightcoral')
                        ax.set_title(f'Boxplot de {col}')
                        ax.set_xlabel(col)
                        st.pyplot(fig)
                    elif chart_type == "Dispersión con otra variable (si es aplicable)":
                        if len(numeric_cols) < 2:
                            st.info("Necesitas al menos dos columnas numéricas para un gráfico de dispersión.")
                        else:
                            available_cols_for_scatter = [c for c in numeric_cols if c != col]
                            if available_cols_for_scatter:
                                col2 = st.selectbox("Selecciona otra columna numérica para el eje Y:", available_cols_for_scatter)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.scatterplot(x=df[col], y=df[col2], ax=ax, color='green')
                                ax.set_title(f'Dispersión de {col} vs {col2}')
                                ax.set_xlabel(col)
                                ax.set_ylabel(col2)
                                st.pyplot(fig)
                            else:
                                st.info("No hay otra columna numérica disponible para un gráfico de dispersión.")
    else:
        st.info("Por favor, sube un archivo CSV para empezar.")

def modulo3():
    """
    Contenido del Módulo 3: Clasificación con Regresión Logística.
    Permite usar un dataset precargado (Iris) o subir uno propio para clasificación.
    """
    st.header("3. Clasificación con Regresión Logística")
    st.markdown("""
        La **Regresión Logística** es un algoritmo fundamental para problemas de clasificación,
        especialmente para problemas binarios (dos clases). Predice la probabilidad
        de que una instancia pertenezca a una clase específica.
    """)

    df = None
    use_example = st.checkbox("Usar dataset Iris pre-cargado (Ejemplo)", value=True)

    if use_example:
        st.info("Cargando el dataset Iris como ejemplo.")
        from sklearn.datasets import load_iris
        iris = load_iris(as_frame=True)
        df = iris.frame
        # Convertir la columna objetivo a binaria para un ejemplo simple de clasificación
        # 0 para 'setosa', 1 para 'versicolor' o 'virginica'
        df['target_binary'] = (df['target'] != 0).astype(int)
        st.write("Vista previa del dataset Iris (binarizado):")
        st.write(df.head())
    else:
        file = st.file_uploader("Sube un dataset CSV para clasificación", type="csv", help="Asegúrate de que tenga columnas numéricas y una columna objetivo.")
        if file:
            df = load_data(file)
            if df is not None:
                st.write("Vista previa del dataset cargado:")
                st.write(df.head())

    if df is not None:
        st.subheader("Selección de Variables")
        # Filtrar solo columnas numéricas para características
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if not numeric_cols:
            st.error("El dataset no contiene columnas numéricas. La regresión logística requiere entradas numéricas.")
            return

        X_cols = st.multiselect("Variables independientes (características):", numeric_cols, help="Selecciona las columnas numéricas que se usarán para predecir.")
        
        # Las variables objetivo pueden ser numéricas o categóricas (que luego se codificarán)
        all_cols = df.columns.tolist()
        y_col = st.selectbox("Variable objetivo (la columna que quieres predecir):", all_cols, help="Selecciona la columna que contiene las etiquetas de clase.")

        if X_cols and y_col:
            X = df[X_cols]
            y = df[y_col]

            # Codificación de la variable objetivo si es categórica (no numérica)
            if y.dtype == 'object' or y.dtype == 'category':
                st.info(f"La columna objetivo '{y_col}' es categórica. Realizando codificación de etiquetas.")
                le = LabelEncoder()
                try:
                    y = le.fit_transform(y)
                    st.write(f"Clases originales: {le.classes_}")
                    st.write(f"Clases codificadas: {list(range(len(le.classes_)))}")
                except Exception as e:
                    st.error(f"Error al codificar la columna objetivo: {e}. Asegúrate de que solo contiene valores válidos.")
                    return
            
            # Verificar si la variable objetivo tiene al menos dos clases (clasificación binaria/multiclase)
            if len(y.unique()) < 2:
                st.warning("La variable objetivo debe tener al menos dos clases para realizar una clasificación.")
                return
            if len(y.unique()) > 2 and model.solver == 'liblinear': # Liblinear solo para binario
                st.warning("Para más de dos clases, considera usar un solver diferente como 'lbfgs'.")


            st.subheader("Entrenamiento del Modelo")
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                with st.spinner("Entrenando modelo de Regresión Logística..."):
                    model = LogisticRegression(max_iter=1000) # Aumentar max_iter para convergencia
                    model.fit(X_train, y_train)
                st.success("Modelo entrenado exitosamente.")

                st.subheader("Evaluación del Modelo")
                y_pred = model.predict(X_test)

                st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
                st.write(f"**Precision:** {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
                st.write(f"**Recall:** {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
                st.write(f"**F1-score:** {f1_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")

                st.subheader("Matriz de Confusión")
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax, cmap='Blues')
                plt.title("Matriz de Confusión")
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error al entrenar o evaluar el modelo: {e}. Asegúrate de que las columnas seleccionadas son adecuadas para el modelo.")
                st.info("Posibles causas: columnas no numéricas en X, o problemas con la variable objetivo.")
        else:
            st.info("Por favor, selecciona las variables independientes y la variable objetivo para la clasificación.")
    else:
        st.info("Por favor, carga un dataset o usa el ejemplo de Iris para clasificación.")

def modulo4():
    """
    Contenido del Módulo 4: Regresión Lineal.
    Permite al usuario subir un CSV y realizar una regresión lineal simple.
    """
    st.header("4. Regresión Lineal")
    st.markdown("""
        La **Regresión Lineal** es un modelo estadístico que busca establecer una
        relación lineal entre una variable dependiente (continua) y una o más
        variables independientes. Es uno de los algoritmos más simples y fundamentales
        del aprendizaje supervisado.
    """)

    file = st.file_uploader("Sube un dataset CSV para regresión", type=["csv"], help="El dataset debe contener columnas numéricas para la regresión.")

    if file:
        df = load_data(file)
        if df is not None:
            st.write("Vista previa del dataset cargado:")
            st.write(df.head())

            st.subheader("Selección de Variables")
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            if not numeric_cols:
                st.error("El dataset no contiene columnas numéricas. La regresión lineal requiere entradas numéricas.")
                return

            x_col = st.selectbox("Variable independiente (X):", numeric_cols, help="Selecciona la variable que usará para predecir.")
            y_col = st.selectbox("Variable dependiente (Y):", [col for col in numeric_cols if col != x_col], help="Selecciona la variable numérica que quieres predecir.")

            if x_col and y_col:
                try:
                    X = df[[x_col]] # Asegurarse de que X sea un DataFrame 2D
                    y = df[y_col]

                    st.subheader("Entrenamiento del Modelo")
                    with st.spinner("Entrenando modelo de Regresión Lineal..."):
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                    st.success("Modelo entrenado exitosamente.")

                    st.subheader("Evaluación del Modelo")
                    y_pred = model.predict(X_test)

                    st.write(f"**Error Absoluto Medio (MAE):** {mean_absolute_error(y_test, y_pred):.2f}")
                    st.write(f"**Error Cuadrático Medio (MSE):** {mean_squared_error(y_test, y_pred):.2f}")
                    st.write(f"**Coeficiente de Determinación (R²):** {r2_score(y_test, y_pred):.2f}")
                    st.write(f"**Coeficiente (Pendiente):** {model.coef_[0]:.2f}")
                    st.write(f"**Intercepto:** {model.intercept_:.2f}")

                    st.subheader("Visualización de la Regresión")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(X_test, y_test, color='blue', label='Datos Reales')
                    ax.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de Regresión')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    ax.set_title(f'Regresión Lineal: {y_col} vs {x_col}')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                    st.pyplot(fig)

                except Exception as e:
                    st.error(f"Error al entrenar o evaluar el modelo: {e}. Asegúrate de que las columnas seleccionadas son adecuadas y numéricas.")
            else:
                st.info("Por favor, selecciona las variables independiente y dependiente para la regresión.")
    else:
        st.info("Por favor, sube un archivo CSV para regresión.")

def modulo5():
    """
    Contenido del Módulo 5: Clustering (Próximamente).
    """
    st.header("5. Clustering")
    st.markdown("""
        El **Clustering** es una técnica de aprendizaje no supervisado que agrupa
        puntos de datos similares en "clusters" o conglomerados. A diferencia
        de la clasificación, no utiliza etiquetas predefinidas, sino que descubre
        estructuras ocultas en los datos.

        Este módulo estará disponible en una próxima versión. ¡Mantente atento!
    """)
    st.info("Sugerencia: Podríamos implementar K-Means o DBSCAN aquí, permitiendo al usuario elegir el número de clusters y visualizar los resultados.")

def modulo6():
    """
    Contenido del Módulo 6: Proyecto Final.
    Ofrece sugerencias de proyectos y la estructura de entrega.
    """
    st.header("6. Proyecto Final")
    st.markdown("""
        ¡Es hora de aplicar todo lo aprendido! El proyecto final es tu oportunidad
        para desarrollar un modelo de Machine Learning de principio a fin.
    """)

    st.subheader("Proyectos Sugeridos:")
    st.markdown("""
    - **Clasificador de reseñas:** Desarrollar un modelo para clasificar reseñas de productos (positivas/negativas).
    - **Predicción de precios:** Predecir el precio de casas, acciones o cualquier otro bien.
    - **Segmentación de usuarios:** Agrupar usuarios de una plataforma basándose en su comportamiento.
    - **Detección de fraude:** Construir un modelo para identificar transacciones fraudulentas.
    - **Recomendador simple:** Crear un sistema de recomendación basado en contenido o colaborativo.
    """)

    st.subheader("Estructura de Entrega Sugerida:")
    st.markdown("""
    1.  **Título y Problema:** Define claramente el problema que quieres resolver y el objetivo de tu proyecto.
    2.  **Dataset y Preparación:** Describe el dataset utilizado, cómo lo obtuviste y los pasos de limpieza y preprocesamiento que realizaste.
    3.  **Modelo Aplicado:** Explica qué algoritmo de Machine Learning elegiste y por qué.
        Menciona los parámetros clave utilizados.
    4.  **Evaluación:** Presenta las métricas de evaluación relevantes para tu tipo de problema (accuracy, precision, recall, F1-score para clasificación; MAE, MSE, R² para regresión).
        Incluye visualizaciones si son útiles (ej. matriz de confusión, gráficos de residuos).
    5.  **Conclusiones:** Resume los hallazgos principales, las limitaciones de tu modelo y posibles mejoras futuras.
    """)
    st.info("¡Anímate a elegir un tema que te apasione!")

# --- MENÚ PRINCIPAL EN EL SIDEBAR ---
modulo_seleccionado = st.sidebar.selectbox("Selecciona un módulo:", [
    "1. Introducción a Machine Learning",
    "2. Preparación de Datos",
    "3. Clasificación (Logistic Regression)",
    "4. Regresión (Linear Regression)",
    "5. Clustering (Próximamente)",
    "6. Proyecto Final"
])

# --- EJECUCIÓN MODULAR BASADA EN LA SELECCIÓN ---
if modulo_seleccionado.startswith("1"):
    modulo1()
elif modulo_seleccionado.startswith("2"):
    modulo2()
elif modulo_seleccionado.startswith("3"):
    modulo3()
elif modulo_seleccionado.startswith("4"):
    modulo4()
elif modulo_seleccionado.startswith("5"):
    modulo5()
elif modulo_seleccionado.startswith("6"):
    modulo6()

# --- PIE DE PÁGINA ---
st.markdown("""
    <div class='footer'>Desarrollado por INIBEP S.A.C. - <a href="https://inibepsac.com/" target="_blank">www.inibepsac.com</a></div>
""", unsafe_allow_html=True)
