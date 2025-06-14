import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from io import StringIO
import numpy as np

# CONFIGURACIÓN GENERAL DE LA APP
st.set_page_config(page_title="Curso Básico de ML - INIBEP", layout="wide", initial_sidebar_state="expanded")

# ESTILO PERSONALIZADO
st.markdown("""
    <style>
    /* Estilos generales del cuerpo */
    .stApp {
        background-color: #f4f6f8; /* Un gris claro para el fondo */
        color: #333333; /* Color de texto principal */
    }

    /* Contenedor principal del contenido */
    .main {
        background-color: #ffffff; /* Fondo blanco para el contenido principal */
        padding: 2rem;
        border-radius: 10px; /* Bordes redondeados */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra suave */
    }

    /* Títulos principales */
    .title {
        font-size: 36px;
        color: #005b94; /* Azul oscuro corporativo */
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }

    /* Subtítulos */
    .subtitle {
        font-size: 24px;
        color: #0078b0; /* Azul medio corporativo */
        text-align: center;
        margin-bottom: 1.5rem;
        font-style: italic;
    }

    /* Encabezados de sección */
    h1, h2, h3, h4, h5, h6 {
        color: #005b94; /* Azul oscuro para todos los encabezados */
        font-weight: bold;
        border-bottom: 2px solid #e0e0e0; /* Línea sutil debajo de los encabezados */
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Texto general de Markdown */
    .stMarkdown p {
        font-size: 16px;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    /* Listas */
    .stMarkdown ul {
        list-style-type: disc;
        margin-left: 20px;
        margin-bottom: 1rem;
    }

    /* Cajas de código y bloques de cita */
    .stCode, .stBlockquote {
        background-color: #e6f2f7; /* Un azul muy claro para bloques de código/cita */
        border-left: 5px solid #0078b0; /* Borde azul para destacar */
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }

    /* Botones */
    .stButton>button {
        background-color: #0078b0; /* Color de botón primario */
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #005b94; /* Oscurecer al pasar el ratón */
    }

    /* Selectbox y otros widgets */
    .stSelectbox, .stFileUploader, .stCheckbox {
        margin-bottom: 1rem;
    }

    /* Pie de página */
    .footer {
        font-size: 14px;
        color: #666666; /* Gris más oscuro para el pie de página */
        text-align: center;
        padding-top: 2rem;
        border-top: 1px solid #e0e0e0; /* Línea divisoria superior */
        margin-top: 2rem;
    }

    /* Contenedor de video */
    .stVideo {
        margin-bottom: 2rem;
        border-radius: 10px;
        overflow: hidden; /* Asegura que el borde redondeado se aplique al video */
    }

    /* Columnas */
    .st-emotion-cache-cpgxny { /* Clases generadas por Streamlit para columnas */
        gap: 2rem; /* Espacio entre columnas */
    }

    /* Ajustes específicos para imágenes */
    .stImage > img {
        border-radius: 8px;
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
Bienvenido a la app interactiva donde aprenderás los fundamentos del Machine Learning de forma **práctica**,
usando datasets reales y visualizaciones dinámicas. Usa el menú lateral para navegar entre los módulos.
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

# MÓDULO 1: INTRODUCCIÓN A MACHINE LEARNING
if modulo == "1. Introducción a Machine Learning":
    st.header("¿Qué es Machine Learning?")
    st.markdown("""
    El Machine Learning (Aprendizaje Automático) es una rama de la inteligencia artificial que permite a las
    computadoras aprender de los datos sin ser programadas explícitamente. Es el motor detrás de muchas
    aplicaciones modernas, desde recomendaciones de películas hasta vehículos autónomos.

    **Conceptos Clave:**
    - **Datos:** La materia prima. Cuantos más datos relevantes tengamos, mejor aprenderá el modelo.
    - **Algoritmo:** La "receta" que el modelo usa para aprender de los datos.
    - **Modelo:** El resultado del proceso de aprendizaje, listo para hacer predicciones o decisiones.
    """)

    st.subheader("Tipos de Aprendizaje Automático:")
    st.markdown("""
    Existen tres tipos principales de Machine Learning, cada uno adecuado para diferentes tipos de problemas:
    """)

    st.markdown("### 🔹 Aprendizaje Supervisado")
    st.markdown("""
    En el aprendizaje supervisado, el modelo aprende de un conjunto de datos que incluye **etiquetas**
    (las respuestas correctas). Es como enseñarle a un niño con ejemplos donde ya sabes la respuesta.

    **Ejemplos Comunes:**
    - **Clasificación:** Predecir una categoría (ej: spam/no spam, gato/perro, cliente churn/no churn).
      * *Imagina que le muestras a una computadora miles de fotos de gatos y perros, y le dices cuál es cuál. Después, la computadora puede identificar si una nueva foto es de un gato o un perro.*
    - **Regresión:** Predecir un valor numérico continuo (ej: precio de una casa, temperatura, ventas futuras).
      * *Podrías usarlo para predecir el precio de una vivienda basándose en su tamaño, número de habitaciones y ubicación.*
    """)
    st.markdown("---")

    st.markdown("### 🔹 Aprendizaje No Supervisado")
    st.markdown("""
    Aquí, el modelo trabaja con datos que **no tienen etiquetas**. Su objetivo es encontrar patrones ocultos,
    estructuras o relaciones dentro de los datos por sí mismo. Es como pedirle a alguien que organice un
    montón de objetos sin decirle cómo agruparlos.

    **Ejemplos Comunes:**
    - **Clustering (Agrupamiento):** Identificar grupos naturales en los datos.
      * *Piensa en agrupar clientes con comportamientos de compra similares sin saber de antemano qué grupos existen. La máquina encuentra los patrones por sí misma.*
    - **Reducción de Dimensionalidad:** Simplificar los datos, reduciendo el número de características.
      * *Útil para visualizar datos complejos o para acelerar otros algoritmos de ML.*
    """)
    st.markdown("---")

    st.markdown("### 🔹 Aprendizaje por Refuerzo")
    st.markdown("""
    En este tipo, un "agente" aprende a tomar decisiones interactuando con un entorno para maximizar
    una "recompensa". Es como enseñarle a un perro a hacer trucos dándole premios cuando lo hace bien.

    **Ejemplos Comunes:**
    - **Juegos:** Agentes que aprenden a jugar videojuegos o ajedrez superando a humanos.
      * *Un ejemplo clásico es un programa que aprende a jugar ajedrez, mejorando con cada partida que juega, recibiendo 'recompensas' por movimientos exitosos.*
    - **Robótica:** Robots que aprenden a realizar tareas complejas en el mundo real.
    - **Sistemas de recomendación:** Optimizar la secuencia de recomendaciones para mantener al usuario engagement.
    """)
    st.markdown("---")

    st.subheader("¡Listo para empezar!")
    st.markdown("""
    Ahora que tienes una idea general de qué es el Machine Learning y sus tipos, te invito a explorar los
    siguientes módulos para ver cómo se aplican estos conceptos en la práctica.
    """)
    st.markdown("""
    **Recursos Adicionales:**
    * [Video introductorio a ML (Coursera - Español)](https://www.youtube.com/watch?v=ukzFI9rgEOw)
    * [Artículo: ¿Qué es el Machine Learning? (IBM)](https://www.ibm.com/es-es/topics/machine-learning/what-is-machine-learning)
    """)

# MÓDULO 2: PREPARACIÓN DE DATOS
elif modulo == "2. Preparación de Datos":
    st.header("Exploración y Preparación de Datos")
    st.markdown("""
    La calidad de tus datos es crucial para el rendimiento de cualquier modelo de Machine Learning.
    En este módulo, aprenderemos a explorar y preparar un dataset real.
    """)

    file = st.file_uploader("Sube un archivo CSV para explorar (ej. iris.csv, titanic.csv)", type=["csv"])
    df = None
    if file:
        try:
            df = pd.read_csv(file)
            st.success("Archivo cargado exitosamente.")
            st.subheader("Vista Previa del Dataset:")
            st.write(df.head())

            st.subheader("Información del Dataset:")
            # Usar StringIO para capturar la salida de df.info()
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

            if st.checkbox("Mostrar estadísticas descriptivas"):
                st.subheader("Estadísticas Descriptivas:")
                st.write(df.describe())

            st.subheader("Visualización de Columnas:")
            col = st.selectbox("Selecciona una columna para visualizar su distribución:", df.columns)

            if pd.api.types.is_numeric_dtype(df[col]):
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax, color='#0078b0') # kde para la curva de densidad
                ax.set_title(f"Distribución de '{col}'")
                ax.set_xlabel(col)
                ax.set_ylabel("Frecuencia")
                st.pyplot(fig)

                if st.checkbox(f"Mostrar Box Plot para '{col}'"):
                    fig, ax = plt.subplots()
                    sns.boxplot(y=df[col], ax=ax, color='#005b94')
                    ax.set_title(f"Box Plot de '{col}' (Detección de Outliers)")
                    st.pyplot(fig)

            else:
                st.info(f"La columna '{col}' no es numérica. Mostrando conteo de valores.")
                st.write(df[col].value_counts())
                fig, ax = plt.subplots()
                sns.countplot(y=df[col], ax=ax, palette='viridis')
                ax.set_title(f"Conteo de valores en '{col}'")
                st.pyplot(fig)

            st.subheader("Limpieza y Transformación (Ejemplos):")
            st.markdown("""
            La limpieza de datos es un paso crucial. Aquí, exploraremos cómo manejar valores nulos.
            """)
            if df.isnull().sum().sum() > 0:
                st.warning("¡Tu dataset contiene valores nulos!")
                st.write("Conteo de valores nulos por columna:")
                st.write(df.isnull().sum()[df.isnull().sum() > 0])

                st.markdown("""
                **Opciones para manejar valores nulos:**
                1. **Eliminar:** Eliminar filas o columnas completas con valores nulos. (¡Usar con precaución!)
                2. **Imputar:** Rellenar los valores nulos con un valor (media, mediana, moda, etc.).
                """)

                impute_option = st.radio("¿Qué acción te gustaría simular?", ["No hacer nada", "Eliminar filas con nulos", "Imputar con la media (solo numéricas)"])
                processed_df = df.copy()
                if impute_option == "Eliminar filas con nulos":
                    rows_before = processed_df.shape[0]
                    processed_df.dropna(inplace=True)
                    rows_after = processed_df.shape[0]
                    st.success(f"Se eliminaron {rows_before - rows_after} filas con valores nulos.")
                    st.write("Vista previa del dataset después de eliminar nulos:")
                    st.write(processed_df.head())
                elif impute_option == "Imputar con la media (solo numéricas)":
                    numeric_cols = processed_df.select_dtypes(include=np.number).columns
                    for col_name in numeric_cols:
                        if processed_df[col_name].isnull().any():
                            mean_val = processed_df[col_name].mean()
                            processed_df[col_name].fillna(mean_val, inplace=True)
                            st.success(f"Valores nulos en '{col_name}' imputados con la media ({mean_val:.2f}).")
                    st.write("Vista previa del dataset después de imputar nulos:")
                    st.write(processed_df.head())
            else:
                st.success("¡Tu dataset no contiene valores nulos! Bien hecho.")

            st.subheader("Correlación entre Variables (solo numéricas):")
            numeric_df = df.select_dtypes(include=np.number)
            if not numeric_df.empty:
                st.write("Matriz de correlación:")
                corr_matrix = numeric_df.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
                ax.set_title("Matriz de Correlación")
                st.pyplot(fig)
                st.markdown("""
                *Una correlación cercana a 1 o -1 indica una fuerte relación lineal.*
                """)
            else:
                st.info("No hay columnas numéricas para calcular la correlación.")

    else:
        st.info("Sube un archivo CSV para empezar la exploración de datos.")
        st.markdown("""
        **Consejo:** Puedes descargar datasets de ejemplo en Kaggle.com o UCI Machine Learning Repository.
        """)
        # Carga un dataset de ejemplo si no se ha subido ninguno
        st.subheader("Dataset de Ejemplo (Iris Dataset):")
        if st.button("Cargar Iris Dataset"):
            try:
                iris_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
                df = pd.read_csv(iris_url)
                st.session_state['iris_df'] = df # Guarda en session_state para que persista
                st.write("Iris dataset cargado:")
                st.write(df.head())
                st.success("Iris dataset cargado. Ahora puedes seleccionar columnas para visualizar.")
            except Exception as e:
                st.error(f"Error al cargar el Iris dataset: {e}")

# MÓDULO 3: CLASIFICACIÓN (LOGISTIC REGRESSION)
elif modulo == "3. Clasificación (Logistic Regression)":
    st.header("Clasificación con Regresión Logística")
    st.markdown("""
    La regresión logística es un algoritmo fundamental para problemas de **clasificación binaria**
    (dos clases). Aunque su nombre sugiere "regresión", su salida es una probabilidad que se usa
    para clasificar.
    """)

    st.subheader("Carga tu Dataset de Clasificación")
    st.info("Para este módulo, se recomienda un dataset con una variable objetivo binaria (ej: 0/1, Sí/No).")
    uploaded_file = st.file_uploader("Sube un dataset CSV", type=["csv"], key="logistic_uploader")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset cargado exitosamente.")
            st.write("Vista previa del dataset:")
            st.write(data.head())

            st.subheader("Configuración del Modelo:")
            all_columns = data.columns.tolist()
            target_column = st.selectbox("Selecciona la variable objetivo (dependiente, binaria):", all_columns, key="logistic_target")
            feature_columns = st.multiselect("Selecciona las variables predictoras (independientes):",
                                             [col for col in all_columns if col != target_column],
                                             key="logistic_features")

            if target_column and feature_columns:
                # Asegurar que la columna objetivo es numérica (0 o 1)
                data[target_column] = pd.Categorical(data[target_column]).codes
                if data[target_column].nunique() != 2:
                    st.error("La variable objetivo debe ser binaria (tener solo dos valores únicos).")
                else:
                    X = data[feature_columns]
                    y = data[target_column]

                    # Manejo de datos no numéricos en X (One-Hot Encoding para simplicidad)
                    X = pd.get_dummies(X, drop_first=True)

                    # Escalar características numéricas
                    numeric_features = X.select_dtypes(include=np.number).columns
                    if not numeric_features.empty:
                        scaler = StandardScaler()
                        X[numeric_features] = scaler.fit_transform(X[numeric_features])
                        st.info("Características numéricas escaladas.")

                    if st.button("Entrenar Modelo de Regresión Logística"):
                        if X.empty:
                            st.error("No hay variables predictoras válidas para entrenar el modelo. Asegúrate de que las columnas seleccionadas son adecuadas.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                            model = LogisticRegression(solver='liblinear', random_state=42) # solver='liblinear' para datasets pequeños
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            probabilities = model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva

                            st.subheader("Resultados del Modelo:")

                            # Precisión
                            accuracy = accuracy_score(y_test, predictions)
                            st.metric(label="Precisión (Accuracy)", value=f"{accuracy:.2f}")
                            st.markdown("""
                            *La Precisión mide la proporción de predicciones correctas sobre el total de predicciones.*
                            """)

                            # Matriz de Confusión
                            st.subheader("Matriz de Confusión:")
                            cm = confusion_matrix(y_test, predictions)
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                                        xticklabels=model.classes_, yticklabels=model.classes_)
                            ax.set_xlabel("Predicción")
                            ax.set_ylabel("Valor Real")
                            ax.set_title("Matriz de Confusión")
                            st.pyplot(fig)
                            st.markdown("""
                            *Una Matriz de Confusión muestra el número de aciertos y errores para cada clase.*
                            * - **Verdaderos Positivos (TP):** Predicho positivo, real positivo.*
                            * - **Verdaderos Negativos (TN):** Predicho negativo, real negativo.*
                            * - **Falsos Positivos (FP):** Predicho positivo, real negativo (Error Tipo I).*
                            * - **Falsos Negativos (FN):** Predicho negativo, real positivo (Error Tipo II).*
                            """)

                            # Reporte de Clasificación
                            st.subheader("Reporte de Clasificación:")
                            report = classification_report(y_test, predictions, output_dict=True, target_names=[str(c) for c in model.classes_])
                            st.json(report) # Muestra el reporte como JSON para mayor claridad
                            st.markdown("""
                            * - **Precision:** Proporción de identificaciones positivas que fueron realmente correctas.*
                            * - **Recall (Sensibilidad):** Proporción de positivos reales que fueron identificados correctamente.*
                            * - **F1-Score:** Media armónica de Precision y Recall.*
                            * - **Support:** Número de ocurrencias de cada clase en `y_test`.*
                            """)

                            st.subheader("Probabilidades de Predicción (Primeras 10):")
                            prob_df = pd.DataFrame({'Valor Real': y_test.head(10), 'Probabilidad Clase Positiva': probabilities[:10], 'Predicción': predictions[:10]})
                            st.write(prob_df)

                            # Interpretación de coeficientes (si X tiene columnas con nombres)
                            if len(model.coef_[0]) == len(X.columns):
                                st.subheader("Interpretación de Coeficientes:")
                                coef_df = pd.DataFrame({'Característica': X.columns, 'Coeficiente': model.coef_[0], 'Exp(Coeficiente) (Odds Ratio)': np.exp(model.coef_[0])})
                                st.write(coef_df.sort_values(by='Coeficiente', ascending=False))
                                st.markdown("""
                                *Los coeficientes indican la influencia de cada característica en la probabilidad de la clase positiva.*
                                *Un `Exp(Coeficiente)` (Odds Ratio) > 1 indica que a medida que la característica aumenta, la probabilidad de la clase positiva aumenta, manteniendo lo demás constante.*
                                *Un `Exp(Coeficiente)` (Odds Ratio) < 1 indica lo contrario.*
                                """)
                            else:
                                st.info("No se pudieron interpretar los coeficientes directamente (posiblemente debido a la codificación de variables).")

            else:
                st.warning("Por favor, selecciona al menos una variable predictora y la variable objetivo.")

        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo: {e}. Asegúrate de que el CSV está bien formateado y las columnas son apropiadas.")
            st.markdown("---")
            st.subheader("Dataset de Ejemplo para Clasificación (Titanic Dataset):")
            st.markdown("Puedes usar este dataset para probar el módulo de Clasificación.")
            if st.button("Cargar Titanic Dataset (Supervivencia)"):
                try:
                    titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    titanic_df = pd.read_csv(titanic_url)
                    # Pre-procesamiento simple para el ejemplo
                    titanic_df['Age'].fillna(titanic_df['Age'].mean(), inplace=True)
                    titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)
                    st.session_state['titanic_df'] = titanic_df
                    st.write("Titanic dataset cargado. Intenta usar 'Survived' como objetivo y 'Age', 'Fare', 'Pclass', 'Sex', 'Embarked' como predictoras.")
                    st.write(titanic_df.head())
                except Exception as e:
                    st.error(f"Error al cargar el Titanic dataset: {e}")

# MÓDULO 4: REGRESIÓN (LINEAR REGRESSION)
elif modulo == "4. Regresión (Linear Regression)":
    st.header("Regresión Lineal Simple y Múltiple")
    st.markdown("""
    La regresión lineal es un método estadístico para modelar la relación entre una variable
    dependiente continua y una o más variables independientes.
    """)

    st.subheader("Carga tu Dataset para Regresión")
    st.info("Para este módulo, se recomienda un dataset con una variable objetivo numérica y continua (ej: precios de casas).")
    uploaded_file = st.file_uploader("Sube un dataset CSV", type=["csv"], key="linear_uploader")

    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Dataset cargado exitosamente.")
            st.write("Vista previa del dataset:", data.head())

            st.subheader("Configuración del Modelo:")
            columns = data.columns.tolist()
            y_col = st.selectbox("Selecciona la variable dependiente (numérica):", columns, key="linear_target")
            x_cols = st.multiselect("Selecciona las variables independientes (predictoras):",
                                    [col for col in columns if col != y_col],
                                    key="linear_features")

            if y_col and x_cols:
                # Filtrar solo columnas numéricas para X e Y
                numeric_data = data.select_dtypes(include=np.number)
                if y_col not in numeric_data.columns:
                    st.error(f"La columna objetivo '{y_col}' no es numérica. Por favor, selecciona una columna numérica.")
                else:
                    X = data[x_cols]
                    y = data[y_col]

                    # Manejar columnas no numéricas en X (One-Hot Encoding)
                    X = pd.get_dummies(X, drop_first=True)

                    if X.empty:
                        st.error("No hay variables predictoras numéricas o codificables válidas seleccionadas. Por favor, revisa tus selecciones.")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        st.subheader("Resultados del Modelo de Regresión Lineal:")

                        # Métricas de Evaluación
                        mae = mean_absolute_error(y_test, predictions)
                        mse = mean_squared_error(y_test, predictions)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(y_test, predictions)

                        st.metric(label="Error Absoluto Medio (MAE)", value=f"{mae:.2f}")
                        st.markdown("""
                        *El MAE es el promedio de las diferencias absolutas entre las predicciones y los valores reales. Es robusto a outliers.*
                        """)
                        st.metric(label="Error Cuadrático Medio (MSE)", value=f"{mse:.2f}")
                        st.markdown("""
                        *El MSE penaliza errores más grandes. Es sensible a outliers.*
                        """)
                        st.metric(label="Raíz del Error Cuadrático Medio (RMSE)", value=f"{rmse:.2f}")
                        st.markdown("""
                        *El RMSE está en las mismas unidades que la variable objetivo, facilitando la interpretación.*
                        """)
                        st.metric(label="Coeficiente de Determinación ($R^2$)", value=f"{r2:.2f}")
                        st.markdown("""
                        *El $R^2$ mide qué tan bien las variables independientes explican la variabilidad de la variable dependiente.
                        Un valor más cercano a 1 indica un mejor ajuste del modelo.*
                        """)

                        # Visualización (solo para regresión simple con una X numérica)
                        if len(x_cols) == 1 and pd.api.types.is_numeric_dtype(data[x_cols[0]]):
                            st.subheader("Gráfico de Regresión Simple:")
                            fig, ax = plt.subplots()
                            sns.scatterplot(x=X_test[x_cols[0]], y=y_test, ax=ax, label="Datos reales", color='#0078b0')
                            sns.lineplot(x=X_test[x_cols[0]], y=predictions, ax=ax, color="red", label="Predicción (Línea de Regresión)")
                            ax.set_xlabel(x_cols[0])
                            ax.set_ylabel(y_col)
                            ax.set_title(f"Regresión Lineal: {y_col} vs {x_cols[0]}")
                            ax.legend()
                            st.pyplot(fig)
                            st.markdown("""
                            *La línea roja representa la relación lineal que el modelo ha aprendido entre la variable independiente
                            y la dependiente, intentando minimizar la distancia a todos los puntos de datos.*
                            """)
                        else:
                            st.info("Para visualizar la línea de regresión, selecciona exactamente una variable independiente numérica.")

                        # Interpretación de Coeficientes
                        st.subheader("Interpretación de Coeficientes:")
                        coef_df = pd.DataFrame({'Característica': X.columns, 'Coeficiente': model.coef_})
                        st.write(coef_df.sort_values(by='Coeficiente', ascending=False))
                        st.write(f"Intercepto (Bias): {model.intercept_:.2f}")
                        st.markdown("""
                        *El **Intercepto** es el valor predicho de la variable dependiente cuando todas las variables independientes son cero.*
                        *Los **Coeficientes** indican cuánto cambia la variable dependiente por cada unidad de cambio en la variable independiente correspondiente, manteniendo las demás constantes.*
                        """)

            else:
                st.warning("Por favor, selecciona la variable dependiente y al menos una variable independiente.")

        except Exception as e:
            st.error(f"Error al cargar o procesar el archivo: {e}. Asegúrate de que el CSV está bien formateado y las columnas son apropiadas.")
            st.markdown("---")
            st.subheader("Dataset de Ejemplo para Regresión (Precios de Viviendas):")
            st.markdown("Puedes usar este dataset para probar el módulo de Regresión. Intenta predecir 'medv' (precio medio) usando 'rm' (número de habitaciones) o 'lstat' (% de población de bajo estatus).")
            if st.button("Cargar Boston Housing Dataset (modificado)"):
                try:
                    # Dataset Boston Housing ya no está disponible directamente en sklearn, usar una versión pública
                    boston_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
                    boston_df = pd.read_csv(boston_url)
                    st.session_state['boston_df'] = boston_df
                    st.write("Boston Housing dataset cargado.")
                    st.write(boston_df.head())
                except Exception as e:
                    st.error(f"Error al cargar el Boston Housing dataset: {e}")

# MÓDULO 5: CLUSTERING (PRÓXIMAMENTE)
elif modulo == "5. Clustering (Próximamente)":
    st.header("Clustering (Agrupamiento)")
    st.markdown("""
    Este módulo se enfocará en algoritmos de aprendizaje no supervisado, donde el objetivo es
    encontrar grupos o "clusters" naturales dentro de tus datos sin etiquetas predefinidas.
    """)
    st.info("¡Este módulo estará disponible en la próxima versión con K-Means y visualizaciones interactivas!")
    st.image("https://placehold.co/600x300/0078b0/ffffff?text=Clustering%20Pronto", caption="Imagen de placeholder para el módulo de Clustering", use_column_width=True)
    st.markdown("""
    Podrás subir tu dataset, seleccionar las características para el clustering y visualizar
    los grupos encontrados, así como experimentar con diferentes números de clusters.
    """)

# MÓDULO 6: PROYECTO FINAL
elif modulo == "6. Proyecto Final":
    st.header("Tu Proyecto Final de Machine Learning")
    st.markdown("""
    ¡Felicidades por llegar a este punto! Aquí es donde aplicarás todo lo aprendido.
    El proyecto final es una oportunidad para demostrar tus habilidades en el ciclo de vida
    completo de un proyecto de Machine Learning, desde la comprensión del problema hasta la
    evaluación del modelo.
    """)

    st.subheader("🎯 Requisitos del Proyecto:")
    st.markdown("""
    - **1. Definición del Problema:**
        - ¿Qué problema quieres resolver con Machine Learning?
        - ¿Por qué es importante?
        - ¿Qué tipo de problema de ML es (clasificación, regresión, clustering, etc.)?
    - **2. Selección y Limpieza de Dataset:**
        - ¿De dónde obtuviste tus datos? (Ej: Kaggle, UCI ML Repository, datos propios)
        - Justifica la elección de tu dataset.
        - Describe el proceso de limpieza y preprocesamiento de datos (manejo de nulos, outliers, codificación, escalado).
    - **3. Exploración de Datos (EDA):**
        - Visualizaciones clave para entender tu dataset.
        - Insights o descubrimientos importantes sobre tus datos.
    - **4. Selección y Aplicación del Modelo:**
        - ¿Qué modelo(s) de ML elegiste y por qué?
        - Describe cómo entrenaste tu modelo.
    - **5. Evaluación de Resultados:**
        - ¿Qué métricas usaste para evaluar tu modelo?
        - Interpreta los resultados de las métricas.
        - ¿Qué tan bien se desempeñó tu modelo?
    - **6. Conclusiones y Futuras Mejoras:**
        - Resumen de tus hallazgos clave.
        - Limitaciones de tu modelo o dataset.
        - Posibles mejoras o siguientes pasos para el proyecto.
    """)

    st.subheader("🚀 ¡Anímate a aplicar todo lo aprendido!")
    st.markdown("""
    Puedes usar esta misma aplicación como base para presentar tus resultados,
    o crear tu propia aplicación Streamlit. ¡La creatividad es clave!
    """)
    st.markdown("---")
    st.subheader("Ideas para tu Proyecto:")
    st.markdown("""
    - **Clasificación:**
        - Predecir si un cliente va a "churn" (dejar un servicio).
        - Clasificar emails como spam o no spam.
        - Predecir la supervivencia en el Titanic.
    - **Regresión:**
        - Predecir el precio de la vivienda basado en características.
        - Estimar las ventas de un producto.
        - Predecir la calificación de un estudiante.
    - **Clustering (cuando el módulo esté disponible):**
        - Segmentar clientes para campañas de marketing.
        - Agrupar especies de flores por características.
    """)

    st.subheader("Recursos Útiles para tu Proyecto:")
    st.markdown("""
    - [Kaggle Datasets](https://www.kaggle.com/datasets): Gran fuente de datasets públicos.
    - [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php): Otro repositorio popular de datasets.
    - [Documentación de Streamlit](https://docs.streamlit.io/): Para construir tu propia app interactiva.
    """)

# PIE DE PÁGINA
st.markdown("""
<div class='footer'>Desarrollado con ❤️ por INIBEP S.A.C. - <a href='https://www.inibepsac.com' target='_blank'>www.inibepsac.com</a></div>
""", unsafe_allow_html=True)
