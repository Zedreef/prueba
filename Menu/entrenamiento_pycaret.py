import tensorflow as tf
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Menu.utilidades import RUTA_PATENTES, RUTA_PUBLICACIONES, procesar_fecha_publicacion

# Keras
# https://mathadventure1.github.io/sm64/sm64/index.html

def mostrar_pycaret():
    st.title('Predicción de Publicaciones y Patentes')

    # Definir el año actual
    current_year = 2024

    # Cargar Datos
    @st.cache_data
    def load_data():
        publications = pd.read_csv(RUTA_PUBLICACIONES, encoding='latin-1')
        patents = pd.read_csv(RUTA_PATENTES, encoding='latin-1')
        return publications, patents

    publications, patents = load_data()

    # Preprocesamiento de Datos
    def preprocess_data(publications, patents, current_year):
        # Procesar fechas de publicaciones
        publications['Publication Date'] = publications['Publication Date'].apply(procesar_fecha_publicacion)
        publications['Publication Year'] = publications['Publication Date'].dt.year
        publications['Years Since Last Publication'] = current_year - publications['Publication Year']
        publications['New Publication Next Year'] = (publications['Years Since Last Publication'] < 2).astype(int)

        # Procesar fechas de patentes
        patents['Filing Date'] = patents['Filing Date'].apply(procesar_fecha_publicacion)
        patents['Filing Year'] = patents['Filing Date'].dt.year
        patents['Years Since Last Patent'] = current_year - patents['Filing Year']

        # Fusionar 'Years Since Last Patent' con publicaciones
        patent_years = patents.groupby('Inventor')['Years Since Last Patent'].min().reset_index()
        patent_years.rename(columns={'Years Since Last Patent': 'Years Since Last Patent'}, inplace=True)
        publications = publications.merge(patent_years, how='left', left_on='Authors', right_on='Inventor')
        publications['Years Since Last Patent'].fillna(0, inplace=True)
        publications = publications.drop(columns=['Inventor', 'Publication Date'])
        patents = patents.drop(columns=['Filing Date'])

        return publications, patents

    publications, patents = preprocess_data(publications, patents, current_year)

    # Seleccionar Características y Etiqueta
    def select_features(publications):
        # Seleccionar solo columnas numéricas
        X = publications.select_dtypes(include=[np.number])

        # Definir la etiqueta
        y = X['New Publication Next Year']
        X = X.drop(columns=['New Publication Next Year'])

        return X, y

    X, y = select_features(publications)

    # Dividir Datos en Entrenamiento y Prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar las Características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir el Modelo
    def create_model(input_shape):
        model = tf.keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='new_publication')
        ])

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model = create_model(input_shape=(X_train_scaled.shape[1],))

    # Entrenar el Modelo
    with st.spinner('Entrenando el modelo...'):
        model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test))
    st.success('Modelo entrenado exitosamente.')

    # Interfaz de Usuario para Predicción
    # Crear una lista de autores únicos
    unique_authors = publications['Authors'].unique()
    selected_author = st.selectbox('Seleccione un Autor', unique_authors)

    # Filtrar las publicaciones del autor seleccionado
    author_publications = publications[publications['Authors'] == selected_author]
    # Filtrar las patentes del autor
    author_patents = patents[patents['Inventor'] == selected_author]

    # Obtener los años desde la última publicación del autor
    if not author_publications.empty:
        last_publication_year = author_publications['Publication Year'].max()
        years_since_last_publication = current_year - last_publication_year
    else:
        years_since_last_publication = 0

    # Obtener los años desde la última patente del autor
    if not author_patents.empty:
        last_patent_year = author_patents['Filing Year'].max()
        years_since_last_patent = current_year - last_patent_year
    else:
        years_since_last_patent = 0

    # Mostrar los valores calculados en la interfaz
    st.write(f"Años desde la última publicación: {years_since_last_publication}")
    st.write(f"Años desde la última patente: {years_since_last_patent}")

    # Preparar los datos de entrada para la predicción
    def prepare_input(author_publications, years_since_last_patent, X_columns):
        if not author_publications.empty:
            # Seleccionar la última publicación del autor
            input_features = author_publications.select_dtypes(include=[np.number]).iloc[-1].drop(labels=['New Publication Next Year'])
        else:
            # Si el autor no tiene publicaciones, crear un array de ceros con el mismo número de características
            input_features = pd.Series([0] * (X_columns.shape[0]))

        # Asegurarse de que 'Years Since Last Patent' esté incluida
        if 'Years Since Last Patent' not in input_features.index:
            input_features['Years Since Last Patent'] = years_since_last_patent
        else:
            # Actualizar el valor de 'Years Since Last Patent'
            input_features['Years Since Last Patent'] = years_since_last_patent

        # Reordenar las características para que coincidan con el entrenamiento
        input_features = input_features[X_columns]

        # Rellenar cualquier valor faltante con cero
        input_features = input_features.fillna(0)

        # Escalar las características
        input_scaled = scaler.transform([input_features.values])

        return input_scaled

    input_scaled = prepare_input(author_publications, years_since_last_patent, X.columns)

    print(X.head())  # Muestra las primeras 5 filas
    print(X.columns)
    print(X_train)



    # Botón para realizar la predicción
    if st.button('Predecir'):
        try:
            prediction = model.predict(input_scaled)[0][0]
            st.write(f"**Probabilidad de nueva publicación:** {prediction:.2f}")
        except Exception as e:
            st.error(f"Error en la predicción: {e}")