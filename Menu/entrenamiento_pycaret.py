

# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import streamlit as st
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from Menu.utilidades import RUTA_PATENTES, RUTA_PUBLICACIONES, procesar_fecha_publicacion

# def mostrar_pycaret():

#     # Cargar datos
#     publications = pd.read_csv(RUTA_PUBLICACIONES, encoding='latin-1')
#     patents = pd.read_csv(RUTA_PATENTES, encoding='latin-1')

#     # Preprocesamiento de datos
#     def preprocess_data(publications, patents):
#         # Procesar fechas de publicaciones
#         publications['Publication Date'] = publications['Publication Date'].apply(procesar_fecha_publicacion)
#         publications['Publication Year'] = publications['Publication Date'].dt.year
#         current_year = 2024
#         publications['Years Since Last Publication'] = current_year - publications['Publication Year']
#         publications['New Publication Next Year'] = (publications['Years Since Last Publication'] < 2).astype(int)

#         # Procesar fechas de patentes
#         patents['Filing Date'] = patents['Filing Date'].apply(procesar_fecha_publicacion)
#         patents['Filing Year'] = patents['Filing Date'].dt.year
#         patents['Years Since Last Patent'] = current_year - patents['Filing Year']

#         # Fusionar 'Years Since Last Patent' con publicaciones
#         patent_years = patents.groupby('Inventor')['Years Since Last Patent'].min().reset_index()
#         patent_years.rename(columns={'Years Since Last Patent': 'Years Since Last Patent'}, inplace=True)
#         publications = publications.merge(patent_years, how='left', left_on='Authors', right_on='Inventor')
#         publications['Years Since Last Patent'].fillna(0, inplace=True)
#         publications = publications.drop(columns=['Inventor', 'Publication Date'])
#         patents = patents.drop(columns=['Filing Date'])

#         # Crear etiquetas adicionales (Reemplaza estas etiquetas falsas con tus etiquetas reales)
#         # A continuación, se crean etiquetas aleatorias para ilustrar cómo manejar múltiples salidas
#         publications['specific_topic'] = np.random.randint(0, 2, size=len(publications))
#         publications['conference'] = np.random.randint(0, 2, size=len(publications))
#         publications['citation'] = (publications['Total Citations'] > 10).astype(int)
#         publications['new_patent'] = np.random.randint(0, 2, size=len(publications))
#         publications['post_patent_publication'] = np.random.randint(0, 2, size=len(publications))

#         return publications, patents

#     # Procesar datos
#     publications, patents = preprocess_data(publications, patents)

#     # Crear el modelo
#     def create_model(input_shape):
#         inputs = layers.Input(shape=input_shape)
#         x = layers.Dense(128, activation='relu')(inputs)
#         x = layers.Dense(64, activation='relu')(x)

#         # Definir múltiples salidas
#         output_new_publication = layers.Dense(1, activation='sigmoid', name='new_publication')(x)
#         output_specific_topic = layers.Dense(1, activation='sigmoid', name='specific_topic')(x)
#         output_conference = layers.Dense(1, activation='sigmoid', name='conference')(x)
#         output_citation = layers.Dense(1, activation='sigmoid', name='citation')(x)
#         output_new_patent = layers.Dense(1, activation='sigmoid', name='new_patent')(x)
#         output_post_patent_publication = layers.Dense(1, activation='sigmoid', name='post_patent_publication')(x)

#         model = tf.keras.Model(inputs=inputs, outputs=[
#             output_new_publication,
#             output_specific_topic,
#             output_conference,
#             output_citation,
#             output_new_patent,
#             output_post_patent_publication
#         ])

#         model.compile(optimizer='adam',
#                       loss='binary_crossentropy',
#                       metrics=['accuracy'])
#         return model

#     print("Columnas de Publicaciones:", publications.columns)
#     print("Columnas de Patentes:", patents.columns)

#     # Separar características y etiquetas
#     X = publications.drop(columns=[
#         'New Publication Next Year',
#         'specific_topic',
#         'conference',
#         'citation',
#         'new_patent',
#         'post_patent_publication'
#     ])
#     X = X.select_dtypes(include=[np.number])

#     # Definir y como DataFrame con todas las etiquetas
#     y = publications[['New Publication Next Year',
#                       'specific_topic',
#                       'conference',
#                       'citation',
#                       'new_patent',
#                       'post_patent_publication']]

#     # Verificar que X y y tengan el mismo número de muestras
#     if len(X) != len(y):
#         st.error("El número de muestras en X y y no coincide.")
#         st.stop()

#     # Dividir en conjuntos de entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Convertir y_train y y_test a diccionarios
#     y_train_dict = {col: y_train[col].values for col in y_train.columns}
#     y_test_dict = {col: y_test[col].values for col in y_test.columns}

#     # Escalar las características
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     # Crear el modelo con la forma correcta de las características
#     model = create_model(input_shape=(X_train_scaled.shape[1],))

#     # Entrenar el modelo
#     model.fit(X_train_scaled, y_train_dict, epochs=10, batch_size=32, validation_data=(X_test_scaled, y_test_dict))

#     # Crear la interfaz en Streamlit
#     st.title('Predicción de Publicaciones y Patentes')

#     # Crear una lista de autores únicos
#     unique_authors = publications['Authors'].unique()

#     # Input para seleccionar un autor de la lista
#     selected_author = st.selectbox('Seleccione un Autor', unique_authors)

#     # Filtrar las publicaciones del autor seleccionado
#     author_publications = publications[publications['Authors'] == selected_author]
#     # Filtrar las patentes del autor
#     author_patents = patents[patents['Inventor'] == selected_author]

#     # Obtener los años desde la última publicación del autor
#     if not author_publications.empty:
#         last_publication_year = author_publications['Publication Year'].max()
#         years_since_last_publication = 2024 - last_publication_year
#     else:
#         years_since_last_publication = 0

#     # Obtener los años desde la última patente del autor
#     if not author_patents.empty:
#         last_patent_year = author_patents['Filing Year'].max()
#         years_since_last_patent = 2024 - last_patent_year
#     else:
#         years_since_last_patent = 0

#     # Mostrar los valores calculados en la interfaz
#     st.write(f"Años desde la última publicación: {years_since_last_publication}")
#     st.write(f"Años desde la última patente: {years_since_last_patent}")

#     # Filtrar solo columnas numéricas
#     input_features = author_publications.select_dtypes(include=[np.number])

#     # Asegúrate de que haya al menos una fila
#     if not input_features.empty:
#         input_data = input_features.iloc[-1].copy()  # Copiar para evitar SettingWithCopyWarning
#     else:
#         st.error("No hay datos disponibles para el autor seleccionado.")
#         st.stop()

#     # Agregar 'Years Since Last Publication' y 'Years Since Last Patent' si no están ya en las características
#     # En este caso, ya están incluidas
#     # Verificar que todas las características necesarias están presentes
#     required_features = X.columns
#     missing_features = set(required_features) - set(input_data.index)
#     if missing_features:
#         for feature in missing_features:
#             input_data[feature] = 0  # O un valor apropiado

#     # Reordenar las características para que coincidan con el entrenamiento
#     input_data = input_data[required_features]

#     # Escalar el input_data usando el mismo scaler
#     input_data_scaled = scaler.transform([input_data.values])

#     # Hacer la predicción
#     if st.button('Predecir'):
#         try:
#             predictions = model.predict(input_data_scaled)
#             # Las predicciones estarán en el orden de las salidas definidas
#             st.write(f"Probabilidad de nueva publicación: {predictions[0][0][0]:.2f}")
#             st.write(f"Probabilidad de que sea en un tema específico: {predictions[1][0][0]:.2f}")
#             st.write(f"Probabilidad de que tenga conferencia: {predictions[2][0][0]:.2f}")
#             st.write(f"Probabilidad de ser citada: {predictions[3][0][0]:.2f}")
#             st.write(f"Probabilidad de nueva patente: {predictions[4][0][0]:.2f}")
#             st.write(f"Probabilidad de nueva publicación después de patente: {predictions[5][0][0]:.2f}")
#         except Exception as e:
#             st.error(f"Error en la predicción: {e}")