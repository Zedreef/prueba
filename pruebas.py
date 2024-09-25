from Menu.utilidades import RUTA_PUBLICACIONES, RUTA_PUBLICACIONES_KERAS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import keras.saving as saving

# Cargar el modelo previamente entrenado
model = saving.load_model(RUTA_PUBLICACIONES_KERAS)

# Cargar el archivo RUTA_PUBLICACIONES
ruta_publicaciones = RUTA_PUBLICACIONES
data = pd.read_csv(ruta_publicaciones)

# Mostrar las primeras filas para asegurarnos de que los datos se cargaron correctamente
print(data.head())

# Función para obtener los datos de un autor
def obtener_datos_autor(nombre_autor, data, preprocessor):
    # Filtrar los datos del autor seleccionado
    autor_info = data[data['Authors'] == nombre_autor]
    
    if autor_info.empty:
        print(f"No se encontraron datos para el autor: {nombre_autor}")
        return None

    # Seleccionar las columnas necesarias para el modelo
    columnas_requeridas = [
        'Total Citations', 'Average per Year', '2005', '2006', '2007', '2008', '2009', '2010',
        '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023',
        'Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title'
    ]
    
    # Extraer los datos del autor en el formato correcto
    input_data = autor_info[columnas_requeridas]

    # Imprimir los datos extraídos para validación
    print(f"Datos encontrados para el autor {nombre_autor}:")
    print(input_data)

    # Aplicar el preprocesamiento (estandarización y codificación OneHot)
    input_data_encoded = preprocessor.transform(input_data)
    
    # Convertir csr_matrix a un array denso para el modelo
    input_data_encoded = input_data_encoded.toarray()
    
    return input_data_encoded

# Creación del preprocessor una vez
numeric_features = ['Total Citations', 'Average per Year'] + [str(year) for year in range(2005, 2023)]
categorical_features = ['Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Asegúrate de ajustar tu preprocesador a los datos completos antes de usar la función.
preprocessor.fit(data)

# El usuario ingresa el nombre del autor
nombre_autor = input("Ingrese el nombre del autor a buscar: ")

# Obtener los datos del autor
input_data = obtener_datos_autor(nombre_autor, data, preprocessor)

if input_data is not None:
    # Hacer la predicción usando el modelo cargado previamente

    # Hacer la predicción con el modelo
    probabilidad_publicacion_2024 = model.predict(input_data)
    
    # Mostrar el resultado
    print(f"Probabilidad de que el autor {nombre_autor} publique en 2024: {probabilidad_publicacion_2024[0][0]:.2f}")
