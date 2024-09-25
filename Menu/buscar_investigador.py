import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from Menu.utilidades import procesar_autor, calcular_resumen, graficar_citas_publicaciones, RUTA_PUBLICACIONES, RUTA_PUBLICACIONES_KERAS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import keras.saving as saving

# Cargar el modelo previamente entrenado
model = saving.load_model(RUTA_PUBLICACIONES_KERAS)

# Cargar el archivo RUTA_PUBLICACIONES
ruta_publicaciones = RUTA_PUBLICACIONES
data = pd.read_csv(ruta_publicaciones)

# Creación del preprocessor una vez
numeric_features = ['Total Citations', 'Average per Year'] + [str(year) for year in range(2005, 2023)]
categorical_features = ['Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Ajustar el preprocesador a los datos completos
preprocessor.fit(data)

# Función para obtener los datos de un autor
def obtener_datos_autor(nombre_autor, data, preprocessor):
    # Filtrar los datos del autor seleccionado
    autor_info = data[data['Authors'] == nombre_autor]
    
    if autor_info.empty:
        return None

    # Seleccionar las columnas necesarias para el modelo
    columnas_requeridas = [
        'Total Citations', 'Average per Year', '2005', '2006', '2007', '2008', '2009', '2010',
        '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023',
        'Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title'
    ]
    
    # Extraer los datos del autor en el formato correcto
    input_data = autor_info[columnas_requeridas]

    # Aplicar el preprocesamiento (estandarización y codificación OneHot)
    input_data_encoded = preprocessor.transform(input_data)
    
    # Convertir csr_matrix a un array denso para el modelo
    input_data_encoded = input_data_encoded.toarray()
    
    return input_data_encoded

# Función principal para mostrar el análisis y predicción
def mostrar_buscar_investigador(ruta_publicaciones):
    # Cargar el archivo CSV y eliminar duplicados de autores
    df_publicaciones = pd.read_csv(ruta_publicaciones)
    autores_unicos = df_publicaciones['Authors'].drop_duplicates().sort_values()

    # Configuración de la app en Streamlit
    st.title("Análisis de Publicaciones")

    # Selector de autor
    autor_seleccionado = st.selectbox("Selecciona un autor", autores_unicos)

    # Mostrar automáticamente los datos del autor seleccionado
    if autor_seleccionado:
        try:
            # Procesar la información del autor seleccionado
            df_resultado = procesar_autor(df_publicaciones, autor_seleccionado)

            # Mostrar el DataFrame resultante
            st.write(f"Datos de {autor_seleccionado}: ")

            # Obtener los datos para la predicción
            input_data = obtener_datos_autor(autor_seleccionado, df_publicaciones, preprocessor)

            if input_data is not None:
                # Hacer la predicción usando el modelo cargado previamente
                probabilidad_publicacion_2024 = model.predict(input_data)
                probabilidad = probabilidad_publicacion_2024[0][0]

                # Calcular el color basado en la probabilidad
                if probabilidad == 0:
                    color = "red"  
                elif probabilidad == 1:
                    color = "green" 
                else:
                    # Interpolación de colores entre rojo y verde
                    r = int(255 * (1 - probabilidad))  
                    g = int(255 * probabilidad)        
                    color = f"rgb({r}, {g}, 0)"

                # Mostrar la probabilidad de publicación en 2024
                st.markdown(f"Probabilidad de que el autor publique en 2024:  <span style='color: {color}; font-weight: bold;'>{probabilidad:.2f}</span>", unsafe_allow_html=True)

            else:
                st.error(f"No se encontraron datos suficientes.")

            # Datos del autor en la tabla
            st.dataframe(df_resultado)

            # Calcular el resumen
            df_resumen = calcular_resumen(df_resultado)

            col1, col2 = st.columns([0.25, 1])

            with col1:
                # Mostrar el resumen
                st.write(f"Métrica de citas: ")
                st.table(df_resumen)

            with col2:
                # Gráfica con los datos
                graficar_citas_publicaciones(df_resultado, autor_seleccionado)

        except Exception as e:
            st.error(f"Error procesando los datos: {e}")