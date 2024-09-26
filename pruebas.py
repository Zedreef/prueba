import re
import time
import unicodedata
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from Menu.utilidades import RUTA_PUBLICACIONES, RUTA_PATENTES

def mostrar_Prueba():

    start_time = time.time()

    # Función para agregar trazados en la gráfica
    def agregar_trazado(fig, x, y, name, color, dash='solid'):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, 
                                marker=dict(color=color), line=dict(color=color, dash=dash)))


    # Cargar solo las columnas necesarias
    df_patentes = pd.read_csv(RUTA_PATENTES, encoding='latin1', usecols=['Inventor', 'Filing Date'])
    df_publicaciones = pd.read_csv(RUTA_PUBLICACIONES, usecols=['Authors', 'Publication Date', 'Title', 'Total Citations'])

    # Normalizar y procesar fechas de forma vectorizada
    df_patentes['Filing Date'] = pd.to_datetime(df_patentes['Filing Date'], format='%d/%m/%Y', dayfirst=True)
    df_patentes['Inventor'] = df_patentes['Inventor'].str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8').str.upper()
    df_publicaciones['Publication Date'] = pd.to_datetime(df_publicaciones['Publication Date'], format='%d/%m/%Y', errors='coerce')
    df_publicaciones['Authors'] = df_publicaciones['Authors'].str.strip()

    # Filtrar datos entre 2005 y 2023 de manera eficiente
    df_publicaciones = df_publicaciones[(df_publicaciones['Publication Date'].dt.year.between(2005, 2023))]
    df_patentes = df_patentes[(df_patentes['Filing Date'].dt.year.between(2005, 2023))]

    # Eliminar duplicados
    df_patentes_unique = df_patentes.drop_duplicates(subset=['Inventor', 'Filing Date'])

    # Agrupar publicaciones por autor y fecha de publicación
    publicaciones_autor_fecha = df_publicaciones.groupby(['Authors', df_publicaciones['Publication Date'].dt.year]).size().reset_index(name='Publicaciones')

    # Calcular publicaciones antes y después
    def calcular_publicaciones(fecha_patente, autor):
        publicaciones = publicaciones_autor_fecha[publicaciones_autor_fecha['Authors'] == autor]
        antes = publicaciones[publicaciones['Publication Date'] < fecha_patente]
        despues = publicaciones[publicaciones['Publication Date'] >= fecha_patente]
        return len(antes), len(despues)

    # Aplicar cálculo a todo el dataframe de forma vectorizada
    df_patentes_unique[['Publicaciones antes', 'Publicaciones después']] = df_patentes_unique.apply(
        lambda row: calcular_publicaciones(row['Filing Date'].year, row['Inventor']),
        axis=1, result_type='expand'
    )

    # Calcular el cambio en publicaciones
    df_patentes_unique['Cambio en Publicaciones'] = df_patentes_unique['Publicaciones después'] - df_patentes_unique['Publicaciones antes']

    # Selección de autor
    autor_seleccionado = st.selectbox("Seleccione un autor", df_patentes_unique['Inventor'].unique())

    # Filtrar por autor seleccionado
    df_autor = df_patentes_unique[df_patentes_unique['Inventor'] == autor_seleccionado]
    df_autor_publicaciones = df_publicaciones[df_publicaciones['Authors'] == autor_seleccionado]

    # Agrupar publicaciones por año
    df_autor_publicaciones['Año'] = df_autor_publicaciones['Publication Date'].dt.year
    publicaciones_por_año = df_autor_publicaciones.groupby('Año').size().reset_index(name='Publicaciones')

    # Agrupar patentes por año
    df_autor['Año'] = df_autor['Filing Date'].dt.year
    patentes_por_año = df_autor.groupby('Año').size().reset_index(name='Patentes')

    # Unir datos de publicaciones y patentes
    datos_combinados = pd.merge(publicaciones_por_año, patentes_por_año, on='Año', how='outer').fillna(0)

    # Crear la gráfica después de realizar todos los cálculos
    fig = go.Figure()

    # Añadir datos de publicaciones y patentes
    agregar_trazado(fig, datos_combinados['Año'], datos_combinados['Publicaciones'], 'Publicaciones', 'blue')
    
    # Mostrar gráfica
    st.plotly_chart(fig)

    # Mostrar resumen
    st.dataframe(df_autor[['Inventor', 'Filing Date', 'Publicaciones antes', 'Publicaciones después', 'Cambio en Publicaciones']], use_container_width=True)

    end_time = time.time()
    st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")

