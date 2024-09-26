import re
import time
import unicodedata
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from Menu.utilidades import RUTA_PUBLICACIONES, RUTA_PATENTES

def mostrar_analisis_patentes():

    st.markdown("<h2 style='text-align: center;'>Análisis de Patentes por Autor</h2>", unsafe_allow_html=True)
    start_time = time.time()

    # Cargar archivos CSV
    df_patentes = pd.read_csv(RUTA_PATENTES, encoding='latin1')
    df_publicaciones = pd.read_csv(RUTA_PUBLICACIONES)

    # Normalización de nombres
    def normalizar_nombre(nombre):
        return unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('utf-8').upper()

    # Calcular publicaciones antes y después de cada patente
    def calcular_publicaciones(df, fecha_patente, autor):
        antes = df[(df['Authors'] == autor) & (df['Publication Date'] < fecha_patente)]
        despues = df[(df['Authors'] == autor) & (df['Publication Date'] >= fecha_patente)]
        return len(antes), len(despues)

    # Procesar datos
    df_patentes['Filing Date'] = pd.to_datetime(df_patentes['Filing Date'], format='%d/%m/%Y', dayfirst=True).dt.normalize()
    df_patentes['Inventor'] = df_patentes['Inventor'].apply(normalizar_nombre)
    df_publicaciones['Authors'] = df_publicaciones['Authors'].str.strip()
    df_publicaciones['Publication Date'] = pd.to_datetime(df_publicaciones['Publication Date'], format='%d/%m/%Y', errors='coerce')

    # Filtrar datos entre 2005 y 2023
    df_publicaciones = df_publicaciones[(df_publicaciones['Publication Date'].dt.year >= 2005) & (df_publicaciones['Publication Date'].dt.year <= 2023)]
    df_patentes = df_patentes[(df_patentes['Filing Date'].dt.year >= 2005) & (df_patentes['Filing Date'].dt.year <= 2023)]

    # Eliminar duplicados en patentes
    df_patentes_unique = df_patentes.drop_duplicates(subset=['Inventor', 'Filing Date'])

    # Aplicar cálculo de publicaciones antes y después
    df_patentes_unique[['Publicaciones antes', 'Publicaciones después']] = df_patentes_unique.apply(
        lambda row: calcular_publicaciones(df_publicaciones, row['Filing Date'], row['Inventor']),
        axis=1, result_type='expand'
    )

    # Calcular 'Cambio en Publicaciones'
    df_patentes_unique['Cambio en Publicaciones'] = df_patentes_unique['Publicaciones después'] - df_patentes_unique['Publicaciones antes']

    # Selección del autor
    autor_seleccionado = st.selectbox("Seleccione un autor", df_patentes_unique['Inventor'].unique())

    # Filtrar por autor seleccionado
    df_autor = df_patentes_unique[df_patentes_unique['Inventor'] == autor_seleccionado]

    # Filtrar publicaciones del autor
    df_autor_publicaciones = df_publicaciones[df_publicaciones['Authors'] == autor_seleccionado]

    # Agrupar publicaciones por año
    df_autor_publicaciones['Año'] = df_autor_publicaciones['Publication Date'].dt.year
    publicaciones_por_año = df_autor_publicaciones.groupby('Año')['Publication Date'].count().reset_index(name='Publicaciones')

    # Agrupar patentes por año
    df_autor['Año'] = df_autor['Filing Date'].dt.year
    patentes_por_año = df_autor.groupby('Año').size().reset_index(name='Patentes')

    # Combinar datos de publicaciones y patentes
    datos_combinados = pd.merge(publicaciones_por_año, patentes_por_año, on='Año', how='outer').fillna(0)

    # Crear gráfico de barras para comparar publicaciones antes y después de las patentes
    fig = go.Figure()

    # Añadir barras para las publicaciones antes de la patente
    fig.add_trace(go.Bar(
        x=df_autor['Filing Date'].dt.year,
        y=df_autor['Publicaciones antes'],
        name='Publicaciones Antes',
        marker_color='blue'
    ))

    # Añadir barras para las publicaciones después de la patente
    fig.add_trace(go.Bar(
        x=df_autor['Filing Date'].dt.year,
        y=df_autor['Publicaciones después'],
        name='Publicaciones Después',
        marker_color='green'
    ))

    # Añadir líneas verticales para las patentes
    for year in df_autor['Filing Date'].dt.year:
        fig.add_vline(x=year, line_dash="dash", line_color="red", annotation_text=f"Patente {year}")

    # Añadir línea de tendencia para publicaciones antes de la patente
    if len(publicaciones_por_año) > 1:
        X = publicaciones_por_año['Año'].values.reshape(-1, 1)
        y = publicaciones_por_año['Publicaciones'].values
        modelo = LinearRegression().fit(X, y)
        predicciones = modelo.predict(X)
        fig.add_trace(go.Scatter(x=publicaciones_por_año['Año'], y=predicciones, mode='lines', name='Tendencia Publicaciones', line=dict(color='orange')))

    # Personalización del gráfico
    fig.update_layout(
        title=f"Comparación de Publicaciones Antes y Después de Patentes - {autor_seleccionado}",
        xaxis_title="Año",
        yaxis_title="Número de Publicaciones",
        barmode='group',  # Agrupa las barras de publicaciones antes y después
        legend_title="Leyenda"
    )

    # Mostrar gráfico
    st.plotly_chart(fig)

    # **Agregar tabla con la información de patentes y publicaciones**
    st.subheader(f"Resumen de Patentes y Publicaciones de {autor_seleccionado}")

    col1, col2 = st.columns([0.5, 1])

    with col1:

        # Crear una tabla con las columnas relevantes
        tabla_resumen = df_autor[['Inventor', 'Filing Date', 'Publicaciones antes', 'Publicaciones después', 'Cambio en Publicaciones']]

        # Mostrar la tabla en Streamlit
        st.dataframe(tabla_resumen)

    with col2:

        # Filtrar las publicaciones del autor seleccionado para la tabla
        df_autor_publicaciones = df_publicaciones[df_publicaciones['Authors'] == autor_seleccionado]

        publicaciones_tabla = df_autor_publicaciones[['Title', 'Publication Date', 'Total Citations']]

        # Mostrar la tabla de publicaciones en Streamlit
        st.dataframe(publicaciones_tabla)

    # Calcular el tiempo de ejecución
    end_time = time.time()
    st.write(f"Tiempo total de ejecución: {end_time - start_time:.2f} segundos")