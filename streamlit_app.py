import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu
# ----------------------- Funciones --------------------------------------------
# Definir la función para procesar los archivos
def procesar_archivos(carpeta):
    correctos = 0
    incorrectos = 0

    archivos_incorrectos = []

    for filename in os.listdir(carpeta):
        if filename.endswith(".txt"):
            ruta_archivo = os.path.join(carpeta, filename)

            try:
                df = pd.read_csv(ruta_archivo, sep=',', quotechar='"',
                                 engine='python')

                correctos += 1
            except Exception as e:
                incorrectos += 1
                archivos_incorrectos.append(filename)

    return correctos, incorrectos, archivos_incorrectos

# Función para generar las estadísticas por autor
def procesar_estadisticas_autores(ruta_final):
    # Cargue los datos del archivo CSV
    data = pd.read_csv(ruta_final)

    # Obtener la lista de columnas de años presentes en el DataFrame
    year_columns = [col for col in data.columns if col.isdigit()]

    # Sumar las citas por año presente en el DataFrame
    data['Sum Of Times Cited'] = data[year_columns].fillna(0).sum(axis=1)

    # Contar el número de publicaciones (títulos) por autor
    publications_per_author = data.groupby('Authors')['Title'].count().reset_index()
    publications_per_author.columns = ['Authors', 'Publications']

    # Sumar las citas totales por autor
    citations_per_author = data.groupby('Authors')['Sum Of Times Cited'].sum().reset_index()

    # Combinar el total de citas y publicaciones por autor
    author_stats = pd.merge(publications_per_author, citations_per_author, on='Authors')

    # Eliminar autores vacíos o no válidos
    author_stats = author_stats[author_stats['Authors'].notna()]
    return author_stats
# ----------------------- Definicion -------------------------------------------

# Definir las rutas
ruta_guardado = '/mount/src/prueba/Datos Completos'
ruta_Publicaciones = 'Analisis/Publicaciones.csv'
#ruta_guardado = "/content/drive/MyDrive/Investigadores IA/Archivos Limpios"
#ruta_final = '/content/drive/MyDrive/Investigadores IA/Datos Para Analizar/Publicaciones.csv'

# Procesar los archivos
correctos, incorrectos, archivos_incorrectos = procesar_archivos(ruta_guardado)
author_stats = procesar_estadisticas_autores(ruta_Publicaciones)

# Filtrar y ordenar los datos de los autores
min_articles = 1
min_citations = 0
filtered_stats = author_stats[
    (author_stats['Publications'] >= min_articles) |
    (author_stats['Sum Of Times Cited'] >= min_citations)
]
filtered_stats = filtered_stats.sort_values(by='Sum Of Times Cited',
                                            ascending=False)

#-------------------------------------------------------------------------------

# Configuración de la página
st.set_page_config(page_title="Investigadores", layout="wide")

#----------------------------- Menú lateral ------------------------------------
with st.sidebar:
    st.title("Investigadores")

    selected = option_menu(
        "Navegación",
        ["Inicio", "Buscar Investigador", "Todos los investigadores",
         "Análisis de citas", "Análisis de patentes",
         "Análisis de conferencias"],
        icons=['house', 'search', 'people', 'graph-up',
               'file-earmark-text', 'calendar'],
        menu_icon="cast",
        default_index=0
    )
    st.text("User: Adrian fdz")
    st.text("Version: 0.0.1")

# Usar 'selected' para controlar qué se muestra en la página
st.write(f"Has seleccionado: {selected}")
#-------------------------------------------------------------------------------

#---------------------- Dashboard principal ------------------------------------
if selected == "Inicio":
    # Encabezado
    st.title("Inicio")

    # Métricas clave con los resultados de `procesar_archivos`
    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos correctos", correctos)
    col2.metric("Archivos con error", incorrectos)
    col3.metric("Archivos en total", correctos + incorrectos)

    # Mostrar detalles adicionales de los archivos incorrectos si existen
    if incorrectos > 0:
        st.subheader("Archivos con error:")
        for archivo in archivos_incorrectos:
            st.write(f"- {archivo}")

#-------Gráfico de barras para mostrar archivos correctos vs incorrectos--------
    st.subheader("Gráfico de Archivos Procesados")

    # Datos para la gráfica
    data = {
        'Categoría': ['Correctos', 'Incorrectos'],
        'Cantidad': [correctos, incorrectos]
    }
    df = pd.DataFrame(data)

    # Crear gráfica de barras
    fig = px.bar(df, x='Categoría', y='Cantidad', color='Categoría',
                 title="Archivos Procesados Correctamente vs Incorrectamente",
                 labels={'Cantidad': 'Número de Archivos'},
                 height=400)

    st.plotly_chart(fig)

#--------Número de Artículos y Citas Totales por Autor--------------------------
    fig = go.Figure()

    # Añadir la serie de datos para el número de artículos (gráfico de barras)
    fig.add_trace(go.Bar(
        x=filtered_stats['Authors'],
        y=filtered_stats['Publications'],
        name='Número de Artículos',
        marker_color='blue',
        opacity=0.6,
        yaxis='y1'  # Asocia esta serie al primer eje Y
    ))

    # Añadir la serie de datos para las citas totales (gráfico de líneas)
    fig.add_trace(go.Scatter(
        x=filtered_stats['Authors'],
        y=filtered_stats['Sum Of Times Cited'],
        name='Citas Totales',
        mode='lines+markers',
        line=dict(color='red'),
        marker=dict(size=8),
        yaxis='y2'
    ))

    # Actualizar el diseño del gráfico
    fig.update_layout(
        title='Número de Artículos y Citas Totales por Autor',
        xaxis_title='Autores',
        yaxis_title='Número de Artículos',
        yaxis=dict(
            title='Número de Artículos',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            side='left'
        ),
        yaxis2=dict(
            title='Citas Totales',
            titlefont=dict(color='red'),
            tickfont=dict(color='red'),
            overlaying='y',
            side='right'
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=filtered_stats['Authors'],
            ticktext=filtered_stats['Authors'],
            tickangle=-90
        ),
        autosize=True,
        height=800
    )
    st.plotly_chart(fig)
#-------------------------------------------------------------------------------

if selected == "Buscar Investigador":
  # Función para procesar los datos del autor seleccionado
  def procesar_autor(df, autor_seleccionado):
      # Filtrar el DataFrame por autor seleccionado y eliminar filas con 'Authors' vacíos
      df_filtrado = df[df['Authors'].notna() & (df['Authors'] != '') & (df['Authors'] == autor_seleccionado)].copy()

      # Mantener solo las columnas específicas que te interesan
      columnas_especificas = ['Title', 'Authors', 'Source Title', 'Publication Date', 'Total Citations', 'Average per Year']
      
      # Filtrar dinámicamente columnas de años (desde 1960 en adelante)
      columnas_de_años = [col for col in df.columns if col.isdigit() and int(col) >= 1960]
      
      # Mantener solo las columnas de años que contienen al menos un valor distinto de 0 en el DataFrame filtrado
      columnas_de_años_validas = [col for col in columnas_de_años if (df_filtrado[col].notna() & (df_filtrado[col] != 0)).any()]
      
      # Combinar las columnas específicas con las columnas de años válidas
      df_final = pd.concat([df_filtrado[columnas_especificas], df_filtrado[columnas_de_años_validas]], axis=1)

      return df_final

  # Función para calcular el resumen de citas
  def calcular_resumen(df):
      # Calcular la suma de 'Total Citations' y el promedio de 'Average per Year'
      total_citations = df['Total Citations'].sum()
      average_per_year = df['Average per Year'].mean()

      # Crear y retornar un DataFrame resumen
      return pd.DataFrame({
          'Métrica': ['Total de Citas', 'Promedio por Año'],
          'Valor': [total_citations, average_per_year]
      })

  # Función para graficar las citas y publicaciones por año
  def graficar_citas_publicaciones(df_autor, autor_seleccionado):
      # Extraer los años de las columnas y convertirlas en formato numérico
      columnas_de_años = [col for col in df_autor.columns if col.isdigit() and int(col) >= 1960]
      
      # Crear DataFrame para Publicaciones y Citas
      publicaciones_por_año = (df_autor[columnas_de_años] > 0).sum()  # Número de publicaciones (conteo de > 0 por año)
      citas_por_año = df_autor[columnas_de_años].sum()  # Total de citas por año
      
      # Crear la gráfica de líneas con Plotly
      fig = go.Figure()

      # Agregar la línea para las publicaciones (Eje izquierdo)
      fig.add_trace(go.Scatter(
          x=columnas_de_años,
          y=publicaciones_por_año,
          mode='lines+markers',
          name='Publications',
          yaxis='y1'
      ))

      # Agregar la línea para las citas (Eje derecho)
      fig.add_trace(go.Scatter(
          x=columnas_de_años,
          y=citas_por_año,
          mode='lines+markers',
          name='Times Cited',
          yaxis='y2'
      ))

      # Configurar los ejes
      fig.update_layout(
          title=f"Times Cited and Publications Over Time for {autor_seleccionado}",
          xaxis_title='Year',
          yaxis=dict(
              title='Publications',
              side='left'
          ),
          yaxis2=dict(
              title='Times Cited',
              overlaying='y',
              side='right'
          ),
          legend=dict(
              orientation="h",
              yanchor="bottom",
              y=1.02,
              xanchor="right",
              x=1
          )
      )

      # Mostrar la gráfica en Streamlit
      st.plotly_chart(fig)

  # Cargar el archivo CSV y eliminar duplicados de autores
  df_publicaciones = pd.read_csv(ruta_Publicaciones)
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
          st.write(f"Datos procesados para {autor_seleccionado}:")
          st.dataframe(df_resultado)
          
          # Calcular y mostrar el resumen
          df_resumen = calcular_resumen(df_resultado)
          st.table(df_resumen)

          #Grafica con los datos
          graficar_citas_publicaciones(df, autor_seleccionado)

      except Exception as e:
          st.error(f"Error procesando los datos: {e}")