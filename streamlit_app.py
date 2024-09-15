# Importaciones estándar de Python
import os
import re
import glob
import time
import unicodedata
from io import StringIO

# Librerías de análisis de datos y matemáticas
import pandas as pd
import numpy as np

# Librerías de grafos y comunidades
import networkx as nx
from networkx.algorithms.community import louvain_communities

# Librerías de visualización
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Otras librerías
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
# ----------------------- Definición -------------------------------------------
# Definir las rutas
# ----------------------- Ruta App ---------------------------------------------
ruta_brutos = '/mount/src/prueba/Datos Brutos'
ruta_guardado = '/mount/src/prueba/Datos Completos'
ruta_Publicaciones = 'Analisis/Publicaciones.csv'
# ----------------------- Ruta GitHub ------------------------------------------
#ruta_brutos = '/workspaces/prueba/Datos Brutos'
#ruta_guardado = '/workspaces/prueba/Datos Completos'
#ruta_Publicaciones = 'Analisis/Publicaciones.csv'
# ----------------------- Ruta Colab -------------------------------------------
#ruta_guardado = "/content/drive/MyDrive/Investigadores IA/Archivos Limpios"
#ruta_final = '/content/drive/MyDrive/Investigadores IA/Datos Para Analizar/Publicaciones.csv'
#-------------------------------------------------------------------------------
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
# Los iconos que usa son de Bootstrap Icons
with st.sidebar:
    st.title("Análisis de Investigadores")

    selected = option_menu(
        "Menú",
        ["Inicio", "Buscar Investigador","Comparar investigadores", "Todos los investigadores",
         "Análisis de Coautoría", "Análisis de patentes",
         "Análisis de conferencias"],
        icons=['house', 'search', 'person-arms-up', 'people-fill', 'graph-up',
               'file-earmark-text', 'calendar'],
        menu_icon="clipboard-data-fill",
        default_index=0
    )
    st.text("User: Adrian fdz")
    st.text("Version: 0.0.1")

#-------------------------------------------------------------------------------

#---------------------- Dashboard principal ------------------------------------
if selected == "Inicio":
    # Encabezado
    st.title("📊 Informe de Archivos Procesados")

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
  
  # Función para calcular el índice h
  def calcular_indice_h(df):
    # Ordenar las publicaciones por número de citas en orden descendente
    citas = df['Total Citations'].sort_values(ascending=False).values
    h_index = 0

    # Calcular el índice h
    for i, c in enumerate(citas):
        if c >= i + 1:
            h_index = i + 1
        else:
            break
    return h_index
  
  # Función para calcular el resumen de citas
  def calcular_resumen(df):
    resumen = []

    # Obtener los autores únicos
    autores = df['Authors'].unique()
    
    for autor in autores:
        # Filtrar los datos para el autor actual
        df_autor = df[df['Authors'] == autor]

        # Calcular la suma de 'Total Citations', el promedio de 'Average per Year', y el índice h
        total_citations = df_autor['Total Citations'].sum()
        average_per_year = df_autor['Average per Year'].mean()
        h_index = calcular_indice_h(df_autor)

        # Agregar los datos al resumen
        resumen.append({
            'Total Citas': total_citations,
            'Promedio Año': average_per_year,
            'Índice h': h_index
        })

    # Convertir el resumen en un DataFrame
    return pd.DataFrame(resumen)

  # Función para graficar las citas y publicaciones por año
  def graficar_citas_publicaciones(df_autor, autor_seleccionado):
      # Extraer el año de 'Publication Date' usando una expresión regular para capturar solo el año
      df_autor['Year'] = df_autor['Publication Date'].apply(lambda x: re.search(r'\d{4}', str(x)).group() if re.search(r'\d{4}', str(x)) else None)
      
      # Eliminar las filas donde no se pudo extraer un año válido
      df_autor = df_autor[df_autor['Year'].notna()].copy()
      
      # Convertir la columna 'Year' a entero
      df_autor['Year'] = df_autor['Year'].astype(int)
      
      # Agrupar por el año y contar el número de publicaciones
      publicaciones_por_año = df_autor.groupby('Year').size()  # Número de publicaciones por año
      
      # Agrupar por el año y sumar el total de citas
      citas_por_año = df_autor.groupby('Year')['Total Citations'].sum()  # Total de citas por año
      
      # Obtener los años únicos para la gráfica
      años = sorted(publicaciones_por_año.index)
      
      # Obtener el valor máximo para escalar ejes
      max_publicaciones = publicaciones_por_año.max()
      max_citas = citas_por_año.max()
      
      # Crear la gráfica con Plotly
      fig = go.Figure()

      # Agregar las barras para las publicaciones (Eje izquierdo)
      fig.add_trace(go.Bar(
          x=años,
          y=publicaciones_por_año,
          name='Publications',
          yaxis='y1'
      ))

      # Agregar la línea para las citas (Eje derecho)
      fig.add_trace(go.Scatter(
          x=años,
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
              side='left',
              range=[0, max_publicaciones + 1] 
          ),
          yaxis2=dict(
              title='Times Cited',
              overlaying='y',
              side='right',
              range=[0, max_citas + 1]
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
          st.write(f"Datos de {autor_seleccionado}: ")
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
#-------------------------------------------------------------------------------

#----------------Comparación de autores------------------------------------------
if selected == "Comparar investigadores":
    # Función para procesar los datos filtrados por rango de fechas
    def procesar_autores(df, cantidad_autores, fecha_inicio, fecha_fin):
        # Filtrar por rango de fechas
        df_filtrado = df[(df['Publication Date'].notna()) & 
                         (pd.to_datetime(df['Publication Date'], errors='coerce').dt.year >= fecha_inicio) &
                         (pd.to_datetime(df['Publication Date'], errors='coerce').dt.year <= fecha_fin)]

        # Filtrar solo las columnas de interés
        columnas_especificas = ['Title', 'Authors', 'Source Title', 'Publication Date', 'Total Citations', 'Average per Year']
        columnas_de_años = [col for col in df.columns if col.isdigit() and int(col) >= 1960]
        columnas_de_años_validas = [col for col in columnas_de_años if (df_filtrado[col].notna() & (df_filtrado[col] != 0)).any()]

        # Agrupar por autor y contar cuántas publicaciones tiene cada autor
        df_agrupado = df_filtrado.groupby('Authors').size().reset_index(name='Publicaciones')

        # Seleccionar los autores con más publicaciones (según la cantidad seleccionada por el usuario)
        autores_seleccionados = df_agrupado.nlargest(cantidad_autores, 'Publicaciones')['Authors']

        # Filtrar el DataFrame original por los autores seleccionados
        df_final = df_filtrado[df_filtrado['Authors'].isin(autores_seleccionados)].copy()

        # Mantener solo las columnas relevantes
        df_final = pd.concat([df_final[columnas_especificas], df_final[columnas_de_años_validas]], axis=1)

        return df_final

    # Función para calcular el índice h
    def calcular_indice_h(df):
        # Ordenar las publicaciones por número de citas en orden descendente
        citas = df['Total Citations'].sort_values(ascending=False).values
        h_index = 0
        
        # Calcular el índice h
        for i, c in enumerate(citas):
            if c >= i + 1:
                h_index = i + 1
            else:
                break
        return h_index

    # Función para calcular el resumen de citas para cada autor
    def calcular_resumen(df):
        resumen = []

        # Obtener los autores únicos
        autores = df['Authors'].unique()
        
        for autor in autores:
            # Filtrar los datos para el autor actual
            df_autor = df[df['Authors'] == autor]

            # Calcular la suma de 'Total Citations', el promedio de 'Average per Year', y el índice h
            total_citations = df_autor['Total Citations'].sum()
            average_per_year = df_autor['Average per Year'].mean()
            h_index = calcular_indice_h(df_autor)

            # Agregar los datos al resumen
            resumen.append({
                'Autor': autor,
                'Total de Citas': total_citations,
                'Promedio por Año': average_per_year,
                'Índice h': h_index
            })

        # Convertir el resumen en un DataFrame
        return pd.DataFrame(resumen)

    # Función para graficar citas y publicaciones por año
    def graficar_citas_publicaciones(df):
        df['Year'] = df['Publication Date'].apply(lambda x: re.search(r'\d{4}', str(x)).group() if re.search(r'\d{4}', str(x)) else None)
        df = df[df['Year'].notna()].copy()
        df['Year'] = df['Year'].astype(int)
        publicaciones_por_año = df.groupby('Year').size()
        citas_por_año = df.groupby('Year')['Total Citations'].sum()

        años = sorted(publicaciones_por_año.index)
        max_publicaciones = publicaciones_por_año.max()
        max_citas = citas_por_año.max()

        fig = go.Figure()
        for autor in df['Authors'].unique():
            df_autor = df[df['Authors'] == autor]

            publicaciones_por_año = df_autor.groupby('Year').size()
            citas_por_año = df_autor.groupby('Year')['Total Citations'].sum()

            # Agregar las barras para las publicaciones por autor (Eje izquierdo)
            fig.add_trace(go.Bar(
                x=publicaciones_por_año.index,
                y=publicaciones_por_año,
                name=f'Publications ({autor})',
                yaxis='y1'
            ))

            # Agregar la línea para las citas por autor (Eje derecho)
            fig.add_trace(go.Scatter(
                x=citas_por_año.index,
                y=citas_por_año,
                mode='lines+markers',
                name=f'Times Cited ({autor})',
                yaxis='y2'
            ))

        fig.update_layout(
            title="Times Cited and Publications Over Time",
            xaxis_title='Year',
            yaxis=dict(title='Publications', side='left', range=[0, max_publicaciones + 1]),
            yaxis2=dict(title='Times Cited', overlaying='y', side='right', range=[0, max_citas + 1]),
            legend=dict(orientation="v", yanchor="middle",y=0.5, xanchor="left", x=1.1)
        )

        st.plotly_chart(fig)

    # Cargar el archivo CSV y procesar
    df_publicaciones = pd.read_csv(ruta_Publicaciones)

    # Configuración de la app en Streamlit
    st.title("Análisis de Publicaciones por Rango de Fechas")

    col3, col4 = st.columns([0.25, 1])
    with col3:
        # Seleccionar cantidad de autores
        cantidad_autores = st.number_input("Selecciona la cantidad de autores", min_value=1, value=5)
    with col4:
        # Seleccionar rango de fechas
        rango_fechas = st.slider("Selecciona el rango de fechas", min_value=1960, max_value=2024, value=(2000, 2020))

    # Procesar los datos según la cantidad de autores y el rango de fechas seleccionado
    if cantidad_autores and rango_fechas:
        try:
            df_resultado = procesar_autores(df_publicaciones, cantidad_autores, rango_fechas[0], rango_fechas[1])

            st.write(f"Datos de los {cantidad_autores} autores más productivos entre {rango_fechas[0]} y {rango_fechas[1]}: ")
            st.dataframe(df_resultado)

            # Calcular el resumen
            df_resumen = calcular_resumen(df_resultado)

            col1, col2 = st.columns([0.5, 1])

            with col1:
                # Mostrar los autores seleccionados en una tabla extra
                st.write(f"Autores encontrados entre {rango_fechas[0]} y {rango_fechas[1]}:")
                st.table(df_resumen)

            with col2:
                graficar_citas_publicaciones(df_resultado)

        except Exception as e:
            st.error(f"Error procesando los datos: {e}")

#-------------------------------------------------------------------------------


#--------Número de Artículos y Citas Totales por Autor--------------------------
if selected == "Todos los investigadores":
    st.subheader("Todos los investigadores")
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

#---------------------------Análisis de Coautoría-------------------------------

if selected == "Análisis de Coautoría":

    # Obtener la lista de archivos .txt usando glob
    archivos_txt = glob.glob(os.path.join(ruta_brutos, '*.txt'))

    # Función para procesar los autores y reemplazar
    def procesar_autores(df, nombre_archivo):
        # Renombrar columna si existe
        if 'Corporate Authors' in df.columns:
            df = df.rename(columns={'Corporate Authors': 'co-author'})
        df['co-author'] = df['co-author'].fillna('') + '; ' + df['Authors'].fillna('')
        
        # Reemplazar los autores por el nombre del archivo sin extensión
        df['Authors'] = os.path.splitext(nombre_archivo)[0]
        
        # Seleccionar solo las columnas Title, Authors y co-author
        columnas_a_mantener = ['Title', 'Authors', 'co-author']
        columnas_existentes = [col for col in columnas_a_mantener if col in df.columns]
        
        # Si alguna de las columnas no existe, puedes manejarlo (por ejemplo, con un log)
        if len(columnas_existentes) < len(columnas_a_mantener):
            columnas_faltantes = set(columnas_a_mantener) - set(columnas_existentes)
            print(f"Advertencia: Columnas faltantes {columnas_faltantes}")
        
        # Retornar solo las columnas que existen en el DataFrame
        df = df[columnas_existentes]
        
        return df


    # Función para limpiar texto
    def limpiar_texto(texto):
        # Normaliza el texto a forma canónica de descomposición
        texto = unicodedata.normalize('NFKD', texto)
        # Elimina los acentos (diacríticos), pero mantiene otros caracteres especiales
        texto = ''.join(c for c in texto if not unicodedata.combining(c))
        # Elimina caracteres no imprimibles y de control
        texto = ''.join(c for c in texto if c.isprintable())
        # Elimina números
        texto = re.sub(r'\d+', '', texto)
        # Elimina múltiples espacios y palabras de una sola letra
        texto = ' '.join(word for word in re.sub(r'\s+', ' ', texto).strip().split() if len(word) > 1)
        
        return texto

    def create_node_trace(G, pos, node_colors='lightblue', node_size=20):
        """Crea un trace de Plotly para los nodos de un grafo."""
        x_nodes, y_nodes, node_texts = zip(*[(pos[node][0], pos[node][1], node) for node in G.nodes()])
        return go.Scatter(x=x_nodes, y=y_nodes,
                          mode='markers+text',
                          text=node_texts,
                          textposition='top center',
                          marker=dict(size=node_size, color=node_colors, line=dict(width=2, color='black')),
                          textfont=dict(size=10))

    def create_edge_trace(G, pos, edge_color='gray'):
        """Crea un trace de Plotly para las aristas de un grafo."""
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])  # Añadimos None para separar las aristas
            edge_y.extend([y0, y1, None])
        return go.Scatter(x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=0.5, color=edge_color),
                        hoverinfo='none')  # hoverinfo desactivado para las aristas

    @st.cache_data
    def procesar_archivos(archivos_txt):
        # Inicializar una lista para almacenar los DataFrames
        dfs = []
        
        for archivo_txt in archivos_txt:
            try:
                # Leer el archivo directamente con pandas
                df = pd.read_csv(archivo_txt, skiprows=2, quotechar='"')
                
                # Procesar los autores
                df = procesar_autores(df, os.path.basename(archivo_txt))
                
                # Añadir el DataFrame procesado a la lista
                dfs.append(df)
            
            except Exception as e:
                logging.error(f"Error procesando el archivo {os.path.basename(archivo_txt)}: {e}")
        
        # Concatenar todos los DataFrames en uno solo
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    # Procesar archivos
    df_todos = procesar_archivos(archivos_txt)

    if df_todos.empty:
        st.write("No se procesaron archivos.")
    else:
        #st.write("DataFrame combinado:")
        #st.write(df_todos.head())

        start_time = time.time()

        # Crear la red de coautoría
        G = nx.Graph()

        for _, row in df_todos.iterrows():
            autor_principal = row['Authors']
            coautores = row['co-author'].split('; ')
            
            for coautor in coautores:
                if coautor:  # Asegurarse de que el coautor no esté vacío
                    G.add_edge(autor_principal, coautor)

        st.subheader('Detección de Comunidades con mas Coautoría')

        # Algoritmo de Louvain
        comunidades_louvain = louvain_communities(G)

        # Mostrar cada comunidad en una gráfica separada
        for i, comunidad in enumerate(comunidades_louvain):
            G_comunidad = G.subgraph(comunidad)
            pos_comunidad = nx.kamada_kawai_layout(G_comunidad)

            # Personalizar los colores y tamaños de los nodos según el grado
            node_sizes = [G_comunidad.degree(node) * 5 for node in G_comunidad.nodes()]

            # Encontrar el nodo (autor) con más conexiones (mayor grado) en la comunidad
            autor_principal = max(G_comunidad.nodes, key=lambda node: G_comunidad.degree(node))

            # Crear la figura de Plotly para la comunidad
            fig_comunidad = go.Figure()

            # Añadir nodos
            fig_comunidad.add_trace(create_node_trace(G_comunidad, pos_comunidad, node_size=node_sizes))

            # Añadir aristas
            fig_comunidad.add_trace(create_edge_trace(G_comunidad, pos_comunidad))

            # Configurar la visualización para la comunidad
            fig_comunidad.update_layout(title=f'Comunidad liderada por {autor_principal}',
                                        showlegend=False,
                                        xaxis=dict(showgrid=False, zeroline=False),
                                        yaxis=dict(showgrid=False, zeroline=False))
            
            # Mostrar la gráfica de la comunidad en Streamlit
            st.plotly_chart(fig_comunidad)

        end_time = time.time()
        st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")
#-------------------------------------------------------------------------------
