# Importaciones estándar de Python
import os
import re
import glob
import time
import unicodedata
import logging
import itertools
from io import StringIO
from Menu.utilidades import procesar_archivos, procesar_estadisticas_autores, RUTA_GUARDADO, RUTA_PUBLICACIONES
from Menu.inicio import mostrar_inicio
from Menu.buscar_investigador import mostrar_buscar_investigador


# Librerías de análisis de datos y matemáticas
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

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

# def procesar_archivos(carpeta):
#     correctos = 0
#     incorrectos = 0

#     archivos_incorrectos = []

#     for filename in os.listdir(carpeta):
#         if filename.endswith(".txt"):
#             ruta_archivo = os.path.join(carpeta, filename)

#             try:
#                 df = pd.read_csv(ruta_archivo, sep=',', quotechar='"',
#                                  engine='python')

#                 correctos += 1
#             except Exception as e:
#                 incorrectos += 1
#                 archivos_incorrectos.append(filename)

#     return correctos, incorrectos, archivos_incorrectos

# Función para generar las estadísticas por autor

# def procesar_estadisticas_autores(ruta_final):
#     # Cargue los datos del archivo CSV
#     data = pd.read_csv(ruta_final)

#     # Obtener la lista de columnas de años presentes en el DataFrame
#     year_columns = [col for col in data.columns if col.isdigit()]

#     # Sumar las citas por año presente en el DataFrame
#     data['Sum Of Times Cited'] = data[year_columns].fillna(0).sum(axis=1)

#     # Contar el número de publicaciones (títulos) por autor
#     publications_per_author = data.groupby(
#         'Authors')['Title'].count().reset_index()
#     publications_per_author.columns = ['Authors', 'Publications']

#     # Sumar las citas totales por autor
#     citations_per_author = data.groupby(
#         'Authors')['Sum Of Times Cited'].sum().reset_index()

#     # Combinar el total de citas y publicaciones por autor
#     author_stats = pd.merge(publications_per_author,
#                             citations_per_author, on='Authors')

#     # Eliminar autores vacíos o no válidos
#     author_stats = author_stats[author_stats['Authors'].notna()]
#     return author_stats


# ----------------------- Definición -------------------------------------------
# Definir las rutas
# ----------------------- Ruta App ---------------------------------------------
ruta_brutos = '/mount/src/prueba/Datos Brutos'
ruta_guardado = '/mount/src/prueba/Datos Completos'
ruta_Publicaciones = 'Analisis/Publicaciones.csv'
ruta_Patentes = 'Analisis/Investigadores PATENTES.csv'
# ----------------------- Ruta GitHub ------------------------------------------
# ruta_brutos = '/workspaces/prueba/Datos Brutos'
# ruta_guardado = '/workspaces/prueba/Datos Completos'
# ruta_Publicaciones = 'Analisis/Publicaciones.csv'
# ruta_Patentes = 'Analisis/Investigadores PATENTES.csv'
# -------------------------------------------------------------------------------
# Procesar los archivos
# correctos, incorrectos, archivos_incorrectos = procesar_archivos(ruta_guardado)
# author_stats = procesar_estadisticas_autores(ruta_Publicaciones)

# # Filtrar y ordenar los datos de los autores
# min_articles = 1
# min_citations = 0
# filtered_stats = author_stats[
#     (author_stats['Publications'] >= min_articles) |
#     (author_stats['Sum Of Times Cited'] >= min_citations)
# ]
# filtered_stats = filtered_stats.sort_values(by='Sum Of Times Cited',
#                                             ascending=False)

# -------------------------------------------------------------------------------

# Configuración de la página
st.set_page_config(page_title="Investigadores", layout="wide")

# ----------------------------- Menú lateral ------------------------------------
# Los iconos que usa son de Bootstrap Icons
with st.sidebar:
    st.title("Análisis de Investigadores")

    selected = option_menu(
        "Menú",
        ["Inicio", "Buscar Investigador", "Comparar investigadores", "Todos los investigadores",
         "Análisis de Coautoría", "Análisis de patentes",
         "Análisis de conferencias"],
        icons=['house', 'search', 'person-arms-up', 'people-fill', 'graph-up',
               'file-earmark-text', 'calendar'],
        menu_icon="clipboard-data-fill",
        default_index=0
    )
# -------------------------------------------------------------------------------
# Dashboard principal
if selected == "Inicio":
    mostrar_inicio()
elif selected == "Buscar Investigador":
    mostrar_buscar_investigador(RUTA_PUBLICACIONES)
# --------------------- Buscar Investigador -------------------------------------
# if selected == "Buscar Investigador":


#     # Cargar el archivo CSV y eliminar duplicados de autores
#     df_publicaciones = pd.read_csv(ruta_Publicaciones)
#     autores_unicos = df_publicaciones['Authors'].drop_duplicates(
#     ).sort_values()

#     # Configuración de la app en Streamlit
#     st.title("Análisis de Publicaciones")

#     # Selector de autor
#     autor_seleccionado = st.selectbox("Selecciona un autor", autores_unicos)

#     # Mostrar automáticamente los datos del autor seleccionado
#     if autor_seleccionado:
#         try:
#             # Procesar la información del autor seleccionado
#             df_resultado = procesar_autor(df_publicaciones, autor_seleccionado)

#             # Mostrar el DataFrame resultante
#             st.write(f"Datos de {autor_seleccionado}: ")
#             st.dataframe(df_resultado)

#             # Calcular el resumen
#             df_resumen = calcular_resumen(df_resultado)

#             col1, col2 = st.columns([0.25, 1])

#             with col1:
#                 # Mostrar el resumen
#                 st.write(f"Métrica de citas: ")
#                 st.table(df_resumen)

#             with col2:
#                 # Gráfica con los datos
#                 graficar_citas_publicaciones(df_resultado, autor_seleccionado)

#         except Exception as e:
#             st.error(f"Error procesando los datos: {e}")
# -------------------------------------------------------------------------------

# ----------------Comparación de autores-----------------------------------------
if selected == "Comparar investigadores":
    # Función para procesar los datos filtrados por rango de fechas
    def procesar_autores(df, cantidad_autores, fecha_inicio, fecha_fin):
        # Filtrar por rango de fechas
        df_filtrado = df[(df['Publication Date'].notna()) &
                         (pd.to_datetime(df['Publication Date'], errors='coerce').dt.year >= fecha_inicio) &
                         (pd.to_datetime(df['Publication Date'], errors='coerce').dt.year <= fecha_fin)]

        # Filtrar solo las columnas de interés
        columnas_especificas = ['Title', 'Authors', 'Source Title',
                                'Publication Date', 'Total Citations', 'Average per Year']
        columnas_de_años = [
            col for col in df.columns if col.isdigit() and int(col) >= 1960]
        columnas_de_años_validas = [col for col in columnas_de_años if (
            df_filtrado[col].notna() & (df_filtrado[col] != 0)).any()]

        # Agrupar por autor y contar cuántas publicaciones tiene cada autor
        df_agrupado = df_filtrado.groupby(
            'Authors').size().reset_index(name='Publicaciones')

        # Seleccionar los autores con más publicaciones (según la cantidad seleccionada por el usuario)
        autores_seleccionados = df_agrupado.nlargest(
            cantidad_autores, 'Publicaciones')['Authors']

        # Filtrar el DataFrame original por los autores seleccionados
        df_final = df_filtrado[df_filtrado['Authors'].isin(
            autores_seleccionados)].copy()

        # Mantener solo las columnas relevantes
        df_final = pd.concat(
            [df_final[columnas_especificas], df_final[columnas_de_años_validas]], axis=1)

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
        df['Year'] = df['Publication Date'].apply(lambda x: re.search(
            r'\d{4}', str(x)).group() if re.search(r'\d{4}', str(x)) else None)
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
            yaxis=dict(title='Publications', side='left',
                       range=[0, max_publicaciones + 1]),
            yaxis2=dict(title='Times Cited', overlaying='y',
                        side='right', range=[0, max_citas + 1]),
            legend=dict(orientation="v", yanchor="middle",
                        y=0.5, xanchor="left", x=1.1)
        )

        st.plotly_chart(fig)

    # Cargar el archivo CSV y procesar
    df_publicaciones = pd.read_csv(ruta_Publicaciones)

    # Configuración de la app en Streamlit
    st.title("Análisis de Publicaciones por Rango de Fechas")

    col3, col4 = st.columns([0.25, 1])
    with col3:
        # Seleccionar cantidad de autores
        cantidad_autores = st.number_input(
            "Selecciona la cantidad de autores", min_value=1, value=5)
    with col4:
        # Seleccionar rango de fechas
        rango_fechas = st.slider(
            "Selecciona el rango de fechas", min_value=1960, max_value=2024, value=(2000, 2020))

    # Procesar los datos según la cantidad de autores y el rango de fechas seleccionado
    if cantidad_autores and rango_fechas:
        try:
            df_resultado = procesar_autores(
                df_publicaciones, cantidad_autores, rango_fechas[0], rango_fechas[1])

            st.write(
                f"Datos de los {cantidad_autores} autores más productivos entre {rango_fechas[0]} y {rango_fechas[1]}: ")
            st.dataframe(df_resultado)

            # Calcular el resumen
            df_resumen = calcular_resumen(df_resultado)

            col1, col2 = st.columns([0.5, 1])

            with col1:
                # Mostrar los autores seleccionados en una tabla extra
                st.write(
                    f"Autores encontrados entre {rango_fechas[0]} y {rango_fechas[1]}:")
                st.table(df_resumen)

            with col2:
                graficar_citas_publicaciones(df_resultado)

        except Exception as e:
            st.error(f"Error procesando los datos: {e}")

# -------------------------------------------------------------------------------


# -------- Número de Artículos y Citas Totales por Autor ------------------------
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
# -------------------------------------------------------------------------------

# -------------------------- Análisis de Coautoría ------------------------------

if selected == "Análisis de Coautoría":

    st.subheader('Detección de comunidades con Algoritmo de Louvain')
    st.text('Este algoritmo agrupa nodos del grafo en comunidades basadas en la densidad de sus conexiones.')

    # Obtener la lista de archivos .txt usando glob
    archivos_txt = glob.glob(os.path.join(ruta_brutos, '*.txt'))

    # Función para procesar los autores y reemplazar
    def procesar_autores(df, nombre_archivo):
        if 'Corporate Authors' in df.columns:
            df = df.rename(columns={'Corporate Authors': 'co-author'})
        df['co-author'] = df['co-author'].fillna('') + \
            '; ' + df['Authors'].fillna('')
        df['Authors'] = os.path.splitext(nombre_archivo)[0]
        columnas_a_mantener = ['Title', 'Authors', 'co-author']
        columnas_existentes = [
            col for col in columnas_a_mantener if col in df.columns]
        if len(columnas_existentes) < len(columnas_a_mantener):
            columnas_faltantes = set(
                columnas_a_mantener) - set(columnas_existentes)
            print(f"Advertencia: Columnas faltantes {columnas_faltantes}")
        df = df[columnas_existentes]
        return df

    # Función para limpiar texto
    def limpiar_texto(texto):
        texto = unicodedata.normalize('NFKD', texto)
        texto = ''.join(c for c in texto if not unicodedata.combining(c))
        texto = ''.join(c for c in texto if c.isprintable())
        texto = re.sub(r'\d+', '', texto)
        texto = ' '.join(word for word in re.sub(
            r'\s+', ' ', texto).strip().split() if len(word) > 1)
        return texto

    # Crea un objeto Scatter de Plotly para los nodos del grafo.
    def create_node_trace(G, pos, node_colors, node_size=10):
        x_nodes = []
        y_nodes = []
        node_texts = []
        for node in G.nodes():
            x, y = pos[node]
            x_nodes.append(x)
            y_nodes.append(y)
            node_texts.append(node)
        return go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers',
            marker=dict(size=node_size, color=node_colors,
                        line=dict(width=1, color='black')),
            hovertext=node_texts,
            hoverinfo='text'
        )

    # Crea un objeto Scatter de Plotly para las aristas (conexiones) del grafo.
    def create_edge_trace(G, pos, edge_color='gray'):
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        return go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=0.5, color=edge_color),
            hoverinfo='none'
        )

    @st.cache_data
    def procesar_archivos(archivos_txt):
        dfs = []
        for archivo_txt in archivos_txt:
            try:
                df = pd.read_csv(archivo_txt, skiprows=2, quotechar='"')
                df = procesar_autores(df, os.path.basename(archivo_txt))
                dfs.append(df)
            except Exception as e:
                logging.error(
                    f"Error procesando el archivo {os.path.basename(archivo_txt)}: {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()

    start_time = time.time()

    # Procesar archivos
    df_todos = procesar_archivos(archivos_txt)

    if df_todos.empty:
        st.write("No se procesaron archivos.")
    else:
        # Crear la red de coautoría
        G = nx.Graph()

        for _, row in df_todos.iterrows():
            autor_principal = row['Authors']
            coautores = row['co-author'].split('; ')
            for coautor in coautores:
                if coautor:
                    G.add_edge(autor_principal, coautor)

        col1, col2 = st.columns([0.5, 1])

        with col1:
            # Aplicar filtros interactivos
            grado_minimo = st.slider(
                'Grado mínimo de los nodos', min_value=1, max_value=10, value=5)
            nodos_filtrados = [n for n, d in G.degree() if d >= grado_minimo]
            G_filtrado = G.subgraph(nodos_filtrados)

            # Detectar comunidades
            comunidades_louvain = louvain_communities(G_filtrado)
            comunidades_louvain_ordenadas = sorted(
                comunidades_louvain, key=lambda c: len(c), reverse=True)
            numero_comunidades_a_mostrar = st.slider(
                'Número de comunidades a mostrar', min_value=1, max_value=len(comunidades_louvain_ordenadas), value=5)
            comunidades_seleccionadas = comunidades_louvain_ordenadas[:numero_comunidades_a_mostrar]

            # Asignar un color a cada comunidad
            colors = itertools.cycle(px.colors.qualitative.Plotly)
            comunidad_por_nodo = {}
            for i, comunidad in enumerate(comunidades_seleccionadas):
                color = next(colors)
                for nodo in comunidad:
                    comunidad_por_nodo[nodo] = color

            # Generar posiciones para todos los nodos
            pos = nx.spring_layout(G_filtrado, k=0.15, iterations=20)

            # Crear trazas de nodos y aristas
            node_colors = [comunidad_por_nodo.get(
                nodo, 'grey') for nodo in G_filtrado.nodes()]
            node_trace = create_node_trace(G_filtrado, pos, node_colors)

            edge_trace = create_edge_trace(G_filtrado, pos)

            # Crear la figura
            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Red de Coautoría',
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                xaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )
        with col2:
            # Mostrar la gráfica en Streamlit
            st.plotly_chart(fig)

        end_time = time.time()
        st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")
# -------------------------------------------------------------------------------

# ---------------------- Análisis de patentes -----------------------------------
if selected == "Análisis de patentes":

    # Cargar los archivos CSV utilizando las rutas
    df_patentes = pd.read_csv(ruta_Patentes, encoding='latin1')
    df_publicaciones = pd.read_csv(ruta_Publicaciones)

    # Función para normalizar nombres
    def normalizar_nombre(nombre):
        return unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('utf-8').upper()
    
    # Procesar las fechas de publicación en diferentes formatos
    def procesar_fecha_publicacion(fecha):
        if pd.isnull(fecha):
            return pd.NaT  # Manejar valores nulos

        fecha = str(fecha).strip().upper()  # Convertir a mayúsculas y quitar espacios

        # Expresión regular para identificar formatos complejos
        if "JAN-FEB" in fecha:
            # Manejar el rango de meses como una fecha arbitraria del primer mes
            return pd.to_datetime("01 " + fecha.split('-')[0] + " 2000", format='%d %b %Y', errors='coerce')

        try:
            if re.match(r'^\w{3} \d{1,2}, \d{4}$', fecha):
                # Formato: "DEC 28, 2015"
                return pd.to_datetime(fecha, format='%b %d, %Y', errors='coerce')
            elif re.match(r'^\w{3} \d{1,2} \d{4}$', fecha):
                # Formato: "MAY 3 2017"
                return pd.to_datetime(fecha, format='%b %d %Y', errors='coerce')
            elif re.match(r'^\w{3} \d{4}$', fecha):
                # Formato: "FEB 2020"
                return pd.to_datetime(fecha, format='%b %Y', errors='coerce')
            elif re.match(r'^\d{1,2} \w{3} \d{4}$', fecha):
                # Formato: "28 DEC 2015"
                return pd.to_datetime(fecha, format='%d %b %Y', errors='coerce')
            elif re.match(r'^\w{3} \d{2}$', fecha):
                # Formato: "JAN 20" (asumiendo que el año es actual)
                return pd.to_datetime(fecha + ' ' + str(pd.datetime.now().year), format='%b %y', errors='coerce')
            else:
                # Formato no reconocido
                return pd.NaT
        except Exception as e:
            return pd.NaT

    # Función para calcular publicaciones antes y después de cada patente
    def calcular_publicaciones(df, fecha_patente, autor):
        antes = df[(df['Authors'] == autor) & (df['Publication Date'] < fecha_patente)]
        despues = df[(df['Authors'] == autor) & (df['Publication Date'] >= fecha_patente)]
        return len(antes), len(despues)

    # Función para agregar líneas de tendencia
    def agregar_tendencia(datos, fig, name, color, dash):
        if len(datos) > 1:
            X = datos['Año'].values.reshape(-1, 1)
            y = datos['Publicaciones'].values
            modelo = LinearRegression().fit(X, y)
            predicciones = modelo.predict(X)
            agregar_trazado(fig, datos['Año'], predicciones, name, color, dash)

    # Función para agregar trazados en la gráfica
    def agregar_trazado(fig, x, y, name, color, dash='solid'):
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, marker=dict(color=color), line=dict(color=color, dash=dash)))

    start_time = time.time()

    # Procesar la fecha de patente y normalizar inventores/autores
    df_patentes['Filing Date'] = pd.to_datetime(df_patentes['Filing Date'], format='%d/%m/%Y', dayfirst=True).dt.normalize()

    df_patentes['Inventor'] = df_patentes['Inventor'].apply(normalizar_nombre)
    df_publicaciones['Authors'] = df_publicaciones['Authors'].apply(normalizar_nombre)

    df_publicaciones['Publication Date'] = df_publicaciones['Publication Date'].apply(procesar_fecha_publicacion)

    # Eliminar duplicados en patentes
    df_patentes_unique = df_patentes.drop_duplicates(subset=['Inventor', 'Filing Date'])

    # Aplicar cálculo de publicaciones
    df_patentes_unique[['Publicaciones antes', 'Publicaciones después']] = df_patentes_unique.apply(
        lambda row: calcular_publicaciones(df_publicaciones, row['Filing Date'], row['Inventor']),
        axis=1, result_type='expand'
    )

    # Calcular cambio en publicaciones
    df_patentes_unique['Cambio en Publicaciones'] = df_patentes_unique['Publicaciones después'] - df_patentes_unique['Publicaciones antes']

    # Selección del autor
    autor_seleccionado = st.selectbox("Seleccione un autor", df_patentes_unique['Inventor'].unique())

    # Filtrar por autor seleccionado
    df_autor = df_patentes_unique[df_patentes_unique['Inventor'] == autor_seleccionado]
    df_autor_publicaciones = df_publicaciones[df_publicaciones['Authors'] == autor_seleccionado]

    # Agrupar publicaciones por año
    df_autor_publicaciones['Año'] = df_autor_publicaciones['Publication Date'].dt.year
    publicaciones_por_año = df_autor_publicaciones.groupby('Año')['Publication Date'].count().reset_index(name='Publicaciones')
    publicaciones_por_año['Año'] = publicaciones_por_año['Año'].astype(int)

    # Agrupar patentes por año
    df_autor['Año'] = df_autor['Filing Date'].dt.year
    patentes_por_año = df_autor.groupby('Año').size().reset_index(name='Patentes')

    # Asegúrate de que estas líneas estén antes de mostrar los datos en la tabla
    df_autor['Filing Date'] = df_autor['Filing Date'].dt.date
    df_autor_publicaciones['Publication Date'] = df_autor_publicaciones['Publication Date'].dt.date

    # Unir datos de publicaciones y patentes
    datos_combinados = pd.merge(publicaciones_por_año, patentes_por_año, on='Año', how='outer').fillna(0)

    print(publicaciones_por_año.head())
    print(patentes_por_año.head())

    # Crear figura
    fig = go.Figure()

    # Obtener los años de las patentes en una lista
    años_patentes = df_autor['Año'].tolist()

    # Para cada año de patente
    for index, año_patente in enumerate(años_patentes):
        # Identificar la patente anterior y siguiente (si existen)
        año_patente_anterior = años_patentes[index - 1] if index > 0 else None
        año_patente_siguiente = años_patentes[index + 1] if index < len(años_patentes) - 1 else None

        # Dividir los datos antes de la patente
        if año_patente_anterior:
            datos_antes_patente = datos_combinados[
                (datos_combinados['Año'] < año_patente) & 
                (datos_combinados['Año'] >= año_patente_anterior)
            ]
        else:
            datos_antes_patente = datos_combinados[datos_combinados['Año'] < año_patente]

        # Dividir los datos después de la patente
        if año_patente_siguiente:
            datos_despues_patente = datos_combinados[
                (datos_combinados['Año'] >= año_patente) & 
                (datos_combinados['Año'] < año_patente_siguiente)
            ]
        else:
            datos_despues_patente = datos_combinados[datos_combinados['Año'] >= año_patente]

        # Agregar líneas de tendencia
        agregar_tendencia(datos_antes_patente, fig, f"Tendencia publicaciones antes de {año_patente}", 'orange', 'dash')
        agregar_tendencia(datos_despues_patente, fig, f"Tendencia publicaciones después de {año_patente}", 'green', 'solid')

        # Añadir línea vertical para la patente
        fig.add_vline(x=año_patente, line_dash="dash", line_color="red", 
                    annotation_text=f"Patente en {año_patente}")

    # Agregar publicaciones y patentes
    agregar_trazado(fig, datos_combinados['Año'], datos_combinados['Publicaciones'], 'Publicaciones', 'blue')
    agregar_trazado(fig, datos_combinados['Año'], datos_combinados['Patentes'], 'Patentes', 'orange')

    # Agregar título a la gráfica
    fig.update_layout(title=f"Tendencia de Publicaciones y Patentes de {autor_seleccionado} Antes y Después de Cada Patente")

    # Mostrar gráfica
    st.plotly_chart(fig)

    # Mostrar resumen de publicaciones y patentes
    st.write("Resumen de publicaciones y patentes de ", autor_seleccionado)

    # Mostrar la tabla ajustada al tamaño del contenedor
    st.dataframe(df_autor[['Inventor', 'Patent', 'Filing Date', 'Publicaciones antes', 'Publicaciones después', 'Cambio en Publicaciones']], use_container_width=True)

    # Calcular el tiempo de ejecución
    end_time = time.time()
    st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")

# -------------------------------------------------------------------------------

# ---------------------- Análisis de conferencias- ------------------------------

# -------------------------------------------------------------------------------