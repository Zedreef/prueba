import os
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----------------------- Funciones --------------------------------------------
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

def procesar_archivos(carpeta):
    correctos = 0
    incorrectos = 0
    archivos_incorrectos = []

    for filename in os.listdir(carpeta):
        if filename.endswith(".txt"):
            ruta_archivo = os.path.join(carpeta, filename)

            try:
                df = pd.read_csv(ruta_archivo, sep=',', quotechar='"', engine='python')
                correctos += 1
            except Exception as e:
                incorrectos += 1
                archivos_incorrectos.append(filename)

    return correctos, incorrectos, archivos_incorrectos

def procesar_estadisticas_autores(ruta_final):
    data = pd.read_csv(ruta_final)
    year_columns = [col for col in data.columns if col.isdigit()]
    data['Sum Of Times Cited'] = data[year_columns].fillna(0).sum(axis=1)
    publications_per_author = data.groupby('Authors')['Title'].count().reset_index()
    publications_per_author.columns = ['Authors', 'Publications']
    citations_per_author = data.groupby('Authors')['Sum Of Times Cited'].sum().reset_index()
    author_stats = pd.merge(publications_per_author, citations_per_author, on='Authors')
    author_stats = author_stats[author_stats['Authors'].notna()]
    return author_stats

# Función para procesar los datos del autor seleccionado
def procesar_autor(df, autor_seleccionado):
    # Filtrar el DataFrame por autor seleccionado y eliminar filas con 'Authors' vacíos
    df_filtrado = df[df['Authors'].notna() & (df['Authors'] != '') & (
        df['Authors'] == autor_seleccionado)].copy()

    # Mantener solo las columnas específicas que te interesan
    columnas_especificas = ['Title', 'Authors', 'Source Title',
                            'Publication Date', 'Total Citations', 'Average per Year']

    # Filtrar dinámicamente columnas de años (desde 1960 en adelante)
    columnas_de_años = [
        col for col in df.columns if col.isdigit() and int(col) >= 1960]

    # Mantener solo las columnas de años que contienen al menos un valor distinto de 0 en el DataFrame filtrado
    columnas_de_años_validas = [col for col in columnas_de_años if (
        df_filtrado[col].notna() & (df_filtrado[col] != 0)).any()]

    # Combinar las columnas específicas con las columnas de años válidas
    df_final = pd.concat([df_filtrado[columnas_especificas],
                            df_filtrado[columnas_de_años_validas]], axis=1)

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

# Función para gráfica las citas y publicaciones por año
def graficar_citas_publicaciones(df_autor, autor_seleccionado):
    # Extraer el año de 'Publication Date' usando una expresión regular para capturar solo el año
    df_autor['Year'] = df_autor['Publication Date'].apply(lambda x: re.search(
        r'\d{4}', str(x)).group() if re.search(r'\d{4}', str(x)) else None)

    # Eliminar las filas donde no se pudo extraer un año válido
    df_autor = df_autor[df_autor['Year'].notna()].copy()

    # Convertir la columna 'Year' a entero
    df_autor['Year'] = df_autor['Year'].astype(int)

    # Agrupar por el año y contar el número de publicaciones
    publicaciones_por_año = df_autor.groupby(
        'Year').size()  # Número de publicaciones por año

    # Agrupar por el año y sumar el total de citas
    citas_por_año = df_autor.groupby(
        'Year')['Total Citations'].sum()  # Total de citas por año

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

# ----------------------- Ruta App ---------------------------------------------
RUTA_BRUTOS = '/mount/src/prueba/Datos Brutos'
RUTA_GUARDADO = '/mount/src/prueba/Datos Completos'
RUTA_PUBLICACIONES = 'Analisis/Publicaciones.csv'
RUTA_PATENTES = 'Analisis/Investigadores PATENTES.csv'
# ----------------------- Ruta GitHub ------------------------------------------
# RUTA_BRUTOS  = '/workspaces/prueba/Datos Brutos'
# RUTA_GUARDADO  = '/workspaces/prueba/Datos Completos'
# RUTA_PUBLICACIONES  = 'Analisis/Publicaciones.csv'
# RUTA_PATENTES  = 'Analisis/Investigadores PATENTES.csv'
# -------------------------------------------------------------------------------