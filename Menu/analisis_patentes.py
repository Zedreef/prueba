import re
import time
import unicodedata
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from Menu.utilidades import RUTA_PUBLICACIONES, RUTA_PATENTES

def mostrar_analisis_patentes():

    # Cargar los archivos CSV utilizando las rutas
    df_patentes = pd.read_csv(RUTA_PATENTES, encoding='latin1')
    df_publicaciones = pd.read_csv(RUTA_PUBLICACIONES)

    # Función para normalizar nombres
    def normalizar_nombre(nombre):
        return unicodedata.normalize('NFKD', nombre).encode('ascii', 'ignore').decode('utf-8').upper()

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
    df_publicaciones['Authors'] = df_publicaciones['Authors'].str.strip()

    # Procesar fechas de publicaciones (en formato dd/mm/yyyy)
    df_publicaciones['Publication Date'] = pd.to_datetime(df_publicaciones['Publication Date'], format='%d/%m/%Y', errors='coerce')

    # **Filtrar publicaciones entre 2005 y 2023**
    df_publicaciones = df_publicaciones[
        (df_publicaciones['Publication Date'].dt.year >= 2005) & 
        (df_publicaciones['Publication Date'].dt.year <= 2023)
    ]

    # **Filtrar patentes entre 2005 y 2023**
    df_patentes = df_patentes[
        (df_patentes['Filing Date'].dt.year >= 2005) & 
        (df_patentes['Filing Date'].dt.year <= 2023)
    ]

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
    # Verificar la existencia de la columna 'Total Citations'
    print(df_publicaciones[['Title', 'Total Citations']].head())

    # Filtrar las publicaciones del autor seleccionado y obtener 'Title' y 'Total Citations'
    df_publicaciones_autor = df_publicaciones[df_publicaciones['Authors'] == autor_seleccionado][['Title', 'Total Citations']]

    # Eliminar las publicaciones con 'Total Citations' vacías o nulas
    df_publicaciones_autor = df_publicaciones_autor.dropna(subset=['Total Citations'])

    # Crear figura
    fig = go.Figure()

    # Obtener los años de las patentes en una lista
    años_patentes = df_autor['Año'].tolist()

    # Para cada año de patente
    for index, año_patente in enumerate(años_patentes):
        # Establecer el rango de dos años antes y después de la patente
        inicio_rango = año_patente - 2
        fin_rango = año_patente + 2

        # Filtrar los datos para incluir solo el rango de dos años antes y dos años después de la patente
        datos_en_rango = datos_combinados[(datos_combinados['Año'] >= inicio_rango) & (datos_combinados['Año'] <= fin_rango)]

        # Dividir los datos antes de la patente
        datos_antes_patente = datos_en_rango[datos_en_rango['Año'] < año_patente]

        # Dividir los datos después de la patente
        datos_despues_patente = datos_en_rango[datos_en_rango['Año'] >= año_patente]

        # Si no hay datos suficientes antes de la patente, agregar una línea predeterminada
        if len(datos_antes_patente) == 0:
            # Crear una línea plana predeterminada antes de la patente
            datos_antes_patente = pd.DataFrame({
                'Año': [inicio_rango, año_patente - 1],
                'Publicaciones': [0, 0]  # Puedes ajustar el valor predeterminado según sea necesario
            })

        # Agregar líneas de tendencia
        agregar_tendencia(datos_antes_patente, fig, f"Tendencia publicaciones antes de {año_patente}", 'orange', 'dash')
        agregar_tendencia(datos_despues_patente, fig, f"Tendencia publicaciones después de {año_patente}", 'green', 'solid')

        # Añadir línea vertical para la patente
        fig.add_vline(x=año_patente, line_dash="dash", line_color="red", 
                    annotation_text=f"Patente en {año_patente}")

    # Agregar publicaciones
    agregar_trazado(fig, datos_combinados[datos_combinados['Publicaciones'] > 0]['Año'],
                datos_combinados[datos_combinados['Publicaciones'] > 0]['Publicaciones'],
                'Publicaciones', 'blue')
    # Agregar trazado de citas por publicación
    agregar_trazado(fig, datos_combinados[datos_combinados['Publicaciones'] > 0]['Año'], 
                    df_publicaciones_autor['Total Citations'], 'Citas por Publicación', 'purple')

    # Agregar título a la gráfica
    fig.update_layout()

    # Actualizar el diseño de la gráfica para mostrar mejor los títulos de las publicaciones en el eje x
    fig.update_layout(
        title=f"Tendencia de Publicaciones de {autor_seleccionado}",
        xaxis_title="Año",
        yaxis_title="Publicaciones",
        xaxis_tickangle=-45
    )

    # Mostrar gráfica
    st.plotly_chart(fig)

    # Mostrar resumen de publicaciones y patentes
    st.write("Resumen de publicaciones y patentes de ", autor_seleccionado)

    # Mostrar la tabla ajustada al tamaño del contenedor
    st.dataframe(df_autor[['Inventor', 'Patent', 'Filing Date', 'Publicaciones antes', 'Publicaciones después', 'Cambio en Publicaciones']], use_container_width=True)

    # Calcular el tiempo de ejecución
    end_time = time.time()
    st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")