import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import re
from .utilidades import calcular_indice_h,graficar_citas_publicaciones_Comparados,procesar_autores, RUTA_PUBLICACIONES

def mostrar_comparar_investigadores():

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
                'Autors': autor,
                'Total de Citas': total_citations,
                'Promedio por Año': average_per_year,
                'Índice h': h_index
            })

        # Convertir el resumen en un DataFrame
        return pd.DataFrame(resumen)

    df_publicaciones = pd.read_csv(RUTA_PUBLICACIONES)

    st.title("Análisis de Publicaciones por Rango de Fechas")

    col3, col4 = st.columns([0.25, 1])
    with col3:
        cantidad_autores = st.number_input(
            "Selecciona la cantidad de autores", min_value=1, value=5)
    with col4:
        rango_fechas = st.slider(
            "Selecciona el rango de fechas", min_value=2005, max_value=2024, value=(2010, 2020))

    if cantidad_autores and rango_fechas:
        try:
            df_resultado = procesar_autores(
                df_publicaciones, cantidad_autores, rango_fechas[0], rango_fechas[1])

            df_resumen = calcular_resumen(df_resultado)

            col1, col2 = st.columns([0.5, 1])
            with col1:
                st.write(f"Autores encontrados entre {rango_fechas[0]} y {rango_fechas[1]}:")
                st.table(df_resumen)
            with col2:
                graficar_citas_publicaciones_Comparados(df_resultado)

            st.write(f"Datos de los {cantidad_autores} autores más productivos entre {rango_fechas[0]} y {rango_fechas[1]}: ")
            st.dataframe(df_resultado)

        except Exception as e:
            st.error(f"Error procesando los datos: {e}")