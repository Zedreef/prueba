import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import re
from Menu.utilidades import procesar_autor, calcular_resumen, graficar_citas_publicaciones

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
