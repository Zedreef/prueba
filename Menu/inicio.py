import streamlit as st
import pandas as pd
import plotly.express as px
from Menu.utilidades import procesar_archivos, procesar_estadisticas_autores, RUTA_GUARDADO, RUTA_PUBLICACIONES

def mostrar_inicio():
    correctos, incorrectos, archivos_incorrectos = procesar_archivos(RUTA_GUARDADO)
    author_stats = procesar_estadisticas_autores(RUTA_PUBLICACIONES)

    # Filtrar y ordenar los datos de los autores
    min_articles = 1
    min_citations = 0
    filtered_stats = author_stats[
        (author_stats['Publications'] >= min_articles) |
        (author_stats['Sum Of Times Cited'] >= min_citations)
    ].sort_values(by='Sum Of Times Cited', ascending=False)

    st.title("📊 Informe de Archivos Procesados")
    col1, col2, col3 = st.columns(3)
    col1.metric("Archivos correctos", correctos)
    col2.metric("Archivos con error", incorrectos)
    col3.metric("Archivos en total", correctos + incorrectos)

    if incorrectos > 0:
        st.subheader("Archivos con error:")
        for archivo in archivos_incorrectos:
            st.write(f"- {archivo}")

    # Gráfico de barras
    data = {
        'Categoría': ['Correctos', 'Incorrectos'],
        'Cantidad': [correctos, incorrectos]
    }
    df = pd.DataFrame(data)

    fig = px.bar(df, x='Categoría', y='Cantidad', color='Categoría',
                 title="Archivos Procesados Correctamente vs Incorrectamente",
                 labels={'Cantidad': 'Número de Archivos'},
                 height=400)

    st.plotly_chart(fig)
