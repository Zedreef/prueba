import streamlit as st
import plotly.graph_objects as go
from Menu.utilidades import RUTA_PUBLICACIONES, procesar_estadisticas_autores


def mostrar_todos_investigadores():
    st.subheader("Todos los investigadores")

    author_stats = procesar_estadisticas_autores(RUTA_PUBLICACIONES)

    # Filtrar y ordenar los datos de los autores
    min_articles = 1
    min_citations = 0
    filtered_stats = author_stats[
        (author_stats['Publications'] >= min_articles) |
        (author_stats['Sum Of Times Cited'] >= min_citations)
    ].sort_values(by='Sum Of Times Cited', ascending=False)

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
