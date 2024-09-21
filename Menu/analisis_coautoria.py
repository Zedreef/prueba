import glob
import itertools
import logging
import os
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import plotly.express as px
from networkx.algorithms.community import louvain_communities
from .utilidades import RUTA_BRUTOS, procesar_autores_Brutos, create_node_trace, create_edge_trace

def mostrar_analisis_coautoria():
    st.subheader('Detección de comunidades con Algoritmo de Louvain')
    st.text('Este algoritmo agrupa nodos del grafo en comunidades basadas en la densidad de sus conexiones.')

    # Obtener la lista de archivos .txt usando glob
    archivos_txt = glob.glob(os.path.join(RUTA_BRUTOS, '*.txt'))

    @st.cache_data
    def procesar_archivos_Brutos(archivos_txt):
        dfs = []
        for archivo_txt in archivos_txt:
            try:
                df = pd.read_csv(archivo_txt, skiprows=2, quotechar='"')
                df = procesar_autores_Brutos(df, os.path.basename(archivo_txt))
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
    df_todos = procesar_archivos_Brutos(archivos_txt)

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