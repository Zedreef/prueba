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
    st.markdown("<h2 style='text-align: center;'>Detección de comunidades con Algoritmo de Louvain</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Este algoritmo agrupa nodos del grafo en comunidades basadas en la densidad de sus conexiones</p>", unsafe_allow_html=True)

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

        col1, col2 = st.columns([0.3, 1])

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

            # Generar posiciones 3D para todos los nodos
            pos_3d = nx.spring_layout(G_filtrado, dim=3, k=0.15, iterations=50)

            # Crear las trazas de nodos y aristas en 3D
            node_x, node_y, node_z = [], [], []
            node_colors = []
            for node in G_filtrado.nodes():
                x, y, z = pos_3d[node]
                node_x.append(x)
                node_y.append(y)
                node_z.append(z)
                node_colors.append(comunidad_por_nodo.get(node, 'grey'))

            edge_x, edge_y, edge_z = [], [], []
            for edge in G_filtrado.edges():
                x0, y0, z0 = pos_3d[edge[0]]
                x1, y1, z1 = pos_3d[edge[1]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]
                edge_z += [z0, z1, None]

            # Crear la figura 3D
            fig = go.Figure()

            # Añadir las aristas
            fig.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                                       mode='lines',
                                       line=dict(color='black', width=0.5),
                                       hoverinfo='none'))

            # Añadir los nodos
            fig.add_trace(go.Scatter3d(x=node_x, y=node_y, z=node_z,
                                       mode='markers',
                                       marker=dict(size=6, color=node_colors, opacity=0.8),
                                       text=list(G_filtrado.nodes()),
                                       hoverinfo='text'))

            # Personalizar el diseño del gráfico 3D
            fig.update_layout(
                title='Red de Coautoría en 3D',
                scene=dict(
                    xaxis=dict(showbackground=False),
                    yaxis=dict(showbackground=False),
                    zaxis=dict(showbackground=False)
                ),
                margin=dict(l=0, r=0, b=0, t=40)
            )

        with col2:
            # Mostrar la gráfica 3D en Streamlit
            st.plotly_chart(fig)

        end_time = time.time()
        st.write(f"Tiempo de ejecución: {end_time - start_time} segundos")