import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
from Menu.utilidades import RUTA_PUBLICACIONES, RUTA_PATENTES

def cargar_datos():
    df_datosC = pd.read_csv(RUTA_PUBLICACIONES, encoding='latin-1')
    df_PreP = pd.read_csv(RUTA_PATENTES, encoding='latin-1')
    return df_datosC, df_PreP

def preprocesar_datos(df_datosC, df_PreP):
    df_datosC = df_datosC.groupby('Authors').agg(
        {'Authors': 'count', 'Total Citations': 'sum'}).rename(columns={'Authors': 'Publications'})
    
    df_PreP = df_PreP.rename(columns={'Inventor': 'Authors'})
    df_PreP = df_PreP[['Authors', 'Patent']]
    df_datosP = df_PreP.groupby('Authors')['Patent'].nunique()

    df_combinado = pd.merge(df_datosC, df_datosP, on='Authors', how='inner')
    return df_datosC, df_datosP, df_combinado.fillna(0)

def graficar_kmeans(X, labels, centers, title, xlabel, ylabel):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=labels, colorscale='Viridis', showscale=True),
        name='Data Points'
    ))
    fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        marker=dict(color='red', size=12, symbol='x'),
        name='Centroids'
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

def graficar_kmedoids(X, labels, medoids, title, xlabel, ylabel):
    # Crear un array para las etiquetas
    cluster_labels = [None] * len(X)
    for cluster_id, indices in enumerate(labels):
        for index in indices:
            cluster_labels[index] = cluster_id

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=X[:, 0],
        y=X[:, 1],
        mode='markers',
        marker=dict(color=cluster_labels, colorscale='Plasma', showscale=True),
        name='Data Points'
    ))
    fig.add_trace(go.Scatter(
        x=medoids[:, 0],
        y=medoids[:, 1],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='x'),
        name='Medoids'
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

def mostrar_analisis_kmeans():
    df_datosC, df_PreP = cargar_datos()
    
    st.markdown("<h2 style='text-align: center;'>Análisis de Clustering de los Autores</h2>", unsafe_allow_html=True)
    
    df_datosC, df_datosP, df_combinado = preprocesar_datos(df_datosC, df_PreP)

    # Análisis KMeans y KMedoids en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 style='text-align: center;'>KMeans</h2>", unsafe_allow_html=True)
        X_datosC = df_datosC[['Publications', 'Total Citations']].values
        X_datosC_scaled = StandardScaler().fit_transform(X_datosC)

        kmeans_model = KMeans(n_clusters=3, n_init='auto')
        kmeans_model.fit(X_datosC_scaled)
        graficar_kmeans(X_datosC_scaled, kmeans_model.labels_, kmeans_model.cluster_centers_, 
                        "KMeans: Autores Publicados (Escalados)", 'Publications (Escaladas)', 'Total Citations (Escaladas)')

        X_combinado = df_combinado[['Publications', 'Patent']].values
        kmeans_model_combinado = KMeans(n_clusters=3, n_init='auto')
        kmeans_model_combinado.fit(X_combinado)
        graficar_kmeans(X_combinado, kmeans_model_combinado.labels_, kmeans_model_combinado.cluster_centers_, 
                        "KMeans: Autores Publicados vs Patentes", 'Publications', 'Patents')

    with col2:
        st.markdown("<h2 style='text-align: center;'>KMedoids</h2>", unsafe_allow_html=True)
        initial_medoids = [0, 1, 2]  # Indices de los medoids iniciales
        kmedoids_model = kmedoids(X_datosC.tolist(), initial_medoids)
        kmedoids_model.process()
        labels = kmedoids_model.get_clusters()
        medoids = kmedoids_model.get_medoids()
        
        # Convertir medoids a array para graficar
        medoids_coords = X_datosC[medoids]
        graficar_kmedoids(X_datosC, labels, medoids_coords, 
                          "KMedoids: Autores Publicados", 'Publications', 'Total Citations')

        initial_medoids_combined = [0, 1, 2]  # Indices de los medoids iniciales
        kmedoids_model_combined = kmedoids(X_combinado.tolist(), initial_medoids_combined)
        kmedoids_model_combined.process()
        labels_combined = kmedoids_model_combined.get_clusters()
        medoids_combined = kmedoids_model_combined.get_medoids()
        
        medoids_combined_coords = X_combinado[medoids_combined]
        graficar_kmedoids(X_combinado, labels_combined, medoids_combined_coords, 
                          "KMedoids: Autores Publicados vs Patentes", 'Publications', 'Patents')