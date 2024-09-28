import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.metrics import silhouette_score
import numpy as np
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

    df_combinado = pd.merge(df_datosC, df_datosP, on='Authors', how='outer').fillna(0)
    return df_datosC, df_datosP, df_combinado

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
        marker=dict(color='red', size=12, symbol='x'),
        name='Medoids'
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    st.plotly_chart(fig)

def calcular_numero_optimo_kmeans(X):
    inertia = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(K_range), y=inertia, mode='lines+markers'))
    fig.update_layout(title='Método del Codo para KMeans', xaxis_title='Número de Clusters', yaxis_title='Inercia')
    st.plotly_chart(fig)

def mostrar_analisis_kmeans():
    df_datosC, df_PreP = cargar_datos()
    
    st.markdown("<h2 style='text-align: center;'>Análisis de Clustering de los Autores</h2>", unsafe_allow_html=True)
    
    df_datosC, df_datosP, df_combinado = preprocesar_datos(df_datosC, df_PreP)

    st.markdown("<h3 style='text-align: center;'>Distribución de los Datos</h3>", unsafe_allow_html=True)

    # Escalar los datos
    X_datosC = df_datosC[['Publications', 'Total Citations']].values
    X_datosC_scaled = StandardScaler().fit_transform(X_datosC)
    
    X_combinado = df_combinado[['Publications', 'Patent']].values
    X_combinado_scaled = StandardScaler().fit_transform(X_combinado)

    # Selección del número de clusters por el usuario
    n_clusters = st.slider('Selecciona el número de clusters:', min_value=2, max_value=10, value=4)

    col3, col4 = st.columns(2)
    fig = go.Figure()

    with col3:
        # Mostrar la gráfica del método del codo y distribución de los datos antes de clustering
        calcular_numero_optimo_kmeans(X_combinado_scaled)
    with col4:
        # Visualizar la distribución de los datos antes de clustering
        fig.update_layout(title='Distribución: Publicaciones vs Patentes', xaxis_title='Publications', yaxis_title='Patents')
        fig.add_trace(go.Scatter(x=df_combinado['Publications'], y=df_combinado['Patent'], mode='markers'))
        st.plotly_chart(fig)

    # Análisis KMeans y KMedoids en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h2 style='text-align: center;'>KMeans</h2>", unsafe_allow_html=True)
        kmeans_model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        kmeans_model.fit(X_datosC_scaled)
        graficar_kmeans(X_datosC_scaled, kmeans_model.labels_, kmeans_model.cluster_centers_, 
                        "KMeans: Autores Publicados", 'Publications (Escaladas)', 'Total Citations (Escaladas)')

        # Métrica silhouette para evaluar el clustering de KMeans
        silhouette_avg = silhouette_score(X_datosC_scaled, kmeans_model.labels_)
        st.write(f"Coeficiente de Silhouette para KMeans (Publicaciones): {silhouette_avg:.2f}")

        kmeans_model_combinado = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        kmeans_model_combinado.fit(X_combinado_scaled)
        graficar_kmeans(X_combinado_scaled, kmeans_model_combinado.labels_, kmeans_model_combinado.cluster_centers_, 
                        "KMeans: Autores Publicados vs Patentes", 'Publications (Escaladas)', 'Patents (Escaladas)')

        silhouette_combined_avg = silhouette_score(X_combinado_scaled, kmeans_model_combinado.labels_)
        st.write(f"Coeficiente de Silhouette para KMeans (Publicaciones vs Patentes): {silhouette_combined_avg:.2f}")

    with col2:
        st.markdown("<h2 style='text-align: center;'>KMedoids</h2>", unsafe_allow_html=True)
        initial_medoids = np.random.choice(len(X_datosC_scaled), n_clusters, replace=False).tolist()  # Elegir n_clusters medoids aleatorios
        kmedoids_model = kmedoids(X_datosC_scaled.tolist(), initial_medoids)
        kmedoids_model.process()
        labels = kmedoids_model.get_clusters()
        medoids = kmedoids_model.get_medoids()

        # Convertir las etiquetas de clusters a un array de etiquetas
        labels_flat = np.zeros(len(X_datosC_scaled))
        for cluster_id, cluster in enumerate(labels):
            for index in cluster:
                labels_flat[index] = cluster_id

        medoids_coords = X_datosC_scaled[medoids]
        graficar_kmedoids(X_datosC_scaled, labels, medoids_coords, 
                        "KMedoids: Autores Publicados", 'Publications (Escaladas)', 'Total Citations(Escaladas)')
        
        # Calcular el coeficiente de Silhouette para KMedoids
        silhouette_kmedoids_avg = silhouette_score(X_datosC_scaled, labels_flat)
        st.write(f"Coeficiente de Silhouette para KMedoids (Publicaciones): {silhouette_kmedoids_avg:.2f}")

        kmedoids_model_combined = kmedoids(X_combinado_scaled.tolist(), initial_medoids)
        kmedoids_model_combined.process()
        labels_combined = kmedoids_model_combined.get_clusters()
        medoids_combined = kmedoids_model_combined.get_medoids()

        # Convertir las etiquetas combinadas a un array de etiquetas
        labels_combined_flat = np.zeros(len(X_combinado_scaled))
        for cluster_id, cluster in enumerate(labels_combined):
            for index in cluster:
                labels_combined_flat[index] = cluster_id

        medoids_combined_coords = X_combinado_scaled[medoids_combined]
        graficar_kmedoids(X_combinado_scaled, labels_combined, medoids_combined_coords, 
                        "KMedoids: Autores Publicados vs Patentes", 'Publications', 'Patents')
        
        # Calcular el coeficiente de Silhouette para KMedoids combinado
        silhouette_kmedoids_combined_avg = silhouette_score(X_combinado_scaled, labels_combined_flat)
        st.write(f"Coeficiente de Silhouette para KMedoids (Publicaciones vs Patentes): {silhouette_kmedoids_combined_avg:.2f}")
