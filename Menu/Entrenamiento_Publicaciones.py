import os
import streamlit as st
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from sklearn.metrics import classification_report, roc_auc_score
import plotly.graph_objects as go

def mostrar_keras():

    # Función para cargar y preprocesar los datos
    @st.cache_data
    def load_data():
        data = pd.read_csv('Analisis/Publicaciones.csv', encoding='ISO-8859-1')
        data['Publish_2024'] = data['2024'] > 0  # Target: Si el autor publicará en 2024
        return data

    def preprocess_data(data):
        features = data[['Total Citations', 'Average per Year'] + [col for col in data.columns if col.isdigit() and 1960 <= int(col) <= 2023] +
                        ['Title', 'Authors', 'Corporate Authors', 'Book Editors', 'Source Title']]
        target = data['Publish_2024']

        numeric_features = ['Total Citations', 'Average per Year'] + \
            [col for col in data.columns if col.isdigit() and 1960 <= int(col) <= 2023]
        categorical_features = ['Title', 'Authors', 'Corporate Authors', 'Book Editors', 'Source Title']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        X = preprocessor.fit_transform(features)
        y = target.values
        return X, y

    # Función para crear y compilar el modelo
    def create_model(input_shape):
        model = Sequential([
            Input(shape=(input_shape,)), 
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='mae',
                    metrics=['accuracy', 'AUC'])
        return model

    # Carga de datos
    st.title("Entrenamiento de predicción de Publicaciones en 2024")

    data = load_data()

    # Preprocesar los datos
    X, y = preprocess_data(data)

    # Dividir los datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir y entrenar el modelo
    model = create_model(X_train.shape[1])

    epochs = st.slider('Selecciona el número de épocas', min_value=10, max_value=100, value=50, step=10)


    if st.button('Entrenar modelo'):

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)

        # Guardar el modelo entrenado en session_state
        st.session_state.model = model

        # Crear columnas para organizar la tabla y los datos
        col1, col2 = st.columns([2, 1])

        # Mostrar las métricas del entrenamiento
        st.write("Resultados del entrenamiento:")

        # Datos a la izquierda
        with col1:
            history_df = pd.DataFrame(history.history)
            
            # Crear gráfico con Plotly (go)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['loss'],
                                    mode='lines', name='Pérdida (Entrenamiento)'))
            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['val_loss'],
                                    mode='lines', name='Pérdida (Validación)'))
            fig.update_layout(title='Pérdida vs Épocas', xaxis_title='Épocas', yaxis_title='Pérdida')
            st.plotly_chart(fig, use_container_width=True)  # Mostrar gráfica a la izquierda

        # Tabla a la derecha
        with col2:
            st.write(f"Mínimo loss en validación: {history_df['val_loss'].min()}")
            
            # Realizar predicciones y mostrar resultados
            y_pred = model.predict(X_test).ravel()
            y_pred_class = (y_pred > 0.5).astype(int)

            # Mostrar el reporte de clasificación y el AUC
            st.write("Evaluación del modelo:")
            st.text(classification_report(y_test, y_pred_class))
            st.write(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")

            # Marcar en el estado de la sesión que el modelo ha sido entrenado
            st.session_state.model_trained = True

    # Guardar el modelo entrenado
    if st.button('Guardar modelo'):
        if st.session_state.model is not None:
            try:
                # Ruta del archivo
                # model_path = '/workspaces/prueba/Analisis/Entrena_Publicaciones.keras'
                model_path = 'Analisis/Entrena_Publicaciones.keras'

                # Verificar si el archivo ya existe
                if os.path.exists(model_path):
                    st.warning("El archivo ya existe y será sobrescrito.")
                    print("El archivo ya existe y será sobrescrito.")

                # Agregar un mensaje antes de guardar el modelo
                st.info("Guardando el modelo...")
                print("Guardando el modelo...")

                # Guardar el modelo en la ruta especificada
                keras.saving.save_model(st.session_state.model, model_path)

                # Confirmar que el modelo ha sido guardado
                st.success("Modelo guardado exitosamente.")
                print("Modelo guardado exitosamente.")

            except Exception as e:
                # Mostrar un mensaje de error si algo sale mal
                st.error(f"Error al guardar el modelo: {e}")
                print(f"Error al guardar el modelo: {e}")
        else:
            st.warning("No se ha entrenado ningún modelo aún.")
            print("Por favor, entrena el modelo primero para habilitar la opción de guardarlo.")
