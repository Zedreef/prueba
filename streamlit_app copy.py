import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#Page configuration
st.set_page_config (
  page_title = 'Iris prediccion',
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

# Titulo de la app
st.title ('Prediccion clase de orquidea')
col1, col2 = st.columns(2)

# Lectura de datos
iris = load_iris ()
df = pd.DataFrame (iris.data, columns = iris.feature_names)

df['target'] = iris.target

# Análisis exploratorio de datos
col1.subheader('Análisis Exploratorio de Datos')
groupby_species_mean = df.groupby ('target').mean ()
col1.write (groupby_species_mean)

# Sidebar
st.sidebar.subheader('Parametros del modelo')
sepal_length = st.sidebar.slider('Sepal length', float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider('Sepal width', float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider('Petal length', float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider('Petal width', float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# Dividir dataset y creación del modelo
X_train, X_test, y_train, y_test = train_test_split (
    df.drop (['target'], axis='columns'), iris.target,
    test_size = 0.2
)

model = RandomForestClassifier (n_estimators=40, max_depth=4)
model.fit (X_train, y_train)

# Aplicar el modelo para hacer predicciones con los datos leídos
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                         columns=iris.feature_names)
prediction = model.predict(input_data)
predicted_class = iris.target_names[prediction[0]]

# Escribir los datos que el usuario seleccionó
col1.subheader('Datos seleccionados:')
col1.write(input_data)

col1.subheader('Clase predecida:')
col1.write(predicted_class)

# Mostrar imagen según la predicción
if predicted_class == 'setosa':
  col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/800px-Kosaciec_szczecinkowaty_Iris_setosa.jpg", caption="Iris Setosa", width= 500)
elif predicted_class == 'versicolor':
  col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Blue_Flag%2C_Ottawa.jpg/800px-Blue_Flag%2C_Ottawa.jpg", caption="Iris Versicolor", width= 500)
else:
  col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/800px-Iris_virginica.jpg", caption="Iris Virginica",  width= 500)
