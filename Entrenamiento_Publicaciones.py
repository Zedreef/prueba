import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score
from Menu.utilidades import RUTA_PUBLICACIONES

# Cargar los datos
data = pd.read_csv(RUTA_PUBLICACIONES)

# Definir features y target
# Supongamos que queremos predecir si el autor publicará en 2024
data['Publish_2024'] = data['2024'] > 0  # Ejemplo binario

features = data[['Total Citations', 'Average per Year', '2005', '2006', '2007','2008','2009','2010'
                 ,'2011','2012','2013','2014','2015','2016','2017','2018','2019','2020','2021','2022', '2023',
                'Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title']]
target = data['Publish_2024']

# Preprocesamiento
numeric_features = ['Total Citations', 'Average per Year'] + [str(year) for year in range(2005, 2023)]
categorical_features = ['Authors', 'Corporate Authors', 'Book Editors', 'Source Title', 'Conference Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

X = preprocessor.fit_transform(features)
y = target.values

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Predicciones
y_pred = model.predict(X_test).ravel()
y_pred_class = (y_pred > 0.5).astype(int)

# Evaluación
print(classification_report(y_test, y_pred_class))
print("AUC:", roc_auc_score(y_test, y_pred))

# Después de entrenar el modelo lo guardamos
keras.saving.save_model(model, 'my_model.keras')