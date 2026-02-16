import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Carga de datos
print("Cargando datos...")
df = load_penguins()

# 2. Exploración y Limpieza
# Eliminamos filas con valores nulos (estrategia del notebook original)
print(f"Dimensiones originales: {df.shape}")
df = df.dropna()
print(f"Dimensiones tras eliminar nulos: {df.shape}")

# 3. Preprocesamiento
# Codificación de variables categóricas
le_species = LabelEncoder()
le_island = LabelEncoder()
le_sex = LabelEncoder()

df['species'] = le_species.fit_transform(df['species'])
df['island'] = le_island.fit_transform(df['island'])
df['sex'] = le_sex.fit_transform(df['sex'])

# Separar características y objetivo
X = df.drop(['species', 'year'], axis=1) # 'year' no aporta valor predictivo
y = df['species']

# División train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Creación y Entrenamiento de Modelos

# Modelo 1: Random Forest
print("\nEntrenando Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(f"Accuracy Random Forest: {accuracy_score(y_test, y_pred_rf):.4f}")

# Modelo 2: SVM (Support Vector Machine)
# Para SVM es recomendable escalar los datos, pero para mantener similitud con el flujo simple, 
# usaremos los datos tal cual o podríamos añadir un scaler. 
# En este caso, usaremos SVM sin escalar para simplificar, pero en prod se recomienda Pipeline.
print("\nEntrenando SVM...")
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(f"Accuracy SVM: {accuracy_score(y_test, y_pred_svm):.4f}")

# 5. Guardado de Modelos y Encoders
if not os.path.exists('models'):
    os.makedirs('models')

print("\nGuardando modelos y encoders en la carpeta 'models'...")
joblib.dump(rf_model, 'models/random_forest_model.pkl')
joblib.dump(svm_model, 'models/svm_model.pkl')
joblib.dump(le_island, 'models/label_encoder_island.pkl')
joblib.dump(le_sex, 'models/label_encoder_sex.pkl')
joblib.dump(le_species, 'models/label_encoder_species.pkl')

print("¡Entrenamiento finalizado y artefactos guardados!")
