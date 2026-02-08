import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv('caracteristicas_solapadas_irving.csv')
#print(df)
df_0 = df[df['Etiqueta']== 0]
#print(df_0)

X = df.drop(['Etiqueta','Ventana'], axis=1)
y = df['Etiqueta']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# parámetros para GridSearch
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']  # Solo aplicable para 'rbf'
}

grid_search = GridSearchCV(SVC(max_iter=5000), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

mejor_modelo = grid_search.best_estimator_
accuracy = mejor_modelo.score(X_test_scaled, y_test)
print(f"Mejor precisión: {accuracy:.2f}")
print("Mejores parámetros encontrados:", grid_search.best_params_)


# Predicciones del modelo
y_pred = mejor_modelo.predict(X_test_scaled)

# Calcula la matriz de confusión
matriz = confusion_matrix(y_test, y_pred)

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
plt.figure(figsize=(6, 4))
sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', xticklabels=mejor_modelo.classes_, yticklabels=mejor_modelo.classes_)
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.tight_layout()
plt.show()


# Guarda el modelo entrenado
joblib.dump(mejor_modelo, 'modelo_entrenadoSVM1.pkl')

# Guarda el escalador
joblib.dump(scaler, 'escaladorSVM1.pkl')

