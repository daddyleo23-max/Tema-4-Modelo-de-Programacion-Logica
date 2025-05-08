# Importamos las librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, 
                             confusion_matrix, 
                             classification_report)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 1. Cargar el dataset Iris
iris = load_iris()
X = iris.data  # Características (4 features)
y = iris.target  # Etiquetas (3 clases)

# 2. EDA Rápido (Exploratory Data Analysis)
print("\n=== Información del Dataset ===")
print(f"Número de muestras: {X.shape[0]}")
print(f"Número de características: {X.shape[1]}")
print(f"Clases: {iris.target_names}")
print("\nPrimeras 5 muestras:")
print(pd.DataFrame(X, columns=iris.feature_names).head())

# 3. Dividir datos (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Escalado de características (mejora la convergencia)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Crear y entrenar modelo (con parámetros optimizados)
model = LogisticRegression(
    max_iter=1000,  # Más iteraciones para convergencia
    multi_class='multinomial',  # Para problemas multiclase
    solver='lbfgs',  # Buen solver para datasets pequeños
    C=0.1  # Regularización para evitar overfitting
)
model.fit(X_train, y_train)

# 6. Evaluación del modelo
y_pred = model.predict(X_test)

print("\n=== Métricas de Evaluación ===")
print(f"Precisión: {accuracy_score(y_test, y_pred):.2%}")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, 
                           target_names=iris.target_names))

# 7. Interpretación de coeficientes
print("\n=== Interpretación del Modelo ===")
coef_df = pd.DataFrame(model.coef_, 
                       columns=iris.feature_names,
                       index=iris.target_names)
print("Coeficientes del modelo (por clase):")
print(coef_df)