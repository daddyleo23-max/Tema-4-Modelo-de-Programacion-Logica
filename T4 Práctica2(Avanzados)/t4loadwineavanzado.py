# ===== PASO 1: Importar librerías =====
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, 
                            confusion_matrix, 
                            classification_report,
                            precision_recall_fscore_support)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===== PASO 2: Cargar y explorar datos =====
wine = load_wine()
X = wine.data  # 13 características químicas
y = wine.target  # 3 clases de vino

# Exploración rápida
print("\n=== Información del dataset ===")
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
print(f"Clases: {wine.target_names}")
print("\nDistribución de clases:")
print(pd.Series(y).value_counts())

# ===== PASO 3: Preprocesamiento =====
# Escalado de características (crucial para regresión logística)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# División estratificada (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

# ===== PASO 4: Modelado y optimización =====
# Configuración profesional del modelo
model = LogisticRegression(
    max_iter=5000,  # Más iteraciones para convergencia
    multi_class='multinomial',  # Para problemas multiclase
    solver='lbfgs',  # Algoritmo óptimo para este caso
    C=0.1,  # Regularización para evitar overfitting
    random_state=42
)

# Entrenamiento
model.fit(X_train, y_train)

# ===== PASO 5: Evaluación exhaustiva =====
# Predicciones
y_pred = model.predict(X_test)

# Métricas principales
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

print("\n=== Métricas principales ===")
print(f"Accuracy: {accuracy:.2%}")
print(f"Precision (weighted): {precision:.2%}")
print(f"Recall (weighted): {recall:.2%}")
print(f"F1-score (weighted): {f1:.2%}")

# Reporte completo por clase
print("\n=== Reporte por clase ===")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Matriz de confusión visual
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# ===== PASO 6: Validación cruzada =====
cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
print("\n=== Validación cruzada (5 folds) ===")
print(f"Accuracy promedio: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

# ===== PASO 7: Interpretación del modelo =====
# Coeficientes por clase
coef_df = pd.DataFrame(
    model.coef_,
    columns=wine.feature_names,
    index=[f"Clase {i} ({name})" for i, name in enumerate(wine.target_names)]
)
print("\n=== Coeficientes del modelo ===")
print(coef_df.T.sort_values(by="Clase 0 (class_0)", ascending=False))