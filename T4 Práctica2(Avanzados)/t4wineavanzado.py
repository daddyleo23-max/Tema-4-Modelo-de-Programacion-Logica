from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, 
                            confusion_matrix, 
                            classification_report,
                            PrecisionRecallDisplay,
                            RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

## 1. Carga y exploración de datos ==============================
wine = load_wine()
X = wine.data  # 13 características químicas
y = wine.target  # 3 clases de vino

# Análisis exploratorio rápido
print("\n=== Información del Dataset ===")
print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
print("Distribución de clases:", np.bincount(y))
print("\nPrimeras 5 muestras:")
print(pd.DataFrame(X, columns=wine.feature_names).head())

## 2. Pipeline de modelado ======================================
# Creación de pipeline con escalado y modelo
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000,
                      multi_class='multinomial',
                      solver='lbfgs',
                      random_state=42)
)

## 3. Búsqueda de hiperparámetros ===============================
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10],
    'logisticregression__penalty': ['l2', 'none']
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)  # Usamos todos los datos para la búsqueda

print("\n=== Mejores Parámetros ===")
print(grid_search.best_params_)

## 4. Evaluación final ==========================================
# División estratificada 80-20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Entrenamiento con mejores parámetros
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Predicciones
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)

## 5. Métricas completas ========================================
print("\n=== Métricas de Evaluación ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")

print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Matriz de confusión
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=wine.target_names,
            yticklabels=wine.target_names)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()

## 6. Curvas de evaluación ======================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Curva Precision-Recall
PrecisionRecallDisplay.from_estimator(
    best_model, X_test, y_test, ax=ax1)
ax1.set_title('Curva Precision-Recall')

# Curva ROC (para clasificación multiclase)
RocCurveDisplay.from_estimator(
    best_model, X_test, y_test, ax=ax2)
ax2.set_title('Curva ROC')

plt.tight_layout()
plt.savefig('curvas_evaluacion.png', dpi=300)
plt.show()

## 7. Validación cruzada ========================================
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print("\n=== Validación Cruzada (5 folds) ===")
print(f"Accuracy promedio: {cv_scores.mean():.2%}")
print(f"Desviación estándar: {cv_scores.std():.2%}")

## 8. Interpretación del modelo =================================
# Extraer coeficientes estandarizados
coefs = best_model.named_steps['logisticregression'].coef_
features = wine.feature_names

coef_df = pd.DataFrame(coefs.T, index=features,
                      columns=[f"Clase {i}" for i in range(3)])
coef_df['Importancia_abs'] = np.abs(coef_df).mean(axis=1)

print("\n=== Coeficientes Estandarizados ===")
print(coef_df.sort_values('Importancia_abs', ascending=False).drop('Importancia_abs', axis=1))