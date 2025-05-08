# Importamos las librerías necesarias desde scikit-learn:

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el dataset Iris. Este dataset contiene datos de 150 flores,

iris = load_iris()

# Asignamos las características (features) a la variable X
# y las etiquetas (labels) a la variable y.
X = iris.data # variables predictoras (features)
y = iris.target # variable objetivo (label)

# Dividir los datos en conjunto de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
random_state=42)

# Crear un modelo de regresión logística.
# max_iter se establece en 200 para asegurar que el algoritmo converge.
model = LogisticRegression(max_iter=200)
# Entrenar el modelo con los datos de entrenamiento.
model.fit(X_train, y_train)
# Realizar predicciones sobre el conjunto de prueba.
y_pred = model.predict(X_test)

# Evaluar el rendimiento del modelo calculando la precisión (accuracy),
print("Precision:", accuracy_score(y_test, y_pred))

