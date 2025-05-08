from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Para escalar datos

wine = load_wine()
X = wine.data  # Características químicas del vino
y = wine.target  # Clases de vino (0, 1, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)  # Aumentamos iteraciones para evitar advertencias
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
precision = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", precision)