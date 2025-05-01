import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Crear arreglo Numpy y mostrar promedio y máximo
productividad = np.array([75, 80, 90, 85, 70])
print(f"Promedio de productividad: {productividad.mean():.2f}")
print(f"Valor máximo de productividad: {productividad.max()}")

# 2. Leer archivo empleados.csv
try:
    empleados = pd.read_csv(r"C:\Users\herna\OneDrive\Documents\8vo Semestre\Programación_Logica_y_Funcional\4to Parcial\empleados.csv", encoding='latin1')
    # 3. Mostrar nombres del departamento Ventas
    ventas = empleados[empleados['Departamento'] == 'Ventas']
    print("\nEmpleados de Ventas:")
    print(ventas['Nombre'].to_string(index=False))
    
    # 4. Agregar columna Bono (10% del salario)
    empleados['Bono'] = empleados['Salario'] * 0.10
    
    # 5. Gráfica de barras con salarios
    plt.figure(figsize=(10, 6))
    plt.bar(empleados['Nombre'], empleados['Salario'], color='skyblue')
    
    # Agregar título y etiquetas
    plt.title('Salarios de Empleados', fontsize=16)
    plt.xlabel('Empleados', fontsize=12)
    plt.ylabel('Salario ($)', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.show()

except FileNotFoundError:
    print("Error: No se encontró el archivo 'empleados.csv'")
except Exception as e:
    print(f"Ocurrió un error: {str(e)}")