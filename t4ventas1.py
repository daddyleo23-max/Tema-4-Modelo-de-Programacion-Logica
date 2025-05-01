#Importación de librerias
import numpy as np
import pandas as pd
import matplotlib .pyplot as plt

#Crear arreglos con Numpy
#Arreglo de ventas por semana
ventas_semana = np.array([150,200,170,220,300,250,190])
print("Ventas por semaba: ",ventas_semana)

# Operaciones con arreglos
print("Promedio de ventas:",
    np.mean(ventas_semana))
print("Ventas máxima:",
    np.max(ventas_semana))
print("Ventas mínima:",
    np.min(ventas_semana))

# Lectura de archivos CSV con pandas
datos_ventas = pd.read_csv(r"C:\Users\herna\OneDrive\Documents\8vo Semestre\Programación_Logica_y_Funcional\4to Parcial\ventas.csv",encoding='utf-8')
print("\nDatos de ventas:\n", datos_ventas)


colores = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta']
# Gráfico de pastel con la proporción de unidades vendidas
plt.pie(datos_ventas['Unidades Vendidas'], labels=datos_ventas['Producto'], autopct='%1.1f%%', startangle=90, colors=colores)
plt.title('Proporción de Unidades Vendidas')
plt.axis('equal')  # Asegura que el gráfico sea un círculo
plt.show()