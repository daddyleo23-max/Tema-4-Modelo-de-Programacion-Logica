import cv2
import mediapipe as mp
import numpy as np
import requests
import mysql.connector
import time
import socket
from datetime import datetime


def obtener_ip_privada():
    try:
        ip_privada = socket.gethostbyname(socket.gethostname())
        return ip_privada
    except socket.error as e:
        return f"Error al obtener la IP privada: {e}"

def obtener_ip_publica():
    try:
        respuesta = requests.get('https://api.ipify.org?format=json')
        respuesta.raise_for_status()
        ip_publica = respuesta.json()['ip']
        return ip_publica
    except requests.RequestException as e:
        return f"Error al obtener la IP pública: {e}"

def insertar_datos(ip_privada, ip_publica, usuario):
    try:
        conexion = mysql.connector.connect(
            host='localhost',
            database='practica_tensorflow',
            user='root',
            password='daddyLEO99##',
            auth_plugin='mysql_native_password'
        )
        cursor = conexion.cursor()
        query = "INSERT INTO detecciones_faciales (fecha, ip_privada, ip_publica, usuario) VALUES (%s, %s, %s, %s)"
        valores = (datetime.now(), ip_privada, ip_publica, usuario)
        cursor.execute(query, valores)

        conexion.commit()
        print("Se guardó el registro en la base de datos...")
    except Exception as e:
        print(f"Error al insertar datos: {e}")
    finally:
        if 'conexion' in locals() and conexion.is_connected():
            cursor.close()
            conexion.close()

# Inicializar Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

def calcular_distancia(punto1, punto2):
    return np.linalg.norm(np.array(punto1) - np.array(punto2))

ip_publica = obtener_ip_publica()
ip_privada = obtener_ip_privada()
usuario = "Pascual"
boca_abierta = False
inicio_boca_abierta = 0
umbral_duracion = 2  # Segundos que la boca debe estar abierta para enviar datos
umbral_boca_abierta = 10  # Ajusta según pruebas

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("No se puede acceder a la cámara")
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Ajusta los índices según la referencia de landmarks
            labio_superior = face_landmarks.landmark[13]  # Cambia si no detecta bien
            labio_inferior = face_landmarks.landmark[14]  # Cambia si no detecta bien
            altura, ancho, _ = image.shape
            labio_superior_px = (int(labio_superior.x * ancho), int(labio_superior.y * altura))
            labio_inferior_px = (int(labio_inferior.x * ancho), int(labio_inferior.y * altura))
            distancia_boca = calcular_distancia(labio_superior_px, labio_inferior_px)

            if distancia_boca > umbral_boca_abierta:
                if not boca_abierta:
                    boca_abierta = True
                    inicio_boca_abierta = time.time()
                elif time.time() - inicio_boca_abierta > umbral_duracion:
                    cv2.putText(image, '¡Boca Abierta!', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    insertar_datos(ip_privada, ip_publica, usuario)
                    inicio_boca_abierta = time.time()  # Reiniciar el temporizador
            else:
                boca_abierta = False

    cv2.imshow('Reconocimiento de Gestos Faciales', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
