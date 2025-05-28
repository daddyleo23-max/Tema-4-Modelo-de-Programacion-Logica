-- 1. Crear la base de datos (si no existe)
CREATE DATABASE IF NOT EXISTS practica_tensorflow;

-- 2. Usar la base de datos
USE practica_tensorflow;

-- 3. Crear la tabla para guardar las detecciones
CREATE TABLE IF NOT EXISTS detecciones_faciales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fecha DATETIME NOT NULL,
    ip_privada VARCHAR(45) NOT NULL,
    ip_publica VARCHAR(45) NOT NULL,
    usuario VARCHAR(100) NOT NULL
);

-- 4. (Opcional) Ejemplo de inserción de datos de prueba
INSERT INTO detecciones_faciales (fecha, ip_privada, ip_publica, usuario)
VALUES (NOW(), '192.168.1.100', '123.456.789.101', 'Usuario de prueba');

ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'daddyLEO99##';
FLUSH PRIVILEGES;

SELECT * FROM detecciones_faciales;
