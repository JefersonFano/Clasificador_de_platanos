import cv2
import numpy as np
import os

def cargar_imagenes_de_carpeta(carpeta, etiqueta, tamaño=(64, 64)):
    datos = []
    for archivo in os.listdir(carpeta):
        ruta = os.path.join(carpeta, archivo)
        imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if imagen is not None:
            imagen_redimensionada = cv2.resize(imagen, tamaño)
            imagen_vector = imagen_redimensionada.flatten() / 255.0
            datos.append((imagen_vector, etiqueta))
    return datos
