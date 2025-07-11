from modelo import RedNeuronal
import pickle
import cv2
import numpy as np

def predecir_imagen(ruta, modelo):
    imagen = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    imagen = cv2.resize(imagen, (64, 64)).flatten() / 255.0
    imagen = imagen.reshape(-1, 1)

    salida = modelo.hacia_adelante(imagen)
    clases = ['bueno', 'maduro', 'malogrado']

    print("\nCompatibilidad con cada clase:")
    for i, prob in enumerate(salida):
        print(f" - {clases[i].capitalize():10}: {prob[0] * 100:.2f}%")

    clase = np.argmax(salida)
    print(f"\nClasificación final: {clases[clase].upper()} con {salida[clase][0] * 100:.2f}% de confianza.")


# Cargar modelo entrenado
with open("modelo_entrenado.pkl", "rb") as f:
    modelo = pickle.load(f)

# Predecir imagen desde la carpeta "prueba"
ruta_imagen = "prueba/imagen_prueba.jpg"  # Cambia esto al nombre de tu imagen
predecir_imagen(ruta_imagen, modelo)
