from preprocesamiento import cargar_imagenes_de_carpeta
from modelo import RedNeuronal
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

# Cargar imágenes y etiquetas
datos = []
datos += cargar_imagenes_de_carpeta('dataset/bueno', [1, 0, 0])
datos += cargar_imagenes_de_carpeta('dataset/maduro', [0, 1, 0])
datos += cargar_imagenes_de_carpeta('dataset/malogrado', [0, 0, 1])

# Mezclar y dividir en entrenamiento y prueba
random.shuffle(datos)
division = int(len(datos) * 0.8)
datos_entrenamiento = datos[:division]
datos_prueba = datos[division:]

X_entrenamiento = np.array([x for x, _ in datos_entrenamiento]).T
Y_entrenamiento = np.array([y for _, y in datos_entrenamiento]).T

X_prueba = np.array([x for x, _ in datos_prueba]).T
Y_prueba = np.array([y for _, y in datos_prueba]).T

# Inicializar red neuronal
modelo = RedNeuronal(entrada=X_entrenamiento.shape[0], ocultas=128, salida=3, tasa_aprendizaje=0.1)

# Entrenamiento y almacenamiento de pérdidas
historial_perdida = []

for epoca in range(5000):
    salida = modelo.hacia_adelante(X_entrenamiento)
    perdida = modelo.calcular_error(Y_entrenamiento, salida)
    historial_perdida.append(perdida)
    modelo.hacia_atras(X_entrenamiento, Y_entrenamiento)
    if epoca % 100 == 0:
        print(f"Época {epoca}, Error: {perdida:.4f}")

# Guardar el modelo entrenado
with open("modelo_entrenado.pkl", "wb") as f:
    pickle.dump(modelo, f)

# Crear gráfico de pérdida
plt.figure(figsize=(10, 5))
plt.plot(historial_perdida, label="Error cuadrático medio", color="blue")
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Pérdida (MSE)")
plt.legend()
plt.grid(True)
plt.savefig("grafico_perdida.png")
plt.close()

# Mostrar resumen de datos
total = len(datos)
entrenamiento = len(datos_entrenamiento)
prueba = len(datos_prueba)

print("Resumen de imágenes:")
print(f" - Total: {total} imágenes")
print(f" - Entrenamiento: {entrenamiento} imágenes ({entrenamiento/total*100:.1f}%)")
print(f" - Prueba: {prueba} imágenes ({prueba/total*100:.1f}%)\n")
print("Modelo guardado y gráfico 'grafico_perdida.png' generado.")
print("Modelo entrenado y guardado como 'modelo_entrenado.pkl'")
