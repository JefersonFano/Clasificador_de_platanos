import numpy as np

class RedNeuronal:
    def __init__(self, entrada, ocultas, salida, tasa_aprendizaje=0.1):
        self.lr = tasa_aprendizaje
        self.W1 = np.random.randn(ocultas, entrada)
        self.b1 = np.zeros((ocultas, 1))
        self.W2 = np.random.randn(salida, ocultas)
        self.b2 = np.zeros((salida, 1))

    def sigmoide(self, z):
        return 1 / (1 + np.exp(-z))

    def derivada_sigmoide(self, a):
        return a * (1 - a)

    def hacia_adelante(self, x):
        self.Z1 = self.W1 @ x + self.b1
        self.A1 = self.sigmoide(self.Z1)
        self.Z2 = self.W2 @ self.A1 + self.b2
        self.A2 = self.sigmoide(self.Z2)
        return self.A2

    def calcular_error(self, y_real, y_predicho):
        return np.mean((y_real - y_predicho) ** 2)

    def hacia_atras(self, x, y):
        m = y.shape[1]
        A2 = self.A2
        A1 = self.A1

        dZ2 = (A2 - y) * self.derivada_sigmoide(A2)
        dW2 = dZ2 @ A1.T / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = (self.W2.T @ dZ2) * self.derivada_sigmoide(A1)
        dW1 = dZ1 @ x.T / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
