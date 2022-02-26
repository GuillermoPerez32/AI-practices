import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

boston = datasets.load_boston()

# Seleccionar columna 6
X_ADR = boston.data[:, np.newaxis, 5]

Y_ADR = boston.target

# plt.scatter(X_ADR, Y_ADR)
# plt.show()

# Creamos modelos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X_ADR, Y_ADR, test_size=0.2)

# Defino algoritmo a usar
from sklearn.tree import DecisionTreeRegressor
algoritmo = DecisionTreeRegressor(max_depth=5)

# Entrenar el modelo
algoritmo.fit(X_TRAIN, Y_TRAIN)

# Realizar una predccion
algoritmo.predict(X_TEST)

# Graficar todos los datos juntos
X_GRID = np.arange(min(X_TEST), max(X_TEST), 0.01)
X_GRID = X_GRID.reshape(len(X_GRID), 1)

plt.scatter(X_TEST, Y_TEST)
plt.plot(X_GRID, algoritmo.predict(X_GRID), color = 'red', linewidth = 3)
plt.show()

print( algoritmo.score(X_TRAIN, Y_TRAIN))