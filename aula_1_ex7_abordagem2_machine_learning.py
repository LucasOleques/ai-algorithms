# GO0106-Abordagem2MachineLearning
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import make_regression

# 1. Gerar dados sintéticos (100 amostras)
np.random.seed(42)
X_train = 2 * np.random.rand(100, 1)  # 100 valores entre 0 e 2
y_train = 4 + 3 * X_train + np.random.randn(100, 1)  # Relação linear com ruído 

modelo = LinearRegression()
modelo.fit(X_train, y_train) # Aprende automaticamente!
print(f"Coeficiente angular (m): {modelo.coef_[0][0]:.2f}")
print(f"Coeficiente linear (b): {modelo.intercept_[0]:.2f}")

# 2. Gerar dados de teste
X_test = np.array([[0], [1], [2]])  # Testar em 0, 1 e 2
y_test = 4 + 3 * X_test  # Valores reais sem ruído

# 3. Fazer predições
y_pred = modelo.predict(X_test)
print("Predições:", y_pred.flatten())

# 4. Avaliar com MSE
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

# 5. Gerar dados de regressão mais complexos
X_train_complex, y_train_complex = make_regression(
    n_samples=100, n_features=1, noise=20, random_state=42)
modelo_complexo = LinearRegression()
modelo_complexo.fit(X_train_complex, y_train_complex)
print(f"Coeficiente angular complexo (m): {modelo_complexo.coef_[0]:.2f}")
print(f"Coeficiente linear complexo (b): {modelo_complexo.intercept_:.2f}")