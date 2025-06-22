import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import FiniteDiffRegressor


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = FiniteDiffRegressor("Ridge", lr=1e-4)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
print(reg.loss_history_)
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RidgeCV", alphas=[0.01, 0.1, 1, 10], lr=1e-2)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("LinearRegression", optimizer="adam", lr=1e-2)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RidgeCV", alphas=[0.01, 0.1, 1, 10], optimizer="adam", lr=1e-2)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RandomForestRegressor", lr=1e-2)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RandomForestRegressor", optimizer="adam", lr=1e-2)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()
