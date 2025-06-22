import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import FiniteDiffRegressor


X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = FiniteDiffRegressor("GBDTRegressor", model_type='xgboost', n_estimators=200)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
print(reg.loss_history_)
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("GBDTRegressor", model_type='lightgbm', n_estimators=200)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("GBDTRegressor", model_type='catboost', n_estimators=200)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

