import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import FiniteDiffRegressor



df = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/tabular/regression/boston_dataset2.csv")

assert 'training_index' in df.columns, \
"must have a column 'training_index' specifying which sample belongs to the training set" 

X_train = df[df['training_index'] == 1].drop(columns=['target', 'training_index'], 
                axis=1) 
y_train = df[df['training_index'] == 1]['target'].values
X_test = df[df['training_index'] == 0].drop(columns=['target', 'training_index'], 
                axis=1) 
y_test = df[df['training_index'] == 0]['target'].values 


lr = 1e-4
optimizer = "sgd"

reg = FiniteDiffRegressor("Ridge", lr=lr, optimizer=optimizer, n_hidden_features=10)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
print(reg.loss_history_)
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RidgeCV", alphas=[0.01, 0.1, 1, 10], lr=lr, optimizer=optimizer, n_hidden_features=10)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("LinearRegression", lr=lr, optimizer=optimizer, n_hidden_features=10)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()

reg = FiniteDiffRegressor("RandomForestRegressor", lr=lr, optimizer=optimizer, n_hidden_features=10)
reg.fit(X_train, y_train)
print(reg.model.W_)
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
plt.plot(reg.loss_history_)
plt.show()
