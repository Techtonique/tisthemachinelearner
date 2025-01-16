import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import Classifier, Regressor

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = Classifier("LogisticRegression", random_state=42)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))

clf = Classifier("RandomForestClassifier", n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print(clf.predict(X_test))
print(clf.score(X_test, y_test))

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = Regressor("LinearRegression")
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))

reg = Regressor("RidgeCV", alphas=[0.01, 0.1, 1, 10])
reg.fit(X_train, y_train)
print(reg.predict(X_test))
print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))

