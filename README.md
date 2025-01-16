machinelearner
--------

<hr>

Lightweight interface to scikit-learn with 2 classes, Classifier and Regressor. 

![PyPI](https://img.shields.io/pypi/v/machinelearner) [![PyPI - License](https://img.shields.io/pypi/l/machinelearner)](https://github.com/thierrymoudiki/machinelearner/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/machinelearner)](https://pepy.tech/project/machinelearner) 
[![HitCount](https://hits.dwyl.com/Techtonique/machinelearner.svg?style=flat-square)](http://hits.dwyl.com/Techtonique/machinelearner)
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/machinelearner/)


## Installing (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install machinelearner
```

- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/machinelearner.git
```

or 

```bash
git clone https://github.com/Techtonique/machinelearner.git
cd machinelearner
make install
```

# Examples

```python
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from machinelearner import Classifier, Regressor

# Classification
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

# Regression
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
```

## License

[BSD 3-Clause](LICENSE) © T. Moudiki, 2025. 
