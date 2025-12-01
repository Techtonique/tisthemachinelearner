tisthemachinelearner
--------

<hr>

Lightweight interface to scikit-learn with 2 classes, `Classifier` and `Regressor`. Home of `FiniteDiffRegressor` (see **Backpropagating quasi-randomized neural networks** [https://thierrymoudiki.github.io/blog/2025/06/23/python/backprop-qrnn](https://thierrymoudiki.github.io/blog/2025/06/23/python/backprop-qrnn)). 

![PyPI](https://img.shields.io/pypi/v/tisthemachinelearner) [![PyPI - License](https://img.shields.io/pypi/l/tisthemachinelearner)](LICENSE) [![Downloads](https://pepy.tech/badge/tisthemachinelearner)](https://pepy.tech/project/tisthemachinelearner) 
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/tisthemachinelearner/)


## Installing (for Python and R)

### Python 

- __1st method__: by using `pip` at the command line for the stable version

```bash
pip install tisthemachinelearner
```

- __2nd method__: from Github, for the development version

```bash
pip install git+https://github.com/Techtonique/tisthemachinelearner.git
```

or 

```bash
git clone https://github.com/Techtonique/tisthemachinelearner.git
cd tisthemachinelearner
make install
```

# Examples

```python
import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from tisthemachinelearner import Classifier, Regressor

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This software contains code derived from [scikit-learn](https://scikit-learn.org/), 
which is licensed under the BSD 3-Clause License. See [NOTICE](NOTICE) file for details.
