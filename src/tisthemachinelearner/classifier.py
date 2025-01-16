from .base import BaseModel
from sklearn.base import ClassifierMixin

class Classifier(BaseModel, ClassifierMixin):
    """
    Wrapper for scikit-learn classifier models.

    Parameters:
    - model_name (str): The name of the scikit-learn classifier model.
    - **kwargs: Additional parameters to pass to the scikit-learn model.

    Examples:
        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_breast_cancer
        from tisthemachinelearner import Classifier

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
        ```
    """
    pass