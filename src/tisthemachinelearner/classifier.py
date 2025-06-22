import nnetsauce as ns
from .base import BaseModel
from sklearn.base import ClassifierMixin
from .wrappedcalibratedclassifier import WrappedCalibratedClassifier

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
    def __init__(self, base_model, custom=False, calibrate=False, **kwargs):
        super().__init__(base_model, custom, **kwargs)
        if self.custom:
            self.model = ns.CustomClassifier(self.model, **self.custom_kwargs)
        self.calibrate = calibrate
        if self.calibrate: 
            raise NotImplementedError

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        super().fit(X, y, **kwargs)
        if self.calibrate:
            self.model = WrappedCalibratedClassifier(self.model, method='sigmoid', cv='prefit')
            self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for the input data.

        Parameters:
        - X (array-like): Input data features.

        Returns:
        - array-like: Predicted class probabilities.
        """
        return self.model.predict_proba(X)
