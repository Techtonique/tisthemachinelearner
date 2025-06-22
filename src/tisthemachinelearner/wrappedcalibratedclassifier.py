from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

class WrappedCalibratedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, method='sigmoid', cv=5):
        self.base_estimator = base_estimator
        self.method = method
        self.cv = cv

    def fit(self, X, y):
        self.calibrated_clf_ = CalibratedClassifierCV(
            estimator=self.base_estimator,
            method=self.method,
            cv=self.cv
        )
        self.calibrated_clf_.fit(X, y)
        return self

    def predict(self, X):
        return self.calibrated_clf_.predict(X)

    def predict_proba(self, X):
        return self.calibrated_clf_.predict_proba(X)

    @property
    def classes_(self):
        return self.calibrated_clf_.classes_

    @property
    def _estimator_type(self):
        return "classifier"