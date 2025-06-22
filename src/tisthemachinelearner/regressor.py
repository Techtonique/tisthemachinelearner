import nnetsauce as ns
from .base import BaseModel
from sklearn.base import RegressorMixin

class Regressor(BaseModel, RegressorMixin):
    """
    Wrapper for scikit-learn regressor models.

    Parameters:
    - model_name (str): The name of the scikit-learn regressor model.
    - **kwargs: Additional parameters to pass to the scikit-learn model.

    Examples:
        ```python
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_diabetes
        from tisthemachinelearner import Regressor

        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = Regressor("LinearRegression")
        reg.fit(X_train, y_train)
        print(reg.predict(X_test))
        print(reg.score(X_test, y_test))

        reg = Regressor("RidgeCV", alphas=[0.01, 0.1, 1, 10])
        reg.fit(X_train, y_train)
        print(reg.predict(X_test))
        print(np.sqrt(np.mean((reg.predict(X_test) - y_test) ** 2)))
        ```
    """
    def __init__(self, base_model, custom=False, **kwargs):
        super().__init__(base_model, custom, **kwargs)
        if self.custom:
            self.model = ns.CustomRegressor(self.model, **self.custom_kwargs)

    def fit(self, X, y, **kwargs):
        """Fit the model."""
        super().fit(X, y, **kwargs)
        if self.custom:
            self.model.fit(X, y)
        return self