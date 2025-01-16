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
        from machinelearner import Regressor

        X, y = load_diabetes(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        reg = Regressor("LinearRegression")
        reg.fit(X_train, y_train)
        print(reg.predict(X_test))
        print(reg.score(X_test, y_test))
        ```
    """
    pass