from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import importlib


class BaseModel(BaseEstimator):
    """
    Base class for dynamically loading and wrapping scikit-learn models.
    """
    def __init__(self, base_model, **kwargs):
        """
        Initialize a scikit-learn model dynamically.

        Parameters:
        - base_model (str): The class name of the scikit-learn model (e.g., 'LogisticRegression').
        - **kwargs: Additional parameters to pass to the scikit-learn model constructor.
        """
        sklearn_modules = [
            "linear_model",
            "ensemble",
            "neural_network",
            "svm",
            "neighbors",
            "tree",
            "discriminant_analysis",
            "gaussian_process",
            "naive_bayes",
            "kernel_ridge",
        ]
        self.base_model = base_model
        self.model_params = kwargs
        self.model = self._load_model(base_model, sklearn_modules)(**kwargs)

    def _load_model(self, base_model, modules):
        """
        Load a model class from scikit-learn modules.

        Parameters:
        - base_model (str): The class name of the scikit-learn model.
        - modules (list): List of scikit-learn submodules to search.

        Returns:
        - class: The loaded scikit-learn model class.
        """
        for module_name in modules:
            try:
                module = importlib.import_module(f"sklearn.{module_name}")
                return getattr(module, base_model)
            except (ImportError, AttributeError):
                continue

        raise ImportError(f"Model '{base_model}' not found in scikit-learn modules.")

    def fit(self, X, y, **kwargs):
        """
        Fit the model to the training data.

        Parameters:
        - X (array-like): Training data features.
        - y (array-like): Target values.
        - **kwargs: Additional parameters to pass to the scikit-learn model fit method.
        """
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """
        Predict using the trained model.

        Parameters:
        - X (array-like): Input data.
        - **kwargs: Additional parameters to pass to the scikit-learn model predict method.
        Returns:
        - array-like: Predictions.
        """
        return self.model.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        """
        Return the score of the model on the given test data and labels.

        Parameters:
        - X (array-like): Test data features.
        - y (array-like): True labels.
        - **kwargs: Additional parameters to pass to the scikit-learn model score method.
        
        Returns:
        - float: The score.
        """
        return self.model.score(X, y, **kwargs)




