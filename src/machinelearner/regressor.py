from .base import BaseModel
from sklearn.base import RegressorMixin

class Regressor(BaseModel, RegressorMixin):
    """
    Wrapper for scikit-learn regressor models.
    """
    pass
