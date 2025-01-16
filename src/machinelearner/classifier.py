from .base import BaseModel
from sklearn.base import ClassifierMixin

class Classifier(BaseModel, ClassifierMixin):
    """
    Wrapper for scikit-learn classifier models.
    """
    pass
