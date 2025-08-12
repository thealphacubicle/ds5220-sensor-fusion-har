from sklearn.svm import SVC
from src.models.base import Model

class SVMModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = SVC(**self.kwargs)
