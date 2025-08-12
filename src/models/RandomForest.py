from sklearn.ensemble import RandomForestClassifier
from src.models.base import Model

class RFClassifierModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = RandomForestClassifier(**self.kwargs)
