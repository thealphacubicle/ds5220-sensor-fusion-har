from sklearn.ensemble import ExtraTreesClassifier
from src.models.base import Model


class ExtraTreesModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = ExtraTreesClassifier(**self.kwargs)
