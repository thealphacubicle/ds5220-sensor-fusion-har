from sklearn.linear_model import LogisticRegression
from src.models.base import Model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticRegressionModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**self.kwargs))
