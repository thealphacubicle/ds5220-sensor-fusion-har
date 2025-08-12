from sklearn.naive_bayes import GaussianNB
from src.models.base import Model

class NaiveBayesModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def build_model(self):
        self.model = GaussianNB(**self.kwargs)
