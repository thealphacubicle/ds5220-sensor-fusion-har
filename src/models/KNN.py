from sklearn.neighbors import KNeighborsClassifier
from src.models.base import Model

class KNNModel(Model):
    def __init__(self, **kwargs):
        super().__init__()
        # KNeighborsClassifier does not accept a random_state parameter.
        # Remove it gracefully if supplied to maintain a consistent interface
        # with other models that use random_state.
        kwargs.pop("random_state", None)
        self.kwargs = kwargs

    def build_model(self):
        self.model = KNeighborsClassifier(**self.kwargs)
