from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import os


class Model(ABC):
    """
    Abstract base class for all models.
    All models should inherit from this class and implement the `build_model` method.
    """

    def __init__(self):
        self.model = None  # Subclasses must set this in build_model()

    @abstractmethod
    def build_model(self):
        """
        Construct the underlying sklearn model and assign it to self.model.
        """
        pass

    def train(self, X_train, y_train):
        """
        Train the model using provided features and labels.
        """
        if self.model is None:
            self.build_model()
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict labels using the trained model.
        """
        return self.model.predict(X)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """
        Evaluate model on both training and test sets.

        Returns:
            dict: accuracy, f1, generalization gap, confusion matrix
        """
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')

        conf_matrix = confusion_matrix(y_test, y_test_pred)

        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
            "generalization_gap": train_acc - test_acc,
            "confusion_matrix": conf_matrix
        }

    def save(self, filepath: str):
        """
        Save the trained model to a file.
        """
        if self.model is None:
            raise ValueError("No model found. Train or load a model first.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"[✓] Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load a trained model from a file.
        """
        self.model = joblib.load(filepath)
        print(f"[✓] Model loaded from {filepath}")
