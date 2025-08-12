import tempfile
import joblib
import uuid
import os
import wandb
import yaml
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_single_model(
    trainer_args: Tuple,
) -> Dict[str, Any]:
    """Train a single model using a Trainer instance."""
    trainer = Trainer(*trainer_args)
    config = {"model": trainer.model_name, "sensor_config": trainer.sensor_config}
    config.update(getattr(trainer.model, "kwargs", {}))
    run_name = f"{trainer.model_name}_{trainer.sensor_config}_{uuid.uuid4().hex[:6]}"

    wandb_kwargs = {
        "project": "sensor-fusion-har",
        "name": run_name,
        "config": config,
        "settings": wandb.Settings(start_method="thread"),
    }
    if not os.getenv("WANDB_API_KEY"):
        logger.info("WANDB_API_KEY not found. Using local offline logging.")
        wandb_kwargs["mode"] = "offline"

    with wandb.init(**wandb_kwargs):
        results = trainer.train_and_evaluate()
    log_msg = f"[✓] Finished training {trainer.model_name} on {trainer.sensor_config}"
    logger.info(log_msg.replace("`", "'"))
    return results


class Trainer:
    """
    Trainer class for supervised model training and evaluation.

    This class handles model training, evaluation, logging to Weights & Biases,
    and saving models as artifacts. It supports loading a training configuration
    from YAML and can be used in multiprocessing setups.
    """

    def __init__(
            self,
            model: Any,
            model_name: str,
            sensor_config: str,
            X_train: Any,
            y_train: Any,
            X_test: Any,
            y_test: Any,
            log_to_wandb: bool = True,
            wandb_project: str = "sensor-fusion-har",
            save_path: str = "models/",
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.sensor_config = sensor_config

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.log_to_wandb = log_to_wandb
        self.wandb_project = wandb_project
        self.save_path = Path(save_path)

    def train_and_evaluate(self) -> Dict[str, Any]:
        """
        Train the model, evaluate it on both train and test data,
        log metrics and artifacts to W&B, and return results.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"Training model: {self.model_name} on sensor config: {self.sensor_config}")

        self.model.train(self.X_train, self.y_train)
        results = self.model.evaluate(self.X_train, self.y_train, self.X_test, self.y_test)

        if self.log_to_wandb and wandb.run:
            self._log_metrics(results)
            self._log_confusion_matrix(results["confusion_matrix"])
            self._save_model()

        return results

    def _save_model(self) -> None:
        """
        Save the trained model to Weights & Biases as an artifact.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_file:
            joblib.dump(self.model.model, tmp_file.name)

            artifact = wandb.Artifact(
                name=f"{self.model_name}_{self.sensor_config}_model",
                type="model",
                description=f"{self.model_name} trained on {self.sensor_config}",
                metadata={
                    "model": self.model_name,
                    "sensor_config": self.sensor_config,
                },
            )
            artifact.add_file(tmp_file.name)
            if wandb.run:
                wandb.log_artifact(artifact)

    def _log_metrics(self, results: Dict[str, Any]) -> None:
        """
        Log evaluation metrics to Weights & Biases.

        Args:
            results (Dict[str, Any]): Dictionary containing metric results.
        """
        if not wandb.run:
            return
        metrics = {k: v for k, v in results.items() if k != "confusion_matrix"}
        # Log metrics using the active run to ensure consistency
        wandb.log(metrics)

    def _log_confusion_matrix(self, conf_matrix: Any) -> None:
        """
        Save and log the confusion matrix to W&B.

        Args:
            conf_matrix (np.ndarray): Confusion matrix.
        """
        output_dir = Path("outputs/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()

        plot_path = output_dir / f"conf_matrix_{self.model_name}_{self.sensor_config}.png"
        plt.savefig(plot_path)
        if wandb.run:
            wandb.log({"confusion_matrix": wandb.Image(str(plot_path))})
        plt.close()

    @staticmethod
    def load_training_config(config_path: str = "config/training_config.yaml") -> dict:
        """
        Loads training configuration from a YAML file.

        Args:
            config_path (str): Relative or absolute path to the YAML config file.

        Returns:
            dict: Parsed YAML configuration.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Training config file not found at {config_path}")
        with open(config_file, "r") as f:
            return yaml.safe_load(f)
