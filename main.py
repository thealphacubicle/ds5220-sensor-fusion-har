import yaml
import polars as pl
import logging
from pathlib import Path
from src.models.LogisticRegression import LogisticRegressionModel
from src.models.RandomForest import RFClassifierModel
from src.models.KNN import KNNModel
from src.models.SVM import SVMModel
from src.models.NaiveBayes import NaiveBayesModel
from src.models.ExtraTrees import ExtraTreesModel
from src.utils.trainer import Trainer, train_single_model
from knockknock.desktop_sender import desktop_sender
from src.utils import preprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Map string class names from YAML to actual classes
MODEL_CLASS_MAP = {
    "LogisticRegressionModel": LogisticRegressionModel,
    "KNNModel": KNNModel,
    "RandomForestClassifier": RFClassifierModel,
    "GaussianNBModel": NaiveBayesModel,
    "SVMModel": SVMModel,
    "ExtraTreesClassifier": ExtraTreesModel
}

VALID_MODEL_PARAMS = {
    "logistic_ovr": ["max_iter", "multi_class", "solver", "random_state"],
    "logistic_multinomial": ["max_iter", "multi_class", "solver", "random_state"],
    "knn": ["n_neighbors"],
    "svm": ["C", "kernel", "gamma"],
    "random_forest_classifier": ["n_estimators", "max_depth", "random_state"],
    "extra_trees_classifier": ["n_estimators", "max_depth", "random_state"],
    "naive_bayes": [],
}


def sanitize_params(model_name: str, params: dict) -> dict:
    """Return only parameters valid for the specified model."""
    allowed = VALID_MODEL_PARAMS.get(model_name, [])
    return {k: v for k, v in params.items() if k in allowed}


def get_model_instance(class_name: str, params: dict = None):
    """
    Instantiates the model class dynamically using provided parameters.
    """
    model_cls = MODEL_CLASS_MAP.get(class_name)
    if not model_cls:
        logging.error(f"Unknown model class: {class_name}")
        raise ValueError(f"Unknown model class: {class_name}")
    logging.info(f"Instantiating model: {class_name} with params: {params}")
    return model_cls(**params) if params else model_cls()


def load_yaml(path: str) -> dict:
    """
    Loads a YAML configuration file.
    """
    logging.info(f"Loading YAML config from: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_feature_groups(cols: list[str]) -> dict[str, list[str]]:
    """Build accelerometer, gyroscope, and fused feature lists."""
    # Exclude angle-based features to keep modality-specific sets
    cols = [c for c in cols if not c.startswith("angle(")]

    accel_prefixes = [
        "tBodyAcc",
        "tGravityAcc",
        "tBodyAccJerk",
        "fBodyAcc",
        "fBodyAccJerk",
        "tBodyAccMag",
        "tGravityAccMag",
        "tBodyAccJerkMag",
        "fBodyAccMag",
        "fBodyAccJerkMag",
    ]
    gyro_prefixes = [
        "tBodyGyro",
        "tBodyGyroJerk",
        "fBodyGyro",
        "tBodyGyroMag",
        "tBodyGyroJerkMag",
        "fBodyGyroMag",
        "fBodyGyroJerkMag",
    ]

    def select(prefixes: list[str]) -> list[str]:
        return [c for c in cols if any(c.startswith(p) for p in prefixes)]

    accel_cols = select(accel_prefixes)
    gyro_cols = select(gyro_prefixes)
    fused_cols = sorted(set(accel_cols + gyro_cols))

    return {"accel": accel_cols, "gyro": gyro_cols, "fused": fused_cols}


def main():
    # === Load Configs ===
    logging.info("Loading model and training configs...")
    models_config = load_yaml("config/models.yaml")
    training_config = Trainer.load_training_config("config/training_config.yaml")

    notify_config = training_config.get("notify", {"send": False, "type": "desktop"})
    send_notifications = (
        notify_config.get("send", False)
        and notify_config.get("type", "desktop") == "desktop"
    )

    if send_notifications:
        @desktop_sender(title="Training experiment started")
        def _notify_start():
            pass

        _notify_start()

    models = [model_name for model_name in models_config["models"]
              if models_config["models"][model_name].get("enabled", False)]

    logging.info(f"Trainable models: {models}")

    sensors = models_config["sensors"]
    logging.info(f"Trainable sensors: {sensors}")

    # Hyperparameter grid for each model
    hyper_grid = training_config.get("hyperparameters", {})

    # === Load Data ===
    logging.info("Loading training and test data...")
    train_path = Path("data/processed/train.parquet")
    test_path = Path("data/processed/test.parquet")

    if not train_path.exists() or not test_path.exists():
        logging.info("Processed parquet files not found. Running preprocessing...")
        preprocess.download_and_extract()
        preprocess.preprocess_and_save()

    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)

    X_train = train_df.drop(["activity", "subject"])
    y_train = train_df["activity"]
    X_test = test_df.drop(["activity", "subject"])
    y_test = test_df["activity"]

    # Pre-compute feature subsets for each sensor configuration
    all_cols = X_train.columns
    sensor_feature_map = build_feature_groups(all_cols)

    # === Train Models Sequentially ===
    for model_name in models:
        details = models_config["models"][model_name]
        class_name = details["class"]
        base_params = sanitize_params(model_name, details.get("params", {}))

        # Load hyperparameter combinations; default to one set if none provided
        param_grid = hyper_grid.get(model_name, [{}])

        for sensor in sensors:
            # Select features based on sensor configuration
            sensor_X_train = X_train.select(sensor_feature_map.get(sensor, all_cols))
            sensor_X_test = X_test.select(sensor_feature_map.get(sensor, all_cols))

            def run_combinations():
                for combo in param_grid:
                    combo_params = sanitize_params(model_name, {**base_params, **combo})
                    logging.info(
                        f"Training model: {model_name}, sensor: {sensor}, params: {combo_params}"
                    )
                    try:
                        model_instance = get_model_instance(class_name, combo_params)
                    except Exception as exc:
                        logging.exception(
                            f"Failed to instantiate model {model_name} with params {combo_params}: {exc}"
                        )
                        continue

                    trainer_args = (
                        model_instance,
                        model_name,
                        sensor,
                        sensor_X_train,
                        y_train,
                        sensor_X_test,
                        y_test,
                    )

                    train_single_model(trainer_args)

            if send_notifications:
                notifier = desktop_sender(
                    title=f"Finished {model_name} on {sensor}"
                )(run_combinations)
                notifier()
            else:
                run_combinations()

    logging.info("All training jobs completed.")

    if send_notifications:
        @desktop_sender(title="Training experiment completed")
        def _notify_end():
            pass

        _notify_end()


if __name__ == "__main__":
    main()
