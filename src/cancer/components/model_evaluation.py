"""Model evaluation component for ThoraxGuard."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import mlflow
import mlflow.keras
from tensorflow import keras

from cancer.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import save_json


class ModelEvaluation:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    @staticmethod
    def load_model(path: Path) -> keras.Model:
        if not path.is_file():
            raise FileNotFoundError(f"Trained model not found: {path}")
        return keras.models.load_model(path)

    def valid_generator(self) -> None:
        datagenerator_kwargs = {
            "rescale": 1.0 / 255,
            "validation_split": 0.20,
        }
        dataflow_kwargs = {
            "target_size": tuple(self.config.params_image_size[:-1]),
            "batch_size": self.config.params_batch_size,
            "interpolation": "bilinear",
            "class_mode": "categorical",
        }

        valid_datagenerator = keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator_obj = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.valid_generator_obj.samples == 0:
            raise ValueError(f"No validation images found in {self.config.training_data}")

    def evaluation(self) -> dict[str, float]:
        self.model = self.load_model(self.config.path_of_model)
        self.valid_generator()

        loss, accuracy = self.model.evaluate(self.valid_generator_obj)
        self.score = {
            "loss": float(loss),
            "accuracy": float(accuracy),
        }
        self.save_score()
        return self.score

    def save_score(self) -> None:
        if not hasattr(self, "score"):
            raise ValueError("Call evaluation() before save_score().")

        self.config.scores_file.parent.mkdir(parents=True, exist_ok=True)
        save_json(path=self.config.scores_file, data=self.score)

    def log_into_mlflow(self) -> None:
        if not hasattr(self, "model") or not hasattr(self, "score"):
            self.evaluation()

        mlflow.set_tracking_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(self.score)

            if tracking_url_type_store != "file":
                mlflow.keras.log_model(
                    self.model,
                    "model",
                    registered_model_name="VGG16Model",
                )
            else:
                mlflow.keras.log_model(self.model, "model")


Evaluation = ModelEvaluation
