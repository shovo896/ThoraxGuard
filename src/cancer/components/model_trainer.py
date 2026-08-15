"""Model training component for ThoraxGuard."""

from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf

from cancer.entity.config_entity import TrainingConfig


class ModelTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def get_base_model(self) -> None:
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

    def train_valid_generator(self) -> None:
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

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs,
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs,
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs,
        )

        if self.train_generator.samples == 0:
            raise ValueError(f"No training images found in {self.config.training_data}")
        if self.valid_generator.samples == 0:
            raise ValueError(f"No validation images found in {self.config.training_data}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

    def train(self) -> tf.keras.callbacks.History:
        if not hasattr(self, "model"):
            raise ValueError("Call get_base_model() before train().")
        if not hasattr(self, "train_generator") or not hasattr(self, "valid_generator"):
            raise ValueError("Call train_valid_generator() before train().")

        steps_per_epoch = math.ceil(
            self.train_generator.samples / self.train_generator.batch_size
        )
        validation_steps = math.ceil(
            self.valid_generator.samples / self.valid_generator.batch_size
        )

        history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_data=self.valid_generator,
        )

        self.save_model(path=self.config.trained_model_path, model=self.model)
        return history


Training = ModelTrainer
