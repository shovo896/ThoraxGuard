"""Flask prediction API for the ThoraxGuard CT classifier."""

from __future__ import annotations

import base64
import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cnnClassifier.utils.common import read_yaml  # noqa: E402


CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH = PROJECT_ROOT / "params.yaml"

CLASS_LABELS = {
    0: "adenocarcinoma",
    1: "normal",
}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    @app.get("/")
    def index():
        return """
        <h2>ThoraxGuard Prediction API</h2>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required />
            <button type="submit">Predict</button>
        </form>
        """

    @app.get("/health")
    def health():
        model_path = get_model_path()
        return jsonify(
            {
                "status": "ok",
                "model_path": str(model_path),
                "model_exists": model_path.is_file(),
            }
        )

    @app.post("/predict")
    def predict():
        try:
            image = read_image_from_request()
            result = predict_image(image)
            return jsonify(result)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    return app


def get_model_path() -> Path:
    config = read_yaml(CONFIG_FILE_PATH)
    model_path = Path(config.training.trained_model_path).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path.resolve()


def get_target_size() -> tuple[int, int]:
    params = read_yaml(PARAMS_FILE_PATH)
    image_size = params.IMAGE_SIZE
    return int(image_size[0]), int(image_size[1])


@lru_cache(maxsize=1)
def load_prediction_model() -> keras.Model:
    model_path = get_model_path()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run training first."
        )
    return keras.models.load_model(model_path)


def read_image_from_request() -> Image.Image:
    if "file" in request.files:
        uploaded_file = request.files["file"]
        if not uploaded_file.filename:
            raise ValueError("Uploaded file is empty.")
        return Image.open(uploaded_file.stream).convert("RGB")

    payload: dict[str, Any] = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    if image_data:
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    raise ValueError("Send an image using form field 'file' or JSON key 'image'.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    target_size = get_target_size()
    image = image.resize(target_size)
    image_array = keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(image: Image.Image) -> dict[str, Any]:
    model = load_prediction_model()
    batch = preprocess_image(image)
    probabilities = model.predict(batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])

    return {
        "prediction": CLASS_LABELS[predicted_index],
        "confidence": confidence,
        "probabilities": {
            CLASS_LABELS[index]: float(probability)
            for index, probability in enumerate(probabilities)
        },
    }


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
