"""Root Flask entry point for ThoraxGuard prediction service."""

from cnnClassifier.pipeline.prediction import app


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
