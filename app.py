"""Root Flask entry point for ThoraxGuard prediction service."""

import os

from cnnClassifier.pipeline.prediction import app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    debug = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
    app.run(host="0.0.0.0", port=port, debug=debug)
