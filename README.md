# ThoraxGuard

ThoraxGuard is an end-to-end deep learning project for chest CT scan classification. It includes a TensorFlow/Keras training pipeline, model evaluation artifacts, a Flask prediction API, Docker packaging, and GitHub Actions based deployment to AWS EC2 through Amazon ECR.

The current prediction service classifies CT images into:

- `adenocarcinoma`
- `normal`

## Project Highlights

- TensorFlow/Keras based image classification model
- Flask web interface and REST prediction API
- Dockerized production server with Gunicorn
- Health endpoint for deployment verification
- DVC-style ML pipeline stages for ingestion, base model preparation, training, and evaluation
- GitHub Actions workflow for ECR image build/push and EC2 deployment
- Git LFS support for the trained Keras model artifact

## Repository Structure

```text
ThoraxGuard/
├── app.py                         # Flask/Gunicorn entry point
├── Dockerfile                     # Production Docker image
├── requirements.txt               # Python dependencies
├── setup.py                       # Editable package setup
├── config/
│   └── config.yaml                # Pipeline and model paths
├── params.yaml                    # Training hyperparameters
├── src/
│   ├── cnnClassifier/             # Prediction and shared utilities
│   └── cancer/                    # Training pipeline components
├── artifacts/
│   ├── training/model.keras       # Trained model artifact
│   └── evaluation/scores.json     # Evaluation metrics
├── templates/
│   └── index.html                 # Web UI
├── static/
│   └── css/style.css              # UI styling
├── dvc.yaml                       # ML pipeline definition
└── .github/workflows/main.yaml    # AWS deployment workflow
```

## Local Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the Flask app locally:

```bash
python app.py
```

The app will start at:

```text
http://127.0.0.1:8080
```

## API Endpoints

Health check:

```bash
curl http://127.0.0.1:8080/health
```

Image prediction using multipart upload:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -F "file=@/path/to/ct-image.jpg"
```

The prediction response returns the predicted class, confidence, and class probabilities.

## Docker

Build the image:

```bash
docker build -t thoraxguard:latest .
```

Run the container:

```bash
docker run --rm -p 8080:8080 -e PORT=8080 thoraxguard:latest
```

Verify:

```bash
curl http://127.0.0.1:8080/health
```

## Model Artifact

The Docker image expects the trained model at:

```text
artifacts/training/model.keras
```

This file is tracked with Git LFS because it is a large binary model artifact. Before cloning or deploying, make sure Git LFS is installed:

```bash
git lfs install
git lfs pull
```

If the model file is missing, Docker build will fail before deployment.

## Training Pipeline

The pipeline stages are defined in `dvc.yaml`:

```bash
dvc repro
```

Main stages:

- `data_ingestion`
- `prepare_base_model`
- `training`
- `evaluation`

The dataset paths in `config/config.yaml` currently point to a local dataset directory. Update those paths before running the training pipeline on a different machine.

## AWS Deployment

Deployment is handled by GitHub Actions in:

```text
.github/workflows/main.yaml
```

The workflow:

1. Builds the Docker image on GitHub-hosted Ubuntu runner.
2. Pushes the image to Amazon ECR.
3. Uses a self-hosted EC2 runner to pull and run the latest image.
4. Verifies deployment with `/health`.

Required GitHub Actions repository secrets:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_DEFAULT_REGION
ECR_REPO
```

`ECR_REPO` should be only the ECR repository name, not the full ECR URL.

Example:

```text
ECR_REPO=thoraxguard
```

## Self-Hosted Runner Note

The deploy job requires an EC2 self-hosted runner with these labels:

```text
self-hosted, Linux, X64
```

To start the runner manually on EC2:

```bash
cd ~/actions-runner
./run.sh
```

Keep the terminal open until it shows:

```text
Listening for Jobs
```

For a permanent runner service:

```bash
cd ~/actions-runner
sudo ./svc.sh install ubuntu
sudo ./svc.sh start
sudo ./svc.sh status
```

## Security Notes

- Do not commit `.env`, `.venv/`, AWS credentials, or local secret files.
- Store AWS credentials only in GitHub Actions repository secrets.
- Rotate any AWS access key that was shared publicly or exposed in screenshots.

## Tech Stack

- Python 3.12
- TensorFlow 2.16
- Flask
- Gunicorn
- Docker
- GitHub Actions
- Amazon ECR
- AWS EC2
- Git LFS
- DVC

