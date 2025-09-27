# Breast Cancer Diagnosis Assistant

Interactive analytics project that trains machine-learning models on breast-tissue measurements and exposes an ergonomic web UI to support clinical decision making. The repository bundles reproducible training jobs, exploratory data analysis tooling, and a Dockerized deployment workflow.

## Business Problem

Breast cancer screening produces dozens of numeric measurements per exam. Clinicians need a quick, explainable indicator of malignancy risk so they can prioritize follow-up care. This project streamlines that workflow by:
- Ingesting standardized diagnostic measurements (mean radius, texture, etc.).
- Producing a probability of malignancy with a best-in-class model chosen via cross-validation.
- Recording predictions for later review and filtering so teams can audit previous diagnoses.

## Dataset & Usage

- **Source**: Breast Cancer Wisconsin (Diagnostic) dataset (included as `Breast_cancer_dataset.csv`).
- **Target variable**: `diagnosis` (`M` = malignant, `B` = benign).
- **Features**: Thirty continuous attributes derived from digitized FNA images plus IDs.
- **Usage considerations**: The dataset contains protected health information. Keep artifacts (models, reports) in controlled environments. The UI enforces positive numeric inputs to align with raw data distributions.

## Approach & Architecture

- **Data preparation**: `scripts/run_eda.py` cleans column headers, audits missing values, derives descriptive statistics, and produces Markdown/HTML reports.
- **Model training**: `scripts/train_models.py` builds a preprocessing + classifier pipeline using scikit-learn, performs GridSearchCV across multiple estimators, evaluates on a stratified hold-out set, and persists the top-performing pipeline with `joblib`.
- **Application layer**: `app/app.py` is a Flask service that loads the serialized pipeline, dynamically derives feature metadata, validates user input, and serves predictions through a Bootstrap-based UI.
- **State & storage**: Generated artifacts live in `models/` (best pipeline) and `reports/` (EDA summaries, grid-search metrics), both ignored by git but mounted into containers to persist results.
- **Containerization**: A single `Dockerfile` installs Python 3.11 dependencies. `docker-compose.yml` defines two services:
  - `web`: serves the Flask UI via Gunicorn on port 5000.
  - `trainer`: one-off job for retraining; gated behind the `tools` profile so it runs on demand.

## Code Organization & Conventions

```
app/
  app.py              # Flask routes, prediction logic, filter utilities.
  templates/index.html
scripts/
  train_models.py     # Model selection + persistence workflow (documented CLI flags).
  run_eda.py          # Lightweight EDA pipeline with markdown + heatmap outputs.
reports/
  eda_report.md       # Generated summary (regenerate as needed).
  correlation_heatmap.html
Breast_cancer_dataset.csv
Dockerfile
docker-compose.yml
requirements.txt
```

Coding guidelines:
- Python modules favour dataclasses, type hints, and structured exceptions for clarity.
- Input validation and error messages bubble up to the UI with actionable feedback.
- Comments are reserved for non-obvious logic (e.g., filter parsing) to keep the codebase concise yet readable.

## Running & Testing the System

### Option 1: Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Train / refresh the persisted model
python scripts/train_models.py Breast_cancer_dataset.csv \
  --model-path models/best_model.joblib \
  --output-json reports/model_selection_results.json

# (Optional) regenerate EDA artifacts
python scripts/run_eda.py Breast_cancer_dataset.csv \
  --output reports/eda_report.md \
  --heatmap-html reports/correlation_heatmap.html

# Serve the web UI for manual testing
python app/app.py
```

Visit `http://127.0.0.1:5000`, supply sample measurements, and confirm predictions display alongside probability scores. Stop the server with `Ctrl+C` and deactivate the environment with `deactivate` when finished.

### Option 2: Docker workflow

```bash
docker compose build

# Train inside the container (runs GridSearch and writes artifacts to ./models & ./reports)
docker compose run --rm trainer

# Launch the production-style web stack
docker compose up web
```

Navigate to `http://localhost:5000`. Use `Ctrl+C` followed by `docker compose down` to stop and clean up containers.

### Verification checklist

- Confirm `models/best_model.joblib` exists before starting the UI (otherwise the app flashes a setup error).
- Inspect `reports/model_selection_results.json` to review cross-validation metrics.
- Optional smoke test: submit a known benign record (from the CSV) and ensure the predicted label matches the dataset entry.

## Extending & Customizing

- Pass alternate GridSearch settings:
  ```bash
  docker compose run --rm trainer \
    python scripts/train_models.py Breast_cancer_dataset.csv \
      --scoring roc_auc --test-size 0.3 --cv-folds 10
  ```
- Skip the heatmap visualization by appending `--no-heatmap` to `run_eda.py`.
- Swap the serving stack to local development mode by exporting `FLASK_ENV=development` and running `flask --app app.app run`.

## Troubleshooting

- **Docker command stalls**: Ensure Docker Desktop is running, then rerun `docker compose build`.
- **Model file missing**: Execute the trainer service or local training command to create `models/best_model.joblib` before hitting the UI.
- **Port already bound**: Edit `docker-compose.yml` to change `5000:5000` to `8080:5000`, or stop other services occupying the port.
- **Dependency install errors**: Re-run installs with a stable internet connection; the base image compiles `scikit-learn` wheels when cached binaries are unavailable.

## Ways to Contribute

- Add automated tests (e.g., pytest for form validation and prediction endpoints).
- Experiment with alternative model families or feature engineering steps and log results in `reports/`.
- Integrate authentication or audit trails if deploying in regulated environments.

---

With the model artifacts in place and either workflow (local or Docker) running, the project provides an end-to-end demonstration of ML-assisted breast cancer diagnosis.
