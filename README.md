# Breast Cancer Diagnosis Assistant

Interactive Flask web app and companion data science scripts for predicting breast cancer malignancy from tabular measurements. The repository bundles
model-training utilities, exploratory data analysis helpers, and a lightweight web UI for clinicians or analysts to explore predictions.

## Project layout

- `app/app.py` – Flask application that loads the saved model and serves the diagnosis UI.
- `app/templates/index.html` – Bootstrap-based interface with form validation and record filtering.
- `scripts/train_models.py` – Grid-search training pipeline that selects and persists the best-performing model.
- `scripts/run_eda.py` – Standalone CLI that produces quick dataset summaries and correlations.
- `Breast_cancer_dataset.csv` – Source dataset used by the trainer and to derive feature metadata in the UI.
- `models/` – Created at runtime; stores the persisted `best_model.joblib` used by the app (ignored by git).
- `reports/` – Holds generated analysis artifacts such as the model selection report.
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` – Docker assets covering both the web app and training workflow.

## Requirements

- Docker Engine 24+ (or Docker Desktop)
- Docker Compose V2 (`docker compose` command)
- ~4 GB RAM available for scikit-learn model training

Everything else (Python, dependencies, build tools) is handled inside the container image.

## Quick start with Docker

1. **Build the image**

   ```bash
   docker compose build
   ```

2. **Train or refresh the model**

   ```bash
   docker compose run --rm trainer
   ```

   This runs `scripts/train_models.py`, writes the best pipeline to `models/best_model.joblib`, and stores search metrics in `reports/model_selection_results.json`. Both directories are volume-mounted to your workspace so the artifacts persist between runs.

3. **Launch the web app**

   ```bash
   docker compose up web
   ```

   Visit http://localhost:5000 to access the diagnosis assistant. Stop with `Ctrl+C` or `docker compose down` when finished.

The web container automatically mounts `./models`, `./reports`, and the dataset so the UI always reflects the latest artifacts you generated.

## Common workflows

- **Retrain with different settings**

  Pass any of the CLI flags from `scripts/train_models.py` through the compose service, e.g.

  ```bash
  docker compose run --rm trainer \
    python scripts/train_models.py Breast_cancer_dataset.csv \
      --model-path models/best_model.joblib \
      --output-json reports/model_selection_results.json \
      --scoring roc_auc --test-size 0.3 --cv-folds 10
  ```

- **Run exploratory analysis**

  ```bash
  docker compose run --rm web python scripts/run_eda.py Breast_cancer_dataset.csv \
    --output reports/eda_report.md \
    --heatmap-html reports/correlation_heatmap.html
  ```

  Add `--no-heatmap` to skip the HTML visualization.

- **Iterate locally while the container runs**

  Mounts keep your host files in sync. Modify code on the host, then rebuild when dependencies change:

  ```bash
  docker compose build web
  docker compose up web
  ```

## Service reference

| Service  | Purpose                         | Default command                                                                 |
|----------|---------------------------------|----------------------------------------------------------------------------------|
| `web`    | Serves the Flask UI on port 5000 | `gunicorn --bind 0.0.0.0:5000 app.app:app`                                       |
| `trainer`| One-shot model training job      | `python scripts/train_models.py Breast_cancer_dataset.csv ...` (profile `tools`) |

By default only `web` starts when you run `docker compose up`. The `trainer` service lives behind the `tools` profile so you explicitly opt-in with `docker compose run --rm trainer`.

## Troubleshooting

- **Missing model error in the UI** – Make sure you executed the trainer service at least once so `models/best_model.joblib` exists.
- **Package installation failures on build** – Retry `docker compose build` with internet access; pip installs wheels for `pandas` and `scikit-learn` during the image build.
- **Port already in use** – Override the published port: `docker compose up web -p bcapp` and edit `docker-compose.yml` to change `5000:5000` to another mapping (e.g. `8080:5000`).

## Manual execution (optional)

If you prefer not to use Docker, set up a local Python 3.11 environment and install dependencies manually:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_models.py Breast_cancer_dataset.csv
python app/app.py
```

Ensure `models/best_model.joblib` exists before launching the web app.
