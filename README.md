# Disaster Tweet Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![Angular](https://img.shields.io/badge/Angular-21-red)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

Binary NLP classification: predict whether a tweet refers to a **real disaster** (1) or not (0).  
Stack: scikit-learn · Multinomial Naive Bayes · TF-IDF · MLflow · Optuna · FastAPI · Angular 21.

---

## Repository structure

```
disaster-tweets-mlops/
├── data/
│   ├── raw/
│   └── processed/
├── frontend/             # Angular 21 — single/batch prediction UI
│   ├── src/app/          # Component, service, template, styles
│   ├── nginx.conf        # SPA routing config (Docker)
│   └── package.json
├── models/               # Serialised artifacts (dill)
├── notebooks/            # 01 EDA · 02 Modeling · 03 Interpretability
├── reports/figures/      # Auto-generated plots
├── settings/             # config.yaml, params.py
├── src/
│   ├── api/              # FastAPI app
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   ├── data/             # preprocess.py, make_dataset.py
│   ├── features/         # build.py
│   ├── models/           # evaluate.py
│   └── utils/            # config.py, logger.py, model_loader.py
├── tests/
│   ├── unit/
│   └── integration/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
└── requirements-dev.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/disaster-tweets-mlops.git
cd disaster-tweets-mlops
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
# Version Python utilisée : 3.10.18 (requise pour la compatibilité avec requirements.txt)
pip install -r requirements.txt
pip install -r requirements-api.txt   # FastAPI + uvicorn
pip install -r requirements-dev.txt   # pytest + coverage
```

Copy the environment template:

```bash
cp .env.example .env
```

---

## Data

Le dataset est chargé automatiquement depuis [OpenML](https://www.openml.org) via `fetch_openml` — aucun téléchargement manuel nécessaire.

```python
from sklearn.datasets import fetch_openml
fetch_openml(name="disaster-tweets", as_frame=True, version="active")
```

---

## Running the notebooks

Execute in order — each notebook depends on outputs from the previous one:

| # | Notebook | Output |
|---|---|---|
| 01 | `disastertweets_01_01_eda.ipynb` | EDA report, feature engineering |
| 02 | `disastertweets_02_modeling.ipynb` | `models/NB_optuna.pkl`, MLflow runs |
| 03 | `disastertweets_03_interpretability.ipynb` | SHAP/LIME plots, robustness analysis |

```bash
jupyter notebook notebooks/
```

---

## Running the API locally

```bash
uvicorn src.api.main:app --reload --port 8000
```

- Swagger UI: http://localhost:8000/docs  
- ReDoc:      http://localhost:8000/redoc  
- Health:     http://localhost:8000/health  

---

## Running the frontend locally

```bash
cd frontend
npm install
npm start        # ng serve
```

Open http://localhost:4200 — the API must also be running on port 8000.

---

## Running with Docker

**Full stack (API + frontend):**

```bash
docker-compose up --build
```

| Service | URL |
|---|---|
| Frontend UI | http://localhost:4200 |
| API (Swagger) | http://localhost:8000/docs |
| API (health) | http://localhost:8000/health |

**API only:**

```bash
docker-compose up --build api
```

Override environment variables:

```bash
MODEL_PATH=models/NB_optuna.pkl LOG_LEVEL=DEBUG docker-compose up
```

---

## Tests

```bash
# Full suite with coverage report
make test-cov

# Or directly
pytest tests/ -v --cov=src --cov-report=html --cov-fail-under=80
```

Coverage report is generated in `htmlcov/index.html`.

---

## Example API call

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Massive wildfire spreading across California hills, evacuations ordered"}'
```

Expected response:

```json
{
  "prediction": 1,
  "label": "disaster",
  "probability_disaster": 0.9312,
  "probability_not_disaster": 0.0688,
  "model_version": "NB_optuna",
  "inference_time_ms": 2.451
}
```

Batch endpoint:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"tweets": [{"text": "Earthquake hits Turkey"}, {"text": "Pizza night!"}]}'
```

---

## Model performance

| Model | Optimiser | F1-val | F1-test | Notes |
|---|---|---|---|---|
| NB_optuna | Optuna 30 trials (NB/LR/SGD/LinearSVC) + 50 trials (XGBoost) | **0.8003** | **0.7624** | TF-IDF unigrams+bigrams, α=0.7255 |

Confusion matrix (test set, n=1 706):

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 1 227 TN | 162 FP |
| **Actual 1** | 101 FN | 216 TP |

---

## MLflow experiment tracking

```bash
mlflow ui
# ou avec chemin explicite :
mlflow ui --backend-store-uri mlruns/
# Open http://localhost:5000
```

All training runs (baselines + Optuna) are tracked under experiment `disaster-tweet-classifier`.  
The final model is registered as `disaster-tweet-classifier` in the MLflow Model Registry.

---

## Reproducible pipeline — DVC

Le pipeline ML est reproductible via DVC (Data Version Control) :

```bash
dvc repro
```

Deux étapes sont définies dans `dvc.yaml` :

| Étape | Script | Sortie |
|-------|--------|--------|
| `preprocess` | `scripts/preprocess.py` | `data/processed/*.parquet` |
| `train` | `scripts/train.py` | `models/NB_optuna.pkl`, `reports/metrics.json` |

Afficher les métriques :

```bash
dvc metrics show
```
