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
NLP_Disaster_Tweets-/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI — tests automatiques (push/PR sur main & develop)
├── .dvc/                       # Configuration DVC
├── data/
│   └── processed/              # Parquets générés par DVC (train/val/test)
├── frontend/                   # Angular 21 — UI single/batch prediction
│   ├── src/app/
│   ├── nginx.conf
│   └── package.json
├── models/                     # Artefacts sérialisés (NB_optuna.pkl)
├── notebooks/                  # 01_EDA · 01_splitting · 02_modeling · 03_interpretability
├── reports/
│   ├── figures/                # Plots SHAP/LIME auto-générés
│   └── metrics.json            # Métriques DVC (f1_macro_val, f1_macro_test)
├── scripts/
│   ├── preprocess.py           # Stage DVC 1 — feature engineering + splits
│   └── train.py                # Stage DVC 2 — entraînement NB_optuna + MLflow
├── settings/
│   ├── config.yaml             # Paramètres DVC (tfidf, split, features)
│   └── params.py               # Constantes Python
├── src/
│   ├── api/                    # FastAPI app
│   │   ├── main.py
│   │   ├── routes/
│   │   └── schemas/
│   ├── data/                   # preprocess.py, make_dataset.py
│   ├── features/               # build.py
│   ├── models/                 # train.py, optimize.py, evaluate.py
│   └── utils/                  # config.py, logger.py, model_loader.py
├── tests/
│   ├── unit/
│   └── integration/
├── .dvcignore
├── dvc.lock                    # Verrou DVC — état reproductible du pipeline
├── dvc.yaml                    # Définition des stages DVC
├── Dockerfile
├── MakeFile
├── docker-compose.yml
├── requirements.txt
├── requirements-api.txt
└── requirements-dev.txt
```

---

## Installation

```bash
git clone https://github.com/BachirTra/NLP_Disaster_Tweets-.git
cd NLP_Disaster_Tweets-
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
| 01a | `disastertweets_01_01_eda.ipynb` | EDA report, feature engineering |
| 01b | `disastertweets_01_02_splitting.ipynb` | `data/processed/*.parquet` — splits train/val/test |
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
mlflow ui --backend-store-uri mlflow/
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

---

## Intégration continue — CI

Le pipeline CI est défini dans `.github/workflows/ci.yml` et se déclenche automatiquement à chaque **push** ou **pull request** sur `main` et `develop`.

| Étape | Action |
|---|---|
| Checkout | Récupère le code source |
| Setup Python 3.10 | Installe l'environnement |
| Install dependencies | `requirements.txt` + `requirements-api.txt` + `requirements-dev.txt` |
| Run tests | `pytest tests/ -v --cov=src --cov-fail-under=80` |

Le seuil de couverture est fixé à **80 %** — le CI échoue si la couverture tombe en dessous.

> Ce projet n'a pas de pipeline CD (Continuous Deployment) car il n'y a pas de cible de déploiement en production.
