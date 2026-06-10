"""
Stage DVC 2 — train
Charge les splits, entraîne MultinomialNB (meilleur alpha trouvé par Optuna),
sauvegarde le modèle et les métriques.
Reproduit la partie NB_optuna du notebook 02_modeling.ipynb.
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import mlflow.sklearn
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MaxAbsScaler

from settings.params import (
    FEATURES_LINEAR,
    STEMMED_COL,
    TARGET_COL,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TFIDF_SUBLINEAR_TF,
)

DATA_DIR     = Path("data/processed")
MODEL_PATH   = Path("models/NB_optuna.pkl")
METRICS_PATH = Path("reports/metrics.json")

# Meilleur alpha trouvé par Optuna (30 trials, notebook 02_modeling)
BEST_ALPHA = 0.7255685482622868

train_df = pd.read_parquet(DATA_DIR / "train.parquet")
val_df   = pd.read_parquet(DATA_DIR / "val.parquet")
test_df  = pd.read_parquet(DATA_DIR / "test.parquet")

# TF-IDF sur le texte stemmé
tfidf = TfidfVectorizer(
    ngram_range=TFIDF_NGRAM_RANGE,
    max_features=TFIDF_MAX_FEATURES,
    min_df=TFIDF_MIN_DF,
    sublinear_tf=TFIDF_SUBLINEAR_TF,
)
X_tr_tfidf   = tfidf.fit_transform(train_df[STEMMED_COL])
X_val_tfidf  = tfidf.transform(val_df[STEMMED_COL])
X_test_tfidf = tfidf.transform(test_df[STEMMED_COL])

# Features numériques — MaxAbsScaler requis (MultinomialNB exige des valeurs >= 0)
scaler = MaxAbsScaler()
X_tr_feat   = scaler.fit_transform(train_df[FEATURES_LINEAR])
X_val_feat  = scaler.transform(val_df[FEATURES_LINEAR])
X_test_feat = scaler.transform(test_df[FEATURES_LINEAR])

# Combinaison TF-IDF + features numériques
X_tr   = hstack([X_tr_tfidf,   X_tr_feat])
X_val  = hstack([X_val_tfidf,  X_val_feat])
X_test = hstack([X_test_tfidf, X_test_feat])

y_tr   = train_df[TARGET_COL].astype(int)
y_val  = val_df[TARGET_COL].astype(int)
y_test = test_df[TARGET_COL].astype(int)

# Entraînement + logging MLflow
mlflow.set_experiment("disaster-tweet-classifier")
with mlflow.start_run(run_name="NB_optuna_dvc"):
    model = MultinomialNB(alpha=BEST_ALPHA)
    model.fit(X_tr, y_tr)

    y_pred_val  = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    f1_val  = round(f1_score(y_val,  y_pred_val,  average="macro"), 4)
    f1_test = round(f1_score(y_test, y_pred_test, average="macro"), 4)

    mlflow.log_params({"alpha": BEST_ALPHA, "model_type": "MultinomialNB"})
    mlflow.log_metrics({"f1_macro_val": f1_val, "f1_macro_test": f1_test})
    mlflow.sklearn.log_model(model, "model")

# Sauvegarde du modèle (même structure dict que l'original)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump({
        "model":      model,
        "tfidf":      tfidf,
        "scaler":     scaler,
        "model_name": "NB_optuna",
        "features":   FEATURES_LINEAR,
        "f1_val":     f1_val,
        "f1_test":    f1_test,
    }, f)

# Métriques pour DVC
METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(METRICS_PATH, "w") as f:
    json.dump({"f1_macro_val": f1_val, "f1_macro_test": f1_test}, f, indent=2)

print(f"F1-macro val  : {f1_val:.4f}  (référence : 0.8003)")
print(f"F1-macro test : {f1_test:.4f}  (référence : 0.7624)")
print(classification_report(y_test, y_pred_test, target_names=["Not Disaster", "Real Disaster"], digits=4))
