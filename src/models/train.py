import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, accuracy_score

from src.utils.logger import logger


def train_and_log(model, X_tr, y_tr, X_val, y_val,
                  name: str, params: dict = None,
                  tags: dict = None, results: list = None) -> dict:
    with mlflow.start_run(run_name=name):
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)

        metrics = {
            "model":     name,
            "f1_macro":  round(f1_score(y_val, y_pred, average="macro"), 4),
            "f1_class0": round(f1_score(y_val, y_pred, pos_label=0), 4),
            "f1_class1": round(f1_score(y_val, y_pred, pos_label=1), 4),
            "accuracy":  round(accuracy_score(y_val, y_pred), 4),
        }

        mlflow.log_params(params or {})
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "model"})
        if tags:
            mlflow.set_tags(tags)
        mlflow.sklearn.log_model(model, "model")

    if results is not None:
        results.append(metrics)
    logger.info(f"{name:25} | F1-macro={metrics['f1_macro']:.4f} | acc={metrics['accuracy']:.4f}")
    return metrics
