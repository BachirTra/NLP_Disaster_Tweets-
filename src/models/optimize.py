from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from xgboost import XGBClassifier


def make_objective_lr(X_train, y_train, X_val, y_val, random_state):
    def objective(trial):
        C        = trial.suggest_float("C", 1e-3, 10.0, log=True)
        solver   = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
        max_iter = trial.suggest_int("max_iter", 500, 2000)
        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter,
                                   class_weight="balanced", random_state=random_state)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average="macro")
    return objective


def make_objective_nb(X_train, y_train, X_val, y_val):
    def objective(trial):
        alpha     = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
        fit_prior = trial.suggest_categorical("fit_prior", [True, False])
        model = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average="macro")
    return objective


def make_objective_sgd(X_train, y_train, X_val, y_val, random_state):
    def objective(trial):
        loss     = trial.suggest_categorical("loss", ["modified_huber", "log_loss"])
        alpha    = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        penalty  = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet"])
        max_iter = trial.suggest_int("max_iter", 500, 2000)
        model = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, max_iter=max_iter,
                              class_weight="balanced", random_state=random_state)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average="macro")
    return objective


def make_objective_svc(X_train, y_train, X_val, y_val, random_state):
    def objective(trial):
        C        = trial.suggest_float("C", 1e-3, 10.0, log=True)
        max_iter = trial.suggest_int("max_iter", 1000, 5000)
        model = LinearSVC(C=C, max_iter=max_iter,
                          class_weight="balanced", random_state=random_state)
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average="macro")
    return objective


def make_objective_xgb(X_train, y_train, X_val, y_val, scale_pos_weight, random_state):
    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return f1_score(y_val, model.predict(X_val), average="macro")
    return objective
