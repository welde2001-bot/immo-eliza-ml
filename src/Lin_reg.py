# train_linear_models.py

import os
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib


def evaluate_split(model, X, y, name=""):
    """Compute MAE, RMSE, and R² for a fitted pipeline/model on a given split."""
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"\n--- {name} Evaluation ---")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    return {"Split": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_tuned_linear_models(
    X_train,
    y_train,
    preprocessor,
    cv: int = 3,
    random_state: int = 42,
):
    """
    Tune Ridge, Lasso, and ElasticNet using RandomizedSearchCV.

    Notes:
    - Lasso/ElasticNet often show convergence warnings on sparse one-hot data.
      We reduce those warnings by increasing max_iter, using selection="random",
      and using a slightly looser tol.
    - The returned estimators are full pipelines: (preprocessor -> model).
    """
    models = {
        "Ridge": Ridge(random_state=random_state),
        "Lasso": Lasso(max_iter=20000, tol=1e-3, selection="random", random_state=random_state),
        "ElasticNet": ElasticNet(max_iter=20000, tol=1e-3, selection="random", random_state=random_state),
    }

    # Regularization grids; kept simple and stable for beginner baselines
    param_grids = {
        "Ridge": {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        "Lasso": {"model__alpha": [0.01, 0.1, 1, 10, 100, 1000]},
        "ElasticNet": {
            "model__alpha": [0.01, 0.1, 1, 10, 100],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    }

    best_models = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")

        # Pipeline ensures preprocessing happens inside CV (no leakage)
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        # Avoid "space smaller than n_iter" warnings by capping n_iter
        alpha_size = len(param_grids[name].get("model__alpha", []))
        l1_size = len(param_grids[name].get("model__l1_ratio", [1]))
        n_iter = min(10, alpha_size * l1_size)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
        )

        search.fit(X_train, y_train)

        print(f"Best params for {name}: {search.best_params_}")
        best_models[name] = search.best_estimator_

    return best_models


def run_tuned_linear_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    models_dir: str = "models",
):
    """
    Full workflow (train/test only):
    - Tune models on training data with CV
    - Report metrics on train and test
    - Save best pipelines to disk
    """
    os.makedirs(models_dir, exist_ok=True)

    best_models = train_tuned_linear_models(X_train, y_train, preprocessor)

    all_results = []

    for name, model in best_models.items():
        print(f"\n===== {name} Results =====")

        # Train metrics (diagnose over/underfit)
        all_results.append(evaluate_split(model, X_train, y_train, f"{name} - Train"))

        # Test metrics (true generalization)
        all_results.append(evaluate_split(model, X_test, y_test, f"{name} - Test"))

        # Save full pipeline (preprocessing + model) for reuse
        model_path = os.path.join(models_dir, f"{name.lower()}_tuned.pkl")
        joblib.dump(model, model_path)
        print(f"{name} model saved to {model_path}")

    return pd.DataFrame(all_results)
