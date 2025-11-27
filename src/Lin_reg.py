# train_linear_models.py

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import joblib


# -------------------------------------------------------------
# Evaluate a single split
# -------------------------------------------------------------
def evaluate_split(model, X, y, name=""):
    preds = model.predict(X)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print(f"\n--- {name} Evaluation ---")
    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    return {"Split": name, "MAE": mae, "RMSE": rmse, "R2": r2}


# -------------------------------------------------------------
# Train tuned linear models (Ridge, Lasso, ElasticNet)
# -------------------------------------------------------------
def train_tuned_linear_models(X_train, y_train, preprocessor):

    models = {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=10000),
        "ElasticNet": ElasticNet(max_iter=10000)
    }

    param_grids = {
        "Ridge": {"model__alpha": [0.001, 0.01, 0.1, 1, 10, 100]},
        "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1, 10]},
        "ElasticNet": {
            "model__alpha": [0.001, 0.01, 0.1, 1],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
        }
    }

    best_models = {}

    for name, model in models.items():
        print(f"\nTuning {name}...")

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[name],
            n_iter=10,
            scoring="neg_mean_absolute_error",
            cv=3,
            n_jobs=-1,
            random_state=42
        )

        search.fit(X_train, y_train)

        print(f"Best params for {name}: {search.best_params_}")
        best_models[name] = search.best_estimator_

    return best_models


# -------------------------------------------------------------
# Full workflow: Train/Test only (NO validation)
# -------------------------------------------------------------
def run_tuned_linear_models(
    X_train_clean, X_test,
    y_train_clean, y_test,
    preprocessor
):

    best_models = train_tuned_linear_models(X_train_clean, y_train_clean, preprocessor)

    all_results = []

    for name, model in best_models.items():
        print(f"\n===== {name} Results =====")

        # Train performance
        all_results.append(evaluate_split(
            model, X_train_clean, y_train_clean, f"{name} - Train"
        ))

        # Test performance
        all_results.append(evaluate_split(
            model, X_test, y_test, f"{name} - Test"
        ))

        # Save model
        model_path = f"models/{name.lower()}_tuned.pkl"
        joblib.dump(model, model_path)
        print(f"{name} model saved to {model_path}")

    return pd.DataFrame(all_results)
