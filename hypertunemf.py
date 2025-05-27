import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import dagshub
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# -------------------------------
# DagsHub Remote MLflow Tracking
# -------------------------------
dagshub.init(repo_owner='anshu989856', repo_name='TESTING', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Anshu989856/TESTING.mlflow")

# -------------------------------
# Load dataset
# -------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model and hyperparameter grid
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Set experiment
mlflow.set_experiment('breast-cancer-rf-hp')

# -------------------------------
# Start Parent MLflow Run
# -------------------------------
with mlflow.start_run(run_name="RandomForest_GridSearchCV") as parent_run:
    # Fit GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log each trial as a nested run
    for i, params in enumerate(grid_search.cv_results_['params']):
        with mlflow.start_run(run_name=f"trial_{i+1}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", grid_search.cv_results_['mean_test_score'][i])

    # Log best model info
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_accuracy", best_score)

    # -------------------------------
    # Log dataset using mlflow.data
    # -------------------------------
    train_dataset = mlflow.data.from_pandas(X_train.assign(target=y_train))
    test_dataset = mlflow.data.from_pandas(X_test.assign(target=y_test))

    mlflow.log_input(train_dataset, context="training")
    mlflow.log_input(test_dataset, context="testing")

    # -------------------------------
    # Log model with input_example and signature
    # -------------------------------
    input_example = X_train.iloc[:5]
    signature = infer_signature(X_train, best_model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="best_model",
        input_example=input_example,
        signature=signature
    )

    # -------------------------------
    # Set Metadata Tags
    # -------------------------------
    mlflow.set_tag("author", "ANSHU")

    # -------------------------------
    # Output
    # -------------------------------
    print("Best Parameters:", best_params)
    print("Best Cross-Validated Accuracy:", best_score)
    print("🧪 View experiment at: https://dagshub.com/Anshu989856/TESTING.mlflow/#/experiments")
