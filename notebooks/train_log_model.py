# Databricks notebook source

import logging

import mlflow
from mlflow.models import infer_signature
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------
# Configure the logger (basic configuration)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# COMMAND ----------
iris = datasets.load_iris(as_frame=True)
X = iris.data
y = iris.target.apply(lambda x: iris.target_names[x])
logger.info("The dataset is loaded.")
# COMMAND ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=28)
logger.info("The dataset is split into train and test sets.")

# COMMAND ----------
# Machine Learning with scikit-learn
preprocessor = ColumnTransformer(
    transformers=[("std_scaler", StandardScaler(), iris.feature_names)]
)
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()),
    ]
)
logger.info("🚀 Starting training...")
pipeline.fit(X_train, y_train)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/iris-demo")
mlflow.autolog(disable=True)
# COMMAND ----------
with mlflow.start_run() as run:
    run_id = run.info.run_id
    y_proba = pipeline.predict_proba(X_test)
    y_pred = pipeline.predict(X_test)

    # Evaluation metrics
    auc_test = roc_auc_score(y_test, y_proba, multi_class="ovr")
    logger.info(f"AUC Report: {auc_test}")

    # Log parameters and metrics
    mlflow.log_param("model_type", "LogisticRegression Classifier with preprocessing")
    mlflow.log_metric("auc", auc_test)

    # Log the model
    signature = infer_signature(model_input=X_train, model_output=y_pred)
    dataset = mlflow.data.from_pandas(iris.frame, name="train_set")
    mlflow.log_input(dataset, context="training")
    mlflow.sklearn.log_model(
        sk_model=pipeline, artifact_path="sklearn-lg-pipeline-model", signature=signature
    )

# COMMAND ----------
