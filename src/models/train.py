import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data.load_data import load_data
from src.pipelines.pipeline import create_pipeline
from src.models.evaluate import evaluate_model
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("TRAIN")

def train(version="v1", force_train=False):
    """
    Treina modelo apenas se não existir ou se force_train=True
    """

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("mutual_model")

    os.makedirs("models/artifacts", exist_ok=True)
    model_path = f"models/artifacts/model_{version}.pkl"
    features_path = f"models/artifacts/features_{version}.pkl"

    # Se já existe, não treina
    if os.path.exists(model_path) and not force_train:
        logger.info(f"Modelo {version} já existe. Carregando...")
        return joblib.load(model_path)
    
    config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")
    model_config = model_config["model"]

    data = load_data(config["data"]["path"])

    X = data.drop(config["data"]["target"], axis=1)
    y = data[config["data"]["target"]]
    stratify = y if model_config["training"]["stratify"] else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_config["training"]["test_size"],
        random_state=model_config["random_state"],
        stratify=stratify
    )

    pipeline = create_pipeline(model_config)

    with mlflow.start_run(run_name=f"model_{version}"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        metrics, report = evaluate_model(y_test, y_pred)

        # log params
        mlflow.log_param("version", version)

        # log métricas
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1_score"])

        # salvar modelo no MLflow
        mlflow.sklearn.log_model(pipeline, f"model_{version}")

    joblib.dump(pipeline, model_path)

    feature_names = X_train.columns.tolist()
    joblib.dump(feature_names, features_path)

    logger.info(f"Modelo {version} carregando.")

    return pipeline