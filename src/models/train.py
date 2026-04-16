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

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("mutual_model")

def train(version="v1", force_train=False):
    """
    Treina modelo apenas se não existir ou se force_train=True
    """

    os.makedirs("models/artifacts", exist_ok=True)
    model_path = f"models/artifacts/model_{version}.pkl"

    # Se já existe, não treina
    if os.path.exists(model_path) and not force_train:
        logger.info(f"Modelo {version} já existe. Carregando...")
        return joblib.load(model_path)
    
    config = load_config("configs/data.yaml")
    model_config = load_config("configs/model.yaml")

    data = load_data(config["data"]["path"])

    X = data.drop(config["data"]["target"], axis=1)
    y = data[config["data"]["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=model_config["training"]["test_size"],
        random_state=model_config["random_state"],
        stratify=model_config["training"]["stratify"]
    )

    pipeline = create_pipeline(model_config)

    with mlflow.start_run(run_name=f"model_{version}"):
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)

        # log params
        mlflow.log_param("version", version)

        # log métricas
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        # salvar modelo no MLflow
        mlflow.sklearn.log_model(pipeline, f"model_{version}")

    joblib.dump(pipeline, model_path)

    logger.info(f"Modelo {version} carregando.")

    return pipeline