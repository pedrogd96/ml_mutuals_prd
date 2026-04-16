import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger("PREDICT")

def load_model(version: str):
    """
    Carrega modelo baseado na versão
    """
    model_path = f"models/artifacts/model_{version}.pkl"

    if not os.path.exists(model_path):
        logger.error(f"Modelo versão {version} não encontrado")
        raise Exception(f"Modelo versão {version} não encontrado")

    model = joblib.load(model_path)
    return model


def predict(data, version="v1"):
    """
    Realiza predição usando modelo versionado
    """
    model = load_model(version)

    prediction = model.predict(data)
    prediction_proba = None

    # Alguns modelos não possuem predict_proba (ex: Perceptron)
    if hasattr(model, "predict_proba"):
        prediction_proba = model.predict_proba(data).tolist()

    logger.info(f"Modelo versão {version} gerou prediction {prediction} e probability {prediction_proba} para os dados {data}")

    return {
        "prediction": prediction,
        "probability": prediction_proba
    }