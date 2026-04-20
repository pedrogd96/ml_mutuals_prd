import joblib
import pandas as pd
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PREPROCESS")

def preprocess_input(data_json, version="v1"):
    """
    Recebe JSON da API, aplica defaults e retorna DataFrame pronto para o modelo
    """
    config = load_config("configs/api.yaml")
    defaults = config["defaults"]

    # Garantir lista
    if isinstance(data_json, dict):
        data_list = [data_json]
    else:
        data_list = data_json

    processed_list = []

    for data in data_list:
        data = data.copy()

        # aplicar defaults
        for key, default_value in defaults.items():
            if key not in data or data[key] is None:
                data[key] = default_value

        processed_list.append(data)

    # Criar DataFrame
    df = pd.DataFrame(processed_list)

    # carregar ordem correta das features
    features_path = f"models/artifacts/features_{version}.pkl"
    feature_names = joblib.load(features_path)

    # garantir mesma estrutura do treino
    df = df.reindex(columns=feature_names)

    logger.info(f"Dados processados {df} na versão {version}")

    return df