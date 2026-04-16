import pandas as pd
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("PREPROCESS")

def preprocess_input(data_json):
    """
    Recebe JSON da API, aplica defaults e retorna DataFrame pronto para o modelo
    """

    config = load_config("configs/api.yaml")
    defaults = config["defaults"]

    # Converter para dict (caso venha como lista)
    data = data_json.copy()

    # Aplicar valores padrão se nulos ou ausentes
    for key, default_value in defaults.items():
        if key not in data or data[key] is None:
            data[key] = default_value

    # Converter para DataFrame
    df = pd.DataFrame([data])

    logger.info(f"Dados processados {data}")

    return df