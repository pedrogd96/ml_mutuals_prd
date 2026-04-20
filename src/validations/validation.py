import os
import joblib
from typing import List, Dict, Any
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

logger = setup_logger("VALIDATION")

def load_feature_schema(version: str) -> List[str]:
    """
    Carrega a lista de features usada no treinamento do modelo
    """
    feature_path = f"models/artifacts/features_{version}.pkl"

    if not os.path.exists(feature_path):
        logger.error(f"Arquivo de features não encontrado para versão {version}")
        raise Exception(f"Features da versão '{version}' não encontradas")

    features = joblib.load(feature_path)

    # Garantir que é lista
    if not isinstance(features, list):
        logger.error(f"Formato inválido no arquivo de features {feature_path}")
        raise Exception("Formato inválido de features")

    return features


def validate_model_version(version: str) -> List[str]:
    """
    Verifica se o modelo da versão existe
    """
    errors = []

    model_path = f"models/artifacts/model_{version}.pkl"

    if not os.path.exists(model_path):
        errors.append(f"Modelo versão '{version}' não encontrado")

    return errors


def validate_input(data: Any, version: str) -> Dict[str, Any]:
    """
    Valida payload completo da API
    """
    config = load_config("configs/api.yaml")
    defaults = config["defaults"]

    expected_features = set(load_feature_schema(version))
    optional_features = set(defaults.keys())

    errors = []

    # Validar versão do modelo
    errors.extend(validate_model_version(version))

    # Validar estrutura do payload
    if not isinstance(data, list):
        data = [data]

    if len(data) == 0:
        errors.append("Lista de dados está vazia")
        return {"valid": False, "errors": errors}

    # Validar cada registro
    for idx, record in enumerate(data):
        if not isinstance(record, dict):
            errors.append(f"Registro {idx} não é um objeto JSON válido")
            continue

        record_keys = set(record.keys())

        # Campos desconhecidos
        unknown_fields = record_keys - expected_features
        if unknown_fields:
            errors.append(
                f"Registro {idx} possui campos inválidos: {list(unknown_fields)}"
            )

        # Campos obrigatórios faltando
        required_features = expected_features - optional_features
        missing_fields = required_features - record_keys

        if missing_fields:
            errors.append(
                f"Registro {idx} está faltando campos obrigatórios: {list(missing_fields)}"
            )

        # Validação de nulos
        for field in required_features:
            if field in record and record[field] is None:
                errors.append(
                    f"Registro {idx} possui campo obrigatório nulo: {field}"
                )

    # Resultado final
    if errors:
        logger.error(f"Erros de validação: {errors}")
        return {"valid": False, "errors": errors}

    logger.info("Validação realizada com sucesso")
    return {"valid": True, "errors": None}