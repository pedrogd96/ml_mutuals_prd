from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from src.utils.logger import setup_logger

logger = setup_logger("EVALUATE")

def evaluate_model(y_true, y_pred):
    """
    Retorna métricas do modelo
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    report = classification_report(y_true, y_pred)

    logger.info(f"Metrics {metrics}")
    logger.info(f"Report {report}")

    return metrics, report