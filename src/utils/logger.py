import logging
import os
from datetime import datetime


def get_log_filename():
    """
    Gera nome do arquivo com data atual
    """
    today = datetime.now().strftime("%d-%m-%Y")
    return f"logs/server-{today}.log"


def setup_logger(tag, level=logging.INFO):
    """
    Logger único para todo o sistema com tag identificadora
    """

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("global_logger")
    logger.setLevel(level)

    # evita duplicação de logs
    if logger.hasHandlers():
        return logger

    log_file = get_log_filename()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(tag)s | %(message)s"
    )

    # handler arquivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logging.LoggerAdapter(logger, {"tag": tag})