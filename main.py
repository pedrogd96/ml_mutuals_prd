import argparse
import time
import requests
from src.models.train import train
from src.utils.logger import setup_logger

logger = setup_logger("MAIN")

def wait_for_mlflow(url="http://mlflow:5000/health", timeout=180):
    logger.info("Aguardando MLflow iniciar.")

    for i in range(timeout):
        try:
            r = requests.get(url)
            if r.status_code == 200:
                logger.info("MLflow está pronto!")
                return
        except Exception as e:
            pass

        time.sleep(1)

    raise Exception("MLflow não iniciou a tempo")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    wait_for_mlflow()

    logger.info(f"Iniciando treinamento para o modelo com a versão {args.version} e {args.force}")

    train(version=args.version, force_train=args.force)

if __name__ == "__main__":
    main()