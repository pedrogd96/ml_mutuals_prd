import argparse
from src.models.train import train
from src.utils.logger import setup_logger

logger = setup_logger("MAIN")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v1")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()

    logger.info(f"Iniciando treinamento para o modelo com a versão {args.version} e {args.force}")

    train(version=args.version, force_train=args.force)

if __name__ == "__main__":
    main()