from flask import Flask, request, jsonify
from src.features.preprocess import preprocess_input
from src.models.predict import predict
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

app = Flask(__name__)
logger = setup_logger("API")
config = load_config("configs/api.yaml")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        request_data = request.get_json()

        # Validar presença do campo versão
        version = request_data.get("version", "v1")

        # Dados para predição
        input_data = request_data.get("data")

        if input_data is None:
            logger.error(f"Não foi passado os dados por parâmetro para a versão {version}")
            return jsonify({"error": "Campo 'data' é obrigatório"}), 400

        # Pré-processamento
        processed_data = preprocess_input(input_data)

        # Predição
        result = predict(processed_data, version)

        logger.info(f"Resultado {result} para {input_data} na versão {version}")

        return jsonify({
            "model_version": version,
            "result": result
        })

    except Exception as e:
        logger.error(f"Erro {str(e)} para {input_data} na versão {version}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(
        host=config["api"]["host"],
        port=config["api"]["port"],
        debug=True
    )