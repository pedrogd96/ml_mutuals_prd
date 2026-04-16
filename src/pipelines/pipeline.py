from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def create_pipeline(config: dict):
    hyperparametersCfg = config["hyperparameters"]
    scalerCfg = config["scaler"]

    # Escolha do scaler baseada no YAML
    if scalerCfg["type"] == "standard":
        scaler = StandardScaler()
    else:
        scaler = None  # ou levanta erro

    steps = []

    if scaler:
        steps.append(("scaler", scaler))

    steps.append((
        "svc", SVC(kernel=hyperparametersCfg["kernel"], class_weight=hyperparametersCfg["class_weight"], C=hyperparametersCfg["C"], random_state=config["random_state"])
    ))

    pipeline = Pipeline(steps)
    return pipeline