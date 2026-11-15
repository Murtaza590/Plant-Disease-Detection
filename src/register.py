"""Stage 4 - MODEL REGISTRY.

Registers both models in the MLflow Model Registry with version tags, then
points the alias "champion" at the model with the best test accuracy.
"""
import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from utils import dataset_version, git_commit, load_params, save_json

MODELS = ["cnn", "mobilenet"]


def main():
    mp = load_params()["mlflow"]
    mlflow.set_tracking_uri(mp["tracking_uri"])
    client = MlflowClient()
    reg_name = mp["registered_model_name"]

    test = {m: json.load(open(f"metrics/{m}_test.json")) for m in MODELS}
    best = max(MODELS, key=lambda m: test[m]["test_accuracy"])

    versions = {}
    for m in MODELS:
        run_id = Path(f"models/{m}/run_id.txt").read_text().strip()
        mv = mlflow.register_model(f"runs:/{run_id}/model", reg_name)
        tags = {
            "architecture": m,
            "test_accuracy": f"{test[m]['test_accuracy']:.4f}",
            "dataset_version": dataset_version(),
            "git_commit": git_commit(),
            "validated": "true",
        }
        for k, v in tags.items():
            client.set_model_version_tag(reg_name, mv.version, k, v)
        versions[m] = mv.version

    # alias "champion" = the version the deployment should use
    client.set_registered_model_alias(reg_name, "champion", versions[best])

    save_json(
        {"registered_model": reg_name, "champion": best,
         "champion_version": versions[best], "versions": versions},
        "metrics/registered_model.json",
    )
    print(f"Registered {versions}; champion = {best} (v{versions[best]})")


if __name__ == "__main__":
    main()