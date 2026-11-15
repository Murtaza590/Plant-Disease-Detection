"""Stage 3 - EVALUATION on the held-out test set. Logs back to the SAME MLflow run as training.

Usage: python src/evaluate.py --model cnn|mobilenet
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score

from utils import load_params, load_split, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "mobilenet"])
    name = ap.parse_args().model

    params = load_params()
    pp, tp, mp = params["prepare"], params["train"], params["mlflow"]
    model_dir = Path("models") / name

    model = tf.keras.models.load_model(model_dir / "model.keras")
    test_ds = load_split("test", pp["img_size"], tp["batch_size"])
    class_names = test_ds.class_names

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)

    metrics = {
        "test_accuracy": float((y_true == y_pred).mean()),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, xticks_rotation=90, colorbar=False
    )
    ax.set_title(f"{name} - test confusion matrix")
    plt.tight_layout()
    cm_path = Path(f"plots/{name}_confusion_matrix.png")
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)

    mlflow.set_tracking_uri(mp["tracking_uri"])
    run_id = (model_dir / "run_id.txt").read_text().strip()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(cm_path), artifact_path="plots")

    save_json(metrics, f"metrics/{name}_test.json")
    print(f"[{name}] {metrics}")


if __name__ == "__main__":
    main()