"""Stage 2 - TRAINING + VALIDATION, tracked with MLflow.

Usage: python src/train.py --model cnn|mobilenet
"""
import argparse
from pathlib import Path

import mlflow
import tensorflow as tf

from model import build_model
from utils import dataset_version, git_commit, load_params, load_split, save_json


class MlflowEpochLogger(tf.keras.callbacks.Callback):
    """Send train/val loss + accuracy to MLflow after every epoch."""

    def on_epoch_end(self, epoch, logs=None):
        mlflow.log_metrics({k: float(v) for k, v in (logs or {}).items()}, step=epoch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["cnn", "mobilenet"])
    name = ap.parse_args().model

    params = load_params()
    pp, tp, mp = params["prepare"], params["train"], params["mlflow"]
    cfg = tp[name]
    tf.keras.utils.set_random_seed(tp["seed"])

    train_ds = load_split("train", pp["img_size"], tp["batch_size"], tp["seed"])
    val_ds = load_split("val", pp["img_size"], tp["batch_size"])
    class_names = train_ds.class_names
    train_ds, val_ds = train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)

    model = build_model(name, len(class_names), pp["img_size"], cfg)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    mlflow.set_tracking_uri(mp["tracking_uri"])
    mlflow.set_experiment(mp["experiment_name"])
    out_dir = Path("models") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"{name}-train") as run:
        # ---- hyperparameters & lineage ----
        mlflow.log_params(
            {
                "model": name,
                "epochs": tp["epochs"],
                "batch_size": tp["batch_size"],
                "img_size": pp["img_size"],
                "num_classes": len(class_names),
                "total_params": model.count_params(),
                **cfg,
            }
        )
        mlflow.set_tags(
            {"git_commit": git_commit(), "dataset_version": dataset_version(), "stage": "train"}
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=tp["epochs"],
            callbacks=[MlflowEpochLogger()],
            verbose=2,
        )

        # ---- final validation metrics ----
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        metrics = {
            "train_accuracy": float(history.history["accuracy"][-1]),
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
        }
        mlflow.log_metrics({f"final_{k}": v for k, v in metrics.items()})

        # ---- artifacts ----
        model.save(out_dir / "model.keras")
        save_json(class_names, out_dir / "class_names.json")
        (out_dir / "run_id.txt").write_text(run.info.run_id)
        mlflow.log_artifact(str(out_dir / "class_names.json"))
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact(f"{pp['out_dir']}/metadata.json", artifact_path="dataset")
        mlflow.keras.log_model(model, artifact_path="model")

    save_json(metrics, f"metrics/{name}_train.json")
    print(f"[{name}] {metrics}")


if __name__ == "__main__":
    main()