"""Small helpers shared by all pipeline scripts."""
import json
import subprocess
from pathlib import Path

import tensorflow as tf
import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_split(split, img_size, batch_size, seed=42):
    """Load data/processed/<split> as a tf.data.Dataset (labels = class index)."""
    return tf.keras.utils.image_dataset_from_directory(
        f"data/processed/{split}",
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=(split == "train"),   # keep val/test order fixed for the confusion matrix
        seed=seed,
        label_mode="int",
    )


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def dataset_version():
    """Hash of data/processed recorded by DVC in dvc.lock (our dataset version id)."""
    try:
        with open("dvc.lock") as f:
            lock = yaml.safe_load(f)
        for out in lock["stages"]["prepare"]["outs"]:
            if out["path"] == "data/processed":
                return out["md5"]
    except Exception:
        pass
    return "unknown"