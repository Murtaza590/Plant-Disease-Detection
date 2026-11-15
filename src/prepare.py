"""Stage 1 - PREPROCESSING.

raw PlantVillage images  ->  cleaned, resized, split (train/val/test) images.
Also writes data/processed/metadata.json (dataset metadata that DVC versions).
"""
import random
import shutil
from pathlib import Path

from PIL import Image

from utils import load_params, save_json

IMG_EXT = {".jpg", ".jpeg", ".png"}


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def main():
    p = load_params()["prepare"]
    rng = random.Random(p["seed"])
    raw, out = Path(p["raw_dir"]), Path(p["out_dir"])

    if out.exists():
        shutil.rmtree(out)          # always rebuild from scratch -> reproducible

    classes = sorted(d.name for d in raw.iterdir() if d.is_dir())
    if p["max_classes"]:
        classes = classes[: p["max_classes"]]

    counts, skipped = {}, 0
    for cls in classes:
        files = sorted(f for f in (raw / cls).iterdir() if f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        if p["max_images_per_class"]:
            files = files[: p["max_images_per_class"]]

        n = len(files)
        n_test = int(n * p["test_split"])
        n_val = int(n * p["val_split"])
        splits = {
            "test": files[:n_test],
            "val": files[n_test : n_test + n_val],
            "train": files[n_test + n_val :],
        }
        counts[cls] = {k: 0 for k in splits}
        for split, items in splits.items():
            dest = out / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in items:
                if not is_valid_image(f):       # drop corrupt files
                    skipped += 1
                    continue
                with Image.open(f) as im:
                    im = im.convert("RGB").resize((p["img_size"], p["img_size"]))
                    im.save(dest / f"{f.stem}.jpg", quality=95)
                counts[cls][split] += 1

    save_json(
        {
            "source": "PlantVillage",
            "classes": classes,
            "num_classes": len(classes),
            "img_size": p["img_size"],
            "split_ratios": {"val": p["val_split"], "test": p["test_split"]},
            "seed": p["seed"],
            "skipped_corrupt_images": skipped,
            "images_per_class": counts,
        },
        out / "metadata.json",
    )
    print(f"Prepared {len(classes)} classes -> {out}  (skipped {skipped} corrupt)")


if __name__ == "__main__":
    main()