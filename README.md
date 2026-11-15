# Plant Disease Detection - MLOps Phase 1

Data versioning + repeatable training pipeline + experiment tracking + model registry
for classifying PlantVillage leaf images (simple CNN baseline vs MobileNetV2 transfer learning).

## What is where

| Requirement | Implementation |
|---|---|
| Dataset versioning | DVC tracks the raw dataset (`data/raw/PlantVillage.dvc`) and the processed splits (stage output in `dvc.yaml`, hash in `dvc.lock`); remote = AWS S3 |
| Pipeline | `dvc.yaml` stages: `prepare` -> `train` -> `evaluate` -> `register`; all settings in `params.yaml` |
| Experiment tracking | MLflow (`src/train.py`, `src/evaluate.py`): params, per-epoch metrics, test metrics, confusion matrix, class names, git commit, dataset version |
| Model registry | `src/register.py`: both models registered with version tags; alias `champion` = best test accuracy |

## Run it

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

dvc repro                         # runs the whole pipeline (only re-runs what changed)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

Other useful commands: `dvc dag`, `dvc metrics show`, `dvc params diff`, `dvc pull`, `pytest tests`.