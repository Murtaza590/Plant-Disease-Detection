import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import build_model  # noqa: E402


def test_models_output_probabilities():
    for name, cfg, size in [
        ("cnn", {"dropout": 0.3}, 64),
        ("mobilenet", {"dropout": 0.2, "pretrained": False}, 96),
    ]:
        model = build_model(name, 5, size, cfg)
        x = np.random.rand(2, size, size, 3).astype("float32") * 255
        out = model.predict(x, verbose=0)
        assert out.shape == (2, 5)
        assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)