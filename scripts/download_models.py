"""Pre-download the 3 models into the project's model cache (resumable).

Run:  uv run python scripts/download_models.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure the package is importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_search import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("download_models")


def ensure_models_dir() -> Path:
    """Create/models/ under project root and point HF_HOME there."""
    models_dir = config.PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(models_dir))
    os.environ.setdefault("HF_HUB_CACHE", str(models_dir / "hub"))
    return models_dir


def download_all() -> None:
    models_dir = ensure_models_dir()
    logger.info("HF_HOME=%s", os.environ.get("HF_HOME"))
    logger.info("pre-downloading ECM: %s", config.ECAPA_MODEL_ID)

    from huggingface_hub import snapshot_download

    for model_id in [config.ECAPA_MODEL_ID,
                     config.SEMANTIC_MODEL_ID,
                     "Systran/faster-whisper-small"]:
        logger.info("downloading %s ...", model_id)
        try:
            snapshot_download(model_id, local_dir_use_symlinks=False)
        except TypeError:
            # Older huggingface_hub <1.0 accepts local_dir_use_symlinks=False;
            # newer may drop it.
            snapshot_download(model_id)
    logger.info("done. models cached under %s", models_dir)


if __name__ == "__main__":
    download_all()