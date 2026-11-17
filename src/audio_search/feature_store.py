"""Feature store — persistence of extracted features.

Writes per-file artifacts (one JSON + one NPZ per file) and a batch manifest
that aggregates everything for phase-4 ETL into PostgreSQL/pgvector.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _clean(obj):
    """Convert numpy scalars/arrays to JSON-friendly Python objects."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_voice_artifact(file_id: str, features: dict, filename: str = "", duration_s: float = 0.0) -> Path:
    """Write voice features for one file -> artifacts/features/voice-<file_id>.json."""
    config.ensure_dirs()
    payload = {
        "feature_version": config.FEATURE_VERSION,
        "file": filename,
        "file_id": file_id,
        "duration_s": duration_s,
        "mfcc_stats": _clean(features["mfcc_stats"]),        # 78-D
        "mfcc_std": _clean(features["mfcc_std"]),            # 78-D
        "prosody": _clean(features["prosody"]),
        "ecapa_embedding": _clean(features["ecapa_embedding"]),  # 192-D
    }
    out_path = config.FEATURES_DIR / f"voice-{file_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def write_content_artifact(file_id: str, features: dict, filename: str = "", duration_s: float = 0.0) -> Path:
    """Write content features for one file -> artifacts/features/content-<file_id>.json.

    `runtime` records the ASR backend (e.g. "cuda/float16") so a corpus can
    never mix artifacts produced by different backends unnoticed — Whisper
    transcripts differ between int8 and float16, and the comparison would
    silently be apples-to-oranges.
    """
    config.ensure_dirs()
    payload = {
        "feature_version": config.FEATURE_VERSION,
        "file": filename,
        "file_id": file_id,
        "duration_s": duration_s,
        "n_chunks": features["n_chunks"],
        "transcript": features["transcript"],
        "semantic_embedding": _clean(features["semantic_embedding"]),  # 384-D
        "chunks": features.get("chunks", []),
        "chunk_embeddings": _clean(features.get("chunk_embeddings", [])),  # (K, 384)
        "runtime": features.get("runtime", ""),
    }
    out_path = config.FEATURES_DIR / f"content-{file_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def reembed_content_artifacts() -> dict:
    """Recompute semantic embeddings from the stored transcripts.

    Re-running Whisper on 60 h of audio is the expensive part; the chunking +
    sentence-embedding step is cheap. This rewrites every content artifact
    with the current chunker (feature_version bumped) without touching ASR.
    """
    from . import content_features

    ok = 0
    paths = sorted(config.FEATURES_DIR.glob("content-*.json"))
    for i, path in enumerate(paths, 1):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sem = content_features.embed_content(payload.get("transcript", ""))
        payload.update({
            "feature_version": config.FEATURE_VERSION,
            "n_chunks": sem["n_chunks"],
            "semantic_embedding": _clean(sem["semantic_embedding"]),
            "chunks": sem["chunks"],
            "chunk_embeddings": _clean(sem["chunk_embeddings"]),
        })
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        ok += 1
        if i % 50 == 0:
            logger.info("re-embedded %d/%d", i, len(paths))
    logger.info("re-embedded %d content artifact(s)", ok)
    return {"ok": ok, "total": len(paths)}


def artifact_exists(file_id: str) -> bool:
    """True if both voice & content artifacts already exist for this file (resume)."""
    voice = (config.FEATURES_DIR / f"voice-{file_id}.json").exists()
    content = (config.FEATURES_DIR / f"content-{file_id}.json").exists()
    return voice and content


def build_manifest(manifest_path: Path = config.FEATURES_DIR / "manifest.csv") -> Path:
    """Aggregate all per-file artifacts into one CSV for phase-4 ETL.

    Each row: file_id, filename, duration_s, mfcc_stats(78),
              ecapa_embedding(192), semantic_embedding(384), transcript.
    Vectors are JSON strings — the ETL step converts them to pgvector arrays.
    """
    import pandas as pd

    rows = []
    for voice_path in sorted(config.FEATURES_DIR.glob("voice-*.json")):
        file_id = voice_path.stem.replace("voice-", "")
        content_path = config.FEATURES_DIR / f"content-{file_id}.json"
        if not content_path.exists():
            continue
        v = json.loads(voice_path.read_text(encoding="utf-8"))
        c = json.loads(content_path.read_text(encoding="utf-8"))
        rows.append({
            "file_id": file_id,
            "filename": v["file"],
            "duration_s": v.get("duration_s", 0.0),
            "mfcc_stats": json.dumps(v["mfcc_stats"]),
            "ecapa_embedding": json.dumps(v["ecapa_embedding"]),
            "semantic_embedding": json.dumps(c["semantic_embedding"]),
            "transcript": c["transcript"],
        })

    df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.suffix == ".parquet":
        df.to_parquet(manifest_path, index=False)
    else:
        df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logger.info("manifest written: %d rows -> %s", len(df), manifest_path)
    return manifest_path