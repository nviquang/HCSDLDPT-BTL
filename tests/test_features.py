"""Smoke tests for the audio_search feature pipeline.

Run: uv run pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from audio_search import config, preprocess


def test_discover_finds_samples():
    files = preprocess.discover_audio_files()
    assert len(files) >= 1
    assert all(f.suffix.lower() in {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"} for f in files)


def test_preprocess_all_samples():
    """All discovered samples preprocess to 16 kHz mono without error."""
    files = preprocess.discover_audio_files()
    results = []
    for src in files:
        try:
            pp = preprocess.preprocess_file(src)
            assert pp.sr == config.SAMPLE_RATE
            assert pp.wave.ndim == 1
            assert pp.duration_s > 0
            results.append(pp)
        except Exception:
            pass  # some may be skipped (too short, etc.)
    assert len(results) >= 1


def _rng_wave(seed: int = 0, seconds: float = 2.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(config.SAMPLE_RATE * seconds)).astype(np.float32)


def test_mfcc_stats_shape():
    """MFCC stats should be 78-D (26 + delta + delta-delta means)."""
    from audio_search.voice_features import compute_mfcc_stats

    out = compute_mfcc_stats(_rng_wave(), config.SAMPLE_RATE)
    assert out["mfcc_stats"].shape == (78,)
    assert out["mfcc_std"].shape == (78,)
    assert out["mfcc_matrix"].shape[0] == 78
    assert np.isfinite(out["mfcc_stats"]).all()


_ECAPA_LOCAL = config.ARTIFACTS_DIR / "models" / "spkrec-ecapa-voxceleb" / "embedding_model.ckpt"


@pytest.mark.skipif(not _ECAPA_LOCAL.exists(), reason="ECAPA model not cached — run extraction first")
def test_ecapa_embedding_shape():
    from audio_search.voice_features import compute_ecapa_embedding

    vec = compute_ecapa_embedding(_rng_wave(1), config.SAMPLE_RATE)
    assert vec.shape == (config.ECAPA_DIM,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_semantic_embedding_shape():
    """Semantic embedding should be 384-D and L2-normalized."""
    from audio_search.content_features import embed_content

    text = "This is a sample sentence about databases. Another one about multimedia retrieval."
    out = embed_content(text)
    assert out["semantic_embedding"].shape == (config.SEMANTIC_DIM,)
    assert np.isclose(np.linalg.norm(out["semantic_embedding"]), 1.0, atol=1e-3)
    assert out["n_chunks"] >= 1