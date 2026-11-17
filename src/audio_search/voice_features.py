"""Voice / speaker features.

Two complementary representations:
  1. Classical — MFCC (+ delta + delta-delta) per-file statistics + prosody/spectral
     features computed with librosa. Interpretable, textbook-style features.
  2. Deep — ECAPA-TDNN speaker embedding (192-D) via SpeechBrain, the primary
     voice vector used for speaker similarity search.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Classical features (librosa)
# ---------------------------------------------------------------------------
def compute_mfcc_stats(wave: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    """MFCC + delta + delta-delta per-file statistics.

    Stacks 26 MFCC + 26 delta + 26 delta-delta coefficients -> 78 rows.
    Returns:
      - mfcc_stats: 78-D vector = per-row mean over time (matches report.md vector(78))
      - mfcc_matrix: (78, T) raw stacked matrix (for plots in the report)
      - mfcc_std:   78-D per-row std over time (fuller characterization, report only)
    """
    import librosa

    mfcc = librosa.feature.mfcc(
        y=wave, sr=sr,
        n_mfcc=config.MFCC_N_COEFF,
        n_fft=config.MFCC_N_FFT,
        hop_length=config.MFCC_HOP_LENGTH,
        win_length=config.MFCC_WIN_LENGTH,
        n_mels=config.MFCC_N_MELS,
        dct_type=2,
        norm="ortho",
    )
    d1 = librosa.feature.delta(mfcc, order=1)
    d2 = librosa.feature.delta(mfcc, order=2)

    stack = np.vstack([mfcc, d1, d2])  # (78, T)

    return {
        "mfcc_stats": stack.mean(axis=1).astype(np.float32),  # (78,)
        "mfcc_std": stack.std(axis=1).astype(np.float32),     # (78,)
        "mfcc_matrix": stack.astype(np.float32),              # (78, T)
    }


def compute_prosody(wave: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    """Prosody & spectral descriptors for the whole file."""
    import librosa

    # Pitch (F0) over voiced frames
    f0, voiced_flag, _ = librosa.pyin(wave, fmin=60, fmax=400, sr=sr)
    if voiced_flag is not None and voiced_flag.any():
        f0_voiced = f0[voiced_flag]
        f0_mean = float(np.nanmean(f0_voiced)) if len(f0_voiced) else None
        f0_std = float(np.nanstd(f0_voiced)) if len(f0_voiced) else None
    else:
        f0_mean, f0_std = None, None

    # Zero-crossing rate
    zcr = float(librosa.feature.zero_crossing_rate(wave).mean())

    # Spectral centroid & rolloff
    centroid = float(librosa.feature.spectral_centroid(y=wave, sr=sr).mean())
    rolloff = float(
        librosa.feature.spectral_rolloff(y=wave, sr=sr, roll_percent=0.85).mean()
    )

    return {
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "zcr_mean": zcr,
        "spectral_centroid_mean": centroid,
        "spectral_rolloff_mean": rolloff,
    }


# ---------------------------------------------------------------------------
# 2. ECAPA-TDNN deep speaker embedding (SpeechBrain)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_ecapa_model():
    """Load the ECAPA-TDNN classifier once and cache it."""
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy

    device = config.resolve_device()
    logger.info("loading ECAPA-TDNN on device=%s ...", device)
    return EncoderClassifier.from_hparams(
        source=config.ECAPA_MODEL_ID,
        savedir=str(config.ARTIFACTS_DIR / "models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": device},
        local_strategy=LocalStrategy.COPY,   # Windows: avoid symlink permission errors
    )


def compute_ecapa_embedding(wave: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Return a 192-D L2-normalized ECAPA speaker embedding for the clip."""
    import torch

    classifier = get_ecapa_model()

    # Input: float32 mono waveform -> (1, T) tensor on the model's device
    device = next(classifier.parameters()).device
    x = torch.from_numpy(wave.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = classifier.encode_batch(x)          # (1, 1, 192)
    vec = emb.squeeze().detach().cpu().numpy().astype(np.float32)  # (192,)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def compute_voice_features(wave: np.ndarray, sr: int = config.SAMPLE_RATE) -> dict:
    """Bundle all voice features for one clip."""
    mfcc = compute_mfcc_stats(wave, sr)
    prosody = compute_prosody(wave, sr)
    ecapa = compute_ecapa_embedding(wave, sr)
    return {
        "mfcc_stats": mfcc["mfcc_stats"],
        "mfcc_std": mfcc["mfcc_std"],
        "mfcc_matrix": mfcc["mfcc_matrix"],
        "prosody": prosody,
        "ecapa_embedding": ecapa,
    }