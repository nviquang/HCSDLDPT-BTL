"""Audio preprocessing — load, resample to 16 kHz mono, normalize.

Batch-ready: auto-discovers files in the data dir, skips already-cached
WAV files (resume-safe), and logs per-file errors without stopping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Preprocessed:
    """A loaded, normalized audio clip plus metadata."""

    file_id: str          # stem of the source filename (unique key)
    filename: str         # original filename
    wave: np.ndarray      # float32 mono, shape (T,), 16 kHz
    sr: int               # always config.SAMPLE_RATE
    duration_s: float

    @property
    def path(self) -> Path:
        return config.WAV_DIR / f"{self.file_id}.wav"


def discover_audio_files(data_dir: Path = config.DATA_DIR) -> list[Path]:
    """Return all audio files (.mp3 / .wav / .m4a / .flac) in data_dir."""
    exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in exts and p.is_file())


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """Load audio via soundfile (uses libsndfile; reads mp3 with ffmpeg bundles)."""
    return sf.read(str(path), dtype="float32", always_2d=False)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load an audio file -> (float32 mono, sr).

    Falls back to ffmpeg if soundfile fails (e.g. unsupported codec).
    """
    try:
        wave, sr = _load_with_soundfile(path)
    except Exception:
        # Fallback: ffmpeg -> raw PCM via stdout
        wave, sr = _load_with_ffmpeg(path)

    if wave.ndim == 2 and wave.shape[1] > 1:
        wave = wave.mean(axis=1)          # stereo -> mono
    return wave.astype(np.float32), sr


def _load_with_ffmpeg(path: Path) -> tuple[np.ndarray, int]:
    import subprocess

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    wave = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if wave.size == 0:
        raise ValueError(f"ffmpeg decoded empty audio: {path}")
    return wave, config.SAMPLE_RATE


def _to_mono(wave: np.ndarray) -> np.ndarray:
    """Downmix any multi-channel audio to mono."""
    if wave.ndim == 2:
        return wave.mean(axis=1)
    return wave


def _resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    import librosa

    return librosa.resample(wave, orig_sr=orig_sr, target_sr=target_sr)


def _normalize(wave: np.ndarray, peak: float = config.PEAK_NORM) -> np.ndarray:
    """Peak-normalize to `peak` so the loudest sample hits `peak` amplitude."""
    max_val = np.max(np.abs(wave)) if wave.size else 0.0
    if max_val < 1e-9:
        return wave
    return (wave / max_val) * peak


def preprocess_file(src: Path, out_dir: Path = config.WAV_DIR) -> Preprocessed:
    """Load, normalize, and cache one audio file as 16 kHz mono WAV.

    Returns a Preprocessed dataclass. If the cached WAV already exists,
    reuses it (resume-safe). Raises on fatal errors; callers should catch.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    file_id = src.stem
    out_path = out_dir / f"{file_id}.wav"

    # Resume: reuse existing normalized WAV
    if out_path.exists():
        wave, sr = sf.read(str(out_path), dtype="float32", always_2d=False)
        if wave.ndim == 2:
            wave = _to_mono(wave)
        duration_s = wave.shape[0] / sr
        return Preprocessed(file_id=file_id, filename=src.name, wave=wave, sr=sr, duration_s=duration_s)

    wave, orig_sr = load_audio(src)
    wave = _to_mono(wave)
    wave = _resample(wave, orig_sr, config.SAMPLE_RATE)
    wave = _normalize(wave)

    if wave.size == 0:
        raise ValueError(f"empty audio after preprocessing: {src}")

    sf.write(str(out_path), wave, config.SAMPLE_RATE)

    duration_s = wave.shape[0] / config.SAMPLE_RATE
    return Preprocessed(file_id=file_id, filename=src.name, wave=wave, sr=config.SAMPLE_RATE, duration_s=duration_s)


def validate_duration(duration_s: float,
                      min_sec: float | None = None,
                      max_sec: float | None = None) -> bool:
    """Return whether a file's duration falls in the acceptable range.

    Bounds default to the *current* config values (read at call time, so the
    CLI's --min-duration / --max-duration overrides take effect).
    """
    lo = config.MIN_DURATION_S if min_sec is None else min_sec
    hi = config.MAX_DURATION_S if max_sec is None else max_sec
    return lo <= duration_s <= hi


def run(data_dir: Path = config.DATA_DIR, out_dir: Path = config.WAV_DIR) -> dict:
    """Preprocess all files in data_dir. Returns summary counts."""
    files = discover_audio_files(data_dir)
    ok, skipped, failed = 0, 0, 0

    for src in files:
        try:
            pp = preprocess_file(src, out_dir)
            if not validate_duration(pp.duration_s):
                logger.warning("duration %.1fs outside range for %s", pp.duration_s, src.name)
                skipped += 1
            else:
                ok += 1
        except Exception as exc:
            logger.error("preprocess failed for %s: %s", src.name, exc)
            failed += 1

    logger.info("preprocess done: %d ok, %d skipped, %d failed (of %d)", ok, skipped, failed, len(files))
    return {"ok": ok, "skipped": skipped, "failed": failed, "total": len(files)}