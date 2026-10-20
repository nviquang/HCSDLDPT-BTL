"""Central configuration — paths, constants, model IDs."""

from pathlib import Path

# --- Paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WAV_DIR = ARTIFACTS_DIR / "wav"
FEATURES_DIR = ARTIFACTS_DIR / "features"
LOGS_DIR = ARTIFACTS_DIR / "logs"

# --- Audio -----------------------------------------------------------
SAMPLE_RATE = 16000          # All models expect 16 kHz
PEAK_NORM = 0.95             # Target peak amplitude after normalization
MIN_DURATION_S = 10.0        # Skip files shorter than this
MAX_DURATION_S = 1800.0      # Skip files longer than this (30 min)

# --- MFCC parameters -------------------------------------------------
MFCC_N_COEFF = 26            # 26 MFCC coefficients
MFCC_N_FFT = 400             # ~25 ms window at 16 kHz
MFCC_HOP_LENGTH = 160        # 10 ms hop at 16 kHz
MFCC_WIN_LENGTH = 400        # 25 ms window at 16 kHz
MFCC_N_MELS = 80             # Mel filterbank channels
# Stats per coeff: mean + std -> 26 x 2 = 52 dimensions for mfcc_stats
# Or: MFCC + delta + delta-delta -> 78 coefficients, then per-file mean+std = 78-D

# --- ECAPA-TDNN ------------------------------------------------------
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIM = 192              # Output embedding dimension

# --- Whisper ASR -----------------------------------------------------
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = "en"

# --- Sentence-Transformers -------------------------------------------
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_DIM = 384           # Output embedding dimension

# --- Feature version (for artifact resume logic) ---------------------
FEATURE_VERSION = "2.0"


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [WAV_DIR, FEATURES_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
