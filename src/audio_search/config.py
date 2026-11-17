"""Central configuration — paths, constants, model IDs."""

from functools import lru_cache
from pathlib import Path

# --- Paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
WAV_DIR = ARTIFACTS_DIR / "wav"
FEATURES_DIR = ARTIFACTS_DIR / "features"
LOGS_DIR = ARTIFACTS_DIR / "logs"
ANALYSIS_DIR = ARTIFACTS_DIR / "analysis"
ANALYSIS_FIGURES_DIR = ANALYSIS_DIR / "figures"

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
# Extraction stores MFCC + delta + delta-delta (78 rows) as a per-file mean
# (`mfcc_stats`, 78-D) and std (`mfcc_std`, 78-D). The time-mean of a delta
# is ~0 by construction, so the search uses the "clean" classical vector
# instead: mean + std of c1..c25 (c0 = log-energy dropped) = 50-D, z-scored
# with corpus statistics (see database.mfcc_search_vector).

# --- ECAPA-TDNN ------------------------------------------------------
ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_DIM = 192              # Output embedding dimension

# --- Whisper ASR -----------------------------------------------------
WHISPER_MODEL_SIZE = "small"
WHISPER_LANGUAGE = "en"      # Corpus is English — pin the language so Whisper
                             # never has to guess (and never translates).
# device/compute_type are derived from resolve_device() — see below.
# Whisper on GPU needs float16; int8 is the CPU sweet spot.

# --- Sentence-Transformers -------------------------------------------
# NOTE: embeddings from different models are not comparable — changing this
# invalidates every existing content-*.json. Re-extract the whole corpus.
SEMANTIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_DIM = 384           # Output embedding dimension
# all-MiniLM-L6-v2 truncates input beyond 256 word pieces (incl. [CLS]/[SEP]).
# Chunks are packed up to this many tokens so no text is ever cut off.
SEMANTIC_CHUNK_TOKENS = 250

# --- Feature version (for artifact resume logic) ---------------------
# 2.1: token-aware chunking (no dropped sentences, no truncation) + stored chunks.
FEATURE_VERSION = "2.1"

# --- Database & search ---------------------------------------------------
DB_PATH = ARTIFACTS_DIR / "audio_search.db"   # SQLite feature database
EVAL_DIR = ARTIFACTS_DIR / "evaluation"
QUERY_DIR = PROJECT_ROOT / "samples" / "queries"  # demo query recordings
TOP_K = 3                                     # results per feature family
# TED uploads that are music / dance performances, not talks (speech rate
# < 2.5 transcript chars/s against a corpus median of 13.3, or a sung / danced
# act). They violate the "bài diễn thuyết / bài giảng / kể chuyện" criterion,
# so they are stored but excluded from both searches and from the file count.
NON_TALK_FILES = frozenset({
    "A_Brilliant_Blend_of_Ballet,_Hip-Hop_and_African_Dance_Ghetto_Classics_Dance_TED",
    "A_Hypnotic_Exploration_of_Movement_JA_Collective_TED",
    "A_New_Take_on_an_Ancient_Sound_Tina_Lajevardian_TED",
    "“Atalanta”_The_Merian_Ensemble_TED",
    "_Universe_Statues_Liberation_Buzz_TED",
    "A_Musical_Journey_Through_Turkana_Turkana_Sessions_TED",
    "East_African_Sound_Meets_Cosmic_Trap_Akoth_Jumadi_and_Mr._Lu_TED",
    "“Cant_de_la_Sibil·la”_Quien_Más_Me_Ama_Maria_Arnal_TED",
    "_Ye_Pariye_Naz_Simorgh_Darya_Niaz_Nawab_TED",
    "“Bridge_Over_Troubled_Water”_—_Like_You’ve_Never_Heard_It_Before_MAS_Vocal_TED",
    "Music,_Movement_and_Poetry_Tunde_Olaniran_TED",
    "_Blank_Canvas_One_World_Sarah_Elizabeth_Charles_TED",
    "How_I_Imitate_Nature’s_Voices_Snow_Raven_TED",
    "Yo!_Have_You_Ever_Seen_a_Yo-Yo_Dance_Like_This_Shu_Takada_TED",
    "An_Anthem_for_Peace_and_Justice_From_Israeli_and_Palestinian_Youth_Jerusalem_Youth_Chorus_TED",
    # Not a human talk: a spoken-word / music act and an AI-generated voice.
    "Xiuhtezcatl_Performing_Careful_Veils_at_TED2024",
    "The_World's_First_AI_TED_Talk",
    # Song / rap sets (found by transcript review of low speech-rate files).
    "“To_My_Bones”_“As_the_River_Flows”_“In_Dictum”_Wallis_Bird_TED",
    "_FEEL_LOVE_POSER_PARTYOF2_TED",
})
# Montages / promos where several celebrities speak in turn. The filename has
# no "X and Y" pattern, so database._multi_speaker cannot detect them.
MULTI_SPEAKER_FILES = frozenset({
    "Storyline_Online_promo_featuring_Terry_Crews,_Rosario_Dawson,_Jennifer_Garner_and_more!",
    "Storyline_Online®_2025_30-Second_Promo",
    "Why_Storyline_Online®_Inspires_Kids_to_Read_Celebrity_Read-Alouds",
    "How_a_Viral_Choreographer_Makes_His_Moves_Sean_Bankhead_TED",   # dance + host interview
    "What_is_Vooks_Storyteller_",                                     # ad montage, several voices
    # Two TED speakers: corpus.py takes the last two title tokens as the speaker,
    # so "A_B_and_C_D_TED" is parsed as "C_D" and the filename rule misses it.
    "Could_We_Replace_Data_Centers_with_…_Plant_DNA_Cliff_Kapono_and_Keolu_Fox_TED",
    "How_You_Could_See_Inside_Your_Body_—_With_a_Micro-Robot_Alex_Luebke_and_Vivek_Kumbhari_TED",
})
REQUIRED_FILES = 500        # đề bài: ít nhất 500 file (một chủ đề, một người nói)
# Files whose transcript is shorter than this are treated as "no speech"
# (instrumental TED performances) and kept out of the content index.
MIN_TRANSCRIPT_CHARS = 200
# Query-side ASR: None lets Whisper detect the language of a new recording.
QUERY_WHISPER_LANGUAGE: str | None = "en"


@lru_cache(maxsize=1)
def resolve_device() -> str:
    """Pick the compute device: "cuda" when a usable GPU is present, else "cpu".

    Override with the AUDIO_SEARCH_DEVICE env var ("cpu" | "cuda") — the CLI
    sets it from `--device`. Falls back to CPU if CUDA is requested but the
    installed torch is a CPU-only build, so the pipeline never hard-crashes.
    """
    import os

    requested = os.environ.get("AUDIO_SEARCH_DEVICE", "").strip().lower()
    if requested in ("cpu", "cuda"):
        if requested == "cuda":
            import torch

            if not torch.cuda.is_available():
                import logging

                logging.getLogger(__name__).warning(
                    "AUDIO_SEARCH_DEVICE=cuda requested but torch cannot see a GPU "
                    "(torch %s) — falling back to cpu", torch.__version__,
                )
                return "cpu"
        return requested

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def whisper_runtime() -> tuple[str, str]:
    """Return (device, compute_type) for faster-whisper on the resolved device."""
    device = resolve_device()
    return device, ("float16" if device == "cuda" else "int8")


def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in [WAV_DIR, FEATURES_DIR, LOGS_DIR, ANALYSIS_DIR, ANALYSIS_FIGURES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
