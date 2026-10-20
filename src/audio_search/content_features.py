"""Content features.

Two-step pipeline:
  1. ASR — transcribe speech to text with faster-whisper (small, int8).
  2. Semantic embedding — embed the transcript with a sentence-transformer
     (all-MiniLM-L6-v2, 384-D), chunking into sentences and mean-pooling to a
     single file-level vector used for content similarity search.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ASR — faster-whisper
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_whisper_model():
    """Load the faster-whisper ASR model once (small, int8)."""
    from faster_whisper import WhisperModel

    logger.info("loading faster-whisper %s (%s/%s) ...",
                config.WHISPER_MODEL_SIZE, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )


def transcribe_audio(wav_path) -> str:
    """Transcribe a 16 kHz WAV file to text with faster-whisper.

    Accepts a path (str/Path/file-like). Returns the joined transcript.
    """
    model = get_whisper_model()
    segments, _ = model.transcribe(
        wav_path,
        language=config.WHISPER_LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )
    parts = [seg.text.strip() for seg in segments]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 2. Semantic embedding — sentence-transformers
# ---------------------------------------------------------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def chunk_to_sentences(text: str, max_chars: int = 1200, min_chars: int = 40) -> list[str]:
    """Split transcript into chunks aligned to sentence boundaries.

    Merges short consecutive sentences up to `max_chars` so the transformer
    sees coherent, moderately-sized units (< 512 tokens).
    """
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]
    chunks: list[str] = []
    cur = ""
    for sent in raw:
        if cur and len(cur) + len(sent) + 1 > max_chars:
            chunks.append(cur)
            cur = ""
        elif cur:
            cur += " " + sent
        else:
            cur = sent
    if cur:
        chunks.append(cur)
    # Drop punctuation-only scraps; keep everything else
    return [c for c in chunks if len(c) >= min_chars or len(c) == len(chunks) == 1]


@lru_cache(maxsize=1)
def get_semantic_model():
    """Load the sentence-transformer once."""
    from sentence_transformers import SentenceTransformer

    logger.info("loading sentence-transformer %s ...", config.SEMANTIC_MODEL_ID)
    return SentenceTransformer(config.SEMANTIC_MODEL_ID)


def embed_content(text: str) -> dict:
    """Chunk + embed a transcript -> 384-D mean-pooled, L2-normalized vector."""
    embedder = get_semantic_model()
    chunks = chunk_to_sentences(text)
    if not chunks:
        # No usable text -> zero vector (caller should handle)
        return {"semantic_embedding": np.zeros(config.SEMANTIC_DIM, dtype=np.float32),
                "n_chunks": 0}

    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)  # (K, 384)
    vec = embs.mean(axis=0).astype(np.float32)          # (384,)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return {"semantic_embedding": vec, "n_chunks": len(chunks)}


def compute_content_features(wav_path, text: str | None = None) -> dict:
    """Bundle content features for one clip: transcript + semantic embedding.

    If `text` is provided, skips ASR (e.g. transcripts already available from
    the source dataset).
    """
    transcript = text if text is not None else transcribe_audio(wav_path)
    sem = embed_content(transcript)
    return {
        "transcript": transcript,
        "semantic_embedding": sem["semantic_embedding"],
        "n_chunks": sem["n_chunks"],
    }