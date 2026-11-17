"""Content features.

Two-step pipeline:
  1. ASR — transcribe speech to text with faster-whisper (small).
  2. Semantic embedding — embed the transcript with an English sentence-
     transformer (all-MiniLM-L6-v2, 384-D), chunking into sentences and
     mean-pooling to a single file-level vector used for content similarity
     search.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from functools import lru_cache

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. ASR — faster-whisper
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_whisper_model():
    """Load the faster-whisper ASR model once (small; float16 on GPU, int8 on CPU)."""
    from faster_whisper import WhisperModel

    device, compute_type = config.whisper_runtime()
    logger.info("loading faster-whisper %s (%s/%s) ...",
                config.WHISPER_MODEL_SIZE, device, compute_type)
    return WhisperModel(
        config.WHISPER_MODEL_SIZE,
        device=device,
        compute_type=compute_type,
    )


_LANG_UNSET = object()


def transcribe_audio(wav_path, language=_LANG_UNSET) -> str:
    """Transcribe audio to text with faster-whisper.

    Accepts a path (str/Path/file-like) or a 16 kHz float32 numpy waveform.
    `language` defaults to config.WHISPER_LANGUAGE; pass None to let Whisper
    detect it. Returns the joined transcript (one segment per line).
    """
    model = get_whisper_model()
    lang = config.WHISPER_LANGUAGE if language is _LANG_UNSET else language
    segments, _ = model.transcribe(
        wav_path,
        language=lang,
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


def _approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English) used when no tokenizer is given."""
    return max(1, len(text) // 4)


def chunk_to_sentences(
    text: str,
    max_tokens: int = config.SEMANTIC_CHUNK_TOKENS,
    min_chars: int = 40,
    token_len: Callable[[str], int] | None = None,
) -> list[str]:
    """Split a transcript into chunks aligned to sentence boundaries.

    Consecutive sentences are merged while the chunk stays within
    `max_tokens`, so the sentence-transformer (hard limit 256 word pieces for
    all-MiniLM-L6-v2) never truncates a chunk. Every sentence lands in exactly
    one chunk: a sentence that would overflow the current chunk opens the next
    one. A single sentence longer than the budget is split on word boundaries.

    `token_len` counts tokens of a string; pass the model tokenizer for exact
    counts. Defaults to a ~4 chars/token estimate.
    """
    count = token_len or _approx_tokens
    raw = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

    # Break over-long sentences into word windows that fit the budget.
    pieces: list[str] = []
    for sent in raw:
        if count(sent) <= max_tokens:
            pieces.append(sent)
            continue
        cur_words: list[str] = []
        for word in sent.split():
            if cur_words and count(" ".join([*cur_words, word])) > max_tokens:
                pieces.append(" ".join(cur_words))
                cur_words = []
            cur_words.append(word)
        if cur_words:
            pieces.append(" ".join(cur_words))

    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        candidate = f"{cur} {piece}" if cur else piece
        if cur and count(candidate) > max_tokens:
            chunks.append(cur)
            cur = piece          # the overflowing sentence starts the next chunk
        else:
            cur = candidate
    if cur:
        chunks.append(cur)

    if len(chunks) <= 1:
        return chunks
    # Drop punctuation-only scraps between real chunks.
    return [c for c in chunks if len(c) >= min_chars]


def model_token_len(text: str) -> int:
    """Exact token count (incl. [CLS]/[SEP]) under the semantic model's tokenizer."""
    tok = get_semantic_model().tokenizer
    return len(tok(text, add_special_tokens=True, truncation=False)["input_ids"])


@lru_cache(maxsize=1)
def get_semantic_model():
    """Load the sentence-transformer once, on the resolved device."""
    from sentence_transformers import SentenceTransformer

    device = config.resolve_device()
    logger.info("loading sentence-transformer %s on device=%s ...",
                config.SEMANTIC_MODEL_ID, device)
    return SentenceTransformer(config.SEMANTIC_MODEL_ID, device=device)


def embed_content(text: str) -> dict:
    """Chunk + embed a transcript -> 384-D mean-pooled, L2-normalized vector.

    Also returns the per-chunk texts and their (L2-normalized) embeddings so
    the database can store them and the search can show which passage
    matched best.
    """
    embedder = get_semantic_model()
    chunks = chunk_to_sentences(text, token_len=model_token_len)
    if not chunks:
        # No usable text (e.g. instrumental audio) -> zero vector; the index
        # excludes such files from content search.
        return {"semantic_embedding": np.zeros(config.SEMANTIC_DIM, dtype=np.float32),
                "n_chunks": 0, "chunks": [],
                "chunk_embeddings": np.zeros((0, config.SEMANTIC_DIM), dtype=np.float32)}

    embs = embedder.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)  # (K, 384)
    embs = embs.astype(np.float32)
    vec = embs.mean(axis=0)                              # (384,)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return {"semantic_embedding": vec.astype(np.float32), "n_chunks": len(chunks),
            "chunks": chunks, "chunk_embeddings": embs}


def compute_content_features(wav_path, text: str | None = None) -> dict:
    """Bundle content features for one clip: transcript + semantic embedding.

    If `text` is provided, skips ASR (e.g. transcripts already available from
    the source dataset).
    """
    transcript = text if text is not None else transcribe_audio(wav_path)
    sem = embed_content(transcript)
    device, compute_type = config.whisper_runtime()
    return {
        "transcript": transcript,
        "semantic_embedding": sem["semantic_embedding"],
        "n_chunks": sem["n_chunks"],
        "chunks": sem["chunks"],
        "chunk_embeddings": sem["chunk_embeddings"],
        "runtime": f"{device}/{compute_type}",
    }