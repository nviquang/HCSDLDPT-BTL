"""Search engine — new audio file in, top-3 by content + top-3 by voice out.

Pipeline for one query (same extractors as the corpus, so vectors are
comparable):

  query audio ─► preprocess (mono, 16 kHz, peak-normalise)
     ├─► VOICE   : ECAPA-TDNN 192-D ─────────────► cosine vs voice_features.ecapa
     │             MFCC c1..c25 mean+std 50-D ───► cosine vs voice_features.mfcc_clean (baseline)
     └─► CONTENT : Whisper ASR ─► transcript ─► chunk ─► MiniLM 384-D (mean-pool)
                                                 └──► cosine vs content_features.semantic
                                    transcript ───► BM25 over transcripts_fts (baseline)

Results are sorted by decreasing similarity. Every intermediate value
(transcript, vector previews/norms, per-step timings, best-matching passage)
is kept on the result object so the CLI and the demo can display it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, database

logger = logging.getLogger(__name__)

# Default decision thresholds; `evaluate` calibrates them on the corpus and
# writes artifacts/evaluation/thresholds.json, which overrides these.
DEFAULT_THRESHOLDS = {
    "content_known": 0.55,   # top-1 semantic cosine above -> content exists in DB
    "voice_known": 0.55,     # top-1 ECAPA cosine above -> speaker exists in DB
}


def load_thresholds() -> dict:
    path = config.EVAL_DIR / "thresholds.json"
    out = dict(DEFAULT_THRESHOLDS)
    if path.exists():
        try:
            out.update({k: float(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()
                        if k in out})
        except (OSError, ValueError):
            logger.warning("could not read %s — using default thresholds", path)
    return out


# ---------------------------------------------------------------------------
# Query feature extraction
# ---------------------------------------------------------------------------
@dataclass
class QueryFeatures:
    name: str
    duration_s: float
    wave: np.ndarray
    ecapa: np.ndarray                 # (192,)
    mfcc_mean: np.ndarray             # (78,)
    mfcc_std: np.ndarray              # (78,)
    mfcc_clean: np.ndarray            # (50,) z-scored + L2-normalised with corpus stats
    prosody: dict
    transcript: str
    semantic: np.ndarray              # (384,)
    chunks: list[str]
    chunk_embeddings: np.ndarray      # (K, 384)
    timings: dict = field(default_factory=dict)


def load_query_audio(path: str | Path) -> tuple[np.ndarray, float]:
    """Load any audio file -> (float32 mono 16 kHz peak-normalised wave, duration)."""
    from . import preprocess

    wave, sr = preprocess.load_audio(Path(path))
    wave = preprocess._to_mono(wave)
    wave = preprocess._resample(wave, sr, config.SAMPLE_RATE)
    wave = preprocess._normalize(wave).astype(np.float32)
    if wave.size == 0:
        raise ValueError(f"empty audio: {path}")
    return wave, wave.shape[0] / config.SAMPLE_RATE


def extract_query_features(path: str | Path, index: database.VectorIndex,
                           wave: np.ndarray | None = None,
                           transcript: str | None = None,
                           with_prosody: bool = True) -> QueryFeatures:
    """Run the full extraction pipeline on a query.

    `wave` lets the evaluation pass an in-memory clip; `transcript` skips ASR;
    `with_prosody=False` skips the (slow, display-only) pitch tracking.
    """
    from . import content_features, voice_features

    t: dict[str, float] = {}
    t0 = time.perf_counter()
    if wave is None:
        wave, duration = load_query_audio(path)
    else:
        duration = wave.shape[0] / config.SAMPLE_RATE
    t["preprocess_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mf = voice_features.compute_mfcc_stats(wave)
    # pYIN is slow on long audio; two minutes are plenty for pitch statistics.
    prosody = (voice_features.compute_prosody(wave[: 120 * config.SAMPLE_RATE]) if with_prosody
               else {"f0_mean_hz": None, "f0_std_hz": None, "zcr_mean": 0.0,
                     "spectral_centroid_mean": 0.0, "spectral_rolloff_mean": 0.0})
    raw50 = database.mfcc_raw_clean(mf["mfcc_stats"], mf["mfcc_std"])
    mfcc_clean = database.mfcc_search_vector(raw50, index.mfcc_mu, index.mfcc_sigma)
    t["mfcc_prosody_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    ecapa = voice_features.compute_ecapa_embedding(wave)
    t["ecapa_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if transcript is None:
        transcript = content_features.transcribe_audio(wave, language=config.QUERY_WHISPER_LANGUAGE)
    t["asr_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sem = content_features.embed_content(transcript)
    t["embed_s"] = time.perf_counter() - t0

    return QueryFeatures(
        name=Path(path).name if path else "clip",
        duration_s=duration, wave=wave, ecapa=ecapa,
        mfcc_mean=mf["mfcc_stats"], mfcc_std=mf["mfcc_std"], mfcc_clean=mfcc_clean,
        prosody=prosody, transcript=transcript,
        semantic=sem["semantic_embedding"], chunks=sem["chunks"],
        chunk_embeddings=sem["chunk_embeddings"], timings=t,
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------
@dataclass
class Hit:
    rank: int
    file_id: str
    score: float
    meta: dict
    passage: str = ""            # best-matching transcript chunk (content hits)
    passage_score: float = 0.0
    query_passage: str = ""      # query chunk that matched it


@dataclass
class SearchResult:
    query: QueryFeatures
    content: list[Hit]
    voice: list[Hit]
    voice_mfcc: list[Hit]        # classical baseline
    keyword: list[Hit]           # BM25 baseline
    content_top10: list[tuple[str, float]]
    voice_top10: list[tuple[str, float]]
    content_known: bool
    voice_known: bool
    thresholds: dict
    search_ms: float


def _top(scores: np.ndarray, ids: list[str], mask: np.ndarray, k: int,
         exclude: set[str]) -> list[tuple[int, float]]:
    s = np.where(mask, scores, -np.inf)
    if exclude:
        for i, fid in enumerate(ids):
            if fid in exclude:
                s[i] = -np.inf
    order = np.argsort(-s)
    return [(int(i), float(s[i])) for i in order[:k] if np.isfinite(s[i])]


def _best_passage(q: QueryFeatures, index: database.VectorIndex, file_id: str) -> tuple[str, float, str]:
    rows = [i for i, f in enumerate(index.chunk_file) if f == file_id]
    if not rows or not len(q.chunk_embeddings):
        return "", 0.0, ""
    sims = q.chunk_embeddings @ index.chunk_emb[rows].T          # (Kq, Kdb)
    qi, di = np.unravel_index(int(np.argmax(sims)), sims.shape)
    return index.chunk_text[rows[di]], float(sims[qi, di]), q.chunks[qi]


def rank(q: QueryFeatures, index: database.VectorIndex, k: int = config.TOP_K,
         exclude: set[str] | None = None) -> SearchResult:
    """Rank the database against a query's features."""
    exclude = exclude or set()
    th = load_thresholds()
    t0 = time.perf_counter()

    has_text = bool(q.chunks)
    content_scores = index.semantic @ q.semantic
    voice_scores = index.ecapa @ q.ecapa
    mfcc_scores = index.mfcc_clean @ q.mfcc_clean

    c_top = _top(content_scores, index.ids, index.voice_ok, 10, exclude) if has_text else []
    v_top = _top(voice_scores, index.ids, index.voice_ok, 10, exclude)
    m_top = _top(mfcc_scores, index.ids, index.voice_ok, k, exclude)

    def hits(top, with_passage=False):
        out = []
        for r, (i, s) in enumerate(top[:k], 1):
            fid = index.ids[i]
            h = Hit(rank=r, file_id=fid, score=s, meta=index.meta[fid])
            if with_passage:
                h.passage, h.passage_score, h.query_passage = _best_passage(q, index, fid)
            out.append(h)
        return out

    content_hits = hits(c_top, with_passage=True)
    voice_hits = hits(v_top)
    mfcc_hits = hits(m_top)
    kw = index.keyword_search(q.transcript, k, exclude) if has_text else []
    kw_hits = [Hit(rank=r, file_id=f, score=s, meta=index.meta.get(f, {})) for r, (f, s) in enumerate(kw, 1)]
    search_ms = (time.perf_counter() - t0) * 1000

    return SearchResult(
        query=q, content=content_hits, voice=voice_hits, voice_mfcc=mfcc_hits, keyword=kw_hits,
        content_top10=[(index.ids[i], s) for i, s in c_top],
        voice_top10=[(index.ids[i], s) for i, s in v_top],
        content_known=bool(content_hits) and content_hits[0].score >= th["content_known"],
        voice_known=bool(voice_hits) and voice_hits[0].score >= th["voice_known"],
        thresholds=th, search_ms=search_ms,
    )


def search_file(path: str | Path, index: database.VectorIndex | None = None,
                k: int = config.TOP_K, exclude: set[str] | None = None) -> SearchResult:
    """End-to-end: audio file path -> SearchResult."""
    index = index or database.load_index()
    q = extract_query_features(path, index)
    return rank(q, index, k=k, exclude=exclude)


# ---------------------------------------------------------------------------
# Text report (CLI)
# ---------------------------------------------------------------------------
def _vec_preview(v: np.ndarray, n: int = 6) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in np.asarray(v)[:n]) + ", …]"


def format_report(res: SearchResult) -> str:
    q = res.query
    L = []
    L.append(f"QUERY: {q.name}  ({q.duration_s:.1f} s)")
    L.append("")
    L.append("── Intermediate results ─────────────────────────────────────────")
    L.append(f"ECAPA-TDNN 192-D      ‖v‖={np.linalg.norm(q.ecapa):.3f}  {_vec_preview(q.ecapa)}")
    L.append(f"MFCC clean 50-D       ‖v‖={np.linalg.norm(q.mfcc_clean):.3f}  {_vec_preview(q.mfcc_clean)}")
    p = q.prosody
    f0 = f"{p['f0_mean_hz']:.1f}±{p['f0_std_hz']:.1f} Hz" if p.get("f0_mean_hz") else "n/a"
    L.append(f"Prosody               F0={f0}  ZCR={p['zcr_mean']:.4f}  "
             f"centroid={p['spectral_centroid_mean']:.0f} Hz  rolloff={p['spectral_rolloff_mean']:.0f} Hz")
    L.append(f"Semantic 384-D        ‖v‖={np.linalg.norm(q.semantic):.3f}  {_vec_preview(q.semantic)}"
             f"  ({len(q.chunks)} chunk(s))")
    snippet = q.transcript.replace("\n", " ")
    L.append(f"Transcript ({len(q.transcript)} chars): {snippet[:300]}{'…' if len(snippet) > 300 else ''}")
    L.append("Timings: " + ", ".join(f"{k}={v:.2f}" for k, v in q.timings.items())
             + f", search={res.search_ms:.1f} ms")
    L.append("")

    def block(title, hits, note="", label="cos"):
        L.append(f"── {title} {note}")
        if not hits:
            L.append("   (no result)")
        for h in hits:
            m = h.meta
            who = m.get("speaker") or m.get("course") or "?"
            flag = "  [multi-reader]" if m.get("multi_speaker") else ""
            L.append(f"  {h.rank}. {label}={h.score:.4f}  {h.file_id}")
            L.append(f"       source={m.get('source')}  speaker/course={who}  "
                     f"duration={m.get('duration_s', 0):.0f}s{flag}")
            if h.passage:
                L.append(f"       best passage (cos={h.passage_score:.3f}): {h.passage[:160]}…")
        L.append("")

    th = res.thresholds
    block("TOP-3 CONTENT (semantic cosine)", res.content,
          f"→ {'content FOUND in DB' if res.content_known else 'content probably NOT in DB'}"
          f" (threshold {th['content_known']:.2f})")
    block("TOP-3 VOICE (ECAPA cosine)", res.voice,
          f"→ {'speaker FOUND in DB' if res.voice_known else 'speaker probably NOT in DB'}"
          f" (threshold {th['voice_known']:.2f})")
    block("Baseline voice: MFCC 50-D cosine", res.voice_mfcc)
    block("Baseline content: BM25 keyword (FTS5)", res.keyword, label="bm25")
    return "\n".join(L)
