"""Feature database — SQLite store for every file's metadata and features.

Why SQLite: the whole system has to run on a student laptop and in a demo
without a database server. SQLite is a real relational DBMS (transactions,
foreign keys, indexes, SQL queries) shipped with Python, and its FTS5
extension gives a full-text index over the transcripts for free.

Schema (see SCHEMA below):

  audio_files       one row per audio file — labels parsed from the filename,
                    duration, speech / multi-speaker flags
  voice_features    ECAPA 192-D, MFCC mean/std 78-D, clean MFCC 50-D, prosody
  content_features  transcript, mean-pooled semantic vector 384-D
  content_chunks    each transcript chunk with its own 384-D vector
  transcripts_fts   FTS5 full-text index over transcripts (BM25 ranking)
  feature_stats     corpus statistics needed to normalise a query (MFCC z-score)
  meta              model ids, feature version, build time

Vectors are stored as little-endian float32 BLOBs. Similarity search loads
each vector family once into a numpy matrix (`VectorIndex`) and ranks by
cosine = dot product of L2-normalised vectors. With N ≈ 500 files an exact
scan costs N·d ≈ 0.2 M multiply-adds per query (< 1 ms), so an approximate
index (HNSW / IVF) would add complexity without a measurable gain; the
`VectorIndex` interface is where one would plug in for a larger corpus.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config, corpus

logger = logging.getLogger(__name__)

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS audio_files (
    file_id        TEXT PRIMARY KEY,           -- stem of the source filename
    filename       TEXT NOT NULL,
    source         TEXT NOT NULL,              -- TED / KhanAcademy / Storybook / Vooks / Other
    speaker        TEXT,                       -- parsed from filename (may be NULL)
    topic          TEXT,
    course         TEXT,                       -- Khan Academy course (same narrator)
    duration_s     REAL NOT NULL,
    is_speech      INTEGER NOT NULL DEFAULT 1, -- 0 = instrumental / no transcript
    multi_speaker  INTEGER NOT NULL DEFAULT 0, -- 1 = filename names several readers
    src_path       TEXT,
    wav_path       TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_source  ON audio_files(source);
CREATE INDEX IF NOT EXISTS idx_files_speaker ON audio_files(speaker);
CREATE INDEX IF NOT EXISTS idx_files_course  ON audio_files(course);

CREATE TABLE IF NOT EXISTS voice_features (
    file_id         TEXT PRIMARY KEY REFERENCES audio_files(file_id) ON DELETE CASCADE,
    ecapa           BLOB NOT NULL,             -- float32[192], L2-normalised
    mfcc_mean       BLOB NOT NULL,             -- float32[78]  (26 MFCC + Δ + ΔΔ, time mean)
    mfcc_std        BLOB NOT NULL,             -- float32[78]  (time std)
    mfcc_clean      BLOB NOT NULL,             -- float32[50]  z-scored mean+std of c1..c25
    f0_mean_hz      REAL,
    f0_std_hz       REAL,
    zcr_mean        REAL,
    centroid_hz     REAL,
    rolloff_hz      REAL
);

CREATE TABLE IF NOT EXISTS content_features (
    file_id         TEXT PRIMARY KEY REFERENCES audio_files(file_id) ON DELETE CASCADE,
    transcript      TEXT NOT NULL,
    n_chunks        INTEGER NOT NULL,
    semantic        BLOB NOT NULL,             -- float32[384], L2-normalised (zeros if no speech)
    asr_runtime     TEXT,
    feature_version TEXT
);

CREATE TABLE IF NOT EXISTS content_chunks (
    file_id    TEXT NOT NULL REFERENCES audio_files(file_id) ON DELETE CASCADE,
    chunk_idx  INTEGER NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,                  -- float32[384], L2-normalised
    PRIMARY KEY (file_id, chunk_idx)
);

CREATE VIRTUAL TABLE IF NOT EXISTS transcripts_fts USING fts5(
    file_id UNINDEXED, transcript, tokenize = 'porter unicode61'
);

CREATE TABLE IF NOT EXISTS feature_stats (
    name  TEXT PRIMARY KEY,
    mean  BLOB NOT NULL,
    std   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# BLOB helpers
# ---------------------------------------------------------------------------
def to_blob(vec) -> bytes:
    return np.asarray(vec, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4").copy()


def connect(path: Path | None = None) -> sqlite3.Connection:
    con = sqlite3.connect(str(path or config.DB_PATH))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# ---------------------------------------------------------------------------
# Classical voice vector
# ---------------------------------------------------------------------------
MFCC_BASE = slice(1, 26)   # c1..c25 — c0 (log-energy) depends on loudness, not voice


def mfcc_raw_clean(mfcc_mean: np.ndarray, mfcc_std: np.ndarray) -> np.ndarray:
    """Un-normalised 50-D classical voice vector: mean + std of c1..c25.

    Works on a single (78,) vector or a stacked (n, 78) matrix.
    """
    mean = np.asarray(mfcc_mean, dtype=np.float64)
    std = np.asarray(mfcc_std, dtype=np.float64)
    return np.concatenate([mean[..., MFCC_BASE], std[..., MFCC_BASE]], axis=-1)


def mfcc_search_vector(raw50: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Z-score with corpus statistics, then L2-normalise (so cosine = dot)."""
    z = (np.asarray(raw50, dtype=np.float64) - mu) / sigma
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return (z / norm).astype(np.float32)


def _multi_speaker(speaker: str | None) -> bool:
    if not speaker:
        return False
    s = f" {speaker.replace('_', ' ')} "
    return " and " in s or "&" in s or "," in s or " y " in s


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(db_path: Path | None = None) -> dict:
    """(Re)create the database from artifacts/features/*.json. Returns counts."""
    db_path = db_path or config.DB_PATH
    config.ensure_dirs()
    if db_path.exists():
        db_path.unlink()

    voice_paths = sorted(config.FEATURES_DIR.glob("voice-*.json"))
    rows_v, rows_c = [], []
    for vp in voice_paths:
        fid = vp.stem[len("voice-"):]
        cp = config.FEATURES_DIR / f"content-{fid}.json"
        if not cp.exists():
            logger.warning("no content artifact for %s — skipped", fid)
            continue
        rows_v.append(json.loads(vp.read_text(encoding="utf-8")))
        rows_c.append(json.loads(cp.read_text(encoding="utf-8")))

    if not rows_v:
        raise RuntimeError(f"no feature artifacts under {config.FEATURES_DIR}")

    # Corpus statistics for the classical MFCC vector.
    raw = mfcc_raw_clean(np.array([v["mfcc_stats"] for v in rows_v]),
                         np.array([v["mfcc_std"] for v in rows_v]))
    mu, sigma = raw.mean(axis=0), raw.std(axis=0)
    sigma[sigma == 0] = 1.0
    clean = mfcc_search_vector(raw, mu, sigma)

    con = connect(db_path)
    con.executescript(SCHEMA)
    with con:
        for i, (v, c) in enumerate(zip(rows_v, rows_c, strict=True)):
            fid = v["file_id"]
            meta = corpus.parse_file_id(fid)
            transcript = c.get("transcript", "") or ""
            sem = np.asarray(c["semantic_embedding"], dtype=np.float32)
            is_speech = (len(transcript.strip()) >= config.MIN_TRANSCRIPT_CHARS and np.linalg.norm(sem) > 0.5
                         and fid not in config.NON_TALK_FILES)
            src = config.DATA_DIR / v.get("file", "")
            con.execute(
                "INSERT INTO audio_files VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (fid, v.get("file", ""), meta.source, meta.speaker, meta.topic, meta.course,
                 float(v.get("duration_s", 0.0)), int(is_speech), int(_multi_speaker(meta.speaker) or fid in config.MULTI_SPEAKER_FILES),
                 str(src), str(config.WAV_DIR / f"{fid}.wav")),
            )
            p = v.get("prosody", {})
            con.execute(
                "INSERT INTO voice_features VALUES (?,?,?,?,?,?,?,?,?,?)",
                (fid, to_blob(v["ecapa_embedding"]), to_blob(v["mfcc_stats"]), to_blob(v["mfcc_std"]),
                 to_blob(clean[i]), p.get("f0_mean_hz"), p.get("f0_std_hz"), p.get("zcr_mean"),
                 p.get("spectral_centroid_mean"), p.get("spectral_rolloff_mean")),
            )
            con.execute(
                "INSERT INTO content_features VALUES (?,?,?,?,?,?)",
                (fid, transcript, int(c.get("n_chunks", 0)), to_blob(sem),
                 c.get("runtime", ""), c.get("feature_version", "")),
            )
            for j, (txt, emb) in enumerate(zip(c.get("chunks", []), c.get("chunk_embeddings", []),
                                               strict=False)):
                con.execute("INSERT INTO content_chunks VALUES (?,?,?,?)", (fid, j, txt, to_blob(emb)))
            if transcript.strip():
                con.execute("INSERT INTO transcripts_fts(file_id, transcript) VALUES (?,?)",
                            (fid, transcript))

        con.execute("INSERT INTO feature_stats VALUES (?,?,?)",
                    ("mfcc_clean_raw50", to_blob(mu), to_blob(sigma)))
        for k, val in {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_version": config.FEATURE_VERSION,
            "ecapa_model": config.ECAPA_MODEL_ID,
            "whisper_model": config.WHISPER_MODEL_SIZE,
            "semantic_model": config.SEMANTIC_MODEL_ID,
        }.items():
            con.execute("INSERT INTO meta VALUES (?,?)", (k, val))

    counts = summary(con)
    con.close()
    logger.info("database built: %s -> %s", counts, db_path)
    return counts


def summary(con: sqlite3.Connection) -> dict:
    q = lambda sql: con.execute(sql).fetchone()[0]
    return {
        "files": q("SELECT COUNT(*) FROM audio_files"),
        "speech_files": q("SELECT COUNT(*) FROM audio_files WHERE is_speech = 1"),
        "non_speech_files": q("SELECT COUNT(*) FROM audio_files WHERE is_speech = 0"),
        "multi_speaker_files": q("SELECT COUNT(*) FROM audio_files WHERE multi_speaker = 1"),
        # Files meeting the assignment's criteria: a talk, one speaker.
        "eligible_files": q("SELECT COUNT(*) FROM audio_files WHERE is_speech = 1 AND multi_speaker = 0"),
        "chunks": q("SELECT COUNT(*) FROM content_chunks"),
        "total_hours": round(q("SELECT SUM(duration_s) FROM audio_files") / 3600, 2),
    }


# ---------------------------------------------------------------------------
# In-memory vector index loaded from the database
# ---------------------------------------------------------------------------
@dataclass
class VectorIndex:
    """All vectors of the database as numpy matrices, row-aligned with `ids`."""

    ids: list[str]
    meta: dict[str, dict]                 # file_id -> audio_files row
    ecapa: np.ndarray                     # (n, 192)
    mfcc_clean: np.ndarray                # (n, 50)
    semantic: np.ndarray                  # (n, 384)
    is_speech: np.ndarray                 # (n,) bool
    voice_ok: np.ndarray                  # (n,) bool — talk with a single speaker
    mfcc_mu: np.ndarray
    mfcc_sigma: np.ndarray
    chunk_file: list[str] = field(default_factory=list)
    chunk_text: list[str] = field(default_factory=list)
    chunk_emb: np.ndarray = field(default_factory=lambda: np.zeros((0, config.SEMANTIC_DIM), np.float32))
    db_path: Path = field(default_factory=lambda: config.DB_PATH)

    def __len__(self) -> int:
        return len(self.ids)

    def row(self, file_id: str) -> int:
        return self.ids.index(file_id)

    def transcript(self, file_id: str) -> str:
        con = connect(self.db_path)
        try:
            r = con.execute("SELECT transcript FROM content_features WHERE file_id = ?",
                            (file_id,)).fetchone()
            return r[0] if r else ""
        finally:
            con.close()

    def keyword_search(self, text: str, k: int = 3, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        """BM25 full-text baseline over the transcripts (FTS5)."""
        words = [w for w in "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()
                 if len(w) > 3][:64]
        if not words:
            return []
        query = " OR ".join(f'"{w}"' for w in dict.fromkeys(words))
        con = connect(self.db_path)
        try:
            rows = con.execute(
                "SELECT file_id, bm25(transcripts_fts) AS s FROM transcripts_fts "
                "WHERE transcripts_fts MATCH ? ORDER BY s LIMIT ?",
                (query, k + len(exclude or ())),
            ).fetchall()
        finally:
            con.close()
        out = [(r["file_id"], -float(r["s"])) for r in rows if r["file_id"] not in (exclude or set())]
        return out[:k]


def load_index(db_path: Path | None = None) -> VectorIndex:
    db_path = db_path or config.DB_PATH
    if not db_path.exists():
        raise FileNotFoundError(f"{db_path} not found — run `python -m audio_search build-db` first")
    con = connect(db_path)
    try:
        rows = con.execute(
            "SELECT f.*, v.ecapa, v.mfcc_clean, c.semantic FROM audio_files f "
            "JOIN voice_features v USING(file_id) JOIN content_features c USING(file_id) "
            "ORDER BY f.file_id"
        ).fetchall()
        stats = con.execute("SELECT mean, std FROM feature_stats WHERE name = 'mfcc_clean_raw50'").fetchone()
        chunks = con.execute("SELECT file_id, text, embedding FROM content_chunks "
                             "ORDER BY file_id, chunk_idx").fetchall()
    finally:
        con.close()

    meta_cols = ("file_id", "filename", "source", "speaker", "topic", "course", "duration_s",
                 "is_speech", "multi_speaker", "src_path", "wav_path")
    ids = [r["file_id"] for r in rows]
    is_speech = np.array([bool(r["is_speech"]) for r in rows])
    return VectorIndex(
        ids=ids,
        meta={r["file_id"]: {c: r[c] for c in meta_cols} for r in rows},
        ecapa=np.vstack([from_blob(r["ecapa"]) for r in rows]),
        mfcc_clean=np.vstack([from_blob(r["mfcc_clean"]) for r in rows]),
        semantic=np.vstack([from_blob(r["semantic"]) for r in rows]),
        is_speech=is_speech,
        # Non-talks and multi-reader files are kept in the DB but are not
        # results: they do not satisfy "một người nói / bài nói".
        voice_ok=is_speech & ~np.array([bool(r["multi_speaker"]) for r in rows]),
        mfcc_mu=from_blob(stats["mean"]).astype(np.float64),
        mfcc_sigma=from_blob(stats["std"]).astype(np.float64),
        chunk_file=[r["file_id"] for r in chunks],
        chunk_text=[r["text"] for r in chunks],
        chunk_emb=(np.vstack([from_blob(r["embedding"]) for r in chunks]) if chunks
                   else np.zeros((0, config.SEMANTIC_DIM), np.float32)),
        db_path=db_path,
    )
