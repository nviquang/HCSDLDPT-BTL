"""Unit tests for chunking, the feature database and ranking (no models needed).

Run: uv run pytest -q
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from audio_search import (
    config,
    content_features,
    database,
    evaluation,
    preprocess,
    search,
)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def _sentences(n: int) -> str:
    return " ".join(f"Sentence number {i} has some words in it to be long enough." for i in range(n))


def test_chunker_keeps_every_sentence_in_order():
    chunks = content_features.chunk_to_sentences(_sentences(60), max_tokens=60)
    got = [int(x) for x in re.findall(r"number (\d+)", " ".join(chunks))]
    assert got == list(range(60))
    assert len(chunks) > 1


def test_chunker_respects_token_budget():
    count = lambda s: len(s.split())
    chunks = content_features.chunk_to_sentences(_sentences(40), max_tokens=30, token_len=count)
    assert all(count(c) <= 30 for c in chunks)


def test_chunker_splits_overlong_sentence():
    long_sentence = " ".join(f"word{i}" for i in range(500))
    count = lambda s: len(s.split())
    chunks = content_features.chunk_to_sentences(long_sentence, max_tokens=100, token_len=count)
    assert " ".join(chunks).split() == long_sentence.split()
    assert all(count(c) <= 100 for c in chunks)


def test_chunker_empty_and_short():
    assert content_features.chunk_to_sentences("") == []
    assert content_features.chunk_to_sentences("Hi.") == ["Hi."]


# ---------------------------------------------------------------------------
# Duration override (CLI --min/--max-duration)
# ---------------------------------------------------------------------------
def test_validate_duration_reads_config_at_call_time(monkeypatch):
    monkeypatch.setattr(config, "MIN_DURATION_S", 100.0)
    assert not preprocess.validate_duration(50.0)
    monkeypatch.setattr(config, "MIN_DURATION_S", 10.0)
    assert preprocess.validate_duration(50.0)


# ---------------------------------------------------------------------------
# Database round-trip
# ---------------------------------------------------------------------------
def _unit(rng, d):
    v = rng.standard_normal(d).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def tiny_db(tmp_path, monkeypatch):
    """Three fake artifacts -> a real SQLite database in tmp_path."""
    feats = tmp_path / "features"
    feats.mkdir()
    monkeypatch.setattr(config, "FEATURES_DIR", feats)
    monkeypatch.setattr(config, "ANALYSIS_DIR", tmp_path / "analysis")
    monkeypatch.setattr(config, "ANALYSIS_FIGURES_DIR", tmp_path / "analysis" / "fig")
    monkeypatch.setattr(config, "WAV_DIR", tmp_path / "wav")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    rng = np.random.default_rng(0)
    ids = ["'Story_A'_read_by_Alice", "Talk_B_John_Doe_TED", "Music_C_Band_TED"]
    texts = ["The dog buried the brush and ran away from home. " * 10,
             "Bacteria can break down plastic waste in the ocean. " * 10,
             ""]
    for fid, text in zip(ids, texts, strict=True):
        (feats / f"voice-{fid}.json").write_text(json.dumps({
            "file": fid + ".mp3", "file_id": fid, "duration_s": 300.0,
            "mfcc_stats": rng.standard_normal(78).tolist(), "mfcc_std": rng.random(78).tolist(),
            "prosody": {"f0_mean_hz": 150.0, "f0_std_hz": 20.0, "zcr_mean": 0.1,
                        "spectral_centroid_mean": 1500.0, "spectral_rolloff_mean": 3000.0},
            "ecapa_embedding": _unit(rng, 192).tolist()}), encoding="utf-8")
        sem = _unit(rng, 384) if text else np.zeros(384, np.float32)
        (feats / f"content-{fid}.json").write_text(json.dumps({
            "file_id": fid, "transcript": text, "n_chunks": 1 if text else 0,
            "semantic_embedding": sem.tolist(), "chunks": [text] if text else [],
            "chunk_embeddings": [sem.tolist()] if text else []}), encoding="utf-8")
    db = tmp_path / "test.db"
    counts = database.build(db)
    return db, ids, counts


def test_build_counts_and_flags(tiny_db):
    db, ids, counts = tiny_db
    assert counts["files"] == 3
    assert counts["speech_files"] == 2 and counts["non_speech_files"] == 1
    con = database.connect(db)
    row = con.execute("SELECT source, speaker FROM audio_files WHERE file_id = ?", (ids[0],)).fetchone()
    assert row["source"] == "Storybook" and row["speaker"] == "Alice"
    con.close()


def test_blob_roundtrip_and_index(tiny_db):
    db, ids, _ = tiny_db
    idx = database.load_index(db)
    assert idx.ecapa.shape == (3, 192) and idx.semantic.shape == (3, 384)
    assert idx.mfcc_clean.shape == (3, 50)
    assert np.allclose(np.linalg.norm(idx.ecapa, axis=1), 1.0, atol=1e-5)
    assert not idx.is_speech[idx.row(ids[2])]
    hits = idx.keyword_search("plastic bacteria ocean", k=3)
    assert hits and hits[0][0] == ids[1]


def _fake_query(idx, row):
    """A query whose vectors equal database row `row`."""
    return search.QueryFeatures(
        name="q", duration_s=30.0, wave=np.zeros(10, np.float32),
        ecapa=idx.ecapa[row], mfcc_mean=np.zeros(78), mfcc_std=np.zeros(78),
        mfcc_clean=idx.mfcc_clean[row], prosody={}, transcript="the dog buried the brush",
        semantic=idx.semantic[row], chunks=["the dog buried the brush"],
        chunk_embeddings=idx.semantic[row][None, :])


def test_rank_self_is_top1_sorted_and_excludes(tiny_db):
    db, ids, _ = tiny_db
    idx = database.load_index(db)
    q = _fake_query(idx, idx.row(ids[0]))
    res = search.rank(q, idx, k=3)
    assert res.content[0].file_id == ids[0] and res.voice[0].file_id == ids[0]
    assert np.isclose(res.content[0].score, 1.0, atol=1e-5)
    scores = [h.score for h in res.voice]
    assert scores == sorted(scores, reverse=True)
    # instrumental file never appears in content results
    assert ids[2] not in [h.file_id for h in res.content]
    # held-out search: the source file disappears from both lists
    res2 = search.rank(q, idx, k=3, exclude={ids[0]})
    assert ids[0] not in [h.file_id for h in res2.content + res2.voice]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def test_rank_of_and_hit_stats():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    mask = np.array([True, True, True, False])
    assert evaluation.rank_of(scores, mask, 1) == 1
    assert evaluation.rank_of(scores, mask, 2) == 2       # 0.7 is masked out
    s = evaluation.hit_stats([1, 2, 5])
    assert s["hit@1"] == pytest.approx(1 / 3) and s["hit@3"] == pytest.approx(2 / 3)


def test_calibrate_separable():
    c = evaluation.calibrate([0.8, 0.9, 0.85], [0.2, 0.3, 0.25])
    assert c["auc"] == 1.0 and 0.3 <= c["threshold"] <= 0.8


def test_flagged_file_ids_exist_in_corpus():
    """A typo in NON_TALK_FILES / MULTI_SPEAKER_FILES would silently flag nothing."""
    if not config.DATA_DIR.is_dir():
        pytest.skip("corpus not present")
    stems = {p.stem for p in config.DATA_DIR.iterdir() if p.is_file()}
    assert not (config.NON_TALK_FILES | config.MULTI_SPEAKER_FILES) - stems
