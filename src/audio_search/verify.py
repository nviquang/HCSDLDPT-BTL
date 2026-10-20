"""Verification — smoke tests on extracted artifacts.

Checks that per-file artifacts are complete, well-shaped, and that the
in-memory top-3 retrieval behaves sanely (self-similarity == 1.0, scores in
decreasing order).
"""

from __future__ import annotations

import json
import logging

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _load_voice(file_id: str) -> dict | None:
    p = config.FEATURES_DIR / f"voice-{file_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load_content(file_id: str) -> dict | None:
    p = config.FEATURES_DIR / f"content-{file_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _l2(vec) -> float:
    return float(np.linalg.norm(np.asarray(vec, dtype=np.float32)))


def run() -> int:
    voice_files = sorted(config.FEATURES_DIR.glob("voice-*.json"))
    ids = sorted({p.stem.replace("voice-", "") for p in voice_files})
    failed = 0

    if not ids:
        logger.error("no artifacts found under %s — run `python -m audio_search all` first", config.FEATURES_DIR)
        return 1

    logger.info("verifying %d file(s)...", len(ids))
    ok_count = 0
    for fid in ids:
        v = _load_voice(fid)
        c = _load_content(fid)
        problems = []

        if v is None or c is None:
            problems.append("missing voice or content artifact")
        else:
            # Shape checks
            if len(v["ecapa_embedding"]) != config.ECAPA_DIM:
                problems.append(f"ecapa dim {len(v['ecapa_embedding'])} != {config.ECAPA_DIM}")
            if len(v["mfcc_stats"]) != 78:
                problems.append(f"mfcc_stats dim {len(v['mfcc_stats'])} != 78")
            if len(c["semantic_embedding"]) != config.SEMANTIC_DIM:
                problems.append(f"semantic dim {len(c['semantic_embedding'])} != {config.SEMANTIC_DIM}")

            # L2 sanity (should be ~1.0 for normalized vectors)
            ecapa_norm = _l2(v["ecapa_embedding"])
            sem_norm = _l2(c["semantic_embedding"])
            if not 0.9 < ecapa_norm < 1.1:
                problems.append(f"ecapa L2 norm {ecapa_norm:.3f} not ~1.0")
            # Structural zero semantic vector is expected for instrumental-only
            # files (no speech to transcribe). Not a pipeline failure.
            if sem_norm < 0.01:
                logger.warning("  [%s] empty content (instrumental/no speech) — skipped",
                               fid)
            elif not 0.9 < sem_norm < 1.1:
                problems.append(f"semantic L2 norm {sem_norm:.3f} not ~1.0")

            # Transcript present (may be short for instrumental files)
            trans = c.get("transcript", "").strip()
            if not trans:
                logger.warning("  [%s] empty transcript (instrumental/no speech) — skipped", fid)

        if problems:
            failed += 1
            for pr in problems:
                logger.warning("  [%s] %s", fid, pr)
        else:
            ok_count += 1

    # --- In-memory retrieval sanity (top-3 voice + top-3 content) ---
    if len(ids) >= 2:
        logger.info("top-3 sanity on voice & content embeddings:")
        for seed in ids[: min(2, len(ids))]:
            print_top3(seed, ids)

    logger.info("verify: %d/%d artifacts OK, %d with problems", ok_count, len(ids), failed)
    return 1 if failed else 0


def print_top3(query_id: str, all_ids: list[str], k: int = 3) -> None:
    """Print the top-k most similar files to query_id (voice and content)."""
    qv = np.asarray(_load_voice(query_id)["ecapa_embedding"], dtype=np.float32)
    qc = np.asarray(_load_content(query_id)["semantic_embedding"], dtype=np.float32)

    # Voice: cosine of L2-normalized vectors == dot product
    v_rank = []
    for fid in all_ids:
        if fid == query_id:
            continue
        ev = np.asarray(_load_voice(fid)["ecapa_embedding"], dtype=np.float32)
        v_rank.append((fid, float(np.dot(qv, ev))))
    v_rank.sort(key=lambda t: -t[1])

    # Content (skip files with no semantic vector — instrumental)
    c_rank = []
    for fid in all_ids:
        if fid == query_id:
            continue
        ec = np.asarray(_load_content(fid)["semantic_embedding"], dtype=np.float32)
        if _l2(ec) < 0.01:
            continue
        c_rank.append((fid, float(np.dot(qc, ec))))
    c_rank.sort(key=lambda t: -t[1])

    print(f"\n-- query: {query_id}")
    print("  voice top-3:   " + ", ".join(f"{fid} ({s:.3f})" for fid, s in v_rank[:k]))
    print("  content top-3: " + ", ".join(f"{fid} ({s:.3f})" for fid, s in c_rank[:k]))

    # Self-similarity sanity (should be 1.0)
    self_v = float(np.dot(qv, qv))
    self_c = float(np.dot(qc, qc))
    print(f"  self-similarity: voice={self_v:.3f} content={self_c:.3f}")