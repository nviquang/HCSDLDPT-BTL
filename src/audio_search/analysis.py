"""Dataset analysis — audit the extracted feature sets and describe the corpus.

This module answers two questions that the feature extraction step cannot:

1. *Are the extracted vectors actually usable?* — dimensionality analysis,
   integrity checks, and how well each vector family separates the corpus.
2. *What does the corpus look like?* — source mix, durations, label coverage,
   and how far the corpus sits from the targets set out in ``report.md``.

Everything here is read-only with respect to the artifacts; it only reads
``artifacts/features/*.json`` and writes to ``artifacts/analysis/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, corpus

logger = logging.getLogger(__name__)

# Tolerances for the integrity audit.
L2_TOL = 0.1          # |norm - 1| must be below this for normalized vectors
MIN_COS = -1.0        # cosine similarity bounds
MAX_COS = 1.0

# MFCC block layout: 26 base coefficients, then 26 deltas, then 26 delta-deltas.
MFCC_BASE = slice(0, 26)
MFCC_DELTA = slice(26, 52)
MFCC_DDELTA = slice(52, 78)


@dataclass
class VoiceMatrix:
    """Every voice artifact, stacked into matrices."""

    ids: list[str]
    mfcc_stats: np.ndarray          # (n, 78)
    mfcc_std: np.ndarray            # (n, 78)
    ecapa: np.ndarray               # (n, 192)
    prosody: np.ndarray             # (n, 5)
    durations: np.ndarray           # (n,)
    problems: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class ContentMatrix:
    """Every content artifact, stacked into matrices."""

    ids: list[str]
    semantic: np.ndarray            # (n, 384)
    n_chunks: np.ndarray            # (n,)
    durations: np.ndarray           # (n,)
    transcript_chars: np.ndarray    # (n,)
    problems: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.ids)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_voice_matrix() -> VoiceMatrix:
    """Read every ``voice-*.json`` into stacked numpy arrays.

    Files that fail to parse, or whose vector dimensions are wrong, are
    recorded in ``problems`` and excluded from the matrices.
    """
    paths = sorted(config.FEATURES_DIR.glob("voice-*.json"))
    ids: list[str] = []
    mfcc_s: list[list[float]] = []
    mfcc_t: list[list[float]] = []
    ecapa: list[list[float]] = []
    prosody: list[list[float]] = []
    durations: list[float] = []
    problems: list[dict] = []

    for path in paths:
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append({"file_id": path.stem.replace("voice-", ""),
                             "issue": f"unreadable artifact: {exc}"})
            continue

        fid = payload.get("file_id", path.stem.replace("voice-", ""))
        try:
            m_stats = np.asarray(payload["mfcc_stats"], dtype=np.float64)
            m_std = np.asarray(payload["mfcc_std"], dtype=np.float64)
            ec = np.asarray(payload["ecapa_embedding"], dtype=np.float64)
            pr = payload["prosody"]
            pro = np.asarray([pr["f0_mean_hz"], pr["f0_std_hz"], pr["zcr_mean"],
                              pr["spectral_centroid_mean"], pr["spectral_rolloff_mean"]],
                             dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append({"file_id": fid, "issue": f"malformed payload: {exc}"})
            continue

        if m_stats.shape != (78,) or m_std.shape != (78,):
            problems.append({"file_id": fid, "issue": f"mfcc shape {m_stats.shape}"})
            continue
        if ec.shape != (config.ECAPA_DIM,):
            problems.append({"file_id": fid, "issue": f"ecapa shape {ec.shape}"})
            continue

        ids.append(fid)
        mfcc_s.append(m_stats)
        mfcc_t.append(m_std)
        ecapa.append(ec)
        prosody.append(pro)
        durations.append(float(payload.get("duration_s", 0.0)))

    if not ids:
        return VoiceMatrix([], np.zeros((0, 78)), np.zeros((0, 78)),
                           np.zeros((0, config.ECAPA_DIM)), np.zeros((0, 5)),
                           np.zeros(0), problems)

    return VoiceMatrix(
        ids=ids,
        mfcc_stats=np.vstack(mfcc_s),
        mfcc_std=np.vstack(mfcc_t),
        ecapa=np.vstack(ecapa),
        prosody=np.vstack(prosody),
        durations=np.asarray(durations),
        problems=problems,
    )


def load_content_matrix() -> ContentMatrix:
    """Read every ``content-*.json`` into stacked numpy arrays."""
    paths = sorted(config.FEATURES_DIR.glob("content-*.json"))
    ids: list[str] = []
    sem: list[list[float]] = []
    chunks: list[int] = []
    durations: list[float] = []
    chars: list[int] = []
    problems: list[dict] = []

    for path in paths:
        try:
            payload = _load_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            problems.append({"file_id": path.stem.replace("content-", ""),
                             "issue": f"unreadable artifact: {exc}"})
            continue

        fid = payload.get("file_id", path.stem.replace("content-", ""))
        try:
            vec = np.asarray(payload["semantic_embedding"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            problems.append({"file_id": fid, "issue": f"malformed payload: {exc}"})
            continue

        if vec.shape != (config.SEMANTIC_DIM,):
            problems.append({"file_id": fid, "issue": f"semantic shape {vec.shape}"})
            continue

        ids.append(fid)
        sem.append(vec)
        chunks.append(int(payload.get("n_chunks", 0)))
        durations.append(float(payload.get("duration_s", 0.0)))
        chars.append(len(payload.get("transcript", "")))

    if not ids:
        return ContentMatrix([], np.zeros((0, config.SEMANTIC_DIM)), np.zeros(0),
                             np.zeros(0), np.zeros(0), problems)

    return ContentMatrix(
        ids=ids,
        semantic=np.vstack(sem),
        n_chunks=np.asarray(chunks),
        durations=np.asarray(durations),
        transcript_chars=np.asarray(chars),
        problems=problems,
    )


# ---------------------------------------------------------------------------
# Integrity audit
# ---------------------------------------------------------------------------
def audit_integrity(voice: VoiceMatrix, content: ContentMatrix) -> dict:
    """Check every artifact for shape, finiteness, and normalization."""
    issues: list[dict] = []
    issues.extend(voice.problems)
    issues.extend(content.problems)

    def _check(ids, matrix, name, expected_norm: bool) -> None:
        for i, fid in enumerate(ids):
            row = matrix[i]
            if not np.all(np.isfinite(row)):
                issues.append({"file_id": fid, "issue": f"{name}: non-finite values"})
                continue
            norm = float(np.linalg.norm(row))
            if expected_norm and abs(norm - 1.0) > L2_TOL:
                issues.append({"file_id": fid, "issue": f"{name}: L2 norm {norm:.4f} != 1"})

    _check(voice.ids, voice.ecapa, "ecapa_embedding", expected_norm=True)
    _check(content.ids, content.semantic, "semantic_embedding", expected_norm=True)

    for i, fid in enumerate(content.ids):
        if float(np.linalg.norm(content.semantic[i])) < 0.01:
            issues.append({"file_id": fid,
                           "issue": "semantic_embedding is a zero vector (no speech)"})

    for i, fid in enumerate(voice.ids):
        if not np.all(np.isfinite(voice.prosody[i])):
            issues.append({"file_id": fid, "issue": "prosody: non-finite values"})

    return {
        "n_voice": len(voice),
        "n_content": len(content),
        "n_issues": len(issues),
        "issues": issues[:100],
        "voice_dim_ok": voice.mfcc_stats.shape[1] == 78 and voice.ecapa.shape[1] == 192,
        "content_dim_ok": content.semantic.shape[1] == 384,
    }


# ---------------------------------------------------------------------------
# Dimensionality
# ---------------------------------------------------------------------------
def pca_analysis(matrix: np.ndarray, name: str) -> dict:
    """Variance explained by the principal components of one matrix."""
    if matrix.shape[0] < 2:
        return {"name": name, "n": int(matrix.shape[0]), "dim": int(matrix.shape[1])}

    centered = matrix - matrix.mean(axis=0)
    # svd on the (n, d) matrix: right singular vectors are the components.
    _, s, _vt = np.linalg.svd(centered, full_matrices=False)
    var = s**2
    if var.sum() == 0:
        return {"name": name, "n": int(matrix.shape[0]), "dim": int(matrix.shape[1]),
                "explained_variance_ratio": [], "n_components": {}}
    ratio = var / var.sum()
    cumulative = np.cumsum(ratio)

    def _n_at(threshold: float) -> int:
        return int(np.searchsorted(cumulative, threshold) + 1)

    return {
        "name": name,
        "n": int(matrix.shape[0]),
        "dim": int(matrix.shape[1]),
        # The full ratio list is kept: a truncated curve would end below 100%
        # and read as "this family needs fewer dimensions than it does".
        "explained_variance_ratio": [float(x) for x in ratio],
        "n_components": {
            "90%": _n_at(0.90),
            "95%": _n_at(0.95),
            "99%": _n_at(0.99),
        },
        "top5_variance_pct": [float(x * 100) for x in ratio[:5]],
    }


def mfcc_block_analysis(mfcc: np.ndarray) -> dict:
    """How much of the 78-D MFCC vector's energy lives in each block.

    The vector is ``[26 base | 26 delta | 26 delta-delta]``. If the delta
    blocks contribute almost no variance, the "78-D" vector is really 26-D
    with two blocks of numerical noise.
    """
    if mfcc.shape[0] == 0:
        return {}

    centered = mfcc - mfcc.mean(axis=0)
    total_energy = float((centered**2).sum())
    if total_energy == 0:
        return {}

    blocks = {
        "base": MFCC_BASE,
        "delta": MFCC_DELTA,
        "delta_delta": MFCC_DDELTA,
    }
    out: dict = {"total_energy": total_energy}
    for block_name, sl in blocks.items():
        b = centered[:, sl]
        energy = float((b**2).sum())
        s = np.linalg.svd(b, compute_uv=False)
        var = s**2
        cum = np.cumsum(var / var.sum()) if var.sum() > 0 else np.zeros(1)
        out[block_name] = {
            "energy": energy,
            "energy_share": energy / total_energy,
            "abs_mean": float(np.abs(b).mean()),
            "n_components_95pct": int(np.searchsorted(cum, 0.95) + 1),
        }

    # How strongly the leading component is just the log-energy term c0.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = centered @ vt[0]
    c0 = centered[:, 0]
    corr = float(np.corrcoef(pc1, c0)[0, 1]) if c0.std() > 0 else 0.0
    out["pc1_vs_c0_correlation"] = corr
    out["c0_mean"] = float(mfcc[:, 0].mean())
    out["c0_std"] = float(mfcc[:, 0].std())
    out["c1_c25_abs_mean"] = float(np.abs(mfcc[:, 1:26]).mean())
    return out


# ---------------------------------------------------------------------------
# Similarity structure
# ---------------------------------------------------------------------------
def _row_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_matrix(matrix: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Full pairwise cosine similarity matrix."""
    if matrix.shape[0] == 0:
        return np.zeros((0, 0))
    x = _row_normalize(matrix) if normalize else matrix
    return x @ x.T


def similarity_stats(similarity: np.ndarray) -> dict:
    """Summary of the off-diagonal similarity distribution."""
    if similarity.shape[0] < 2:
        return {}
    upper = similarity[np.triu_indices(similarity.shape[0], 1)]
    return {
        "n_pairs": int(upper.size),
        "mean": float(upper.mean()),
        "median": float(np.median(upper)),
        "std": float(upper.std()),
        "min": float(upper.min()),
        "p01": float(np.percentile(upper, 1)),
        "p99": float(np.percentile(upper, 99)),
        "max": float(upper.max()),
    }


def mfcc_clean_variant(mfcc: np.ndarray) -> np.ndarray:
    """The usable form of the 78-D MFCC vector.

    Drops the delta blocks (which carry no variance) and z-scores the 26
    base coefficients so that no single coefficient — above all the
    log-energy term c0 — dominates the cosine.
    """
    base = mfcc[:, MFCC_BASE]
    std = base.std(axis=0)
    std[std == 0] = 1.0
    return (base - base.mean(axis=0)) / std


def _is_missing(label: object) -> bool:
    """True for a label that carries no information.

    ``None`` is the in-memory sentinel, but the labels reach this module from
    a pandas DataFrame, where a missing string becomes ``NaN`` (or ``pd.NA``)
    — and both are *truthy*, so a plain truthiness test would treat every
    missing label as a real group. Worse, building a numpy array from a mixed
    str/NaN list coerces the NaNs to the literal string ``"nan"``, which would
    then group all label-less files together.
    """
    if label is None:
        return True
    if isinstance(label, float) and np.isnan(label):
        return True
    if label is pd.NA:
        return True
    return label == ""


def label_gap(similarity: np.ndarray, labels: list[str | None]) -> dict:
    """Mean similarity within a label vs across labels.

    A large positive gap means the vector separates that label well.
    """
    same: list[float] = []
    diff: list[float] = []
    valid = [lab for lab in labels if not _is_missing(lab)]
    if not valid:
        return {}

    label_arr = np.array([("" if _is_missing(lab) else str(lab)) for lab in labels])
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if not label_arr[i] or not label_arr[j]:
                continue
            value = float(similarity[i, j])
            (same if label_arr[i] == label_arr[j] else diff).append(value)

    if not same or not diff:
        return {"n_same": len(same), "n_diff": len(diff)}

    return {
        "n_same": len(same),
        "n_diff": len(diff),
        "same_mean": float(np.mean(same)),
        "diff_mean": float(np.mean(diff)),
        "gap": float(np.mean(same) - np.mean(diff)),
    }


def nearest_neighbour_stats(similarity: np.ndarray, ids: list[str],
                            group_of: list[str] | None = None) -> dict:
    """How similar each file's closest neighbour is, and whether it matches group."""
    if similarity.shape[0] < 2:
        return {}

    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    best = sim.argmax(axis=1)
    best_sim = sim[np.arange(len(ids)), best]

    out = {
        "nn_sim_mean": float(best_sim.mean()),
        "nn_sim_median": float(np.median(best_sim)),
        "nn_sim_min": float(best_sim.min()),
        "nn_sim_max": float(best_sim.max()),
    }

    if group_of is not None:
        same = sum(1 for i in range(len(ids)) if group_of[i] == group_of[best[i]])
        out["top1_same_group_fraction"] = same / len(ids)
        out["top1_group_distribution"] = {
            g: sum(1 for b in best if group_of[b] == g)
            for g in sorted({g for g in group_of if g})
        }

    out["examples"] = [
        {"file_id": ids[i], "nearest": ids[best[i]], "cosine": round(float(best_sim[i]), 4)}
        for i in np.argsort(-best_sim)[:10]
    ]
    return out


def cluster_analysis(similarity: np.ndarray, ids: list[str],
                     labels: list[str | None] | None = None,
                     thresholds: tuple[float, ...] = (0.4, 0.6, 0.8)) -> dict:
    """Average-linkage clustering on ``1 - cosine`` at several cut levels.

    Reports how many clusters appear and, when labels are supplied, how
    pure they are with respect to those labels.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    if similarity.shape[0] < 3:
        return {}

    dist = 1.0 - similarity
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, None)
    linkage_matrix = linkage(squareform(dist, checks=False), "average")

    out: dict = {}
    for threshold in thresholds:
        clusters = fcluster(linkage_matrix, threshold, "distance")
        sizes = np.bincount(clusters)[1:]
        entry: dict = {
            "n_clusters": len(sizes),
            "largest_cluster": int(sizes.max()) if sizes.size else 0,
            "size_top10": sorted((int(s) for s in sizes), reverse=True)[:10],
        }
        if labels is not None:
            label_arr = np.array([("" if _is_missing(lab) else str(lab)) for lab in labels])
            purity, purity_lab = 0.0, ""
            for cluster_id in np.unique(clusters):
                members = label_arr[clusters == cluster_id]
                members = members[members != ""]
                if members.size:
                    counts = np.bincount(np.unique(members, return_inverse=True)[1])
                    if counts.max() > purity:
                        purity = float(counts.max() / members.size)
                        purity_lab = str(np.unique(members)[counts.argmax()])
            entry["purity"] = round(purity, 4)
            entry["dominant_label"] = purity_lab
        out[f"threshold_{threshold}"] = entry
    return out


def duration_effect(similarity: np.ndarray, durations: np.ndarray) -> dict:
    """Does a longer file make it harder to find a similar one?

    ECAPA embeds a whole utterance at once, so long recordings dilute the
    speaker signal. This measures the size of that effect.
    """
    if similarity.shape[0] < 2:
        return {}

    sim = similarity.copy()
    np.fill_diagonal(sim, -np.inf)
    best_sim = sim.max(axis=1)
    if durations.std() == 0:
        return {}
    return {
        "corr_duration_vs_nn_cos": float(np.corrcoef(durations, best_sim)[0, 1]),
        "corr_duration_vs_mean_cos": float(np.corrcoef(
            durations, (similarity.sum(axis=1) - 1) / (len(durations) - 1))[0, 1]),
    }


# ---------------------------------------------------------------------------
# Corpus description
# ---------------------------------------------------------------------------
def describe_corpus(voice: VoiceMatrix) -> dict:
    """Describe the corpus: source mix, durations, label coverage."""
    table = corpus.build_corpus_table(voice.ids)
    durations = voice.durations

    by_source = table["source"].value_counts().to_dict()
    duration_by_source = (
        table.assign(duration_s=durations).groupby("source")["duration_s"]
        .agg(["count", "mean", "median", "min", "max"]).round(1)
    )

    speakers = [s for s in table["speaker"].dropna().tolist() if s]
    topics = [t for t in table["topic"].dropna().tolist() if t]
    speaker_counts = pd_series_counts(speakers)
    topic_counts = pd_series_counts(topics)

    # Multi-file speakers are what makes a "same voice, different topic" test
    # possible at all.
    multi_file_speakers = {k: v for k, v in speaker_counts.items() if v > 1}

    return {
        "n_files": len(voice),
        "by_source": by_source,
        "duration": {
            "total_hours": float(durations.sum() / 3600),
            "mean_s": float(durations.mean()),
            "median_s": float(np.median(durations)),
            "min_s": float(durations.min()),
            "max_s": float(durations.max()),
            "p05_s": float(np.percentile(durations, 5)),
            "p95_s": float(np.percentile(durations, 95)),
        },
        "duration_by_source": duration_by_source.to_dict("index"),
        "labels": {
            "n_with_speaker": int(table["has_speaker"].sum()),
            "n_with_topic": int(table["has_topic"].sum()),
            "n_with_neither": int((~table["has_speaker"] & ~table["has_topic"]).sum()),
            "n_distinct_speakers": len(speaker_counts),
            "n_speakers_multi_file": len(multi_file_speakers),
            "multi_file_speakers": multi_file_speakers,
            "n_distinct_topics": len(topic_counts),
            "topics_with_4plus_files": sum(1 for v in topic_counts.values() if v >= 4),
            "topics_with_2plus_files": sum(1 for v in topic_counts.values() if v >= 2),
        },
    }


def pd_series_counts(values: list[str]) -> dict:
    """Count occurrences, most frequent first."""
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def compare_to_targets(corpus_desc: dict) -> dict:
    """Compare the corpus against the targets written in report.md section 2.4c.

    Those targets ask for at least 40 speakers (15-20 files each), 15-20
    topics (4-5 files each), and a 25-30 file query split. The corpus is
    measured against them here so the gap is explicit.
    """
    labels = corpus_desc["labels"]
    topics = labels["n_distinct_topics"]
    # Every entry carries the same three keys — wanted / actual / met — so a
    # consumer never has to guess which name a given target used for its
    # measured value.
    return {
        "target_files": {"wanted": 500, "actual": corpus_desc["n_files"],
                         "met": corpus_desc["n_files"] >= 500},
        "target_speakers": {"wanted": 40, "actual": labels["n_distinct_speakers"],
                            "met": labels["n_distinct_speakers"] >= 40},
        "target_speakers_multi_file": {
            "wanted": "≥40 người nói × 15-20 file",
            "actual": labels["n_speakers_multi_file"],
            "met": labels["n_speakers_multi_file"] >= 40,
        },
        "target_topics": {"wanted": "15-20 chủ đề", "actual": topics,
                          "met": 15 <= topics <= 25},
        "target_topics_size": {
            "wanted": "≥4-5 file mỗi chủ đề",
            "actual": labels["topics_with_4plus_files"],
            "met": labels["topics_with_4plus_files"] >= 15,
        },
    }


# ---------------------------------------------------------------------------
# Content-side analysis
# ---------------------------------------------------------------------------
def analyse_content(content: ContentMatrix) -> dict:
    """Similarity structure of the semantic embeddings.

    Zero vectors (files with no speech) carry no information and would
    otherwise dominate the statistics, so they are excluded and counted
    separately.
    """
    if len(content) == 0:
        return {"n": 0}

    norms = np.linalg.norm(content.semantic, axis=1)
    usable = norms > 0.01
    out: dict = {
        "n": len(content),
        "n_zero_vector": int((~usable).sum()),
        "n_usable": int(usable.sum()),
    }
    if usable.sum() < 2:
        return out

    vec = content.semantic[usable]
    sim = cosine_matrix(vec)
    out["pairwise"] = similarity_stats(sim)

    # Does a longer transcript dilute the mean-pooled vector the same way a
    # longer recording dilutes the ECAPA vector?
    chunks = content.n_chunks[usable]
    durations = content.durations[usable]
    sim2 = sim.copy()
    np.fill_diagonal(sim2, -np.inf)
    best = sim2.max(axis=1)
    if chunks.std() > 0:
        out["corr_nchunks_vs_nn_cos"] = float(np.corrcoef(chunks, best)[0, 1])
    if durations.std() > 0:
        out["corr_duration_vs_nn_cos"] = float(np.corrcoef(durations, best)[0, 1])

    out["n_chunks"] = {
        "mean": float(chunks.mean()),
        "median": float(np.median(chunks)),
        "min": int(chunks.min()),
        "max": int(chunks.max()),
    }
    out["transcript_chars"] = {
        "mean": float(content.transcript_chars[usable].mean()),
        "median": float(np.median(content.transcript_chars[usable])),
    }
    return out


def tfidf_baseline(transcripts: list[str], ids: list[str]) -> dict:
    """Build a TF-IDF matrix over the transcripts — the baseline the report
    wants to compare the semantic embedding against.

    Returns the pairwise-cosine statistics so the two can be compared on the
    same footing.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    usable = [(fid, text) for fid, text in zip(ids, transcripts) if text.strip()]
    if len(usable) < 2:
        return {"n": 0}

    texts = [t for _, t in usable]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    matrix = vectorizer.fit_transform(texts)
    dense = matrix.toarray()
    sim = cosine_matrix(dense)
    return {
        "n": len(usable),
        "vocabulary": len(vectorizer.vocabulary_),
        "pairwise": similarity_stats(sim),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run(skip_content: bool = False) -> dict:
    """Run the full analysis and return the metrics dictionary."""
    config.ensure_dirs()

    logger.info("loading voice artifacts from %s ...", config.FEATURES_DIR)
    voice = load_voice_matrix()
    logger.info("loaded %d voice artifact(s)", len(voice))

    content = ContentMatrix([], np.zeros((0, config.SEMANTIC_DIM)), np.zeros(0),
                            np.zeros(0), np.zeros(0))
    if not skip_content:
        content = load_content_matrix()
        logger.info("loaded %d content artifact(s)", len(content))

    metrics: dict = {
        "integrity": audit_integrity(voice, content),
        "dimensionality": {
            "mfcc_stats": pca_analysis(voice.mfcc_stats, "mfcc_stats"),
            "mfcc_std": pca_analysis(voice.mfcc_std, "mfcc_std"),
            "ecapa": pca_analysis(voice.ecapa, "ecapa_embedding"),
            "prosody": pca_analysis(voice.prosody, "prosody"),
        },
        "mfcc_structure": mfcc_block_analysis(voice.mfcc_stats),
    }

    # --- similarity structure -------------------------------------------
    mfcc_clean = mfcc_clean_variant(voice.mfcc_stats)
    sim_mfcc_raw = cosine_matrix(voice.mfcc_stats)
    sim_mfcc_clean = cosine_matrix(mfcc_clean)
    sim_ecapa = cosine_matrix(voice.ecapa)

    metrics["similarity"] = {
        "mfcc_raw_78d": similarity_stats(sim_mfcc_raw),
        "mfcc_clean_26d_zscore": similarity_stats(sim_mfcc_clean),
        "ecapa_192d": similarity_stats(sim_ecapa),
    }

    # --- label separation ----------------------------------------------
    table = corpus.build_corpus_table(voice.ids)
    courses = table["course"].tolist()
    topics = table["topic"].tolist()
    sources = table["source"].tolist()

    metrics["separation"] = {
        "ecapa_by_course": label_gap(sim_ecapa, courses),
        "ecapa_by_topic": label_gap(sim_ecapa, topics),
        "ecapa_by_source": label_gap(sim_ecapa, sources),
        "mfcc_clean_by_course": label_gap(sim_mfcc_clean, courses),
        "mfcc_raw_by_course": label_gap(sim_mfcc_raw, courses),
    }

    # --- nearest neighbours & clustering -------------------------------
    metrics["nearest_neighbour"] = {
        "ecapa": nearest_neighbour_stats(sim_ecapa, voice.ids, sources),
        "mfcc_clean": nearest_neighbour_stats(sim_mfcc_clean, voice.ids, sources),
    }
    metrics["clustering"] = {
        "ecapa": cluster_analysis(sim_ecapa, voice.ids, courses),
        "mfcc_clean": cluster_analysis(sim_mfcc_clean, voice.ids, courses),
    }
    metrics["duration_effect"] = {
        "ecapa": duration_effect(sim_ecapa, voice.durations),
        "mfcc_clean": duration_effect(sim_mfcc_clean, voice.durations),
    }

    # --- corpus ---------------------------------------------------------
    corpus_desc = describe_corpus(voice)
    metrics["corpus"] = corpus_desc
    metrics["targets_comparison"] = compare_to_targets(corpus_desc)

    # --- content --------------------------------------------------------
    if not skip_content and len(content):
        metrics["content"] = analyse_content(content)
        metrics["content"]["manifest_rows"] = _manifest_row_count()

    return metrics


def _manifest_row_count() -> int:
    """Number of data rows in the manifest CSV (excluding the header)."""
    path = config.FEATURES_DIR / "manifest.csv"
    if not path.exists():
        return 0
    import pandas as pd

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        return len(df)
    except Exception:
        return -1


def write_outputs(metrics: dict, corpus_table: object | None = None) -> dict:
    """Write metrics.json and the corpus table CSV."""
    config.ensure_dirs()
    metrics_path = config.ANALYSIS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("metrics written: %s", metrics_path)

    written = {"metrics": str(metrics_path)}

    if corpus_table is not None:
        table_path = config.ANALYSIS_DIR / "corpus_table.csv"
        corpus_table.to_csv(table_path, index=False, encoding="utf-8-sig")
        logger.info("corpus table written: %s", table_path)
        written["corpus_table"] = str(table_path)

    return written
