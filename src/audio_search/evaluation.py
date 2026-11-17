"""Retrieval evaluation (requirement 4c).

Three experiments, all written to artifacts/evaluation/:

E1  Clip retrieval — `n_clips` speech files are sampled (stratified by
    source); a `clip_s`-second excerpt is cut at a random offset and used as
    the query, in two conditions:
      clean      the excerpt as recorded
      degraded   telephone band-pass 300-3400 Hz + white noise at 15 dB SNR
                 (simulates the same talk re-recorded through another mic)
    The source file is the correct answer for content AND voice. Reported:
    Hit@1, Hit@3, MRR for semantic vs BM25 (content) and ECAPA vs MFCC (voice).
    The same queries are then run with the source file *excluded* — the talk is
    now "not in the database" — which yields the score distributions used to
    calibrate the known/unknown thresholds (ROC AUC + Youden-optimal cut).

E2  Speaker leave-one-out — on the stored full-file vectors: for every file
    whose speaker group has ≥2 files (same filename speaker, or same Khan
    Academy course = same narrator), remove the file itself and check whether
    the nearest neighbour belongs to the same group. ECAPA vs MFCC.

E3  Query set — the recordings in samples/queries/ (queries.json gives the
    expected answer): TTS speakers reading known / paraphrased / new content,
    and real-speech clips. Full end-to-end search, as in the demo.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
import soundfile as sf

from . import config, database, search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def degrade(clip: np.ndarray, sr: int = config.SAMPLE_RATE, snr_db: float = 15.0,
            seed: int = 0) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, clip).astype(np.float32)
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(y ** 2)) or 1e-9
    y = y + rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32)


def rank_of(scores: np.ndarray, mask: np.ndarray, target: int) -> int:
    """1-based rank of `target` among masked candidates (inf if not a candidate)."""
    if not mask[target]:
        return 10 ** 9
    s = np.where(mask, scores, -np.inf)
    return int(np.sum(s > s[target])) + 1


def hit_stats(ranks: list[int]) -> dict:
    r = np.asarray(ranks, dtype=float)
    return {"n": len(r), "hit@1": float(np.mean(r <= 1)), "hit@3": float(np.mean(r <= 3)),
            "mrr": float(np.mean(1.0 / r))}


def calibrate(pos: list[float], neg: list[float]) -> dict:
    """ROC AUC and the Youden-optimal threshold separating pos from neg scores."""
    from sklearn.metrics import roc_auc_score, roc_curve

    if not pos or not neg:
        return {"auc": None, "threshold": None, "n_pos": len(pos), "n_neg": len(neg)}
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg]
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    t = float(thr[j])
    return {"auc": float(roc_auc_score(y, s)), "threshold": t,
            "tpr": float(tpr[j]), "fpr": float(fpr[j]),
            "accuracy": float(np.mean((s >= t) == (y == 1))),
            "n_pos": len(pos), "n_neg": len(neg),
            "pos_mean": float(np.mean(pos)), "neg_mean": float(np.mean(neg))}


def speaker_groups(index: database.VectorIndex) -> list[str | None]:
    """Weak speaker label per file: filename speaker, else Khan Academy course."""
    out = []
    for fid in index.ids:
        m = index.meta[fid]
        if m["multi_speaker"] or not m["is_speech"]:
            out.append(None)
        elif m["speaker"]:
            out.append("spk:" + m["speaker"].lower())
        elif m["course"]:
            out.append("course:" + m["course"].lower())
        else:
            out.append(None)
    return out


def sample_files(index: database.VectorIndex, n: int, clip_s: float, rng) -> list[int]:
    """Stratified sample of speech files long enough for a clip."""
    eligible = [i for i, fid in enumerate(index.ids)
                if index.voice_ok[i] and index.meta[fid]["duration_s"] >= clip_s + 20]
    by_src: dict[str, list[int]] = {}
    for i in eligible:
        by_src.setdefault(index.meta[index.ids[i]]["source"], []).append(i)
    picks: list[int] = []
    for src, rows in sorted(by_src.items()):
        k = max(1, round(n * len(rows) / len(eligible)))
        picks += list(rng.choice(rows, size=min(k, len(rows)), replace=False))
    rng.shuffle(picks)
    return [int(i) for i in picks[:n]]


# ---------------------------------------------------------------------------
# E1 — clip retrieval
# ---------------------------------------------------------------------------
def e1_clips(index: database.VectorIndex, n_clips: int, clip_s: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = speaker_groups(index)
    group_size = pd.Series([g for g in groups if g]).value_counts().to_dict()
    rows = []
    picks = sample_files(index, n_clips, clip_s, rng)
    for qi, i in enumerate(picks, 1):
        fid = index.ids[i]
        wave, sr = sf.read(str(config.WAV_DIR / f"{fid}.wav"), dtype="float32")
        dur = wave.shape[0] / sr
        start = float(rng.uniform(0.1 * dur, max(0.1 * dur, dur - clip_s - 0.05 * dur)))
        clip = wave[int(start * sr): int((start + clip_s) * sr)]
        for cond, w in (("clean", clip), ("degraded", degrade(clip, seed=qi))):
            t0 = time.perf_counter()
            q = search.extract_query_features("", index, wave=w, with_prosody=False)
            sem = index.semantic @ q.semantic
            eca = index.ecapa @ q.ecapa
            mfc = index.mfcc_clean @ q.mfcc_clean
            kw = [f for f, _ in index.keyword_search(q.transcript, k=50)]
            excl = index.voice_ok.copy()
            excl[i] = False
            g = groups[i]
            rows.append({
                "file_id": fid, "source": index.meta[fid]["source"], "condition": cond,
                "start_s": round(start, 1), "speaker_group": g,
                "group_size": int(group_size.get(g, 1)) if g else 1,
                "rank_semantic": rank_of(sem, index.voice_ok, i),
                "rank_bm25": kw.index(fid) + 1 if fid in kw else 10 ** 9,
                "rank_ecapa": rank_of(eca, index.voice_ok, i),
                "rank_mfcc": rank_of(mfc, index.voice_ok, i),
                "top1_semantic_in": float(np.max(np.where(index.voice_ok, sem, -np.inf))),
                "top1_semantic_out": float(np.max(np.where(excl, sem, -np.inf))),
                "top1_ecapa_in": float(np.max(np.where(index.voice_ok, eca, -np.inf))),
                "top1_ecapa_out": float(np.max(np.where(excl, eca, -np.inf))),
                "top1_ecapa_out_same_group": (
                    bool(g) and groups[int(np.argmax(np.where(excl, eca, -np.inf)))] == g),
                "transcript_chars": len(q.transcript),
                "seconds": round(time.perf_counter() - t0, 2),
            })
        logger.info("[E1 %d/%d] %s  sem=%d/%d ecapa=%d/%d", qi, len(picks), fid[:60],
                    rows[-2]["rank_semantic"], rows[-1]["rank_semantic"],
                    rows[-2]["rank_ecapa"], rows[-1]["rank_ecapa"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E2 — speaker leave-one-out on full-file vectors
# ---------------------------------------------------------------------------
def e2_speaker_loo(index: database.VectorIndex) -> dict:
    groups = np.array([g or "" for g in speaker_groups(index)], dtype=object)
    sizes = pd.Series(groups[groups != ""]).value_counts()
    multi = set(sizes[sizes >= 2].index)
    out = {}
    for name, mat in (("ecapa", index.ecapa), ("mfcc", index.mfcc_clean)):
        sim = mat @ mat.T
        p1, p3, ap = [], [], []
        for i, g in enumerate(groups):
            if g not in multi:
                continue
            s = np.where(index.voice_ok, sim[i], -np.inf)
            s[i] = -np.inf
            order = np.argsort(-s)
            rel = groups[order] == g
            n_rel = int(sizes[g]) - 1
            p1.append(float(rel[0]))
            p3.append(float(rel[:3].mean()))
            hits = np.cumsum(rel[: len(index)])
            prec = hits / np.arange(1, len(hits) + 1)
            ap.append(float((prec * rel[: len(index)]).sum() / n_rel))
        out[name] = {"n_queries": len(p1), "precision@1": float(np.mean(p1)),
                     "precision@3": float(np.mean(p3)), "mAP": float(np.mean(ap))}
    out["n_groups"] = len(multi)
    out["groups"] = {g: int(sizes[g]) for g in sorted(multi)}
    return out


# ---------------------------------------------------------------------------
# E3 — sample query set
# ---------------------------------------------------------------------------
def e3_queries(index: database.VectorIndex) -> list[dict]:
    qfile = config.QUERY_DIR / "queries.json"
    if not qfile.exists():
        logger.warning("%s missing — run scripts/make_queries.py", qfile)
        return []
    out = []
    for spec in json.loads(qfile.read_text(encoding="utf-8")):
        path = config.QUERY_DIR / spec["file"]
        res = search.search_file(path, index)
        exp_c, exp_v = spec.get("expected_content"), spec.get("expected_speaker_file")
        c_ids = [h.file_id for h in res.content]
        v_ids = [h.file_id for h in res.voice]
        out.append({
            **spec,
            "transcript": res.query.transcript,
            "content_top3": [(h.file_id, round(h.score, 4)) for h in res.content],
            "voice_top3": [(h.file_id, round(h.score, 4)) for h in res.voice],
            "content_known_pred": res.content_known,
            "voice_known_pred": res.voice_known,
            "content_correct": (exp_c in c_ids[:1]) if exp_c else (not res.content_known),
            "content_in_top3": (exp_c in c_ids) if exp_c else None,
            "voice_correct": (exp_v in v_ids[:1]) if exp_v else (not res.voice_known),
            "timings": {k: round(v, 2) for k, v in res.query.timings.items()},
        })
        logger.info("[E3] %s content_ok=%s voice_ok=%s", spec["file"], out[-1]["content_correct"],
                    out[-1]["voice_correct"])
    return out


# ---------------------------------------------------------------------------
# Figures & report
# ---------------------------------------------------------------------------
def _figures(df: pd.DataFrame, cal: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = config.EVAL_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (key, title) in zip(axes, (("semantic", "Nội dung — cosine top-1 (semantic)"),
                                       ("ecapa", "Giọng nói — cosine top-1 (ECAPA)")), strict=True):
        inn = df[f"top1_{key}_in"]
        out = df[f"top1_{key}_out"]
        if key == "ecapa":
            out = df.loc[df["group_size"] <= 1, "top1_ecapa_out"]
        bins = np.linspace(min(inn.min(), out.min()) - 0.02, 1.0, 30)
        ax.hist(inn, bins=bins, alpha=0.65, label="có trong CSDL", color="#2a6fdb")
        ax.hist(out, bins=bins, alpha=0.65, label="chưa có trong CSDL", color="#e8833a")
        t = cal[f"{key}_known"]["threshold"]
        if t is not None:
            ax.axvline(t, color="k", ls="--", lw=1, label=f"ngưỡng {t:.2f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("cosine")
        ax.legend(fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = fig_dir / "known_vs_unknown_scores.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(8, 3.6))
    methods = [("rank_semantic", "Semantic"), ("rank_bm25", "BM25"),
               ("rank_ecapa", "ECAPA"), ("rank_mfcc", "MFCC")]
    x = np.arange(len(methods))
    for k, (cond, color) in enumerate((("clean", "#2a6fdb"), ("degraded", "#e8833a"))):
        sub = df[df["condition"] == cond]
        vals = [float(np.mean(sub[m] <= 1)) for m, _ in methods]
        bars = ax.bar(x + (k - 0.5) * 0.38, vals, 0.38, label=cond, color=color)
        ax.bar_label(bars, labels=[f"{v:.0%}" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(x, [m for _, m in methods])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Hit@1")
    ax.set_title("E1 — tìm lại đúng file nguồn từ đoạn cắt", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = fig_dir / "hit_at_1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(str(p))
    return paths


def _summary_md(results: dict) -> str:
    e1, e2, cal = results["e1"], results["e2"], results["calibration"]
    L = ["# Kết quả đánh giá hệ thống tìm kiếm", "",
         f"Sinh tự động bởi `python -m audio_search evaluate` lúc {results['generated_at']}.", "",
         (f"## E1. Tìm lại file nguồn từ đoạn cắt {results['params']['clip_s']:.0f} s "
          f"({e1['n_files']} file, 2 điều kiện)"), "",
         "| Điều kiện | Phương pháp | Hit@1 | Hit@3 | MRR |", "|---|---|---|---|---|"]
    for cond in ("clean", "degraded"):
        for m in ("semantic", "bm25", "ecapa", "mfcc"):
            s = e1[cond][m]
            L.append(f"| {cond} | {m} | {s['hit@1']:.1%} | {s['hit@3']:.1%} | {s['mrr']:.3f} |")
    L += ["", "## Ngưỡng \"có / chưa có trong CSDL\" (hiệu chỉnh từ E1, ROC)", "",
          "| Bộ đặc trưng | AUC | Ngưỡng | TPR | FPR | Độ chính xác | TB có | TB chưa có |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ("semantic_known", "ecapa_known"):
        c = cal[k]
        if c["auc"] is None:
            L.append(f"| {k} | n/a | | | | | | |")
            continue
        L.append(f"| {k} | {c['auc']:.3f} | {c['threshold']:.3f} | {c['tpr']:.1%} | {c['fpr']:.1%} | "
                 f"{c['accuracy']:.1%} | {c['pos_mean']:.3f} | {c['neg_mean']:.3f} |")
    L += ["", f"## E2. Nhận diện người nói (leave-one-out, {e2['n_groups']} nhóm người nói ≥ 2 file)", "",
          "| Đặc trưng | Số truy vấn | P@1 | P@3 | mAP |", "|---|---|---|---|---|"]
    for m in ("ecapa", "mfcc"):
        s = e2[m]
        L.append(f"| {m} | {s['n_queries']} | {s['precision@1']:.1%} | {s['precision@3']:.1%} | {s['mAP']:.3f} |")
    L += ["", "## E3. Bộ truy vấn mẫu (samples/queries)", "",
          "| File | Loại | Top-1 nội dung (cos) | Đúng? | Top-1 giọng (cos) | Đúng? |",
          "|---|---|---|---|---|---|"]
    for q in results["e3"]:
        c = q["content_top3"][0] if q["content_top3"] else ("—", 0)
        v = q["voice_top3"][0] if q["voice_top3"] else ("—", 0)
        L.append(f"| {q['file']} | {q['kind']} | {c[0][:45]} ({c[1]:.3f}) | "
                 f"{'✅' if q['content_correct'] else '❌'} | {v[0][:40]} ({v[1]:.3f}) | "
                 f"{'✅' if q['voice_correct'] else '❌'} |")
    L += ["", ("Với truy vấn có người nói/nội dung *chưa có*, \"đúng\" nghĩa là hệ thống báo "
               "điểm top-1 dưới ngưỡng (chưa có trong CSDL).")]
    return "\n".join(L) + "\n"


def run(n_clips: int = 60, clip_s: float = 60.0, seed: int = 7) -> dict:
    config.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    index = database.load_index()
    logger.info("E1: %d clips × 2 conditions ...", n_clips)
    df = e1_clips(index, n_clips, clip_s, seed)
    df.to_csv(config.EVAL_DIR / "e1_clips.csv", index=False, encoding="utf-8-sig")

    e1 = {"n_files": int(df["file_id"].nunique())}
    for cond in ("clean", "degraded"):
        sub = df[df["condition"] == cond]
        e1[cond] = {m: hit_stats(sub[f"rank_{m}"].tolist()) for m in ("semantic", "bm25", "ecapa", "mfcc")}

    # Known = the talk/speaker is in the DB; unknown = source excluded.
    # For voice, "unknown" only counts when no other file of that speaker remains;
    # "known" also includes held-out clips whose speaker has other files.
    voice_pos = df["top1_ecapa_in"].tolist() + df.loc[df["group_size"] >= 2, "top1_ecapa_out"].tolist()
    voice_neg = df.loc[df["group_size"] <= 1, "top1_ecapa_out"].tolist()
    cal = {
        "semantic_known": calibrate(df["top1_semantic_in"].tolist(), df["top1_semantic_out"].tolist()),
        "ecapa_known": calibrate(voice_pos, voice_neg),
    }
    thresholds = dict(search.DEFAULT_THRESHOLDS)
    if cal["semantic_known"]["threshold"] is not None:
        thresholds["content_known"] = round(cal["semantic_known"]["threshold"], 4)
    if cal["ecapa_known"]["threshold"] is not None:
        thresholds["voice_known"] = round(cal["ecapa_known"]["threshold"], 4)
    (config.EVAL_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    logger.info("E2: speaker leave-one-out ...")
    e2 = e2_speaker_loo(index)
    logger.info("E3: sample queries ...")
    e3 = e3_queries(index)          # uses the freshly calibrated thresholds

    results = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": {"n_clips": n_clips, "clip_s": clip_s, "seed": seed},
        "e1": e1, "calibration": cal, "thresholds": thresholds, "e2": e2, "e3": e3,
        "figures": _figures(df, cal),
    }
    (config.EVAL_DIR / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (config.EVAL_DIR / "summary.md").write_text(_summary_md(results), encoding="utf-8")
    logger.info("evaluation written to %s", config.EVAL_DIR)
    return results
