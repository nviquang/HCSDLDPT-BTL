"""Command implementations for the audio_search CLI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from . import config, feature_store, preprocess

logger = logging.getLogger(__name__)


def _select_files(data_dir: Path, stems: list[str] | None) -> list[Path]:
    """Filter discovered audio files to a subset by stem, if requested.

    Stems are matched whole — never split on whitespace — so filenames that
    contain spaces survive the round-trip from the CLI. Unknown stems are
    reported rather than silently dropped.
    """
    files = preprocess.discover_audio_files(data_dir)
    if not stems:
        return files
    wanted = set(stems)
    selected = [f for f in files if f.stem in wanted]
    missing = wanted - {f.stem for f in selected}
    if missing:
        logger.warning("--files matched no audio file for %d stem(s): %s",
                       len(missing), ", ".join(sorted(missing)))
    return selected


def _start(step: str, files: list[Path]) -> float:
    """Log the start of a batch step and return a perf_counter timestamp."""
    import time

    logger.info("=== %s: %d file(s) to process ===", step, len(files))
    return time.perf_counter()


def _finish(step: str, ok: int, skipped: int, failed: int, total: int, t0: float) -> None:
    """Log the per-step summary with elapsed time and throughput."""
    import time

    elapsed = time.perf_counter() - t0
    rate = total / elapsed if elapsed > 0 else 0.0
    logger.info(
        "=== %s done: %d ok, %d skipped, %d failed (of %d) in %.1fs (%.2f file/s) ===",
        step, ok, skipped, failed, total, elapsed, rate,
    )


def run_preprocess(stems: list[str] | None = None) -> int:
    files = _select_files(config.DATA_DIR, stems)
    t0 = _start("preprocess", files)
    ok = skipped = failed = 0
    for i, src in enumerate(files, 1):
        try:
            pp = preprocess.preprocess_file(src)
            if not preprocess.validate_duration(pp.duration_s):
                logger.warning("[%d/%d] skip %s (duration %.1fs outside range)",
                               i, len(files), src.name, pp.duration_s)
                skipped += 1
            else:
                logger.info("[%d/%d] ok   %s (%.1fs)", i, len(files), src.name, pp.duration_s)
                ok += 1
        except Exception as exc:
            logger.error("[%d/%d] fail %s: %s", i, len(files), src.name, exc)
            failed += 1
    _finish("preprocess", ok, skipped, failed, len(files), t0)
    return 1 if failed else 0


def run_voice(stems: list[str] | None = None) -> int:
    from . import voice_features

    files = _select_files(config.DATA_DIR, stems)
    t0 = _start("voice", files)
    ok = skipped = failed = 0
    for i, src in enumerate(files, 1):
        file_id = src.stem
        voice_path = config.FEATURES_DIR / f"voice-{file_id}.json"
        if voice_path.exists():
            logger.info("[%d/%d] skip %s (artifact exists)", i, len(files), src.name)
            skipped += 1
            continue
        try:
            pp = preprocess.preprocess_file(src)
            if not preprocess.validate_duration(pp.duration_s):
                logger.warning("[%d/%d] skip %s (duration %.1fs outside range)",
                               i, len(files), src.name, pp.duration_s)
                skipped += 1
                continue
            feats = voice_features.compute_voice_features(pp.wave, pp.sr)
            feature_store.write_voice_artifact(file_id, feats, filename=src.name, duration_s=pp.duration_s)
            logger.info("[%d/%d] ok   %s (%.1fs)", i, len(files), src.name, pp.duration_s)
            ok += 1
        except Exception as exc:
            logger.error("[%d/%d] fail %s: %s", i, len(files), src.name, exc)
            failed += 1
    _finish("voice", ok, skipped, failed, len(files), t0)
    return 1 if failed else 0


def run_content(stems: list[str] | None = None) -> int:
    from . import content_features

    files = _select_files(config.DATA_DIR, stems)
    t0 = _start("content", files)
    ok = skipped = failed = 0
    for i, src in enumerate(files, 1):
        file_id = src.stem
        content_path = config.FEATURES_DIR / f"content-{file_id}.json"
        if content_path.exists():
            logger.info("[%d/%d] skip %s (artifact exists)", i, len(files), src.name)
            skipped += 1
            continue
        try:
            pp = preprocess.preprocess_file(src)
            if not preprocess.validate_duration(pp.duration_s):
                logger.warning("[%d/%d] skip %s (duration %.1fs outside range)",
                               i, len(files), src.name, pp.duration_s)
                skipped += 1
                continue
            feats = content_features.compute_content_features(str(pp.path))
            feature_store.write_content_artifact(file_id, feats, filename=src.name, duration_s=pp.duration_s)
            logger.info("[%d/%d] ok   %s (%d chars, %d chunks)",
                        i, len(files), src.name,
                        len(feats["transcript"]), feats["n_chunks"])
            ok += 1
        except Exception as exc:
            logger.error("[%d/%d] fail %s: %s", i, len(files), src.name, exc)
            failed += 1
    _finish("content", ok, skipped, failed, len(files), t0)
    return 1 if failed else 0


def run_manifest() -> int:
    try:
        feature_store.build_manifest()
        return 0
    except Exception as exc:
        logger.error("manifest failed: %s", exc)
        return 1


def run_verify() -> int:
    from . import verify

    try:
        return verify.run()
    except Exception as exc:
        logger.error("verify failed: %s", exc)
        return 1


def run_reembed() -> int:
    try:
        res = feature_store.reembed_content_artifacts()
        return 0 if res["ok"] == res["total"] else 1
    except Exception as exc:
        logger.error("reembed failed: %s", exc)
        return 1


def run_build_db() -> int:
    from . import database

    try:
        database.build()
        return 0
    except Exception as exc:
        logger.error("build-db failed: %s", exc)
        return 1


def run_search(path: str, k: int, exclude: list[str] | None, as_json: bool) -> int:
    import json
    import sys

    from . import database, search

    try:
        index = database.load_index()
        res = search.search_file(path, index, k=k, exclude=set(exclude or []))
    except Exception as exc:
        logger.error("search failed: %s", exc)
        return 1
    if as_json:
        out = {
            "query": path, "transcript": res.query.transcript, "timings": res.query.timings,
            "content": [{"rank": h.rank, "file_id": h.file_id, "cosine": h.score} for h in res.content],
            "voice": [{"rank": h.rank, "file_id": h.file_id, "cosine": h.score} for h in res.voice],
            "content_known": res.content_known, "voice_known": res.voice_known,
        }
        text = json.dumps(out, ensure_ascii=False, indent=2)
    else:
        text = search.format_report(res)
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.flush()
    return 0


def run_evaluate(n_clips: int, clip_s: float, seed: int) -> int:
    from . import evaluation

    try:
        evaluation.run(n_clips=n_clips, clip_s=clip_s, seed=seed)
        return 0
    except Exception:
        logger.exception("evaluate failed")
        return 1


def run_analyze(skip_content: bool = False, no_figures: bool = False) -> int:
    """Audit the extracted features and describe the corpus."""
    from . import analysis, corpus

    try:
        metrics = analysis.run(skip_content=skip_content)
        table = corpus.build_corpus_table()
        analysis.write_outputs(metrics, corpus_table=table)
        if not no_figures:
            from . import figures

            figures.render_all(metrics, table)
        logger.info("analysis complete")
        return 0
    except Exception as exc:
        logger.error("analyze failed: %s", exc)
        return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (also used by pyproject [project.scripts] audio-search)."""
    import argparse
    import sys

    from . import config

    config.ensure_dirs()

    # Windows consoles default to cp1252; file ids contain non-ASCII characters.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOGS_DIR / "extract.log", encoding="utf-8"),
        ],
    )
    # SpeechBrain lowers its own loggers to DEBUG on import; gate at the handlers.
    for h in logging.getLogger().handlers:
        h.setLevel(logging.INFO)

    for noisy in ("speechbrain", "faster_whisper", "sentence_transformers", "numba", "urllib3",
                  "httpx", "huggingface_hub", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(prog="audio_search")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("preprocess", "Normalize audio files to 16 kHz mono WAV"),
        ("voice", "Extract voice features (MFCC + ECAPA)"),
        ("content", "Extract content features (Whisper ASR + semantic)"),
        ("all", "Run preprocess -> voice -> content -> manifest"),
        ("manifest", "Aggregate per-file artifacts into one CSV"),
        ("verify", "Smoke-test the extracted artifacts"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--files", nargs="*", default=None, help="restrict to these file stems")
        sp.add_argument("--device", default=None, help="cpu | cuda")
        sp.add_argument("--min-duration", type=float, default=None)
        sp.add_argument("--max-duration", type=float, default=None)

    # `analyze` reads the artifacts rather than the audio, so it takes its own
    # flags instead of the file-selection ones the extraction steps use.
    ap = sub.add_parser("analyze", help="Audit extracted features and describe the corpus")
    ap.add_argument("--skip-content", action="store_true",
                    help="audit only the voice features (content artifacts may be incomplete)")
    ap.add_argument("--no-figures", action="store_true", help="skip rendering the figures")

    sub.add_parser("reembed", help="Recompute semantic embeddings from stored transcripts")
    sub.add_parser("build-db", help="Load all feature artifacts into the SQLite database")

    sp = sub.add_parser("search", help="Search the database with a new audio file")
    sp.add_argument("query", help="path to the query audio file")
    sp.add_argument("-k", type=int, default=config.TOP_K, help="results per list (default 3)")
    sp.add_argument("--exclude", nargs="*", default=None,
                    help="file_ids to leave out of the ranking (held-out tests)")
    sp.add_argument("--json", action="store_true", help="machine-readable output")
    sp.add_argument("--device", default=None, help="cpu | cuda")

    ep = sub.add_parser("evaluate", help="Evaluate retrieval on synthetic + recorded queries")
    ep.add_argument("--n-clips", type=int, default=60, help="number of corpus clips to query")
    ep.add_argument("--clip-s", type=float, default=60.0, help="clip length in seconds")
    ep.add_argument("--seed", type=int, default=7)
    ep.add_argument("--device", default=None, help="cpu | cuda")

    args = parser.parse_args(argv)

    # `analyze` has its own flag set, so the shared options are read
    # defensively rather than assumed.
    if getattr(args, "device", None):
        os.environ["AUDIO_SEARCH_DEVICE"] = args.device
    if getattr(args, "min_duration", None) is not None:
        config.MIN_DURATION_S = args.min_duration
    if getattr(args, "max_duration", None) is not None:
        config.MAX_DURATION_S = args.max_duration

    stems = getattr(args, "files", None) or []

    logger.info("compute device: %s", config.resolve_device())

    if args.command == "preprocess":
        return run_preprocess(stems)
    if args.command == "voice":
        return run_voice(stems)
    if args.command == "content":
        return run_content(stems)
    if args.command == "all":
        logger.info("### running full pipeline: preprocess -> voice -> content -> manifest ###")
        for fn in (run_preprocess, run_voice, run_content):
            rc = fn(stems)
            if rc:
                logger.error("step failed with code %d", rc)
                return rc
        return run_manifest()
    if args.command == "manifest":
        return run_manifest()
    if args.command == "verify":
        return run_verify()
    if args.command == "analyze":
        return run_analyze(skip_content=args.skip_content, no_figures=args.no_figures)
    if args.command == "reembed":
        return run_reembed()
    if args.command == "build-db":
        return run_build_db()
    if args.command == "search":
        return run_search(args.query, args.k, args.exclude, args.json)
    if args.command == "evaluate":
        return run_evaluate(args.n_clips, args.clip_s, args.seed)
    return 1