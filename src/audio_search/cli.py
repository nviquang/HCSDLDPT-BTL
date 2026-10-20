"""Command implementations for the audio_search CLI."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from . import config, feature_store, preprocess

logger = logging.getLogger(__name__)


def _device() -> str:
    return os.environ.get("AUDIO_SEARCH_DEVICE", "cpu")


def _select_files(data_dir: Path, stems: list[str] | None) -> list[Path]:
    """Filter discovered audio files to a subset by stem, if requested."""
    files = preprocess.discover_audio_files(data_dir)
    if not stems:
        return files
    wanted = {s for s in stems}
    return [f for f in files if f.stem in wanted]


def run_preprocess(stems: str = "") -> int:
    files = _select_files(config.DATA_DIR, stems.split() if stems else None)
    ok = skipped = failed = 0
    for src in files:
        try:
            pp = preprocess.preprocess_file(src)
            if not preprocess.validate_duration(pp.duration_s):
                logger.warning("duration %.1fs outside range for %s", pp.duration_s, src.name)
                skipped += 1
            else:
                ok += 1
        except Exception as exc:
            logger.error("preprocess failed for %s: %s", src.name, exc)
            failed += 1
    logger.info("preprocess: %d ok, %d skipped, %d failed (of %d)", ok, skipped, failed, len(files))
    return 1 if failed else 0


def run_voice(stems: str = "") -> int:
    from . import voice_features

    files = _select_files(config.DATA_DIR, stems.split() if stems else None)
    ok = skipped = failed = 0
    for src in files:
        file_id = src.stem
        voice_path = config.FEATURES_DIR / f"voice-{file_id}.json"
        if voice_path.exists():
            skipped += 1
            continue
        try:
            pp = preprocess.preprocess_file(src)
            feats = voice_features.compute_voice_features(pp.wave, pp.sr)
            feature_store.write_voice_artifact(file_id, feats, filename=src.name, duration_s=pp.duration_s)
            ok += 1
        except Exception as exc:
            logger.error("voice failed for %s: %s", src.name, exc)
            failed += 1
    logger.info("voice: %d ok, %d skipped, %d failed (of %d)", ok, skipped, failed, len(files))
    return 1 if failed else 0


def run_content(stems: str = "") -> int:
    from . import content_features

    files = _select_files(config.DATA_DIR, stems.split() if stems else None)
    ok = skipped = failed = 0
    for src in files:
        file_id = src.stem
        content_path = config.FEATURES_DIR / f"content-{file_id}.json"
        if content_path.exists():
            skipped += 1
            continue
        try:
            pp = preprocess.preprocess_file(src)
            feats = content_features.compute_content_features(str(pp.path))
            feature_store.write_content_artifact(file_id, feats, filename=src.name, duration_s=pp.duration_s)
            ok += 1
        except Exception as exc:
            logger.error("content failed for %s: %s", src.name, exc)
            failed += 1
    logger.info("content: %d ok, %d skipped, %d failed (of %d)", ok, skipped, failed, len(files))
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


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (also used by pyproject [project.scripts] audio-search)."""
    import argparse
    import sys

    from . import config

    config.ensure_dirs()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOGS_DIR / "extract.log", encoding="utf-8"),
        ],
    )

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

    args = parser.parse_args(argv)

    if args.device:
        os.environ["AUDIO_SEARCH_DEVICE"] = args.device
    if args.min_duration is not None:
        config.MIN_DURATION_S = args.min_duration
    if args.max_duration is not None:
        config.MAX_DURATION_S = args.max_duration

    stems = args.files or []

    if args.command == "preprocess":
        return run_preprocess(" ".join(stems))
    if args.command == "voice":
        return run_voice(" ".join(stems))
    if args.command == "content":
        return run_content(" ".join(stems))
    if args.command == "all":
        for fn in (run_preprocess, run_voice, run_content):
            rc = fn(" ".join(stems))
            if rc:
                logger.error("step failed with code %d", rc)
                return rc
        return run_manifest()
    if args.command == "manifest":
        return run_manifest()
    if args.command == "verify":
        return run_verify()
    return 1