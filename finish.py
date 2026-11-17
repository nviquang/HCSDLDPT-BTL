"""Finish missing voice feature extractions without dying to native crashes.

Run:  uv run python finish.py

Why subprocess per file
-----------------------
The voice step has been killed repeatedly by a native access violation
(0xc0000005 in VCRUNTIME140.dll — see Windows Event Log). A try/except in
the same process cannot catch that; the whole interpreter dies. So every
file is extracted in its own child process (`python -m audio_search voice
--files <stem>`). If the child crashes, only the child dies: this script
records the exit code, logs the outcome, and moves on. Artifacts are also
re-validated with json.load so a half-written file never counts as success.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from audio_search import config, preprocess

MAX_ROUNDS = 5          # each round retries every still-missing file once
ATTEMPT_TIMEOUT_S = 3600  # one CPU extraction must never exceed 1 h
RETRY_DELAY_S = 2         # brief pause between children (native-lib teardown)

logger = logging.getLogger("finish")


def artifact_ok(voice_path: Path) -> bool:
    """True only if the artifact exists and parses as valid JSON."""
    if not voice_path.exists():
        return False
    try:
        json.loads(voice_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def missing_files() -> list[Path]:
    """All discovered audio files whose voice artifact is absent/corrupt."""
    out = []
    for src in preprocess.discover_audio_files(config.DATA_DIR):
        if not artifact_ok(config.FEATURES_DIR / f"voice-{src.stem}.json"):
            out.append(src)
    return out


def extract_one(src: Path) -> tuple[str, int, float]:
    """Extract one file in an isolated subprocess.

    Returns (result, returncode, elapsed_seconds). Never raises: spawn
    failures, timeouts and child crashes are all reported as a result string.
    """
    voice_path = config.FEATURES_DIR / f"voice-{src.stem}.json"
    cmd = [sys.executable, "-m", "audio_search", "voice", "--device", "cpu", "--files", src.stem]
    # Let the child dump a Python stack if it hits the native crash again.
    env = {**os.environ, "PYTHONFAULTHANDLER": "1"}

    t0 = time.perf_counter()
    try:
        rc = subprocess.run(cmd, env=env, timeout=ATTEMPT_TIMEOUT_S, check=False).returncode
    except subprocess.TimeoutExpired:
        return "timeout", -1, time.perf_counter() - t0
    except OSError as exc:
        logger.error("cannot spawn subprocess for %s: %s", src.name, exc)
        return "spawn-error", -1, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0

    if artifact_ok(voice_path):
        # Artifact valid even if rc != 0 (e.g. crash after the write).
        return "ok" if rc == 0 else f"ok (rc={rc:#x} but artifact valid)", rc, elapsed
    if rc < 0 or rc >= 0xC0000000:  # NTSTATUS: native crash (0xC0000005, ...)
        return f"crashed ({rc & 0xFFFFFFFF:#010x})", rc, elapsed
    return f"failed (rc={rc}, no artifact)", rc, elapsed


def main() -> int:
    config.ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOGS_DIR / "finish.log", encoding="utf-8"),
        ],
    )

    all_files = preprocess.discover_audio_files(config.DATA_DIR)
    pending = missing_files()
    logger.info("=== finish: %d/%d file(s) missing voice artifacts ===",
                len(pending), len(all_files))
    if not pending:
        logger.info("nothing to do — all voice artifacts already valid")
        return 0

    done: set[str] = set()
    for rnd in range(1, MAX_ROUNDS + 1):
        if not pending:
            break
        logger.info("=== round %d/%d: %d file(s) to retry ===", rnd, MAX_ROUNDS, len(pending))
        still: list[Path] = []
        for i, src in enumerate(pending, 1):
            logger.info("[round %d, %d/%d] start  %s", rnd, i, len(pending), src.name)
            try:
                result, _rc, elapsed = extract_one(src)
            except Exception as exc:  # finish.py itself must never die
                logger.error("[round %d, %d/%d] guard  %s: %s", rnd, i, len(pending), src.name, exc)
                still.append(src)
                continue
            if result.startswith("ok"):
                done.add(src.stem)
                logger.info("[round %d, %d/%d] %-32s %s (%.1fs)",
                            rnd, i, len(pending), result, src.name, elapsed)
            else:
                logger.error("[round %d, %d/%d] %-32s %s (%.1fs)",
                             rnd, i, len(pending), result, src.name, elapsed)
                still.append(src)
            time.sleep(RETRY_DELAY_S)
        pending = still

    remaining = missing_files()
    logger.info(
        "=== finish done: %d ok this run, %d still missing (of %d) — log: %s ===",
        len(done), len(remaining), len(all_files),
        config.LOGS_DIR / "finish.log",
    )
    for src in remaining:
        logger.error("STILL MISSING: %s", src.name)
    return 1 if remaining else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\ninterrupted — already-written artifacts are kept; rerun to resume", file=sys.stderr)
        raise SystemExit(130) from None
