"""Fetch extra single-speaker TED talks to replace non-talk / multi-reader files.

Uses the same source (YouTube @TED channel), tool (yt-dlp, bestaudio -> mp3)
and filename convention (CrawlData `sanitize_filename`: "Title_Speaker_TED")
as the team's original crawl, so new files parse exactly like the old ones.

    uv run python scripts/fetch_talks.py --dry-run          # list candidates only
    uv run python scripts/fetch_talks.py --count 25         # download into data/

Selection rules (the assignment: one topic, one speaker, a talk):
  * title has the form "<title> | <speaker> | TED" with a single speaker
    (no "and", "&", ",", "x" between names)
  * no performance keywords (music, song, dance, poem, choir, ...)
  * duration 4-15 min (same range as the existing TED files)
  * not already in data/
After downloading, run:
    uv run python -m audio_search all --files <new stems>   (printed at the end)
    uv run python -m audio_search build-db
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHANNEL = "https://www.youtube.com/@TED/videos"
PERFORMANCE = re.compile(
    r"\b(music|musical|song|songs|sing|sings|singing|dance|dancing|danced|poem|poetry|spoken word|"
    r"choir|chorus|accordion|piano|guitar|violin|cello|drums?|flute|orchestra|ensemble|band|concert|perform|performance|beat|rap|dj|live|"
    r"trailer|highlights|countdown|shorts?|q&a|interview|conversation|podcast|in conversation)\b",
    re.IGNORECASE,
)
MULTI = re.compile(r"\s(and|&|x|with)\s|,", re.IGNORECASE)


def sanitize_filename(name: str) -> str:
    """Identical to CrawlData/src/utils.py:sanitize_filename."""
    name = name.strip()
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"[_\s]+", "_", name)
    name = name.strip(".")
    return (name[:200] if len(name) > 200 else name) or "untitled"


def yt_dlp() -> str:
    exe = shutil.which("yt-dlp")
    if not exe:
        sys.exit("yt-dlp not found in PATH (pip install yt-dlp)")
    return exe


def list_channel(start: int, end: int) -> list[tuple[str, int, str]]:
    out = subprocess.run(
        [yt_dlp(), "--encoding", "utf-8", "--flat-playlist", "--playlist-items", f"{start}-{end}",
         "--print", "%(id)s\t%(duration)s\t%(title)s", CHANNEL],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3 and parts[1].replace(".", "").isdigit():
            rows.append((parts[0], int(float(parts[1])), parts[2]))
    return rows


def eligible(title: str, duration: int, existing: set[str]) -> str | None:
    parts = [p.strip() for p in title.split("|")]
    if len(parts) != 3 or parts[2] != "TED":
        return None
    if not 240 <= duration <= 900:
        return None
    if PERFORMANCE.search(parts[0]) or MULTI.search(parts[1]):
        return None
    stem = sanitize_filename(title)
    return None if stem in existing else stem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=25)
    ap.add_argument("--start", type=int, default=1, help="channel position to start scanning")
    ap.add_argument("--scan", type=int, default=600, help="how many channel videos to scan")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    existing = {p.stem for p in DATA.iterdir() if p.is_file()}
    picks: list[tuple[str, str, int]] = []
    for vid, dur, title in list_channel(args.start, args.start + args.scan - 1):
        stem = eligible(title, dur, existing)
        if stem and stem not in {s for _, s, _ in picks}:
            picks.append((vid, stem, dur))
        if len(picks) >= args.count:
            break

    for vid, stem, dur in picks:
        print(f"{dur // 60:2d}:{dur % 60:02d}  {vid}  {stem}")
    print(f"{len(picks)} candidate(s)")
    if args.dry_run or not picks:
        return 0

    done = []
    for vid, stem, _ in picks:
        rc = subprocess.run(
            [yt_dlp(), "--encoding", "utf-8", "-f", "bestaudio/best", "-x", "--audio-format", "mp3",
             "--no-playlist", "-o", str(DATA / f"{stem}.%(ext)s"),
             f"https://www.youtube.com/watch?v={vid}"],
            check=False,
        ).returncode
        if rc == 0 and (DATA / f"{stem}.mp3").exists():
            done.append(stem)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "new_stems.txt").write_text("\n".join(done), encoding="utf-8")
    print(f"downloaded {len(done)}/{len(picks)} -> artifacts/new_stems.txt")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
