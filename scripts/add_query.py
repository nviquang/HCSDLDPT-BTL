"""Add a team recording to the query set (samples/queries/queries.json).

Examples:
  # Đọc lại một đoạn truyện có trong CSDL (nội dung đã có, người nói mới)
  uv run python scripts/add_query.py rec_minh_harry.m4a --content "'Harry_the_Dirty_Dog'_read_by_Betty_White" \
      --note "Minh đọc đoạn đầu truyện Harry the Dirty Dog"
  # Nói về chủ đề không có trong CSDL (nội dung mới, người nói mới)
  uv run python scripts/add_query.py rec_lan_pho.m4a --note "Lan nói về cách nấu phở"

Then re-run:  uv run python -m audio_search evaluate
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_search import config, database


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="recording (wav/mp3/m4a/ogg/flac)")
    ap.add_argument("--content", default=None,
                    help="file_id whose content the recording repeats (omit for a new topic)")
    ap.add_argument("--speaker", default=None,
                    help="file_id spoken by the same person (omit for a new speaker)")
    ap.add_argument("--note", default="", help="short description")
    args = ap.parse_args()

    src = Path(args.audio)
    if not src.is_file():
        sys.exit(f"not found: {src}")
    con = database.connect()
    for fid in filter(None, (args.content, args.speaker)):
        if not con.execute("SELECT 1 FROM audio_files WHERE file_id = ?", (fid,)).fetchone():
            sys.exit(f"file_id not in database: {fid}")
    con.close()

    config.QUERY_DIR.mkdir(parents=True, exist_ok=True)
    dst = config.QUERY_DIR / f"human_{src.name}"
    shutil.copy2(src, dst)
    kind = ("known" if args.content else "new") + "_content_" + ("known" if args.speaker else "new") + "_speaker"
    qfile = config.QUERY_DIR / "queries.json"
    queries = json.loads(qfile.read_text(encoding="utf-8")) if qfile.exists() else []
    queries = [q for q in queries if q["file"] != dst.name]
    queries.append({"file": dst.name, "kind": kind, "expected_content": args.content,
                    "expected_speaker_file": args.speaker, "note": args.note or "giọng người thật"})
    qfile.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"added {dst.name} ({kind}); {len(queries)} queries in {qfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
