"""Corpus metadata — derive labels from audio filenames.

The corpus was collected from four sources whose filename conventions encode
who is speaking and what the talk is about. Recovering those labels lets the
dataset analysis score the extracted features against ground truth without a
separate annotation step.

Filename conventions (verified across all 504 files):

  Khan Academy   ``NNN_Title ｜ Course ｜ Khan Academy``   (separator U+FF5C)
  TED            ``NN_Title_Speaker_TED``
  Storybook      ``Title_read_by_Speaker`` / ``Title_leído_por_Speaker``
  Vooks          ``..._Vooks_Narrated_Storybooks``

The speaker label for TED is a *heuristic* (the last two underscore tokens),
and it is only weakly reliable — see ``analysis.py`` for the measured
agreement with ECAPA-derived speaker clusters. Treat it as a cross-check, not
as ground truth.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config

# The Khan Academy separator is a fullwidth vertical bar, not ASCII "|".
KA_SEPARATOR = "｜"

# "Title_read_by_Speaker" / "Title_leído_por_Speaker" / "Title read by Speaker"
_SPEAKER_PATTERNS = (
    re.compile(r"_read_by_(?P<speaker>.+)$"),
    re.compile(r"\s+read\s+by\s+(?P<speaker>.+)$"),
    re.compile(r"_leído_por_(?P<speaker>.+)$"),
    re.compile(r"_Leído_por_(?P<speaker>.+)$"),
    re.compile(r"_leido_por_(?P<speaker>.+)$"),
)

# A leading numeric index, e.g. "037_Debugging with test cases" or "12_Predictions_..."
_LEADING_INDEX = re.compile(r"^\d+_")


@dataclass
class FileMeta:
    """Labels derived from one audio file's name."""

    file_id: str
    source: str
    speaker: str | None = None
    topic: str | None = None
    course: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def has_speaker(self) -> bool:
        return bool(self.speaker)

    @property
    def has_topic(self) -> bool:
        return bool(self.topic)


def _khan_academy(stem: str) -> FileMeta | None:
    """Parse ``NNN_Title ｜ Course ｜ Khan Academy``."""
    if "Khan Academy" not in stem:
        return None
    parts = [p.strip() for p in stem.split(KA_SEPARATOR)]
    if len(parts) >= 3:
        course = parts[1]
        title = parts[0]
    elif len(parts) == 2:
        course, title = parts[0], parts[0]
    else:
        return FileMeta(file_id=stem, source="KhanAcademy", notes=["unparsed course field"])
    return FileMeta(
        file_id=stem,
        source="KhanAcademy",
        topic=_LEADING_INDEX.sub("", title).strip() or None,
        course=course or None,
    )


def _ted(stem: str) -> FileMeta | None:
    """Parse ``NN_Title_Speaker_TED`` — speaker is the last two tokens."""
    if not stem.endswith("_TED"):
        return None
    body = _LEADING_INDEX.sub("", stem[: -len("_TED")])
    tokens = body.split("_")
    if len(tokens) < 3:
        # Too short to split a title from a speaker.
        return FileMeta(file_id=stem, source="TED", topic=body.strip() or None,
                        notes=["no separable speaker field"])
    speaker = "_".join(tokens[-2:])
    topic = "_".join(tokens[:-2]).strip()
    return FileMeta(file_id=stem, source="TED", speaker=speaker or None, topic=topic or None)


def _storybook(stem: str) -> FileMeta | None:
    """Parse ``Title_read_by_Speaker`` (and its Spanish/space variants)."""
    for pattern in _SPEAKER_PATTERNS:
        m = pattern.search(stem)
        if m:
            speaker = m.group("speaker").strip()
            title = stem[: m.start()].strip()
            return FileMeta(
                file_id=stem,
                source="Storybook",
                speaker=speaker or None,
                topic=_LEADING_INDEX.sub("", title).strip() or None,
            )
    return None


def _vooks(stem: str) -> FileMeta | None:
    """Parse ``..._Vooks_Narrated_Storybooks``."""
    if "Vooks" not in stem:
        return None
    title = stem.split("_Vooks")[0].strip()
    return FileMeta(
        file_id=stem,
        source="Vooks",
        topic=_LEADING_INDEX.sub("", title).strip() or None,
    )


def parse_file_id(stem: str) -> FileMeta:
    """Derive source/speaker/topic/course for one file stem.

    Order matters: the Khan Academy and TED patterns are checked before the
    generic storybook pattern, because a TED title can contain the words a
    storybook title would match.
    """
    for parser in (_khan_academy, _ted, _storybook, _vooks):
        meta = parser(stem)
        if meta is not None:
            return meta
    return FileMeta(file_id=stem, source="Other", notes=["no known naming convention"])


def parse_all(file_ids: list[str]) -> list[FileMeta]:
    """Parse many stems, preserving the input order."""
    return [parse_file_id(fid) for fid in file_ids]


def build_corpus_table(file_ids: list[str] | None = None) -> object:
    """Build a DataFrame of corpus metadata, one row per file.

    Columns: file_id, source, speaker, topic, course, has_speaker, has_topic.

    Returns a ``pandas.DataFrame``. The import is local so that the module
    stays importable without pandas installed.
    """
    import pandas as pd

    if file_ids is None:
        file_ids = sorted(p.stem for p in config.DATA_DIR.glob("*.mp3"))
    metas = parse_all(file_ids)
    return pd.DataFrame(
        [
            {
                "file_id": m.file_id,
                "source": m.source,
                "speaker": m.speaker,
                "topic": m.topic,
                "course": m.course,
                "has_speaker": m.has_speaker,
                "has_topic": m.has_topic,
            }
            for m in metas
        ]
    )


def discover_file_ids() -> list[str]:
    """All file stems present in the data directory, sorted."""
    from . import preprocess

    return [p.stem for p in preprocess.discover_audio_files(config.DATA_DIR)]
