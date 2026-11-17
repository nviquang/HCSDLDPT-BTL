"""Build the demo / evaluation query set under samples/queries/.

Run:  uv run python scripts/make_queries.py

The assignment asks for queries "do một người nói về một nội dung đã có và
chưa có trong dữ liệu". This script produces both kinds, with the expected
answer recorded in samples/queries/queries.json:

  * TTS queries — Windows' built-in speech synthesiser (voices "David" and
    "Zira", speakers that are NOT in the database) reads
      - an excerpt of a corpus transcript       -> content known, speaker new
      - our own paraphrase of a corpus story    -> content known (no word overlap), speaker new
      - texts on topics absent from the corpus  -> content new, speaker new
  * Real-speech queries — a 45 s excerpt cut from a corpus recording, plus a
    "re-recorded" version of it (telephone band-pass + noise at 15 dB SNR)
                                                -> content known, speaker known

Recordings made by the team members (any format) can simply be dropped into
samples/queries/ and added to queries.json by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_search import config, database

OUT = config.QUERY_DIR

# Excerpts are taken from these corpus files' stored transcripts.
TED_EXCERPT = "How_to_Fix_Africa’s_Learning_Crisis_Rapelang_Rabana_TED"
KHAN_EXCERPT = "028_Games and modularity ｜ Intro to CS - Python ｜ Khan Academy"
PARAPHRASE_OF = "'Harry_the_Dirty_Dog'_read_by_Betty_White"
REAL_CLIP_OF = "The_Brilliance_of_Bacteria_(and_How_They_Combat_Waste)_Patricia_Aymà_Maldonado_TED"

# Our own words — no sentence is copied from the book.
PARAPHRASE = (
    "This is the story of a little white dog with black spots who hated taking baths. "
    "One day, when he heard the water running in the tub, he grabbed the scrubbing brush, "
    "carried it out to the back yard and buried it. Then he ran away from home. "
    "He played all day near the roads, at the railway, and around a building site, "
    "and he got so dirty that he turned into a black dog with white spots. "
    "When he finally came home hungry and tired, his family did not recognise him at all. "
    "So the dog dug up the brush he had hidden, ran upstairs and jumped into the bathtub himself. "
    "After a good scrub his spots appeared again, the family shouted that it was their dog, "
    "and he fell asleep happily on his bed, with the brush hidden under his pillow."
)

NEW_TOPIC_BREAD = (
    "Today I want to explain how sourdough bread works. Instead of commercial yeast, a sourdough "
    "starter is a living culture of wild yeasts and lactic acid bacteria that you feed with flour "
    "and water every day. When you mix the starter into the dough, the yeasts produce carbon dioxide "
    "which makes the loaf rise, while the bacteria produce lactic and acetic acids that give the bread "
    "its sour taste. A long, cold fermentation in the fridge develops more flavour and makes the gluten "
    "easier to digest. Bakers judge the dough by its volume, its bubbles and how it springs back when "
    "pressed. Finally the loaf is scored with a blade and baked in a very hot covered pot, so steam "
    "keeps the crust soft at first and lets the bread expand before it turns golden and crunchy."
)

NEW_TOPIC_PLANTS = (
    "In this lesson we will look at photosynthesis, the process plants use to make their own food. "
    "Inside the leaves there are tiny structures called chloroplasts that contain a green pigment, "
    "chlorophyll. Chlorophyll absorbs red and blue light from the sun. The plant takes in carbon "
    "dioxide from the air through small pores called stomata, and water from the soil through its "
    "roots. Using the energy of light, it splits water molecules, releases oxygen as a by-product, "
    "and builds glucose from carbon dioxide in a cycle of reactions known as the Calvin cycle. "
    "The glucose is used for energy and to build cellulose for new cell walls. That is why almost "
    "all life on Earth, including us, ultimately depends on photosynthesis."
)


def excerpt(file_id: str, words: int = 130, start_frac: float = 0.35) -> str:
    con = database.connect()
    t = con.execute("SELECT transcript FROM content_features WHERE file_id = ?", (file_id,)).fetchone()[0]
    con.close()
    w = t.split()
    s = int(len(w) * start_frac)
    return " ".join(w[s:s + words])


def tts(text: str, voice: str, out: Path) -> None:
    """Synthesise `text` with a Windows SAPI voice into a WAV file."""
    txt = out.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    ps = (
        "Add-Type -AssemblyName System.Speech;"
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('Microsoft {voice} Desktop');"
        "$s.Rate = 0;"
        f"$s.SetOutputToWaveFile('{out}');"
        f"$s.Speak([IO.File]::ReadAllText('{txt}', [Text.Encoding]::UTF8));"
        "$s.Dispose()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


def real_clip(file_id: str, out: Path, start_s: float = 120.0, dur_s: float = 45.0) -> np.ndarray:
    wave, sr = sf.read(str(config.WAV_DIR / f"{file_id}.wav"), dtype="float32")
    clip = wave[int(start_s * sr): int((start_s + dur_s) * sr)]
    sf.write(str(out), clip, sr)
    return clip


def degrade(clip: np.ndarray, sr: int = config.SAMPLE_RATE, snr_db: float = 15.0, seed: int = 0) -> np.ndarray:
    """Simulate a re-recording: telephone band (300-3400 Hz) + white noise."""
    from scipy.signal import butter, sosfilt

    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    y = sosfilt(sos, clip).astype(np.float32)
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(y ** 2)) or 1e-9
    noise = rng.standard_normal(y.shape).astype(np.float32) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    y = y + noise
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.95).astype(np.float32)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = []

    def add(name, kind, expected_content, expected_speaker_file, note):
        queries.append({"file": name, "kind": kind, "expected_content": expected_content,
                        "expected_speaker_file": expected_speaker_file, "note": note})

    tts(excerpt(TED_EXCERPT), "David", OUT / "q1_tts_david_ted_excerpt.wav")
    add("q1_tts_david_ted_excerpt.wav", "known_content_new_speaker", TED_EXCERPT, None,
        "TTS (David) đọc một đoạn transcript của bài TED có trong CSDL")

    tts(excerpt(KHAN_EXCERPT), "Zira", OUT / "q2_tts_zira_khan_excerpt.wav")
    add("q2_tts_zira_khan_excerpt.wav", "known_content_new_speaker", KHAN_EXCERPT, None,
        "TTS (Zira) đọc một đoạn bài giảng Khan Academy có trong CSDL")

    tts(PARAPHRASE, "Zira", OUT / "q3_tts_zira_story_paraphrase.wav")
    add("q3_tts_zira_story_paraphrase.wav", "known_content_new_speaker", PARAPHRASE_OF, None,
        "TTS (Zira) kể lại bằng lời khác câu chuyện 'Harry the Dirty Dog'")

    tts(NEW_TOPIC_BREAD, "David", OUT / "q4_tts_david_new_topic_sourdough.wav")
    add("q4_tts_david_new_topic_sourdough.wav", "new_content_new_speaker", None, None,
        "TTS (David) nói về bánh mì sourdough — chủ đề không có trong CSDL")

    tts(NEW_TOPIC_PLANTS, "Zira", OUT / "q5_tts_zira_new_topic_photosynthesis.wav")
    add("q5_tts_zira_new_topic_photosynthesis.wav", "new_content_new_speaker", None, None,
        "TTS (Zira) giảng về quang hợp — chủ đề không có trong CSDL")

    clip = real_clip(REAL_CLIP_OF, OUT / "q6_real_ted_clip.wav")
    add("q6_real_ted_clip.wav", "known_content_known_speaker", REAL_CLIP_OF, REAL_CLIP_OF,
        "45 s cắt từ bản ghi gốc của một bài TED trong CSDL (giọng thật)")

    sf.write(str(OUT / "q7_real_ted_clip_rerecorded.wav"), degrade(clip), config.SAMPLE_RATE)
    add("q7_real_ted_clip_rerecorded.wav", "known_content_known_speaker", REAL_CLIP_OF, REAL_CLIP_OF,
        "q6 qua bộ lọc điện thoại 300-3400 Hz + nhiễu trắng SNR 15 dB (giả lập thu lại)")

    # Keep the team's own recordings (added with scripts/add_query.py).
    qfile = OUT / "queries.json"
    if qfile.exists():
        queries += [q for q in json.loads(qfile.read_text(encoding="utf-8"))
                    if q["file"].startswith("human_")]
    qfile.write_text(json.dumps(queries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(queries)} queries -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
