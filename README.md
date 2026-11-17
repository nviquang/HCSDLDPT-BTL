# HCSDLDPT BTL — Hệ CSDL lưu trữ và tìm kiếm âm thanh theo nội dung

Bài tập lớn môn **Hệ CSDL đa phương tiện và ứng dụng** (HCSDLDPT).

**Đầu vào:** một file âm thanh mới. **Đầu ra:** 3 file giống nhất về **nội dung** và 3 file giống nhất về **giọng nói**, xếp giảm dần theo độ tương đồng. Kèm theo là toàn bộ kết quả trung gian (transcript, vector đặc trưng, độ tương đồng).

| Thành phần | Công cụ | File |
|---|---|---|
| Bộ dữ liệu | 539 file (≈ 67,1 giờ): TED, Khan Academy, Storyline Online, Vooks — **508 file đạt chuẩn** (một người nói, là bài nói của người thật); 31 file biểu diễn / giọng AI / nhiều người nói được gắn cờ và loại khỏi tìm kiếm | `data/` |
| Đặc trưng giọng nói | ECAPA-TDNN 192-D (SpeechBrain) + MFCC/Δ/ΔΔ + prosody | `voice_features.py` |
| Đặc trưng nội dung | faster-whisper (ASR) → chia đoạn ≤ 250 token → MiniLM 384-D | `content_features.py` |
| CSDL | SQLite: 6 bảng + chỉ mục full-text FTS5 | `database.py` → `artifacts/audio_search.db` |
| Tìm kiếm | Cosine top-k, ngưỡng "có/chưa có trong CSDL" | `search.py` |
| Đánh giá | Hit@k, MRR, ROC/AUC, leave-one-out người nói | `evaluation.py` → `artifacts/evaluation/` |
| Demo | Streamlit (tải file / ghi âm / file mẫu) | `app.py` |

Tài liệu chính: **[`he_thong_tim_kiem.md`](he_thong_tim_kiem.md)** (sơ đồ khối, quy trình, kết quả trung gian, đánh giá). Phân tích đặc trưng: [`bao_cao_phan_tich.md`](bao_cao_phan_tich.md). Thuật toán: [`thuat_toan_pipeline.md`](thuat_toan_pipeline.md).

## Hướng dẫn chạy

### 1. Yêu cầu

| Thành phần | Ghi chú |
|---|---|
| Windows / Linux / macOS | Đã chạy thử trên Windows 11 |
| [uv](https://docs.astral.sh/uv/) | Quản lý Python 3.12 và thư viện (`uv sync` tự cài Python nếu thiếu) |
| `ffmpeg` trong `PATH` | Đọc mp3 / m4a. Kiểm tra: `ffmpeg -version` |
| GPU NVIDIA (tuỳ chọn) | torch cài bản CUDA 12.8. Không có GPU thì tự chạy trên CPU (chậm hơn) |
| Dung lượng trống ≈ 15 GB | `data/` ≈ 5,2 GB, `artifacts/` ≈ 7,4 GB (trong đó `artifacts/wav/` ≈ 7,3 GB) |

### 2. Cài đặt

```powershell
uv sync                                     # tạo .venv và cài toàn bộ thư viện
uv run python scripts/download_models.py    # tuỳ chọn: tải trước 3 model (ECAPA, Whisper small, MiniLM)
```

> Trên Windows, nếu gặp lỗi `UnicodeEncodeError` khi in tên file có ký tự đặc biệt, đặt
> `$env:PYTHONIOENCODING = "utf-8"` (PowerShell) trước khi chạy.

### 3. Dữ liệu và CSDL không nằm trong git

`.gitignore` loại `data/`, `artifacts/` và mọi file âm thanh (`*.mp3`, `*.wav`…). Sau khi clone cần chép
từ máy nhóm (hoặc ổ chung) vào đúng chỗ:

```
data/                     539 file mp3 gốc                       (bắt buộc nếu chạy lại từ đầu)
artifacts/audio_search.db CSDL đã dựng (≈ 23 MB)                 (đủ để chạy demo / tìm kiếm)
artifacts/features/       đặc trưng từng file (≈ 62 MB)          (đủ để dựng lại CSDL)
artifacts/wav/            wav 16 kHz đã tiền xử lý (≈ 7,3 GB)    (cần cho bước evaluate)
samples/queries/*.wav     file truy vấn mẫu
```

Có hai cách chạy:

- **Cách nhanh (chỉ cần `artifacts/audio_search.db`)** → nhảy tới [bước 5](#5-tìm-kiếm).
- **Chạy lại từ đầu (cần `data/`)** → làm bước 4.

### 4. Trích đặc trưng và dựng CSDL (chạy lại từ đầu)

```powershell
# 4.1 Tiền xử lý + đặc trưng giọng + đặc trưng nội dung cho toàn bộ data/
#     Mất vài giờ với 539 file. Chạy lại được: file nào đã có đặc trưng sẽ bỏ qua.
uv run python -m audio_search all
#     hoặc từng bước:  preprocess  →  voice  →  content  →  manifest
#     chỉ một số file:  uv run python -m audio_search all --files "<tên file không đuôi>" ...
#     ép chạy CPU:      uv run python -m audio_search all --device cpu

# 4.2 Nếu bước giọng nói bị lỗi (CUDA hết bộ nhớ với file dài, hoặc tiến trình bị crash),
#     chạy finish.py: trích giọng trên CPU, mỗi file một tiến trình riêng, tự thử lại.
uv run python finish.py
uv run python -m audio_search content       # bổ sung đặc trưng nội dung còn thiếu (nếu có)

# 4.3 Kiểm tra đặc trưng rồi nạp vào CSDL SQLite
uv run python -m audio_search verify
uv run python -m audio_search build-db      # → artifacts/audio_search.db
```

`build-db` đọc `artifacts/features/` và gắn cờ các file không đạt chuẩn (biểu diễn, giọng AI, nhiều người nói;
danh sách trong `src/audio_search/config.py`). Các file đó vẫn được lưu nhưng không xuất hiện trong kết quả tìm kiếm.

Khi đổi model nhúng nội dung hoặc cách chia đoạn, không cần chạy lại ASR:

```powershell
uv run python -m audio_search reembed       # tính lại embedding từ transcript đã lưu
uv run python -m audio_search build-db
```

### 5. Tìm kiếm

```powershell
# Trả về top-3 theo nội dung và top-3 theo giọng nói, kèm kết quả trung gian
uv run python -m audio_search search samples/queries/q3_tts_zira_story_paraphrase.wav

uv run python -m audio_search search <file> -k 5                  # lấy 5 kết quả mỗi danh sách
uv run python -m audio_search search <file> --json                # xuất JSON
uv run python -m audio_search search <file> --exclude "<file_id>" # bỏ một file khỏi CSDL → giả lập "chưa có"
```

File truy vấn có thể là wav / mp3 / m4a / flac / ogg, nên dài ≥ 20 giây. Mỗi danh sách kết luận
"đã có / chưa có trong CSDL" dựa trên ngưỡng hiệu chỉnh ở bước đánh giá (`artifacts/evaluation/thresholds.json`).

### 6. Demo (Streamlit)

```powershell
uv run streamlit run app.py                 # mở http://localhost:8501
```

| Tab | Nội dung |
|---|---|
| 🔎 Tìm kiếm | Chọn nguồn truy vấn: **tải file lên**, **ghi âm trực tiếp** bằng micro, hoặc **file mẫu** → top-3 nội dung, top-3 giọng, transcript, vector, độ tương đồng, thời gian |
| 🧩 Sơ đồ khối | Sơ đồ hai giai đoạn: lập chỉ mục (offline) và tìm kiếm (online) |
| 🗄️ CSDL | Duyệt bản ghi, lọc theo nguồn / cờ, xem lược đồ, thử truy vấn FTS5 |
| 📊 Đánh giá | Kết quả E1–E3 đọc từ `artifacts/evaluation/` |

Lần tìm kiếm đầu tiên chậm hơn vì phải nạp model (khoảng vài chục giây); các lần sau nhanh.

### 7. Bộ truy vấn mẫu và đánh giá

```powershell
# Tạo lại truy vấn mẫu q1–q7 (dùng giọng TTS có sẵn của Windows: David, Zira)
uv run python scripts/make_queries.py

# Thêm bản ghi âm của người thật vào bộ truy vấn
uv run python scripts/add_query.py rec_minh.m4a --content "<file_id có cùng nội dung>" --note "Minh đọc lại ..."
uv run python scripts/add_query.py rec_lan.m4a --note "Lan nói về chủ đề mới"      # nội dung mới, người nói mới

# Chạy đánh giá: E1 (đoạn cắt sạch / thu lại), hiệu chỉnh ngưỡng ROC, E2 (nhận diện người nói), E3 (truy vấn mẫu)
uv run python -m audio_search evaluate      # → artifacts/evaluation/summary.md, results.json, thresholds.json
uv run python -m audio_search evaluate --n-clips 100 --clip-s 30 --seed 1          # đổi cấu hình E1
```

`make_queries.py` chỉ chạy được trên Windows (dùng System.Speech). Hai file `human_mark_rober_*` là đoạn cắt
từ một bài TED không có trong CSDL, không do script tạo ra.

### 8. Thêm dữ liệu mới

```powershell
# Tải thêm bài TED một diễn giả (cần yt-dlp trong PATH). Xem trước danh sách bằng --dry-run.
uv run python scripts/fetch_talks.py --dry-run --count 10
uv run python scripts/fetch_talks.py --count 10                    # tải vào data/, ghi tên vào artifacts/new_stems.txt

# Trích đặc trưng cho các file mới rồi dựng lại CSDL
uv run python -m audio_search all --files <các tên trong artifacts/new_stems.txt>
uv run python -m audio_search build-db
uv run python -m audio_search evaluate
```

Nên đọc transcript các file mới (`artifacts/features/`) để chắc là bài nói của một người; nếu không,
thêm tên file vào `NON_TALK_FILES` hoặc `MULTI_SPEAKER_FILES` trong `config.py`.

### 9. Kiểm thử và phân tích

```powershell
uv run pytest -q                            # 16 test (một số test bỏ qua nếu thiếu data/ hoặc CSDL)
uv run ruff check .
uv run python -m audio_search analyze       # phân tích đặc trưng, vẽ hình → artifacts/analysis/
```

### 10. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `CUDA out of memory` ở bước `voice` | `uv run python finish.py` (chạy CPU từng file) hoặc thêm `--device cpu` |
| Tiến trình chết không báo lỗi (0xc0000005) khi trích giọng | `uv run python finish.py` |
| `UnicodeEncodeError` khi in tên file | `$env:PYTHONIOENCODING = "utf-8"` |
| Không đọc được mp3 / m4a | Cài `ffmpeg` và thêm vào `PATH` |
| Demo báo thiếu CSDL | Chép `artifacts/audio_search.db` vào, hoặc chạy `build-db` |
| Cổng 8501 đang bận | `uv run streamlit run app.py --server.port 8502` |

## Cấu trúc thư mục

```
app.py                     demo Streamlit
finish.py                  trích giọng an toàn (CPU, mỗi file một tiến trình)
src/audio_search/
  cli.py                   lệnh: preprocess, voice, content, all, manifest, verify, analyze, reembed, build-db, search, evaluate
  config.py                đường dẫn, tham số, danh sách file bị gắn cờ
  preprocess.py            mono · 16 kHz · chuẩn hoá biên độ
  voice_features.py        MFCC, ECAPA-TDNN, prosody
  content_features.py      Whisper ASR, chia đoạn, MiniLM
  database.py              lược đồ SQLite + FTS5, nạp dữ liệu
  search.py                trích đặc trưng truy vấn, tìm cosine top-k
  evaluation.py            E1–E3, hiệu chỉnh ngưỡng
  corpus.py                đọc tên file → nguồn, người nói
  analysis.py, figures.py  phân tích đặc trưng, vẽ hình
scripts/                   download_models, make_queries, add_query, fetch_talks
samples/queries/           truy vấn mẫu + queries.json (đáp án mong đợi)
tests/                     pytest
```
