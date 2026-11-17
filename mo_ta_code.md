# Báo cáo Mô tả Kiến trúc và Mã nguồn Hệ thống (Audio Search)

> **Môn học**: Hệ cơ sở dữ liệu đa phương tiện và ứng dụng (HCSDLDPT)  
> **Đề tài**: Xây dựng hệ CSDL lưu trữ và tìm kiếm âm thanh theo nội dung  
> **Phiên bản mã nguồn**: `audio-search v0.1.0` (Giai đoạn 2: Trích xuất đặc trưng Giọng nói & Nội dung)  
> **Môi trường & Công cụ**: Python 3.12, `uv`, PyTorch 2.8, SpeechBrain, faster-whisper, sentence-transformers, librosa

---

## 1. Tổng quan hệ thống

Dự án triển khai bài toán **tìm kiếm âm thanh theo nội dung (Content-Based Audio Retrieval)** với yêu cầu đầu ra gồm 2 trục độc lập:

1. **Tìm kiếm theo Giọng nói (Voice / Speaker Retrieval)**: Trả về Top-3 file có giọng người nói tương đồng nhất.
2. **Tìm kiếm theo Nội dung (Content / Semantic Retrieval)**: Trả về Top-3 file có chủ đề, nội dung câu chuyện/bài giảng tương đồng nhất.

Hiện tại, mã nguồn đã hoàn thiện **Giai đoạn 2**:

- Xây dựng hoàn chỉnh 2 bộ thuộc tính (Giọng nói + Nội dung).
- Đóng gói thành package Python `audio_search` có CLI điều phối.
- Xử lý mượt mà trên bộ dữ liệu mẫu (6 file TED Talks), kiến trúc được thiết kế **batch-ready** và **resume-safe** để sẵn sàng mở rộng lên $\ge 500$ files.
- Dữ liệu vector được chuẩn hóa $L_2$ và đóng gói sẵn sàng nạp vào **PostgreSQL + pgvector** ở các giai đoạn sau.

---

## 2. Cấu trúc thư mục dự án

```text
.
├── pyproject.toml              # Khai báo dependencies, cấu hình ghim bản build PyTorch CPU
├── uv.lock                     # Khóa phiên bản các package phụ thuộc (Astral uv)
├── README.md                   # Hướng dẫn cài đặt và sử dụng nhanh
├── plan.md                     # Kế hoạch chi tiết Giai đoạn 2
├── report.md                   # Báo cáo kế hoạch tổng thể 5 giai đoạn của bài tập lớn
├── mo_ta.md                    # Đề bài gốc của môn học
├── mo_ta_code.md               # [Tài liệu này] Báo cáo chi tiết cấu trúc mã nguồn
│
├── data/                       # Chứa file âm thanh đầu vào (.mp3 gốc)
├── scripts/
│   └── download_models.py      # Script tải trước các pretrained models về cache cục bộ
│
├── src/audio_search/           # Package mã nguồn chính
│   ├── __init__.py
│   ├── __main__.py             # Entrypoint khi chạy `python -m audio_search`
│   ├── config.py               # Quản lý cấu hình tập trung, tham số mô hình, đường dẫn
│   ├── preprocess.py           # Chuẩn hóa âm thanh (16 kHz mono, peak normalization)
│   ├── voice_features.py       # Trích xuất đặc trưng giọng nói (MFCC 78-D + ECAPA 192-D)
│   ├── content_features.py     # Trích xuất đặc trưng nội dung (Whisper ASR + MiniLM 384-D)
│   ├── feature_store.py        # Lưu trữ artifact JSON và xuất manifest.csv phục vụ ETL
│   ├── cli.py                  # CLI cung cấp các lệnh tiền xử lý và trích xuất
│   └── verify.py               # Kiểm tra tính toàn vẹn và demo tìm kiếm Top-3 in-memory
│
├── tests/
│   └── test_features.py        # Unit test với pytest (kiểm tra shape, L2 norm, v.v.)
│
└── artifacts/                  # Thư mục chứa dữ liệu sinh ra tự động
    ├── wav/                    # Audio sau khi chuẩn hóa về định dạng 16 kHz mono WAV
    ├── models/                 # Cache lưu trữ trọng số mô hình ECAPA-TDNN
    ├── features/               # File JSON trích xuất cho từng audio + manifest.csv
    └── logs/                   # Log chạy pipeline (extract.log)
```

---

## 3. Chi tiết các module trong `src/audio_search/`

### 3.1. `config.py` — Quản trị cấu hình tập trung

- **Đường dẫn thư mục**: Định nghĩa các hằng số đường dẫn gốc `PROJECT_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`, `WAV_DIR`, `FEATURES_DIR`, `LOGS_DIR`.
- **Chuẩn hóa âm thanh**:
  - `SAMPLE_RATE = 16000`: Cố định 16 kHz mono (chuẩn đầu vào của cả mô hình ECAPA và Whisper).
  - `PEAK_NORM = 0.95`: Biên độ cực đại sau chuẩn hóa để tránh méo tiếng và đồng đều âm lượng.
  - `MIN_DURATION_S = 10.0`, `MAX_DURATION_S = 1800.0`: Lọc các file quá ngắn hoặc vượt quá 30 phút.
- **Kích thước các bộ đặc trưng**:
  - `MFCC_N_COEFF = 26`, phân tích khung $25\text{ ms}$, bước nhảy $10\text{ ms}$ $\rightarrow$ kết hợp đạo hàm bậc 1 và bậc 2 thành **78 chiều**.
  - `ECAPA_DIM = 192`: Kích thước vector nhúng giọng nói của SpeechBrain `spkrec-ecapa-voxceleb`.
  - `WHISPER_MODEL_SIZE = "small"`, `WHISPER_COMPUTE_TYPE = "int8"`, `WHISPER_LANGUAGE = "en"`.
  - `SEMANTIC_DIM = 384`: Kích thước vector ngữ nghĩa của `sentence-transformers/all-MiniLM-L6-v2`.

---

### 3.2. `preprocess.py` — Tiền xử lý âm thanh

- **`discover_audio_files(data_dir)`**: Tự động quét và phát hiện các định dạng âm thanh `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.aac`.
- **Cơ chế tải âm thanh an toàn**: Hàm `load_audio` sử dụng `soundfile` làm bộ giải mã chính; nếu gặp codec không tương thích sẽ tự động fallback sang `ffmpeg` trực tiếp qua pipe `subprocess` để bảo đảm không bị crash giữa chừng.
- **Chuẩn hóa tín hiệu**:
  - `_to_mono`: Chuyển đổi stereo thành mono (lấy trung bình các kênh).
  - `_resample`: Đưa tần số lấy mẫu về đúng 16.000 Hz qua `librosa.resample`.
  - `_normalize`: Chuẩn hóa biên độ đỉnh về 0.95.
- **Resume-safe**: Nếu file `.wav` đã được xử lý và lưu trong `artifacts/wav/<file_id>.wav`, hàm `preprocess_file` sẽ nạp lại trực tiếp từ đĩa, tiết kiệm thời gian xử lý khi chạy lại.

---

### 3.3. `voice_features.py` — Bộ thuộc tính Giọng nói

Module triển khai 2 phương pháp tiếp cận bổ trợ cho nhau:

#### a) Đặc trưng cổ điển (Acoustic / Prosody qua `librosa`)

- **MFCC + $\Delta$ + $\Delta\Delta$ (`compute_mfcc_stats`)**:
  - Trích xuất 26 hệ số MFCC trên thang Mel (80 kênh lọc).
  - Tính đạo hàm bậc 1 ($\Delta$) và bậc 2 ($\Delta\Delta$) phản ánh sự biến thiên đặc trưng âm thanh theo thời gian.
  - Gom 3 ma trận thành ma trận $(78, T)$, sau đó tính trung bình theo trục thời gian $\rightarrow$ thu được vector **78 chiều** (`mfcc_stats`).
- **Đặc trưng cao độ và phổ (`compute_prosody`)**:
  - Tần số cơ bản $F_0$ (Pitch) bằng giải thuật `pyin` trong dải $60 - 400\text{ Hz}$, tính trung bình và độ lệch chuẩn trên các khung có tiếng nói (`f0_mean_hz`, `f0_std_hz`).
  - Tỷ lệ qua điểm 0 (`zcr_mean`).
  - Trọng tâm phổ (`spectral_centroid_mean`) và độ suy giảm phổ (`spectral_rolloff_mean`).

#### b) Đặc trưng Deep Speaker Embedding (qua `SpeechBrain`)

- Mô hình **ECAPA-TDNN** (`speechbrain/spkrec-ecapa-voxceleb`).
- Tự động phát hiện GPU/CPU; nạp mô hình một lần bằng `@lru_cache`.
- Áp dụng `LocalStrategy.COPY` trên Windows nhằm tránh lỗi phân quyền tạo symlink.
- Đưa âm thanh qua mạng trích xuất ra vector đại diện người nói **192 chiều**.
- **Chuẩn hóa $L_2$**: Vector được chuẩn hóa $\|v\|_2 = 1.0$, cho phép tính độ tương đồng Cosine cực nhanh bằng tích vô hướng (Dot Product).

---

### 3.4. `content_features.py` — Bộ thuộc tính Nội dung

Module xây dựng pipeline 2 bước:

#### a) Nhận dạng giọng nói (ASR - Automatic Speech Recognition)

- Sử dụng thư viện **`faster-whisper`** (dựa trên CTranslate2) với mô hình `small`, định dạng lượng tử hóa `int8` tối ưu tốc độ CPU.
- Tích hợp bộ lọc âm thanh hoạt động **VAD** (`vad_filter=True`) để loại bỏ khoảng lặng và âm nhiễu nền trước khi giải mã.
- Xuất ra chuỗi văn bản hoàn chỉnh (`transcript`).

#### b) Nhúng ngữ nghĩa văn bản (Semantic Embedding)

- **`chunk_to_sentences`**: Cắt văn bản theo ranh giới dấu câu (`.`, `!`, `?`), nhóm các câu liên tiếp thành các đoạn tối đa 250 token (đếm bằng tokenizer của MiniLM, giới hạn cứng 256) để không đoạn nào bị cắt; câu làm tràn đoạn sẽ mở đoạn mới, câu quá dài được tách theo từ.
- Sử dụng mô hình **`sentence-transformers/all-MiniLM-L6-v2`**.
- Mã hóa từng chunk thành vector 384 chiều, sau đó áp dụng **Mean-pooling** trên tất cả các chunk để tạo thành 1 vector duy nhất đại diện cho toàn bộ bài nói.
- **Chuẩn hóa $L_2$**: Vector ngữ nghĩa 384 chiều được chuẩn hóa độ dài về 1.0.

---

### 3.5. `feature_store.py` — Lưu trữ đặc trưng và Xuất Manifest

- **Lưu trữ per-file**:
  - `voice-<file_id>.json`: Chứa `mfcc_stats` (78-D), `mfcc_std`, `prosody`, và `ecapa_embedding` (192-D).
  - `content-<file_id>.json`: Chứa `transcript`, `n_chunks`, và `semantic_embedding` (384-D).
- **Hỗ trợ Resume**: Hàm `artifact_exists(file_id)` kiểm tra nếu cả 2 file JSON đã tồn tại thì tự động bỏ qua khi chạy batch.
- **Tạo bảng tổng hợp `manifest.csv`**:
  - Duyệt và ghép nối các file JSON đặc trưng.
  - Xuất ra bảng gồm các cột: `file_id`, `filename`, `duration_s`, `mfcc_stats`, `ecapa_embedding`, `semantic_embedding`, `transcript`.
  - Các vector được lưu dưới dạng chuỗi JSON, là cấu trúc dữ liệu trung gian chuẩn mực để giai đoạn sau import trực tiếp vào cột `vector` của **pgvector**.

---

### 3.6. `cli.py` & `__main__.py` — Giao diện dòng lệnh

Cho phép điều phối toàn bộ pipeline linh hoạt bằng các lệnh:

- `uv run python -m audio_search preprocess`: Chuẩn hóa âm thanh sang 16 kHz WAV.
- `uv run python -m audio_search voice`: Trích xuất bộ đặc trưng giọng nói.
- `uv run python -m audio_search content`: Trích xuất bộ đặc trưng nội dung.
- `uv run python -m audio_search all`: Chạy tuần tự cả 3 bước và tự động tạo `manifest.csv`.
- `uv run python -m audio_search manifest`: Gom các artifact đã có thành file CSV.
- `uv run python -m audio_search verify`: Kiểm tra tính toàn vẹn của dữ liệu đã trích xuất.
- Tham số hỗ trợ: `--files` (chỉ định file cụ thể), `--device` (cpu/cuda), `--min-duration`, `--max-duration`.

---

### 3.7. `verify.py` — Kiểm tra và Thử nghiệm truy vấn Top-3

- **Kiểm tra tính hợp lệ của Artifacts**:
  - Xác thực đúng số chiều: ECAPA đủ 192 chiều, MFCC đủ 78 chiều, Semantic đủ 384 chiều.
  - Xác thực chuẩn $L_2$: $\|\mathbf{v}\|_2 \approx 1.0 \pm 0.1$.
  - Báo cảnh báo nếu file không có lời (instrumental) dẫn đến transcript trống hoặc vector semantic bằng 0.
- **Mô phỏng tìm kiếm Top-3 (In-memory retrieval sanity)**:
  - Chọn file truy vấn (query file).
  - Tính độ tương đồng Cosine trực tiếp với tất cả các file còn lại trong bộ dữ liệu:
    $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} \quad (\text{do } \|\mathbf{u}\| = \|\mathbf{v}\| = 1)$$
  - Sắp xếp giảm dần và in ra **Top-3 file có giọng nói giống nhất** và **Top-3 file có nội dung giống nhất**.
  - Kiểm tra độ tự tương đồng (`self-similarity` $\approx 1.000$).

---

## 4. Kiểm thử và Quản lý phiên bản phụ thuộc

### 4.1. Unit Tests (`tests/test_features.py`)

Kiểm thử tự động bằng `pytest`:

- `test_discover_finds_samples`: Kiểm tra quét file dữ liệu mẫu trong `data/`.
- `test_preprocess_all_samples`: Kiểm tra toàn bộ file mẫu tiền xử lý thành công về 16 kHz mono.
- `test_mfcc_stats_shape`: Kiểm tra vector MFCC đủ 78 chiều, không chứa giá trị NaN/Inf.
- `test_ecapa_embedding_shape`: Kiểm tra vector ECAPA đủ 192 chiều và đạt chuẩn $L_2 = 1.0$.
- `test_semantic_embedding_shape`: Kiểm tra vector ngữ nghĩa đủ 384 chiều và đạt chuẩn $L_2 = 1.0$.

### 4.2. Khắc phục xung đột thư viện (`pyproject.toml`)

Codebase xử lý các vấn đề tương thích thư viện phiên bản mới:

1. **PyTorch & TorchAudio 2.8.0**: Ghim cứng bản 2.8.0 qua `pytorch-cpu` wheel vì từ bản `torchaudio >= 2.9`, hàm `torchaudio.list_audio_backends()` đã bị xóa bỏ, dẫn đến việc `SpeechBrain 1.0.3` bị crash nếu dùng bản mới.
2. **HuggingFace Hub `< 1.0`**: SpeechBrain 1.0.3 vẫn truyền tham số `use_auth_token`, tham số này đã bị loại bỏ ở `huggingface-hub >= 1.0`.
3. **NumPy `< 2.3`**: Đảm bảo tính tương thích với C-extensions của SciPy và Librosa.

---

## 5. Bảng tổng kết hai bộ đặc trưng

| Tiêu chí                         | Bộ thuộc tính Giọng nói (Voice Features)        | Bộ thuộc tính Nội dung (Content Features)   |
|:-------------------------------- |:----------------------------------------------- |:------------------------------------------- |
| **Đặc trưng chính (Primary)**    | **ECAPA-TDNN Embedding**                        | **Sentence-Transformers Embedding**         |
| **Mô hình trích xuất**           | `speechbrain/spkrec-ecapa-voxceleb`             | `all-MiniLM-L6-v2` (sau Whisper ASR)        |
| **Kích thước vector**            | **192 chiều** (Vector float32)                  | **384 chiều** (Vector float32)              |
| **Chuẩn hóa**                    | $L_2$ Normalized ($\|v\|_2 = 1.0$)              | $L_2$ Normalized ($\|v\|_2 = 1.0$)          |
| **Đặc trưng bổ trợ (Auxiliary)** | MFCC stats (78-D), Prosody (F0, ZCR, Spectral)  | Full Transcript, số lượng chunks            |
| **Độ đo tương đồng**             | Cosine Similarity (Dot product)                 | Cosine Similarity (Dot product)             |
| **Ý nghĩa trích xuất**           | Nhận diện âm sắc, đặc trưng sinh học thanh quản | Nhận diện ý nghĩa, ngữ cảnh, chủ đề bài nói |
| **Lưu trữ CSDL đích**            | PostgreSQL `pgvector`: `vector(192)`            | PostgreSQL `pgvector`: `vector(384)`        |

---

## 6. Trạng thái hiện tại & Kế hoạch giai đoạn tiếp theo

### Trạng thái hiện tại

- Đã xử lý và kiểm thử thành công trên 6 file mẫu trong `data/`.
- Đã sinh đầy đủ các file âm thanh chuẩn hóa trong `artifacts/wav/`.
- Đã lưu trữ đầy đủ 6 cặp `voice-*.json` và `content-*.json`.
- Đã tổng hợp thành công `artifacts/features/manifest.csv`.
- Các bài kiểm thử trong `tests/` đều vượt qua (`passed`).

### Các giai đoạn tiếp theo theo đề bài

- **Giai đoạn 3 (Thu thập đủ $\ge 500$ file & Xây dựng CSDL)**:
  - Mở rộng tập dữ liệu lên $\ge 500$ file (từ TED-LIUM 3, MIT OCW, LibriVox).
  - Chạy batch pipeline để trích xuất đặc trưng toàn bộ 500 file.
  - Khởi tạo cơ sở dữ liệu **PostgreSQL + pgvector**, tạo bảng `audio_files`, `voice_features`, `content_features` và đánh chỉ mục HNSW/IVFFlat.
  - Chạy script nạp dữ liệu từ `manifest.csv` vào PostgreSQL.
- **Giai đoạn 4 (Hệ thống tìm kiếm)**:
  - Viết dịch vụ tìm kiếm nhận vào 1 file âm thanh mới.
  - Chạy pipeline trích xuất nhanh file mới $\rightarrow$ thực hiện 2 truy vấn SQL Cosine Distance (`<=>`) để lấy Top-3 file giọng nói giống nhất và Top-3 file nội dung giống nhất.
- **Giai đoạn 5 (Giao diện Demo)**:
  - Xây dựng giao diện web (Streamlit / FastAPI) cho phép người dùng tải lên file âm thanh và hiển thị kết quả trực quan (nghe thử audio, hiển thị điểm tương đồng, trích đoạn transcript và biểu đồ đặc trưng).
