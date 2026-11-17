# Tổng hợp thuật toán trong pipeline hiện tại (Giai đoạn 1 & 2)

> Tài liệu phục vụ viết báo cáo: liệt kê **tên thuật toán / mô hình**, **nguyên lý rút gọn**, và **cách thuật toán được sử dụng trong project** (file code, tham số, đầu ra).
> Pipeline hiện tại: **thu thập & tiền xử lý âm thanh** (GĐ1) → **trích 2 bộ thuộc tính: giọng nói + nội dung** (GĐ2).
> Nguồn code: `src/audio_search/` — tham số tập trung trong `config.py`.

---

## 0. Sơ đồ pipeline

```
data/ (mp3/wav/m4a/flac — diễn thuyết / bài giảng / sách nói)
   │
   ▼  [preprocess.py]  giải mã → mono → resample 16 kHz → peak-normalize
artifacts/wav/*.wav
   │
   ├──► [voice_features.py]   BỘ THUỘC TÍNH A — GIỌNG NÓI
   │      ├─ MFCC (26 hệ số, 25 ms/10 ms) + Δ + ΔΔ  → mean over time → mfcc_stats (78-D)
   │      ├─ Prosody/phổ: pyin (F0), ZCR, spectral centroid, spectral rolloff
   │      └─ ECAPA-TDNN (SpeechBrain) → ecapa_embedding (192-D, L2-normalized)
   │
   └──► [content_features.py] BỘ THUỘC TÍNH B — NỘI DUNG
          ├─ faster-whisper small (CTranslate2, int8, beam=5, VAD) → transcript
          └─ chunk câu → all-MiniLM-L6-v2 → mean-pool → semantic_embedding (384-D, L2-normalized)
   │
   ▼  [feature_store.py]  voice-<id>.json + content-<id>.json + manifest.csv
```

---

## 1. Giai đoạn 1 — Thu thập & tiền xử lý âm thanh

### 1.1. Peak Normalization (chuẩn hóa biên độ đỉnh)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Peak normalization                                                                                                                                                                                                                                                                                 |
| **Nguyên lý**                  | Tìm giá trị tuyệt đối lớn nhất trong tín hiệu, chia toàn bộ tín hiệu cho nó rồi nhân với hệ số đích để mẫu loudest đạt đúng biên độ `peak`.                                                                                                                                                        |
| **Cách sử dụng trong project** | `preprocess.py:_normalize()` — chuẩn hóa mọi file về `PEAK_NORM = 0.95`. Áp dụng **sau** khi downmix và resample, trước khi ghi WAV cache. Mục đích: chuẩn hóa âm lượng đầu vào giữa các nguồn dữ liệu khác nhau (TED, OCW, LibriVox) để các mô hình downstream nhận tín hiệu có độ lớn nhất quán. |

### 1.2. Downmix đa kênh → mono (channel averaging)

| Hạng mục                       | Nội dung                                                                                                               |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Channel averaging / downmix                                                                                            |
| **Nguyên lý**                  | Trung bình cộng các kênh (stereo…) theo trục thời gian → 1 kênh duy nhất.                                              |
| **Cách sử dụng trong project** | `preprocess.py:_to_mono()` — `wave.mean(axis=1)`. Mọi model downstream (ECAPA, Whisper, librosa) yêu cầu đầu vào mono. |

### 1.3. Resampling (tái lấy mẫu)

| Hạng mục                       | Nội dung                                                                                                                                                                                                     |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tên thuật toán**             | Digital resampling (band-limited sinc interpolation qua librosa / soxr)                                                                                                                                      |
| **Nguyên lý**                  | Chuyển đổi tín hiệu từ sampling rate gốc `orig_sr` sang `target_sr` bằng bộ lọc anti-aliasing + nội suy.                                                                                                     |
| **Cách sử dụng trong project** | `preprocess.py:_resample()` — `librosa.resample(orig_sr → SAMPLE_RATE=16000)`. Mọi model trong pipeline (ECAPA-TDNN, Whisper, toàn bộ trích đặc trưng cổ điển) được huấn luyện / kỳ vọng đầu vào **16 kHz**. |

### 1.4. Giải mã audio (soundfile + ffmpeg fallback)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                          |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên công cụ**                | libsndfile (qua `soundfile`) với fallback ffmpeg → raw PCM `f32le`                                                                                                                                                                                                                |
| **Cách sử dụng trong project** | `preprocess.py:load_audio()` — thử `sf.read` trước; nếu exception (codec không hỗ trợ, ví dụ một số mp3/m4a) thì spawn `ffmpeg -f f32le -ac 1 -ar 16000` đọc qua stdout. File được cache thành `artifacts/wav/<file_id>.wav` (resume-safe: file đã tồn tại thì bỏ qua xử lý lại). |

### 1.5. Kiểm định thời lượng (duration validation)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                      |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Ngưỡng lọc theo điều kiện (threshold filter)                                                                                                                                                                                                  |
| **Cách sử dụng trong project** | `preprocess.py:validate_duration()` — chấp nhận file trong khoảng `MIN_DURATION_S = 10s` … `MAX_DURATION_S = 1800s` (30 phút); ngoài khoảng bị đánh dấu skipped. File quá ngắn thường không đủ đặc trưng, file quá dài gây tốn kém tính toán. |

---

## 2. Giai đoạn 2A — Bộ thuộc tính GIỌNG NÓI (`voice_features.py`)

### 2.1. MFCC + Delta + Delta-delta (đặc trưng cổ điển, `librosa`)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Mel-Frequency Cepstral Coefficients (MFCC) + hệ số động Δ, ΔΔ                                                                                                                                                                                                                                                                                                                                      |
| **Nguyên lý**                  | (1) Chia tín hiệu thành khung 25 ms, hop 10 ms → FFT → phổ năng lượng; (2) chiếu qua **mel filterbank** (80 filters) để mô phỏng độ nhạy thính giác người; (3) lấy log; (4) **DCT-II** (ortho-norm) để nén thành các hệ số độc lập — 26 MFCC. Hệ số động Δ (đạo hàm thời gian theo công thức sai phân hữu hạn) và ΔΔ capturing **độ biến thiên và gia tốc của phổ** (nhịp điệu nói, cách nhả chữ). |
| **Cách sử dụng trong project** | `voice_features.py:compute_mfcc_stats()` — stack `[mfcc; Δ; ΔΔ]` → ma trận (78, T); **mean over time** → vector `mfcc_stats` **78-D** (bộ thuộc tính cổ điển chính, dùng cho so sánh / baseline và báo cáo). Ma trận thô và std cũng được lưu phục vụ trực quan hóa. Tham số: `n_mfcc=26, n_fft=400, hop=160, win=400, n_mels=80, dct_type=2, norm="ortho"` (`config.py`).                         |

### 2.2. Trích suất F0 bằng pYIN (prosody)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | pYIN (probabilistic YIN)                                                                                                                                                                                                                        |
| **Nguyên lý**                  | Biến thể xác suất của thuật toán YIN ước lượng **nhịp cơ bản (F0)**: dùng difference function + cumulative mean normalized difference, sau đó HMM/Viterbi xác định biên giới voiced/unvoiced/silence và smoothing — cho F0 ổn định hơn YIN gốc. |
| **Cách sử dụng trong project** | `voice_features.py:compute_prosody()` — `librosa.pyin(wave, fmin=60, fmax=400)`. Từ các frame **voiced** tính `f0_mean_hz`, `f0_std_hz` — đặc trưng cốt lõi của ngữ điệu / cao độ giọng nói (phân biệt nam/nữ, cao thấp giọng).                 |

### 2.3. Zero-Crossing Rate (ZCR)

| Hạng mục                       | Nội dung                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Zero-crossing rate                                                                                                                                      |
| **Nguyên lý**                  | Tỷ lệ số lần tín hiệu đổi dấu (âm ↔ dương) trên khung thời gian — xấp xỉ tần suất trung bình của tín hiệu, cao ở noise/fricatives, thấp ở voiced.       |
| **Cách sử dụng trong project** | `voice_features.py:compute_prosody()` — `librosa.feature.zero_crossing_rate(wave).mean()`, dùng làm đặc trưng bổ sung mô tả "texture" âm sắc của giọng. |

### 2.4. Spectral Centroid & Spectral Rolloff

| Hạng mục                       | Nội dung                                                                                                                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Tên thuật toán**             | Spectral centroid, spectral rolloff (thống kê phổ)                                                                                                                                                                 |
| **Nguyên lý**                  | **Centroid**: trung tâm năng lượng của phổ (độ "sáng" âm thanh). **Rolloff (85%)**: tần suất dưới đó chứa 85% tổng năng lượng phổ — đo độ giàu高频 / độ rõ âm của giọng.                                             |
| **Cách sử dụng trong project** | `voice_features.py:compute_prosody()` — `librosa.feature.spectral_centroid` và `spectral_rolloff(roll_percent=0.85)`, lấy mean toàn file. Bổ sung cho MFCC/ECAPA khi mô tả đặc trưng âm sắc cố định của người nói. |

### 2.5. ECAPA-TDNN (deep speaker embedding — đặc trưng chính)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán / mô hình**   | **ECAPA-TDNN** (Emphasized Channel Attention, Propagation and Aggregation in TDNN) — mô hình nhận dạng người nói (speaker verification), pretrained `speechbrain/spkrec-ecapa-voxceleb` trên VoxCeleb                                                                                                                                                                                                                                    |
| **Nguyên lý**                  | Mạng neural深度: nâng cấp x-vector bằng **SE-attention theo kênh** (tập trung kênh quan trọng), **multi-layer feature aggregation** (gộp đặc trưng nhiều tầng), và **channel-dependent frame-level attention pooling** (gộp theo frame có trọng số thay vì mean thuần) → đầu ra embedding 192-D đại diện "giọng nói" của người phát ngôn bất biến với nội dung. Embedding được **L2-normalize** để khoảng cách Euclid ↔ cosine similarity. |
| **Cách sử dụng trong project** | `voice_features.py:compute_ecapa_embedding()` — model load 1 lần (`lru_cache`), auto chọn CUDA/CPU; input waveform float32 mono 16 kHz → `classifier.encode_batch()` → vector **192-D L2-normalized** = `ecapa_embedding`, **đặc trưng giọng nói primary** dùng để tìm kiếm top-3 "giống giọng" ở GĐ4 (so sánh cosine/cosine distance qua pgvector). Ghi chú kỹ thuật: `LocalStrategy.COPY` để tránh lỗi symlink trên Windows.           |

---

## 3. Giai đoạn 2B — Bộ thuộc tính NỘI DUNG (`content_features.py`)

### 3.1. Nhận dạng giọng nói (ASR) — faster-whisper

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán / mô hình**   | **Whisper** (encoder-decoder Transformer của OpenAI) qua thư viện **faster-whisper** (CTranslate2), model `small`, lượng tử `int8`, chạy CPU (đổi được qua env)                                                                                                                                                                                                                              |
| **Nguyên lý**                  | Whisper chuyển audio thành log-Mel spectrogram (80 mel bins, 25 ms window) rồi dùng **Transformer encoder–decoder** sinh văn bản; decoder autoregressive có thể dùng **beam search**. Tích hợp **Voice Activity Detection (VAD)** để bỏ khoảng lặng trước khi nhận dạng, tăng chất lượng transcript.                                                                                         |
| **Cách sử dụng trong project** | `content_features.py:transcribe_audio()` — tham số `beam_size=5`, `vad_filter=True` (ngưỡng `min_silence_duration_ms=500`), `condition_on_previous_text=False` (giảm lỗi hallucination trên file dài), `language="en"`. Transcript là **đầu vào trực tiếp** của bước embedding ngữ nghĩa; nếu dataset đã có transcript (TED-LIUM STM, LibriVox text) thì truyền thẳng `text=` để bỏ qua ASR. |

### 3.2. Ngắt văn bản theo câu (sentence chunking)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                 |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Rule-based sentence splitting (regex) + cửa sổ ký tự                                                                                                                                                                                     |
| **Nguyên lý**                  | Tách transcript thành các chunk theo dấu `.`/`!`/`?`/xuống dòng (`re` lookbehind), gộp đến tối đa `max_tokens=250` token đếm bằng tokenizer của model (giới hạn cứng 256 của MiniLM; câu tràn mở đoạn mới, câu quá dài tách theo từ; bỏ mẩu < 40 ký tự) — để mỗi embedding đại diện một ý hoàn chỉnh, tránh đầu vào quá dài cho sentence-encoder. |
| **Cách sử dụng trong project** | `content_features.py:chunk_to_sentences()` — chuẩn bị đầu vào cho `embed_content()`; số chunk (`n_chunks`) được ghi vào artifact để kiểm chứng.                                                                                          |

### 3.3. Sentence Embedding — Sentence-BERT (all-MiniLM-L6-v2)

| Hạng mục                       | Nội dung                                                                                                                                                                                                                                                                                                         |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán / mô hình**   | **Sentence-BERT (SBERT)** — mô hình sentence-transformers `all-MiniLM-L6-v2` (Transformer BERT-độ nhỏ, 6 tầng, hidden 384)                                                                                                                                                                                       |
| **Nguyên lý**                  | Khác BERT mean-pool thô, SBERT fine-tune để vector câu có **khoảng cách ngữ nghĩa có ý nghĩa**: mỗi chunk câu → vector 384-D (mean-pooling tokens) đã normalize; hai câu cùng ý ở gần nhau trong không gian embedding, dùng **cosine similarity** để so khớp.                                                    |
| **Cách sử dụng trong project** | `content_features.py:embed_content()` — encode K chunk → ma trận (K, 384), **mean-pool** theo trục chunk → vector **384-D**, L2-normalize lần cuối = `semantic_embedding`, **đặc trưng nội dung primary** dùng để tìm top-3 "giống nội dung" ở GĐ4. Bản ghi không có text → vector 0 (đánh dấu để caller xử lý). |

### 3.4. Mean-pooling toàn file (aggregate embedding)

| Hạng mục                       | Nội dung                                                                                                                                                                                        |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tên thuật toán**             | Mean pooling + L2 normalization                                                                                                                                                                 |
| **Nguyên lý**                  | Trung bình vector các chunk rồi chuẩn hóa về chuẩn Euclid = 1 → một vector duy nhất đại diện ngữ nghĩa cả file, bất biến thứ tự độ dài; L2-norm để cosine similarity = dot product.             |
| **Cách sử dụng trong project** | Áp dụng cho semantic embedding (`embed_content`) và ECAPA embedding (`compute_ecapa_embedding`) — cả hai đều trả về vector đơn vị để so sánh trực tiếp bằng inner product trong pgvector ở GĐ4. |

---

## 4. Tổ chức & lưu trữ đặc trưng (`feature_store.py`) — phục vụ CSDL GĐ3

| Thành phần            | Cách hoạt động                                                                                                                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Artifact per-file** | Mỗi file âm thanh → 2 JSON: `voice-<file_id>.json` (mfcc_stats 78-D, ecapa 192-D, prosody, matrix) và `content-<file_id>.json` (transcript, semantic 384-D, n_chunks), gắn `feature_version = "2.0"` để resume/invalidate.                                          |
| **Manifest**          | `build_manifest()` gộp mọi artifact → `manifest.csv`: mỗi hàng = `file_id, filename, duration_s, mfcc_stats(78), ecapa_embedding(192), semantic_embedding(384), transcript` (vector dạng JSON string) — đầu vào sẵn cho ETL nạp PostgreSQL + **pgvector** ở GĐ3–G4. |

---

## 5. Bảng tổng hợp nhanh (cho slide / mục báo cáo)

| #   | Thuật toán / mô hình                              | Nhóm                | Vai trò trong pipeline               | Đầu ra           | Code                  |
| --- | ------------------------------------------------- | ------------------- | ------------------------------------ | ---------------- | --------------------- |
| 1   | Peak normalization                                | Tiền xử lý          | Chuẩn hóa biên độ                    | wav 0.95 peak    | `preprocess.py`       |
| 2   | Downmix mono                                      | Tiền xử lý          | Gộp đa kênh                          | mono             | `preprocess.py`       |
| 3   | Resampling (sinc/soxr)                            | Tiền xử lý          | Về 16 kHz cho mọi model              | 16 kHz           | `preprocess.py`       |
| 4   | MFCC + Δ + ΔΔ (DCT-II, mel filterbank)            | Giọng nói — cổ điển | Đại diện phổ-cepstral                | 78-D             | `voice_features.py`   |
| 5   | pYIN                                              | Giọng nói — cổ điển | Ước lượng F0 (ngữ điệu)              | f0 mean/std (Hz) | `voice_features.py`   |
| 6   | ZCR / Spectral centroid / rolloff                 | Giọng nói — cổ điển | Đặc trưng âm sắc                     | 3 số thực        | `voice_features.py`   |
| 7   | **ECAPA-TDNN** (SE-attention + attention pooling) | Giọng nói — deep    | **Speaker embedding (primary)**      | 192-D L2         | `voice_features.py`   |
| 8   | **Whisper** (transformer ASR) + VAD + beam search | Nội dung            | Âm thanh → văn bản                   | transcript       | `content_features.py` |
| 9   | Sentence chunking (regex)                         | Nội dung            | Chuẩn bị đầu vào embedding           | K chunk          | `content_features.py` |
| 10  | **Sentence-BERT all-MiniLM-L6-v2**                | Nội dung            | **Semantic embedding (primary)**     | 384-D L2         | `content_features.py` |
| 11  | Mean-pooling + L2-norm                            | Cả hai              | Gộp chunk / chuẩn hóa vector so sánh | vector đơn vị    | cả hai module         |

**Cặp đặc trưng primary cho GĐ4:** `ecapa_embedding` (192-D, cosine) cho tìm kiếm theo **giọng nói** · `semantic_embedding` (384-D, cosine) cho tìm kiếm theo **nội dung**; `mfcc_stats` + prosody là đặc trưng cổ điển bổ trợ / giải thích / baseline.
