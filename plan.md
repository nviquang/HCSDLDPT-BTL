# Kế hoạch Giai đoạn 2 · Xây dựng 2 bộ thuộc tính cho bộ dữ liệu âm thanh

> Dự án: **Hệ CSDL lưu trữ và tìm kiếm âm thanh theo nội dung** (HCSDLDPT · bài tập lớn).
> Giai đoạn 2 — xây dựng 2 bộ thuộc tính: **giọng nói** và **nội dung**.
> Người thực hiện: nhóm 3 người (P1/P2/P3). Trạng thái: **đang triển khai**.

---

## 1. Mục tiêu giai đoạn

Xây dựng 2 bộ thuộc tính (đặc trưng) cho mỗi file âm thanh trong bộ dữ liệu:

| Bộ                         | Module                | Đặc trưng                          | Chiều | Công cụ                             |
| -------------------------- | --------------------- | ---------------------------------- | ----- | ----------------------------------- |
| **A. Giọng nói**           | `voice_features.py`   | **ECAPA-TDNN embedding** (primary) | 192   | SpeechBrain `spkrec-ecapa-voxceleb` |
| **A. Giọng nói (cổ điển)** | `voice_features.py`   | MFCC(+Δ/ΔΔ) + prosody/phổ          | 78    | librosa                             |
| **B. Nội dung**            | `content_features.py` | **Semantic embedding** (primary)   | 384   | faster-whisper → `all-MiniLM-L6-v2` |

Đầu ra:

1. **Môi trường ảo bằng `uv`** (đã tạo)
2. **Tài nguyên cần chuẩn bị** (models, data, hệ thống)
3. **2 module trích đặc trưng** — chạy được trên 6 file mẫu, **batch-ready** cho 500 file
4. **Các artifact đặc trưng** định dạng sẵn để G4 nạp PostgreSQL + pgvector

> Không nằm trong giai đoạn này (thuộc G4–G6): PostgreSQL/pgvector, FastAPI, Streamlit, TF-IDF baseline.

---

## 2. Tổng quan kiến trúc

```
data/ (6 mẫu → 500 file)
   │
   ▼
preprocess.py ── 16 kHz mono, normalize
   │
   ├──► voice_features.py
   │      ├─ MFCC 26 (25ms/10ms) + Δ + ΔΔ → mean → mfcc_stats (78)
   │      ├─ prosody: F0, ZCR, spectral centroid/rolloff
   │      └─ ECAPA-TDNN → ecapa_embedding (192, L2-normalized)
   │
   └──► content_features.py
          ├─ faster-whisper small (int8) → transcript
          └─ chunk câu → all-MiniLM-L6-v2 → mean-pool → semantic (384)
   │
   ▼
feature_store.py ── 1 JSON + 1 JSON/file + manifest.csv (cho G4)
```

---

## 3. Môi trường (đã tạo, vận hành bằng `uv`)

### 3.1. Khởi tạo

```powershell
cd C:\Users\nviquang\Documents\University\FourthYear\HCSDLDPT\BTL
uv init --package --python 3.12
uv python pin 3.12
```

Package đã đổi tên: `src/btl` → `src/audio_search`.

### 3.2. Dependencies (pyproject.toml)

| Gói                                                               | Phiên bản (pin)       | Vì sao                                                                                                          |
| ----------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------- |
| `torch`                                                           | **2.8.0** (CPU wheel) | cực kỳ quan trọng: torchaudio ≥ 2.9 **bỏ** `list_audio_backends()` mà SpeechBrain gọi → crash. Pin 2.8 để tránh |
| `torchaudio`                                                      | **2.8.0**             | khớp torch 2.8, còn `list_audio_backends()`                                                                     |
| `speechbrain`                                                     | 1.0.3                 | ECAPA-TDNN                                                                                                      |
| `huggingface-hub`                                                 | ≥0.20, **<1.0**       | SpeechBrain 1.0.3 vẫn truyền `use_auth_token` → bị gỡ ở hub ≥1.0                                                |
| `faster-whisper`                                                  | 1.2.1                 | ASR (CTranslate2)                                                                                               |
| `sentence-transformers`                                           | 3.4.1                 | semantic embedding                                                                                              |
| `librosa`                                                         | ≥0.10, <0.12          | MFCC, phổ                                                                                                       |
| `numpy`                                                           | **<2.3**              | tránh breaking change                                                                                           |
| `scikit-learn`, `scipy`, `pandas`, `pyarrow`, `soundfile`, `tqdm` | latest                | cos sim, manifest, đọc audio                                                                                    |

**Cập nhật:** `pyproject.toml` hiện dùng torch **CUDA 12.8** (`pytorch-cu128`); `config.resolve_device()` tự rơi về CPU khi không có GPU.

**Lưu ý CPU vs GPU (bản gốc):** mặc định **CPU** (torch CPU wheel từ `https://download.pytorch.org/whl/cpu`, đã khai báo trong `pyproject.toml` qua `[tool.uv.sources]`). GPU (GTX 1650 4GB) chỉ bật khi cần tăng tốc batch lớn:

```powershell
uv remove torch torchaudio
uv add torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3.3. Chạy

```powershell
uv sync                      # tạo .venv + cài deps
uv run python -m audio_search preprocess   # data → artifacts/wav/
uv run python -m audio_search voice        # giọng → features/voice-*.json
uv run python -m audio_search content      # nội dung → features/content-*.json
uv run python -m audio_search all          # chạy cả pipeline (resume-safe)
uv run python -m audio_search manifest     # gom → features/manifest.csv
uv run python -m audio_search verify       # smoke test
```

---

## 4. Tài nguyên cần chuẩn bị

### 4.1. Models (tải tự động, lưu vào cache HF)

| Model                | ID                                       | Mục đích         | Dung lượng |
| -------------------- | ---------------------------------------- | ---------------- | ---------- |
| ECAPA-TDNN           | `speechbrain/spkrec-ecapa-voxceleb`      | giọng → 192-D    | ~70 MB     |
| faster-whisper small | `Systran/faster-whisper-small`           | ASR              | ~250 MB    |
| MiniLM-L6-v2         | `sentence-transformers/all-MiniLM-L6-v2` | nội dung → 384-D | ~90 MB     |

**Tổng ~410 MB.** Đặt dưới `models/` nếu muốn tải trước:

```powershell
uv run python scripts/download_models.py
```

(Tự tải khi chạy lần đầu nếu chưa có — HF cache mặc định ở `~/.cache/huggingface`.)

### 4.2. Dữ liệu

| Khoản                        | Tình trạng                                                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| 6 file mp3 TED mẫu (`data/`) | ✅ có                                                                                                                             |
| Bộ 500+ file lời nói (G1)    | đang triển khai; khi xong: bỏ vào `data/`, chạy `uv run python -m audio_search all` — pipeline tự nhận, resume, bỏ qua file hỏng |

> ⚠️ 6 file mẫu hiện là **màn trình diễn âm nhạc** (bài hát TED) — phù hợp để **kiểm chứng pipeline**; kết quả tìm kiếm **ý nghĩa** cần bổ sung file lời nói.

### 4.3. Hệ thống

| Hạng mục   | Giá trị      | Ghi chú                                    |
| ---------- | ------------ | ------------------------------------------ |
| OS         | Windows 11   | —                                          |
| uv         | 0.11.12      | đã cài                                     |
| Python     | 3.12.10      | uv pin 3.12                                |
| git        | 2.53         | (repo GitHub chưa tạo — nên tạo sớm)       |
| ffmpeg     | 8.1          | đã có, dùng fallback decode                |
| GPU        | GTX 1650 4GB | tùy chọn; mặc định CPU                     |
| RAM        | 15.3 GB      | load model tuần tự, không cùng lúc 3 model |
| Disk trống | 47 GB        | đủ                                         |

---

## 5. Nguyên lý trích rút 2 bộ thuộc tính

### 5.1. Giọng nói

**MFCC (cepstral):** âm thanh → cửa sổ 25 ms, hop 10 ms → thang Mel (gần cảm nhận thính giác) → log → DCT-II → **26 hệ số MFCC**. Lấy thêm Δ (delta) và ΔΔ (delta-delta) để bắt đặc trưng động. **MFCC miêu tả hình dạng đường âm thanh (vocal tract)** của người nói — mỗi người có formant riêng.

- `mfcc_stats` (78-D) = mean theo thời gian của 78 hệ số (26 MFCC + 26 Δ + 26 ΔΔ) → vector đại diện cả file.

**Prosody/phổ:** pitch trung bình (F0, nam/nữ), zero-crossing rate, spectral centroid & rolloff.

**ECAPA-TDNN (primary):** mạng CNN residual + **attentive statistical pooling**, huấn luyện trên VoxCeleb với Additive Margin Softmax. Đầu ra **voiceprint 192-D** đã chuẩn hóa L2. **Cosine similarity** giữa 2 embedding đo độ giống giọng.

### 5.2. Nội dung

**Whisper (ASR):** Transformer seq2seq, huấn luyện ~680k giờ đa ngữ → chuyển **giọng nói → văn bản** (transcript). Đây là cầu nối "âm thanh → nội dung chữ".

**Semantic embedding (primary):** `all-MiniLM-L6-v2` (384-D) — mapper văn bản → vector sao cho **các đoạn cùng chủ đề gần nhau**. Vì transcript có thể rất dài, cắt thành **chunk theo câu** (≤ 250 token theo tokenizer của model — giới hạn cứng của all-MiniLM-L6-v2 là 256), embed từng chunk, **mean-pool** → vector đại diện cả file.

**Cách dùng (G5):** file query → trích 2 vector → `ORDER BY emb <=> :q` (cosine, pgvector) → top-3 giọng + top-3 nội dung, giảm dần độ tương đồng.

---

## 6. Cấu trúc code

```
BTL/
├── data/                      # 6 mẫu → 500 file
├── src/audio_search/
│   ├── config.py              # hằng số: SR, MFCC params, model IDs, paths
│   ├── preprocess.py          # load → 16 kHz mono → normalize → cache wav
│   ├── voice_features.py      # MFCC stats + prosody + ECAPA 192-D
│   ├── content_features.py    # faster-whisper ASR + MiniLM 384-D
│   ├── feature_store.py       # ghi artifact/file + manifest.csv
│   ├── verify.py              # smoke test
│   ├── cli.py + __main__.py   # CLI
├── scripts/download_models.py # pre-download 3 models
├── artifacts/                 # đầu ra: wav/, features/, logs/ (gitignore)
├── pyproject.toml · uv.lock · .python-version
└── plan.md                    # tài liệu này
```

---

## 7. Phân công công việc (P1/P2/P3)

| Người  | Công việc                                                       | Khối lượng |
| ------ | --------------------------------------------------------------- | ---------- |
| **P1** | Env uv + `preprocess.py` + `config.py` + `cli.py`               | 2–3 ngày   |
| **P2** | `voice_features.py` (MFCC + ECAPA) + `verify.py`                | 3–4 ngày   |
| **P3** | `content_features.py` (Whisper + MiniLM) + `plan.md` + manifest | 3–4 ngày   |

Sau đó: join + `all` + `verify` + thuyết minh nguyên lý cho báo cáo.

---

## 8. Kiểm chứng (trên 6 file mẫu)

| Kiểm tra              | Kỳ vọng                            | Lệnh/Module            |
| --------------------- | ---------------------------------- | ---------------------- |
| Preprocess            | 6/6 WAV 16 kHz mono                | `ffprobe`/`preprocess` |
| `mfcc_stats`          | chiều 78                           | `verify`               |
| `ecapa_embedding`     | 192-D, L2 ≈ 1.0                    | `verify`               |
| `semantic_embedding`  | 384-D, L2 ≈ 1.0                    | `verify`               |
| Transcript            | ≥ 3 từ/file                        | `verify`               |
| Top-3 voice & content | tự tương đồng = 1.0; điểm giảm dần | `verify`               |
| manifest.csv          | 6 dòng, vector đủ                  | `python -c`            |

---

## 9. Rủi ro & giảm thiểu

| Rủi ro                            | Ảnh hưởng       | Giảm thiểu                                                        |
| --------------------------------- | --------------- | ----------------------------------------------------------------- |
| SpeechBrain crash trên torch mới  | import lỗi      | **Pin torch==2.8 + torchaudio==2.8 + hub<1.0** (đã xác minh chạy) |
| Lỗi symlink Windows khi tải model | không tải được  | `from_hparams(local_strategy="copy")`                             |
| Whisper kém trên file hát         | transcript lỗi  | Ghi log; bổ sung file lời nói cho đánh giá thật                   |
| VRAM 4GB chạy hết                 | OOM             | Mặc định CPU; int8; `--device cpu`                                |
| Batch 500 file chậm trên CPU      | thiếu thời gian | ECAPA ~2–3s/file; Whisper int8 ~1–2x realtime; ước 2–4h GPU       |
| RAM 15GB thấp khi load 3 model    | crash           | load tuần tự, từng module giải phóng                              |

---

## 10. Định hướng G4 (nạp PostgreSQL + pgvector)

Các vector đã lưu đúng chiều để ETL trực tiếp:

```sql
-- voice:  mfcc_stats vector(78), ecapa_embedding vector(192)
-- content: semantic_embedding vector(384)
-- manifest.csv chứa vector dạng JSON string → CAST thành array khi INSERT
CREATE INDEX idx_voice   ON voice_features   USING hnsw (ecapa_embedding vector_cosine_ops);
CREATE INDEX idx_content ON content_features USING hnsw (semantic_embedding vector_cosine_ops);
```

---

## 11. Tài liệu tham khảo

- report.md (gồm đầy đủ §4 nguyên lý, §5 kiến trúc, §6 schema CSDL)
- SpeechBrain ECAPA: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Sentence-Transformers MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- librosa MFCC: https://librosa.org/doc/latest/generated/librosa.feature.mfcc.html
- pgvector: https://github.com/pgvector/pgvector