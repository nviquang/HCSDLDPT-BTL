# HCSDLDPT BTL — Hệ CSDL lưu trữ và tìm kiếm âm thanh theo nội dung

Bài tập lớn môn **Hệ CSDL đa phương tiện và ứng dụng** (HCSDLDPT).
Giai đoạn 2 hiện tại: **xây dựng 2 bộ thuộc tính (giọng nói + nội dung)** cho bộ dữ liệu âm thanh.

## Cài đặt

```powershell
# Yêu cầu: uv (https://docs.astral.sh/uv/), Python 3.12
uv sync
```

## Chạy pipeline

```powershell
uv run python -m audio_search preprocess   # data/ → artifacts/wav/ (16 kHz mono)
uv run python -m audio_search voice        # giọng → artifacts/features/voice-*.json
uv run python -m audio_search content      # nội dung → artifacts/features/content-*.json
uv run python -m audio_search all          # cả 3 + manifest
uv run python -m audio_search manifest     # gom thành manifest.csv (cho G4 pgvector)
uv run python -m audio_search verify       # smoke test các đặc trưng đã trích
```

Các model (ECAPA, faster-whisper small, MiniLM) tự tải lần đầu. Muốn tải trước:

```powershell
uv run python scripts/download_models.py
```

Xem thêm: **`plan.md`** (kế hoạch & tài nguyên giai đoạn 2) và **`report.md`** (kế hoạch dự án).