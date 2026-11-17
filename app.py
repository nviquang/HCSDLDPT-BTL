"""Demo — hệ thống tìm kiếm âm thanh theo nội dung và giọng nói.

Chạy:  uv run streamlit run app.py
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from audio_search import config, database, search

st.set_page_config(page_title="Tìm kiếm âm thanh", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang nạp CSDL đặc trưng…")
def get_index() -> database.VectorIndex:
    return database.load_index()


@st.cache_resource(show_spinner="Đang nạp các mô hình (ECAPA, Whisper, MiniLM)…")
def warm_models() -> bool:
    from audio_search import content_features, voice_features

    voice_features.get_ecapa_model()
    content_features.get_whisper_model()
    content_features.get_semantic_model()
    return True


@st.cache_data
def db_summary() -> dict:
    con = database.connect()
    try:
        s = database.summary(con)
        s["meta"] = {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM meta")}
        return s
    finally:
        con.close()


@st.cache_data
def files_table() -> pd.DataFrame:
    con = sqlite3.connect(str(config.DB_PATH))
    try:
        return pd.read_sql_query(
            "SELECT f.file_id, f.source, f.speaker, f.course, f.topic, ROUND(f.duration_s,1) AS duration_s, "
            "f.is_speech, f.multi_speaker, c.n_chunks, LENGTH(c.transcript) AS transcript_chars, "
            "ROUND(v.f0_mean_hz,1) AS f0_mean_hz "
            "FROM audio_files f JOIN content_features c USING(file_id) JOIN voice_features v USING(file_id) "
            "ORDER BY f.source, f.file_id", con)
    finally:
        con.close()


def short(fid: str, n: int = 70) -> str:
    return fid if len(fid) <= n else fid[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
if not config.DB_PATH.exists():
    st.error("Chưa có CSDL. Chạy `uv run python -m audio_search build-db` trước.")
    st.stop()

summary = db_summary()
th = search.load_thresholds()
with st.sidebar:
    st.header("CSDL đặc trưng")
    c1, c2 = st.columns(2)
    c1.metric("Số file", summary["files"])
    c2.metric("Tổng giờ", summary["total_hours"])
    c1.metric("Đạt chuẩn đề", summary.get("eligible_files", summary["speech_files"]),
              help="Bài nói, một người nói — tham gia tìm kiếm. Đề yêu cầu ≥ 500.")
    c2.metric("Đoạn văn bản", summary["chunks"])
    st.caption(f"SQLite · {config.DB_PATH.name} · dựng lúc {summary['meta'].get('built_at', '?')}")
    st.divider()
    st.subheader("Mô hình")
    st.markdown(
        f"- Giọng nói: `{summary['meta'].get('ecapa_model')}` (192-D)\n"
        f"- ASR: faster-whisper `{summary['meta'].get('whisper_model')}`\n"
        f"- Nội dung: `{summary['meta'].get('semantic_model')}` (384-D)\n"
        f"- Thiết bị: `{config.resolve_device()}`"
    )
    st.subheader("Ngưỡng (hiệu chỉnh)")
    st.markdown(f"- Nội dung có trong CSDL: cos ≥ **{th['content_known']:.3f}**\n"
                f"- Người nói có trong CSDL: cos ≥ **{th['voice_known']:.3f}**")
    k = st.slider("Số kết quả mỗi danh sách", 3, 10, config.TOP_K)

st.title("🎧 Hệ CSDL lưu trữ và tìm kiếm âm thanh theo nội dung")
st.caption("Đầu vào: một file âm thanh mới · Đầu ra: 3 file giống nhất về **nội dung** "
           "và 3 file giống nhất về **giọng nói**, xếp giảm dần theo độ tương đồng.")

tab_search, tab_flow, tab_db, tab_eval = st.tabs(
    ["🔎 Tìm kiếm", "🧩 Sơ đồ khối", "🗄️ CSDL", "📊 Đánh giá"])

# ---------------------------------------------------------------------------
# Tab: block diagram
# ---------------------------------------------------------------------------
with tab_flow:
    st.graphviz_chart("""
digraph G {
  rankdir=TB; nodesep=0.35; ranksep=0.4; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=11];
  edge [fontname="Helvetica", fontsize=9];
  subgraph cluster_off { label="GIAI ĐOẠN 1 — Lập chỉ mục (offline, 539 file)"; style=dashed; color="#999";
    D [label="Bộ dữ liệu\\n539 file mp3"]; P [label="Tiền xử lý\\nmono · 16 kHz · chuẩn hoá biên độ"];
    V [label="Đặc trưng giọng nói\\nECAPA-TDNN 192-D\\nMFCC 78+78 · prosody"];
    A [label="ASR\\nfaster-whisper small"]; C [label="Đặc trưng nội dung\\nchia đoạn ≤250 token\\nMiniLM 384-D · mean-pool"];
    DB [label="CSDL SQLite\\naudio_files · voice_features\\ncontent_features · content_chunks\\ntranscripts_fts (BM25)", shape=cylinder, fillcolor="#fff4e0"];
    D -> P; P -> V; P -> A; A -> C; V -> DB; C -> DB; A -> DB [label="transcript"];
  }
  subgraph cluster_on { label="GIAI ĐOẠN 2 — Truy vấn (online)"; style=dashed; color="#999";
    Q [label="File âm thanh\\nmới", fillcolor="#e6f6ea"]; QP [label="Tiền xử lý"];
    QV [label="ECAPA 192-D\\n+ MFCC 50-D"]; QC [label="Whisper → MiniLM\\n384-D"];
    SV [label="Cosine với\\nvoice_features.ecapa"]; SC [label="Cosine với\\ncontent_features.semantic"];
    RV [label="Top-3 giọng nói\\n(giảm dần)", fillcolor="#e6f6ea"]; RC [label="Top-3 nội dung\\n(giảm dần)", fillcolor="#e6f6ea"];
    Q -> QP; QP -> QV; QP -> QC; QV -> SV; QC -> SC; SV -> RV; SC -> RC;
  }
  DB -> SV [style=dotted]; DB -> SC [style=dotted];
}
""", width="stretch")
    st.markdown("""
**Quy trình thực hiện**
1. **Tiền xử lý:** đọc file, trộn về mono, lấy mẫu lại 16 kHz, chuẩn hoá biên độ đỉnh 0,95.
2. **Giọng nói:** ECAPA-TDNN (SpeechBrain, huấn luyện trên VoxCeleb) cho vector người nói 192 chiều, chuẩn hoá L2.
   Thêm MFCC c1..c25 (mean + std, z-score theo thống kê toàn CSDL), 50 chiều, làm baseline cổ điển.
3. **Nội dung:** Whisper chuyển giọng nói thành văn bản. Văn bản được chia đoạn theo ranh giới câu, mỗi đoạn ≤ 250 token.
   MiniLM tạo vector cho từng đoạn, rồi lấy trung bình và chuẩn hoá thành một vector 384 chiều. BM25 (FTS5) làm baseline từ khoá.
4. **CSDL:** mọi đặc trưng và metadata được lưu trong SQLite (vector lưu dạng BLOB float32).
5. **Tìm kiếm:** độ tương đồng là cosine (= tích vô hướng của các vector đã chuẩn hoá L2), quét toàn bộ N file.
   Trả về top-k, xếp giảm dần. Nếu điểm top-1 dưới ngưỡng đã hiệu chỉnh, hệ thống báo "chưa có trong CSDL".
""")

# ---------------------------------------------------------------------------
# Tab: database browser
# ---------------------------------------------------------------------------
with tab_db:
    df = files_table()
    c1, c2, c3 = st.columns([1, 1, 2])
    src = c1.multiselect("Nguồn", sorted(df["source"].unique()), default=[])
    only_flag = c2.selectbox("Lọc", ["Tất cả", "Không phải bài nói", "Nhiều người đọc"])
    text = c3.text_input("Tìm theo tên file / người nói")
    view = df
    if src:
        view = view[view["source"].isin(src)]
    if only_flag == "Không phải bài nói":
        view = view[view["is_speech"] == 0]
    elif only_flag == "Nhiều người đọc":
        view = view[view["multi_speaker"] == 1]
    if text:
        mask = view["file_id"].str.contains(text, case=False) | view["speaker"].fillna("").str.contains(text, case=False)
        view = view[mask]
    st.dataframe(view, width="stretch", height=380, hide_index=True)
    st.caption(f"{len(view)} / {len(df)} file")
    cc1, cc2 = st.columns(2)
    cc1.bar_chart(df["source"].value_counts(), horizontal=True)
    cc2.bar_chart(np.histogram(df["duration_s"], bins=20)[0])
    with st.expander("Lược đồ CSDL (SQL)"):
        st.code(database.SCHEMA.strip(), language="sql")
    with st.expander("Truy vấn full-text (FTS5 · BM25) trên transcript"):
        kw = st.text_input("Từ khoá", "photosynthesis OR bacteria")
        if kw:
            con = sqlite3.connect(str(config.DB_PATH))
            try:
                res = pd.read_sql_query(
                    "SELECT file_id, ROUND(-bm25(transcripts_fts),3) AS bm25, "
                    "snippet(transcripts_fts, 1, '[', ']', '…', 12) AS snippet "
                    "FROM transcripts_fts WHERE transcripts_fts MATCH ? ORDER BY bm25(transcripts_fts) LIMIT 10",
                    con, params=(kw,))
                st.dataframe(res, width="stretch", hide_index=True)
            except Exception as exc:  # malformed FTS query
                st.warning(f"Cú pháp truy vấn FTS5 không hợp lệ: {exc}")
            finally:
                con.close()

# ---------------------------------------------------------------------------
# Tab: evaluation
# ---------------------------------------------------------------------------
with tab_eval:
    md = config.EVAL_DIR / "summary.md"
    if md.exists():
        st.markdown(md.read_text(encoding="utf-8"))
        figs = sorted((config.EVAL_DIR / "figures").glob("*.png"))
        for f in figs:
            st.image(str(f), width="stretch")
    else:
        st.info("Chưa có kết quả đánh giá. Chạy `uv run python -m audio_search evaluate`.")

# ---------------------------------------------------------------------------
# Tab: search
# ---------------------------------------------------------------------------
with tab_search:
    samples = {}
    qjson = config.QUERY_DIR / "queries.json"
    if qjson.exists():
        for q in json.loads(qjson.read_text(encoding="utf-8")):
            samples[f"{q['file']} — {q['note']}"] = config.QUERY_DIR / q["file"]

    mode = st.radio("Nguồn truy vấn", ["Tải file lên", "Ghi âm trực tiếp", "File mẫu"], horizontal=True)
    query_path: Path | None = None
    if mode == "Tải file lên":
        up = st.file_uploader("Chọn file âm thanh", type=["wav", "mp3", "m4a", "flac", "ogg", "aac"])
        if up is not None:
            tmp = Path(tempfile.gettempdir()) / f"audio_search_query_{up.name}"
            tmp.write_bytes(up.getvalue())
            query_path = tmp
    elif mode == "Ghi âm trực tiếp":
        rec = st.audio_input("Nhấn để ghi âm (nên nói ≥ 20 giây)")
        if rec is not None:
            tmp = Path(tempfile.gettempdir()) / "audio_search_query_recording.wav"
            tmp.write_bytes(rec.getvalue())
            query_path = tmp
    else:
        if samples:
            choice = st.selectbox("Chọn file mẫu", list(samples))
            query_path = samples[choice]
        else:
            st.info("Chưa có file mẫu. Chạy `uv run python scripts/make_queries.py`.")

    exclude_text = st.text_input("Loại khỏi CSDL khi tìm (file_id, cách nhau bằng dấu ;) — để thử trường hợp 'chưa có'",
                                 "")
    exclude = {s.strip() for s in exclude_text.split(";") if s.strip()}

    if query_path is not None:
        st.audio(str(query_path))
    go = st.button("🔎 Tìm kiếm", type="primary", disabled=query_path is None)

    if go and query_path is not None:
        index = get_index()
        warm_models()
        with st.status("Đang xử lý truy vấn…", expanded=False) as status:
            st.write("Tiền xử lý → ECAPA + MFCC → Whisper → MiniLM → so khớp cosine")
            res = search.search_file(query_path, index, k=k, exclude=exclude)
            status.update(label=f"Xong · {sum(res.query.timings.values()):.1f} s xử lý · "
                                f"so khớp {res.search_ms:.1f} ms", state="complete")
        st.session_state["result"] = res

    res: search.SearchResult | None = st.session_state.get("result")
    if res is not None:
        q = res.query
        st.subheader(f"Kết quả cho: {q.name} ({q.duration_s:.1f} s)")
        col_c, col_v = st.columns(2)

        def card(h: search.Hit, kind: str):
            m = h.meta
            who = m.get("speaker") or m.get("course") or "—"
            with st.container(border=True):
                st.markdown(f"**#{h.rank} · cos = {h.score:.4f}**")
                st.progress(float(max(0.0, min(1.0, h.score))))
                st.markdown(f"`{short(h.file_id, 90)}`")
                st.caption(f"Nguồn: {m.get('source')} · Người nói/khoá: {who} · {m.get('duration_s', 0):.0f} s"
                           + (" · ⚠️ nhiều người đọc" if m.get("multi_speaker") else ""))
                if kind == "content" and h.passage:
                    st.markdown(f"> {h.passage[:400]}…")
                    st.caption(f"Đoạn khớp nhất (cos đoạn = {h.passage_score:.3f})")
                src_path = Path(m.get("src_path", ""))
                if src_path.exists():
                    st.audio(str(src_path))

        with col_c:
            st.markdown("### 📝 Top nội dung")
            if res.content_known:
                st.success(f"Nội dung đã có trong CSDL (top-1 {res.content[0].score:.3f} ≥ {res.thresholds['content_known']:.3f})")
            elif res.content:
                st.warning(f"Nội dung có thể CHƯA có trong CSDL (top-1 {res.content[0].score:.3f} < "
                           f"{res.thresholds['content_known']:.3f}). Dưới đây là các file gần nhất.")
            else:
                st.error("Không nhận dạng được lời nói trong truy vấn.")
            for h in res.content:
                card(h, "content")
        with col_v:
            st.markdown("### 🗣️ Top giọng nói")
            if res.voice_known:
                st.success(f"Người nói đã có trong CSDL (top-1 {res.voice[0].score:.3f} ≥ {res.thresholds['voice_known']:.3f})")
            else:
                st.warning(f"Người nói có thể CHƯA có trong CSDL (top-1 {res.voice[0].score:.3f} < "
                           f"{res.thresholds['voice_known']:.3f}). Dưới đây là các giọng gần nhất.")
            for h in res.voice:
                card(h, "voice")

        st.divider()
        st.subheader("Kết quả trung gian")
        t1, t2, t3, t4 = st.tabs(["Transcript", "Vector đặc trưng", "Độ tương đồng", "Baseline & thời gian"])
        with t1:
            st.text_area("Transcript (Whisper)", q.transcript, height=180)
            st.caption(f"{len(q.transcript)} ký tự · {len(q.chunks)} đoạn")
            for i, ch in enumerate(q.chunks):
                st.markdown(f"**Đoạn {i + 1}:** {ch}")
        with t2:
            v1, v2 = st.columns(2)
            v1.markdown(f"**ECAPA 192-D** · ‖v‖ = {np.linalg.norm(q.ecapa):.3f}")
            v1.bar_chart(pd.DataFrame({"ECAPA": q.ecapa}))
            v2.markdown(f"**Semantic 384-D** · ‖v‖ = {np.linalg.norm(q.semantic):.3f}")
            v2.bar_chart(pd.DataFrame({"semantic": q.semantic}))
            v3, v4 = st.columns(2)
            v3.markdown("**MFCC c0..c25 — trung bình theo thời gian**")
            v3.bar_chart(pd.DataFrame({"mfcc_mean": q.mfcc_mean[:26]}))
            v4.markdown("**Prosody & phổ**")
            p = q.prosody
            v4.dataframe(pd.DataFrame({
                "đặc trưng": ["F0 trung bình (Hz)", "F0 độ lệch (Hz)", "ZCR", "Spectral centroid (Hz)",
                              "Spectral rolloff 85% (Hz)"],
                "giá trị": [p.get("f0_mean_hz"), p.get("f0_std_hz"), p.get("zcr_mean"),
                            p.get("spectral_centroid_mean"), p.get("spectral_rolloff_mean")],
            }), hide_index=True, width="stretch")
            with st.expander("Giá trị số đầy đủ"):
                st.write({"ecapa_192": np.round(q.ecapa, 4).tolist(),
                          "mfcc_clean_50": np.round(q.mfcc_clean, 4).tolist(),
                          "semantic_384": np.round(q.semantic, 4).tolist()})
        with t3:
            s1, s2 = st.columns(2)
            s1.markdown("**Top-10 cosine nội dung**")
            s1.dataframe(pd.DataFrame(res.content_top10, columns=["file_id", "cosine"]),
                         hide_index=True, width="stretch")
            s2.markdown("**Top-10 cosine giọng nói (ECAPA)**")
            s2.dataframe(pd.DataFrame(res.voice_top10, columns=["file_id", "cosine"]),
                         hide_index=True, width="stretch")
        with t4:
            b1, b2 = st.columns(2)
            b1.markdown("**Baseline giọng nói — MFCC 50-D**")
            b1.dataframe(pd.DataFrame([(h.rank, h.file_id, round(h.score, 4)) for h in res.voice_mfcc],
                                      columns=["#", "file_id", "cosine"]), hide_index=True, width="stretch")
            b2.markdown("**Baseline nội dung — BM25 (FTS5)**")
            b2.dataframe(pd.DataFrame([(h.rank, h.file_id, round(h.score, 3)) for h in res.keyword],
                                      columns=["#", "file_id", "bm25"]), hide_index=True, width="stretch")
            st.markdown("**Thời gian từng bước (giây)**")
            st.dataframe(pd.DataFrame([{**{k: round(v, 2) for k, v in q.timings.items()},
                                        "search_ms": round(res.search_ms, 1)}]), hide_index=True)
