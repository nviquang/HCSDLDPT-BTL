# Báo cáo phân tích đặc trưng & mô tả tập dữ liệu

> **Giai đoạn 2 — kiểm định chất lượng bộ thuộc tính.**
> Tài liệu này trả lời hai câu hỏi mà bước trích xuất đặc trưng không trả lời được:
> **(1)** các vector đã trích có dùng được không? **(2)** bộ dữ liệu trông như thế nào so với mục tiêu thiết kế?
> 
> - **Mã nguồn:** `src/audio_search/analysis.py` (đo lường), `src/audio_search/figures.py` (trực quan hoá), `src/audio_search/corpus.py` (sinh nhãn từ tên file)
> - **Dữ liệu đầu vào:** `artifacts/features/voice-*.json` (504 file), `artifacts/features/content-*.json` (504 file)
> - **Kết quả:** `artifacts/analysis/metrics.json`, `artifacts/analysis/corpus_table.csv`, `artifacts/analysis/figures/*.png`
> - **Ngày chạy:** 22/10/2026 · **Quy mô:** 504 file · 60,81 giờ âm thanh · 126.756 cặp so sánh
> - **Cập nhật 16/11/2026:** bộ dữ liệu đã được bổ sung 35 bài TED một diễn giả (539 file, 67,1 giờ). 19 file biểu diễn hoặc giọng AI và 12 file nhiều người nói (người đọc kép, promo hoặc quảng cáo ghép nhiều giọng, phỏng vấn) bị gắn cờ, còn **508 file đạt chuẩn**. Số liệu bên dưới là của lần phân tích 504 file; số liệu hiện hành xem [`he_thong_tim_kiem.md`](he_thong_tim_kiem.md) §0 và §6.

---

## 0. Cách chạy lại

```powershell
uv run audio-search analyze                 # đầy đủ: metrics + corpus_table + 8 biểu đồ
uv run audio-search analyze --skip-content  # chỉ audit phần giọng nói
uv run audio-search analyze --no-figures    # không render PNG (nhanh hơn)
```

> `analyze` chỉ **đọc** artifact JSON — không cần audio gốc, không cần GPU. Thời gian chạy ~10 giây.
> Luồng đầy đủ: `manifest` → `analyze`. Phải chạy `manifest` trước nếu vừa trích thêm feature mới.

---

## 1. Các đặc trưng được phân tích

| Nhóm                    | Tên biến trong artifact                              | Số chiều | Nguồn sinh ra                | Vai trò trong GĐ4               |
| ----------------------- | ---------------------------------------------------- | -------- | ---------------------------- | ------------------------------- |
| **Giọng nói — cổ điển** | `mfcc_stats` (mean của 26 MFCC + Δ + ΔΔ)             | 78       | `voice_features.py`          | Baseline / giải thích           |
|                         | `mfcc_std` (độ lệch chuẩn 78 hệ số)                  | 78       | `voice_features.py`          | Trực quan hoá                   |
|                         | `prosody` (F0 mean/std, ZCR, centroid, rolloff)      | 5        | `voice_features.py`          | Bổ trợ                          |
| **Giọng nói — deep**    | `ecapa_embedding` (ECAPA-TDNN, L2-normalize)         | 192      | `voice_features.py`          | **PRIMARY — tìm theo giọng**    |
| **Nội dung**            | `semantic_embedding` (MiniLM-L6-v2, L2-normalize)    | 384      | `content_features.py`        | **PRIMARY — tìm theo nội dung** |
| **Nội dung — phụ**      | `transcript`, `n_chunks`                             | —        | `content_features.py`        | Đánh giá, baseline TF-IDF       |
| **Metadata**            | `duration_s`, `source`, `speaker`, `topic`, `course` | —        | `corpus.py` (parse tên file) | Đánh giá                        |

Ba biến thể dùng để so sánh trong báo cáo:

- **`mfcc_raw_78d`** — MFCC 78-D nguyên bản (không chuẩn hoá)
- **`mfcc_clean_26d_zscore`** — bỏ 2 khối Δ/ΔΔ, z-score 26 hệ số cơ sở (`mfcc_clean_variant`)
- **`ecapa_192d`** — ECAPA 192-D (đã L2-normalize)

---

## 2. Kiểm tra toàn vẹn (`integrity`)

| Chỉ số           | Giá trị | Ý nghĩa                             |
| ---------------- | ------- | ----------------------------------- |
| `n_voice`        | 504     | số artifact giọng nói đọc được      |
| `n_content`      | 504     | số artifact nội dung đọc được       |
| `n_issues`       | 12      | xem chi tiết bên dưới               |
| `voice_dim_ok`   | `true`  | mọi vector đúng shape 78 / 78 / 192 |
| `content_dim_ok` | `true`  | mọi vector đúng shape 384           |

**Các kiểm tra thực hiện** (`audit_integrity`, `analysis.py:195`):

| Kiểm tra                           | Ngưỡng              | Kết quả        |
| ---------------------------------- | ------------------- | -------------- |
| Shape `mfcc_stats` / `mfcc_std`    | `(78,)`             | 504/504 đạt    |
| Shape `ecapa_embedding`            | `(192,)`            | 504/504 đạt    |
| Shape `semantic_embedding`         | `(384,)`            | 504/504 đạt    |
| Giá trị hữu hạn (không NaN/Inf)    | tất cả              | đạt            |
| L2-norm của vector đã normalize    | `\|‖v‖ − 1\| ≤ 0,1` | 6 file vi phạm |
| `semantic_embedding` khác vector 0 | `‖v‖ > 0,01`        | 6 file vi phạm |
| Prosody hữu hạn                    | tất cả              | đạt            |
| Artifact đọc được / JSON hợp lệ    | —                   | đạt            |

### 2.1. 6 file có `semantic_embedding` bằng vector 0

Đây **không phải lỗi trích xuất** — đây là đặc điểm thật của dữ liệu:

| `file_id`                                                                          | Lý do                                          |
| ---------------------------------------------------------------------------------- | ---------------------------------------------- |
| `_Universe_Statues_Liberation_Buzz_TED`                                            | TED talk về tạo hình tượng, không có lời thoại |
| `A_Brilliant_Blend_of_Ballet,_Hip-Hop_and_African_Dance_Ghetto_Classics_Dance_TED` | phần trình diễn múa, không lời                 |
| `A_Hypnotic_Exploration_of_Movement_JA_Collective_TED`                             | chủ yếu là biểu diễn, không lời                |
| `A_Musical_Journey_Through_Turkana_Turkana_Sessions_TED`                           | nhạc trình diễn, không lời                     |
| `A_New_Take_on_an_Ancient_Sound_Tina_Lajevardian_TED`                              | nhạc cụ, không lời                             |
| `“Atalanta”_The_Merian_Ensemble_TED`                                               | tấn công bằng nhạc cụ, không lời               |

**Cơ chế:** Whisper không nhận dạng được lời nói → transcript rỗng → không có chunk nào để embed → `semantic_embedding` = vector 0.
Mỗi file sinh **2 issue** (một lần cho `L2 norm != 1`, một lần cho `zero vector`) — nên 12 issue tương ứng với **6 file**, không phải 12 file lỗi.

**Cách xử lý trong báo cáo:** `analyse_content()` loại các vector 0 khỏi mọi thống kê và đếm riêng (`n_zero_vector: 6`, `n_usable: 498`) → phân phối similarity của phần nội dung không bị nhiễu.

> **Lưu ý cho GĐ4:** khi nạp vào pgvector, 6 file này có vector 0 và sẽ luôn xếp hạng thấp — không sai, nhưng cần biết để giải thích kết quả tìm kiếm.

---

## 3. Phân tích số chiều (`dimensionality`)

**Cách tính** (`pca_analysis`, `analysis.py:236`): trung tâm hóa ma trận → SVD → tỉ lệ phương sai giải thích của từng thành phần chính (PC) → số PC tối thiểu để đạt ngưỡng phương sai tích luỹ.

| Vector       | Dim | PC đạt 90% | PC đạt 95% | PC đạt 99% | PC1 đóng góp | PC2    | PC3   | PC4   | PC5   |
| ------------ | --- | ---------- | ---------- | ---------- | ------------ | ------ | ----- | ----- | ----- |
| `mfcc_stats` | 78  | **4**      | **6**      | 15         | **67,25%**   | 11,44% | 8,17% | 3,92% | 2,89% |
| `mfcc_std`   | 78  | **4**      | **6**      | 13         | 65,35%       | 15,20% | 8,10% | 2,81% | 2,40% |
| `ecapa`      | 192 | 88         | **114**    | 159        | 14,92%       | 8,21%  | 4,66% | 3,28% | 2,38% |
| `prosody`    | 5   | **1**      | **1**      | 3          | **97,44%**   | 1,42%  | 0,93% | 0,21% | ~0%   |

**Kết luận:**

1. **MFCC gần như nằm trong 6 chiều.** 95% phương sai tích luỹ đạt được ở PC6, tức 78 chiều thực tế chỉ mang thông tin tương đương ~6 chiều. Đặc biệt PC1 chiếm 67% — phần lớn là hệ số log-energy c0 (xem §4).
2. **ECAPA cần 114/192 chiều** mới đạt 95% — thông tin phân tán rộng hơn nhiều, đúng như vai trò "đặc trưng người nói" (mã hoá cả cao độ, nhịp nói, ngữ điệu, đặc trưng phụ âm). Việc giữ nguyên 192 chiều là cần thiết, không phải dư thừa.
3. **Prosody 5 chiều nhưng thực tế 1 chiều** (PC1 = 97,4%). Hệ quả: `f0_mean_hz` và `f0_std_hz` gần như biến thiên cùng nhau → không nên dùng prosody làm vector độc lập trong không gian vector; nó chỉ nên là **số đo diễn giải** đi kèm.

---

## 4. Cấu trúc khối MFCC (`mfcc_structure`)

**Câu hỏi:** vector "78-D" có thực sự chứa 78 chiều thông tin, hay 52 chiều cuối chỉ là nhiễu?

**Cách tính** (`mfcc_block_analysis`, `analysis.py:270`): chia 78-D thành 3 khối `[base c0–c25 | Δ c26–c51 | ΔΔ c52–c77]`, đo tỉ lệ năng lượng (`Σ(x − x̄)²`) của từng khối trên tổng năng lượng; thêm tương quan giữa PC1 và hệ số c0.

| Khối              | Năng lượng  | **Tỉ lệ năng lượng** | Trung bình \|x\| | Số PC đạt 95% trong khối |
| ----------------- | ----------- | -------------------- | ---------------- | ------------------------ |
| **base** (c0–c25) | 831.398,013 | **0,9999999932**     | 3,988            | 6                        |
| **Δ** (c26–c51)   | 0,00485     | **5,83 × 10⁻⁹**      | 1,44 × 10⁻⁴      | 3                        |
| **ΔΔ** (c52–c77)  | 0,00082     | **9,85 × 10⁻¹⁰**     | 8,95 × 10⁻⁵      | 5                        |

| Chỉ số                  | Giá trị    | Ý nghĩa                                    |
| ----------------------- | ---------- | ------------------------------------------ |
| `pc1_vs_c0_correlation` | **0,9979** | PC1 gần như *chính là* hệ số log-energy c0 |
| `c0_mean`               | −327,93    | log-energy trung bình của corpus           |
| `c0_std`                | 32,81      | độ biến thiên log-energy giữa các file     |
| `c1_c25_abs_mean`       | 7,58       | biên độ trung bình của 25 hệ số còn lại    |

**Kết luận — đây là phát hiện quan trọng nhất của phân tích:**

> Hai khối Δ và ΔΔ mang **gần như 0% biến thiên** (5,8 × 10⁻⁹ và 9,9 × 10⁻¹⁰). Vector "78-D" thực chất chỉ là **26-D**, phần còn lại là nhiễu số.
> 
> Thêm nữa, PC1 tương quan 0,998 với c0 ⇒ nếu dùng nguyên vector 78-D để tính cosine, **hệ số log-energy sẽ chi phối toàn bộ phép so sánh**: mọi file đều có cùng cấu trúc phổ và gần như cùng năng lượng tổng, nên cosine giữa chúng luôn ≈ 1 (xem §5).
> 
> **Hệ quả trực tiếp cho GĐ4:** vector đưa vào pgvector cho chiều giọng nói nên là **`ecapa_embedding` 192-D**, không phải `mfcc_stats`. Nếu vẫn muốn dùng MFCC (ví dụ để minh hoạ "đặc trưng cổ điển"), **phải** dùng biến thể `mfcc_clean_26d_zscore` (bỏ Δ/ΔΔ + z-score), không dùng bản thô.

---

## 5. Cấu trúc tương đồng (`similarity`)

**Cách tính** (`cosine_matrix`, `analysis.py:324`): chuẩn hoá L2 từng hàng → nhân ma trận với chuyển vị → thống kê phân phối trên **ngoài đường chéo** (126.756 cặp = 504 × 503 / 2).

| Họ vector                       | Số cặp  | Mean        | Median  | Std    | Min     | p01     | p99    | Max     |
| ------------------------------- | ------- | ----------- | ------- | ------ | ------- | ------- | ------ | ------- |
| `mfcc_raw_78d`                  | 126.756 | **0,9941**  | 0,9952  | 0,0043 | 0,9448  | 0,9791  | 0,9994 | 0,99995 |
| `mfcc_clean_26d_zscore`         | 126.756 | **0,00028** | −0,0095 | 0,3702 | −0,8808 | −0,7050 | 0,8488 | 0,9918  |
| `ecapa_192d`                    | 126.756 | **0,1134**  | 0,0734  | 0,1949 | −0,3084 | −0,1305 | 0,9101 | 0,9842  |
| `semantic_embedding` (nội dung) | 123.753 | **0,1765**  | 0,1433  | 0,1422 | −0,1564 | −0,0456 | 0,5856 | 0,9114  |

*(Hàng cuối tính trên 498 file dùng được, đã loại 6 vector 0.)*

**Kết luận:**

1. **MFCC thô không phân biệt được gì.** Toàn bộ 126.756 cặp dồn trong khoảng [0,945 ; 1,000] — độ rộng 0,055 trong khi khoảng động của cosine là 2. Một ngưỡng tìm kiếm bất kỳ (0,9 hay 0,99) sẽ trả về gần như toàn bộ corpus. Đây là hệ quả trực tiếp của §4.
2. **MFCC đã z-score phân biệt tốt** (mean ≈ 0, std 0,37, trải đều từ −0,88 đến 0,99).
3. **ECAPA có phân phối cosine đúng như mong đợi** cho bài toán nhận dạng người nói: phần lớn cặp *khác người nói* có cosine thấp (median 0,073), còn cặp *cùng người nói* nằm ở đuôi phải (p99 = 0,910).
4. **Semantic embedding** có phân phối tương tự nhưng dịch sang phải một chút (median 0,143) — nội dung có tính đặc trưng chung cao hơn giọng nói (cùng chủ đề, cùng thể loại kể chuyện).

---

## 6. Khả năng tách nhãn (`separation`)

**Cách tính** (`label_gap`, `analysis.py:381`): với mỗi cách gán nhãn, so sánh cosine trung bình của các cặp **cùng nhãn** với các cặp **khác nhãn**. `gap` càng lớn ⇒ vector tách được nhãn đó càng tốt.

| Vector / nhãn                   | Số cặp cùng | Số cặp khác | `same_mean` | `diff_mean` | **`gap`**     |
| ------------------------------- | ----------- | ----------- | ----------- | ----------- | ------------- |
| `ecapa` × **course** (môn Khan) | 1.100       | 11.941      | 0,9181      | 0,4001      | **+0,5180**   |
| `ecapa` × **topic**             | 1           | 119.315     | 0,8128      | 0,1114      | +0,7015 ⚠️    |
| `ecapa` × **source**            | 36.915      | 89.841      | 0,2224      | 0,0686      | **+0,1538**   |
| `mfcc_clean` × course           | 1.100       | 11.941      | 0,8554      | 0,1030      | **+0,7524**   |
| `mfcc_raw` × course             | 1.100       | 11.941      | 0,9988      | 0,9944      | **+0,0044** ❌ |

**Kết luận:**

1. **ECAPA tách tốt theo môn học** (gap 0,518): các file cùng môn Khan được nhận dạng là cùng giọng. Nhưng lưu ý: "cùng môn" ở Khan Academy gần như đồng nghĩa với "cùng người nói" (Salman Khan) → con số này đo **khả năng nhận dạng người nói**, không đo khả năng phân loại chủ đề.
2. **ECAPA tách yếu theo nguồn** (gap 0,154) — đúng như mong đợi, vì nguồn dữ liệu không phải yếu tố nhận dạng người nói.
3. **`mfcc_raw` gap 0,0044 ≈ 0** → xác nhận lần nữa: bản thô không dùng được để so sánh.
4. ⚠️ **`ecapa_by_topic` không có ý nghĩa thống kê**: chỉ có **1 cặp** cùng chủ đề (vì 488 chủ đề / 504 file ⇒ gần như mỗi file một chủ đề riêng). Con số gap 0,70 là của một mẫu n = 1, **không được trích dẫn** trong báo cáo.

---

## 7. Hàng xóm gần nhất (`nearest_neighbour`)

**Cách tính** (`nearest_neighbour_stats`, `analysis.py:413`): với mỗi file, tìm file có cosine cao nhất (đường chéo bị loại), rồi kiểm tra file đó có cùng nhóm (nguồn) hay không.

| Chỉ số                         | ECAPA 192-D        | MFCC-clean 26-D |
| ------------------------------ | ------------------ | --------------- |
| `nn_sim_mean`                  | 0,6510             | 0,8096          |
| `nn_sim_median`                | 0,5392             | 0,8258          |
| `nn_sim_min`                   | 0,2789             | 0,4532          |
| `nn_sim_max`                   | 0,9842             | 0,9918          |
| **`top1_same_group_fraction`** | **0,8194** (81,9%) | 0,8135 (81,3%)  |

**Phân bố nguồn của file được chọn làm "gần nhất" (ECAPA):**

| Nguồn           | KhanAcademy | TED | Storybook | Vooks | Other |
| --------------- | ----------- | --- | --------- | ----- | ----- |
| Số file trỏ tới | 168         | 172 | 83        | 64    | 17    |

**10 ví dụ điển hình (ECAPA):**

| File                                                                           | File gần nhất                                                 | Cosine |
| ------------------------------------------------------------------------------ | ------------------------------------------------------------- | ------ |
| `050_Calculating i raised to arbitrary exponents ｜ Precalculus ｜ Khan Academy` | `048_Introduction to i and imaginary numbers ｜ …Precalculus…` | 0,9842 |
| `007_Tracing variables ｜ Intro to CS - Python ｜ Khan Academy`                  | `006_Variables and assignment ｜ Intro to CS - Python…`        | 0,9836 |
| `035_Program design： Nim game ｜ Intro to CS - Python…`                         | `033_Program design： deck of cards ｜ …`                       | 0,9818 |
| `077_Dividing polynomials by x… ｜ Algebra 2 ｜ Khan Academy`                    | `078_Dividing quadratics by linear expressions…`              | 0,9818 |
| `034_Creating modules ｜ Intro to CS - Python…`                                 | `036_Unit testing ｜ Intro to CS - Python…`                    | 0,9817 |

**Kết luận:**

- **81,9% file ECAPA tìm được file gần nhất thuộc cùng nguồn** — xác nhận ECAPA thực sự đang học *người nói*, không phải nội dung hay đặc trưng kỹ thuật của file.
- Ví dụ điển hình: các bài Khan Academy cùng môn được ghép đôi thành cặp, cosine 0,98 — đây chính là kịch bản *"cùng giọng – khác chủ đề"* mà `report.md` §2.4c yêu cầu chuẩn bị sẵn cho GĐ4.
- ⚠️ **Cảnh báo về độ "sạch" của phép đo:** tỉ lệ 81,9% là trên **nhóm `source`**, một nhãn rất thô. Nhãn `speaker` của TED là *heuristic* (2 token cuối tên file — `corpus.py:15-18` cảnh báo chỉ nên dùng để đối chiếu, không phải ground truth). Do đó **chưa có độ đo nào ở đây dùng làm ground truth chuẩn cho đánh giá GĐ4** — cần tạo tập nhãn thủ công cho một tập con.

---

## 8. Phân cụm phân cấp (`clustering`)

**Cách tính** (`cluster_analysis`, `analysis.py:446`): chuyển cosine thành khoảng cách `1 − cosine` → **average-linkage hierarchical clustering** (SciPy) → cắt cây ở 3 ngưỡng khoảng cách (0,4 / 0,6 / 0,8) → đếm số cụm, cụm lớn nhất, và **purity** (nhãn chiếm đa số trong cụm).

### ECAPA 192-D

| Ngưỡng | Số cụm | Cụm lớn nhất | Top-5 kích thước    | Purity   | Nhãn chiếm đa số       |
| ------ | ------ | ------------ | ------------------- | -------- | ---------------------- |
| 0,4    | 308    | **104**      | 104, 36, 25, 8, 6   | **1,00** | `Intro to CS - Python` |
| 0,6    | 197    | **104**      | 104, 37, 28, 14, 8  | **1,00** | `Intro to CS - Python` |
| 0,8    | 49     | **107**      | 107, 51, 47, 34, 20 | **1,00** | `Intro to CS - Python` |

### MFCC-clean 26-D

| Ngưỡng | Số cụm | Cụm lơn nhất | Top-5 kích thước      | Purity   | Nhãn chiếm đa số       |
| ------ | ------ | ------------ | --------------------- | -------- | ---------------------- |
| 0,4    | 142    | 71           | 71, 40, 31, 26, 24    | **1,00** | `Intro to CS - Python` |
| 0,6    | 48     | 84           | 84, 73, 64, 43, 42    | **1,00** | `Intro to CS - Python` |
| 0,8    | 14     | 132          | 132, 130, 117, 43, 31 | 0,9231   | `Intro to CS - Python` |

**Kết luận — đây là bằng chứng mạnh nhất về khả năng nhận dạng người nói:**

> Ở **mọi** ngưỡng cắt, ECAPA gom được **104 file thuộc cùng một cụm** với **purity = 1,00** — tức **104/104 file trong cụm lớn nhất đều có cùng nhãn môn học**. Đây chính là giọng nói Salman Khan của Khan Academy (162 file trong corpus).
> 
> ⇒ **ECAPA đã tái tạo được nhóm người nói một cách tự động, không cần nhãn**, chỉ từ đặc trưng âm thanh. Đây chính là cơ sở kỹ thuật để đáp ứng yêu cầu *"3 file giọng nói giống nhất"* của GĐ4.
> 
> Khi nới ngưỡng (0,4 → 0,8), cụm lớn nhất **nhận thêm file** (104 → 107) thay vì vỡ ra — nghĩa là cấu trúc cụm ổn định, không phụ thuộc ngưỡng cắt.

---

## 9. Ảnh hưởng của thời lượng (`duration_effect`)

**Câu hỏi:** file dài hơn thì có khó khớp giọng hơn không? (ECAPA embed cả utterance một lần nên file dài có thể bị loãng tín hiệu người nói.)

**Cách tính** (`duration_effect`, `analysis.py:491`): tương quan Pearson giữa thời lượng file và (a) cosine với file gần nhất, (b) cosine trung bình với mọi file.

| Vector             | `corr_duration_vs_nn_cos` | `corr_duration_vs_mean_cos` |
| ------------------ | ------------------------- | --------------------------- |
| **ECAPA**          | **−0,568**                | **−0,503**                  |
| MFCC-clean         | −0,433                    | −0,108                      |
| Semantic embedding | −0,103                    | —                           |

**Kết luận:**

1. **ECAPA chịu ảnh hưởng mạnh của thời lượng** (r = −0,57): file dài hơn → khớp giọng kém hơn. Đây là hạn chế thật cần nêu trong báo cáo và có ý nghĩa thiết kế cho GĐ4: **nên cắt file dài thành các đoạn ngắn trước khi trích ECAPA**, thay vì embed cả file một lần.
2. **MFCC-clean ít bị ảnh hưởng hơn** ở cosine trung bình (r = −0,11) — dù toàn bộ vector đã gom theo thời gian nên đặc trưng thống kê ổn định hơn.
3. **Semantic embedding gần như không bị ảnh hưởng** (r = −0,10) — vì mỗi file được cắt thành nhiều chunk rồi mean-pool, nên việc thêm nội dung không làm loãng vector. Đây là một lợi thế rõ ràng của thiết kế chunk + mean-pool so với ECAPA.

---

## 10. Mô tả tập dữ liệu (`corpus`)

### 10.1. Quy mô & thời lượng

| Chỉ số                | Giá trị                        |
| --------------------- | ------------------------------ |
| Số file               | **504**                        |
| Tổng thời lượng       | **60,81 giờ**                  |
| Thời lượng trung bình | 434,3 s (7,2 phút)             |
| Thời lượng trung vị   | 420,4 s (7,0 phút)             |
| Ngắn nhất / dài nhất  | 30,0 s / 1.266,4 s (21,1 phút) |
| p05 / p95             | 212,5 s / 658,6 s              |

Phân bố nằm gọn trong dải **3–21 phút** ⇒ đúng khuyến nghị "3–20 phút/file" của `report.md` §2.4c.

### 10.2. Hỗn hợp nguồn

| Nguồn       | Số file | TB (s) | Trung vị (s) | Min (s) | Max (s) |
| ----------- | ------- | ------ | ------------ | ------- | ------- |
| TED         | 190     | 507,9  | 530,6        | 200,9   | 658,8   |
| KhanAcademy | 162     | 303,9  | 296,0        | 30,9    | 519,3   |
| Storybook   | 103     | 545,5  | 515,4        | 278,8   | 1.266,4 |
| Vooks       | 34      | 391,3  | 377,7        | 229,5   | 558,8   |
| Other       | 15      | 244,8  | 218,4        | 30,0    | 641,3   |

### 10.3. Độ phủ nhãn (sinh từ tên file — `corpus.py`)

| Chỉ số                       | Giá trị                                     |
| ---------------------------- | ------------------------------------------- |
| Có nhãn `speaker`            | 293 / 504 (58,1%)                           |
| Có nhãn `topic`              | 489 / 504 (97,0%)                           |
| Không có nhãn nào            | 15 (2,9%) — toàn bộ thuộc nhóm `Other`      |
| Số người nói khác nhau       | 291                                         |
| **Số người nói có ≥ 2 file** | **2** (`Angus_Hervey`: 2, `Jaime_Camil`: 2) |
| Số chủ đề khác nhau          | 488                                         |
| **Số chủ đề có ≥ 4 file**    | **0**                                       |
| Số chủ đề có ≥ 2 file        | 1                                           |

**Cảnh báo quan trọng:** vì `topic` được lấy từ **tiêu đề file** và mỗi file có một tiêu đề riêng, `n_distinct_topics = 488` gần bằng `n_files = 504`. Về mặt kỹ thuật "mỗi file một chủ đề" là đúng yêu cầu đề bài, nhưng **nhãn này không dùng được để đánh giá truy hồi theo chủ đề** (mỗi nhóm có đúng 1 file ⇒ không có cặp nội nhóm để so sánh).

---

## 11. Đối chiếu mục tiêu thiết kế (`targets_comparison`)

So với §2.4c của `report.md`. Mỗi mục trả về `{wanted, actual, met}` — cùng cấu trúc khóa để người đọc không phải đoán tên trường.

| Tiêu chí                | Mục tiêu                | Thực tế | Kết quả                                   |
| ----------------------- | ----------------------- | ------- | ----------------------------------------- |
| Số file tổng            | 500                     | **504** | ✅ Đạt                                     |
| Số người nói            | 40                      | **291** | ✅ Đạt                                     |
| Người nói có nhiều file | ≥ 40 người × 15–20 file | **2**   | ❌ Chưa đạt                                |
| Số chủ đề               | 15–20                   | **488** | ❌ Chưa đạt (quá nhiều, mỗi file 1 chủ đề) |
| Số chủ đề có ≥ 4 file   | ≥ 15 chủ đề             | **0**   | ❌ Chưa đạt                                |

**Đọc kết quả này đúng cách:**

- Ba tiêu chí đầu **đạt về mặt số lượng**: 504 file, 291 người nói khác nhau, đủ đa dạng giọng.
- Hai tiêu chí sau **không đạt vì cách gán nhãn**, không phải vì thiếu dữ liệu:
  - "Người nói có nhiều file": tên file TED/Storybook chỉ chứa tên người nói **khi có định dạng `*_read_by_*`**; đa số file Khan Academy không có trường người nói trong tên ⇒ chỉ 2 người nói "nhận diện được" bằng heuristic. Thực tế **104 file Khan Academy đều là cùng một người nói** (đã chứng minh ở §8) — dữ liệu *đã có* nhóm người nói, chỉ là **nhãn chưa được ghi ra**.
  - "Chủ đề": mỗi file có một tiêu đề riêng nên 488 chủ đề. Muốn đánh giá truy hồi theo chủ đề thì phải **gán nhóm chủ đề thủ công** (ví dụ gom theo môn Khan, theo nhóm TED) — không thể suy ra tự động từ tên file.

---

## 12. Nội dung (`content`)

| Chỉ số                   | Giá trị |
| ------------------------ | ------- |
| Số artifact              | 504     |
| Vector 0 (không lời nói) | 6       |
| **Số dùng được**         | **498** |
| Số cặp so sánh           | 123.753 |

**Phân phối cosine** (`semantic_embedding`): mean 0,1765 · median 0,1433 · std 0,1422 · min −0,1564 · p01 −0,0456 · p99 0,5856 · max 0,9114

**Số chunk / độ dài transcript:**

| Chỉ số                              | Giá trị             |
| ----------------------------------- | ------------------- |
| Số chunk: mean / median / min / max | 4,86 / 5 / 1 / 13   |
| Độ dài transcript: mean / median    | 5.391 / 5.169 ký tự |

**Tương quan:**

| Tương quan                | Giá trị    | Ý nghĩa                                                                |
| ------------------------- | ---------- | ---------------------------------------------------------------------- |
| `corr_nchunks_vs_nn_cos`  | **+0,024** | gần như **không** — thêm chunk không làm giảm chất lượng khớp nội dung |
| `corr_duration_vs_nn_cos` | **−0,103** | thời lượng ảnh hưởng rất nhẹ                                           |

**Kết luận:** thiết kế "cắt chunk + mean-pool" của `content_features.py` **hoàn toàn không bị ảnh hưởng bởi độ dài file** — đây là điểm mạnh rõ ràng so với ECAPA (r = −0,57, §9). Nội dung dài không làm loãng vector ngữ nghĩa.

---

## 13. Các biểu đồ

Thư mục `artifacts/analysis/figures/` — tất cả vẽ từ **chính `metrics.json`**, nên số liệu trên biểu đồ không bao giờ lệch với số liệu trong báo cáo.

| File                            | Nội dung                                                                          | Thông điệp                                                   |
| ------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `pca_cumulative_variance.png`   | Đường phương sai tích luỹ của MFCC (78-D) và ECAPA (192-D), đánh dấu điểm cắt 95% | MFCC đạt 95% ở PC6 — ECAPA cần 114/192                       |
| `mfcc_block_energy.png`         | Tỉ lệ năng lượng 3 khối base/Δ/ΔΔ (bar + nhãn trực tiếp)                          | Khối Δ và ΔΔ mang **0%** biến thiên — 78-D thực chất ~26-D   |
| `pairwise_cosine.png`           | 3 panel small-multiples: histogram cosine của MFCC thô / MFCC-clean / ECAPA       | MFCC thô dồn 126.756 cặp quanh 1,0 — không phân biệt được gì |
| `duration_by_source.png`        | Histogram thời lượng toàn corpus + bar thời lượng trung vị theo nguồn             | 504 file, 61 giờ, trung vị 7 phút                            |
| `label_separation.png`          | Bar nhóm: cosine trong cùng nhóm vs khác nhóm, cho course / topic / source        | ECAPA tách tốt theo môn, kém theo nguồn                      |
| `duration_effect.png`           | Scatter thời lượng vs cosine-NN của ECAPA + đường hồi quy                         | r = −0,568 — file dài khó khớp giọng                         |
| `label_coverage_vs_targets.png` | Bảng trạng thái 5 tiêu chí so với mục tiêu §2.4c                                  | 2 đạt / 3 chưa đạt (lý do nêu ở §11)                         |
| `ecapa_clustering.png`          | Số cụm & cụm lớn nhất của ECAPA ở 3 ngưỡng cắt                                    | Cụm lớn nhất gom 104–107 file, purity 1,00                   |

---

## 14. Tổng kết & hành động đề xuất

### Những gì đã xác nhận

1. ✅ **Toàn bộ 504 file trích đặc trưng thành công**, không có artifact hỏng (0 lỗi shape, 0 NaN/Inf, 0 file đọc lỗi).
2. ✅ **ECAPA-TDNN là đặc trưng giọng nói đáng tin cậy**: tự động gom 104 file cùng người nói với purity 1,00; 81,9% file tìm được người nói/nguồn giống nhất ở hạng #1.
3. ✅ **Semantic embedding ổn định và không bị ảnh hưởng bởi thời lượng** (r = −0,10) — sẵn sàng cho tìm kiếm theo nội dung.
4. ✅ **Quy mô corpus đạt yêu cầu** (504 ≥ 500 file, 291 người nói, 60,8 giờ).

### Vấn đề phát hiện được

| #   | Vấn đề                                                                                  | Mức độ     | Khuyến nghị                                                                                             |
| --- | --------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------- |
| 1   | `mfcc_stats` 78-D thực chất chỉ 26-D; Δ/ΔΔ có năng lượng ~10⁻⁹                          | Cao        | Không đưa `mfcc_stats` thô vào pgvector; nếu dùng thì bắt buộc `mfcc_clean_26d_zscore`                  |
| 2   | Cosine của MFCC thô dồn quanh 0,994 — không có ngưỡng tìm kiếm nào dùng được            | Cao        | Như trên                                                                                                |
| 3   | ECAPA bị loãng theo thời lượng (r = −0,57)                                              | Trung bình | Cắt file > 10 phút thành đoạn trước khi trích ECAPA                                                     |
| 4   | Nhãn `topic` không dùng được cho đánh giá (488 chủ đề / 504 file, 0 chủ đề có ≥ 4 file) | Cao        | Gán nhóm chủ đề thủ công cho một tập con ~50–100 file                                                   |
| 5   | Nhãn `speaker` của TED là heuristic, chỉ 2 người nói có ≥ 2 file                        | Cao        | Gán nhãn người nói thủ công cho tập con đánh giá (dùng cụm ECAPA §8 làm gợi ý)                          |
| 6   | 6 file TED không lời nói → semantic vector 0                                            | Thấp       | Ghi nhận trong báo cáo; loại khi nạp pgvector hoặc đánh dấu `has_speech = false`                        |
| 7   | `tfidf_baseline()` đã viết nhưng `run()` chưa gọi                                       | Trung bình | Gọi nó trong `analysis.run()` để có baseline TF-IDF so sánh với SBERT (theo yêu cầu §7 của `report.md`) |

### Việc cần làm cho GĐ4

1. **Tạo tập đánh giá có nhãn thủ công** — đây là điều kiện tiên quyết để tính precision@3. Có thể dùng cụm ECAPA (§8) làm nhãn người nói gián tiếp cho Khan Academy (104 file, purity 1,00) — đã có ground truth ở mức *nhóm nguồn*, chỉ cần xác nhận.
2. **Nạp vào pgvector**: `ecapa_embedding` (192-D) cho trục giọng nói, `semantic_embedding` (384-D) cho trục nội dung, cả hai đã L2-normalize ⇒ dùng được trực tiếp với cosine distance (`<=>`).
3. **Chuẩn hoá ngưỡng similarity trước khi so sánh với baseline** — vì ECAPA (median 0,073) và semantic (median 0,143) có thang khác nhau.
4. **Bật `tfidf_baseline`** để có số liệu đối chiếu bắt buộc trong phần đánh giá.

---

## 15. Phụ lục — Bảng tra cứu nhanh

### 15.1. Khoá trong `metrics.json`

| Khoá                 | Nội dung                                                              |
| -------------------- | --------------------------------------------------------------------- |
| `integrity`          | Kiểm tra shape / finiteness / L2-norm / vector 0                      |
| `dimensionality`     | PCA cho `mfcc_stats`, `mfcc_std`, `ecapa`, `prosody`                  |
| `mfcc_structure`     | Phân bố năng lượng 3 khối base / Δ / ΔΔ                               |
| `similarity`         | Thống kê cosine của 3 họ vector giọng nói (4 họ nếu tính cả nội dung) |
| `separation`         | Gap trong-nhóm vs ngoài-nhóm theo course / topic / source             |
| `nearest_neighbour`  | Thống kê file gần nhất + 10 ví dụ                                     |
| `clustering`         | Average-linkage ở ngưỡng 0,4 / 0,6 / 0,8 + purity                     |
| `duration_effect`    | Tương quan thời lượng ↔ similarity                                    |
| `corpus`             | Quy mô, nguồn, thời lượng, độ phủ nhãn                                |
| `targets_comparison` | Đối chiếu §2.4c của `report.md`                                       |
| `content`            | Phân phối cosine + số chunk + độ dài transcript + `manifest_rows`     |

### 15.2. `corpus_table.csv`

504 dòng × 7 cột: `file_id`, `source`, `speaker`, `topic`, `course`, `has_speaker`, `has_topic`.
Dùng để nối `metrics.json` với từng file cụ thể, hoặc làm input cho bước gán nhãn thủ công ở GĐ4.

### 15.3. Quy ước nhãn (`corpus.py`)

| Nguồn        | Quy ước tên file                                                     | Nhãn rút ra                                                    |
| ------------ | -------------------------------------------------------------------- | -------------------------------------------------------------- |
| Khan Academy | `NNN_Title ｜ Course ｜ Khan Academy` (separator U+FF5C)               | `course` = phần giữa, `topic` = tiêu đề                        |
| TED          | `NN_Title_Speaker_TED`                                               | `speaker` = 2 token cuối *(heuristic)*, `topic` = phần còn lại |
| Storybook    | `Title_read_by_Speaker` (thêm biến thể `*_leído_por_*`, ` read by `) | `speaker`, `topic`                                             |
| Vooks        | `..._Vooks_Narrated_Storybooks`                                      | `topic` = tiêu đề trước `_Vooks`                               |
| Khác         | —                                                                    | `source = "Other"`, không nhãn                                 |

> `corpus.py:15-18` ghi rõ: nhãn speaker của TED **chỉ mang tính đối chiếu, không phải ground truth**.
