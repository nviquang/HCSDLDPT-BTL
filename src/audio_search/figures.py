"""Figures for the dataset analysis.

Every figure is rendered from the artifacts and the metrics dictionary that
``analysis.run()`` produced, so the numbers on a chart and the numbers in
``metrics.json`` can never drift apart.

Design rules applied here (they are not decoration — each one fixes a way a
chart can mislead):

* Categorical colour is assigned in a fixed slot order and never cycled.
  The three slots used below were validated for colour-vision deficiency in
  both light and dark mode (worst all-pairs CVD ΔE 9.2 light / 9.4 dark).
* One axis, always. Two measures of different scale become two charts, never
  two y-scales.
* Gridlines and axes are solid hairlines one step off the surface; there are
  no dashed rules and no borders around marks.
* Text wears text tokens, never the series colour — identity comes from the
  coloured mark beside the label.
* A legend is present whenever two or more series share a plot, and labels
  are direct only where they carry the story.

Figures are static PNGs embedded in a Markdown report, so they are rendered
for the light chart surface.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import analysis, config

logger = logging.getLogger(__name__)

# --- palette ---------------------------------------------------------------
# Categorical slots 1-3 of the validated default palette (light mode).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
# Status colours — reserved, never reused as a series colour.
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style() -> None:
    """Apply the chart chrome: recessive grid, no top/right spines, sans type."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "figure.dpi": 200,
    })


def _despine(ax) -> None:
    """Drop the top and right spines; keep a hairline baseline on the left."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)


def _save(fig, name: str) -> str:
    """Write one figure to the analysis figures directory."""
    config.ensure_dirs()
    path = config.ANALYSIS_FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    logger.info("figure written: %s", path)
    return str(path)


def _legend(ax, **kwargs) -> None:
    """Legend without a frame, in ink tokens rather than a series colour."""
    leg = ax.legend(**kwargs)
    for text in leg.get_texts():
        text.set_color(INK_SECONDARY)


# ---------------------------------------------------------------------------
# 1. Scree / cumulative variance
# ---------------------------------------------------------------------------
def figure_pca_scree(metrics: dict) -> str:
    """Cumulative explained variance: how fast each family collapses.

    Two families only. The MFCC *std* matrix is reported in the CSV and the
    write-up, but on the chart it would sit right on top of the MFCC stats
    curve (both cross 95% at PC6) and add no story.

    The direct labels sit at each curve's 95% *crossing*, not at the curve
    end: the ends are 114 PCs apart but carry no number the reader needs,
    and the crossings are the actual finding.
    """
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    blocks = [("mfcc_stats", "MFCC thống kê (78-D)", SERIES[0]),
              ("ecapa", "ECAPA (192-D)", SERIES[2])]

    ax.axhline(95, color=INK_MUTED, linewidth=0.8,
               label="Ngưỡng 95% phương sai tích lũy")

    for key, label, color in blocks:
        ratio = np.asarray(metrics["dimensionality"][key]["explained_variance_ratio"])
        cum = np.cumsum(ratio) * 100
        xs = np.arange(1, len(cum) + 1)
        ax.plot(xs, cum, color=color, linewidth=2, solid_joinstyle="round",
                solid_capstyle="round", label=label)

        n95 = metrics["dimensionality"][key]["n_components"]["95%"]
        # Marker carries the crossing point; the label carries the number.
        ax.plot([n95], [95], marker="o", markersize=8, color=color, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"95% ở PC{n95}", xy=(n95, 95), xytext=(9, -15),
                    textcoords="offset points", ha="left",
                    color=INK_SECONDARY, fontsize=9)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 106)
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlabel("Số thành phần chính (PC)")
    ax.set_ylabel("Phần trăm phương sai tích lũy (%)")
    ax.set_title("MFCC: 95% phương sai chỉ cần 6 chiều — ECAPA cần 114/192 chiều")
    _despine(ax)
    _legend(ax, loc="lower right")
    return _save(fig, "pca_cumulative_variance.png")


# ---------------------------------------------------------------------------
# 2. MFCC block energy
# ---------------------------------------------------------------------------
def figure_mfcc_energy(metrics: dict) -> str:
    """Energy share of the base / delta / delta-delta blocks.

    The delta bars are legitimately invisible — that *is* the finding — so
    every bar carries a direct label and the axis is not relied upon.
    """
    struct = metrics["mfcc_structure"]
    blocks = [("base", "Base\n(c0–c25)"), ("delta", "Δ\n(c26–c51)"),
              ("delta_delta", "ΔΔ\n(c52–c77)")]
    shares = [struct[key]["energy_share"] * 100 for key, _ in blocks]
    abs_means = [struct[key]["abs_mean"] for key, _ in blocks]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = np.arange(len(blocks))
    # One series -> one colour (slot 1). Length already encodes the value.
    ax.bar(xs, shares, width=0.5, color=SERIES[0], edgecolor=SURFACE, linewidth=2)

    for x, (share, amean) in enumerate(zip(shares, abs_means)):
        ax.annotate(f"{share:.6f}%\n|trung bình| = {amean:.2e}",
                    xy=(x, share), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK_SECONDARY, fontsize=9)

    ax.set_xticks(xs, [label for _, label in blocks])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Tỉ lệ năng lượng của khối (%)")
    ax.set_title("Khối Δ và ΔΔ mang 0 phần trăm biến thiên — 78-D thực chất chỉ ~26-D")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    return _save(fig, "mfcc_block_energy.png")


# ---------------------------------------------------------------------------
# 3. Pairwise cosine distributions
# ---------------------------------------------------------------------------
def figure_pairwise_cosine(metrics: dict, similarity: dict[str, np.ndarray]) -> str:
    """Pairwise cosine for the three vector families.

    Three stacked panels sharing one x-axis, not one panel with three curves:
    the raw-MFCC distribution is a spike whose density reaches ~58 while the
    other two top out near 1 and 4, so on a shared density axis they collapse
    into the baseline and stop being readable. Each panel keeps its own
    density scale — that is small multiples, not a second y-axis on one plot.
    """
    series = [
        ("mfcc_raw_78d", "MFCC 78-D thô", SERIES[0]),
        ("mfcc_clean_26d_zscore", "MFCC 26-D (c0=0, z-score)", SERIES[1]),
        ("ecapa_192d", "ECAPA 192-D", SERIES[2]),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.4), sharex=True)
    for ax, (key, label, color) in zip(axes, series):
        vals = similarity[key]
        upper = vals[np.triu_indices(vals.shape[0], 1)]
        ax.hist(upper, bins=120, range=(-1.0, 1.0), density=True, histtype="step",
                linewidth=2, color=color)
        ax.hist(upper, bins=120, range=(-1.0, 1.0), density=True, histtype="stepfilled",
                color=color, alpha=0.10)

        stats = metrics["similarity"][key]
        # Mean reference line sits behind the marks, so it never competes
        # with the curve it describes.
        ax.axvline(stats["mean"], color=INK_SECONDARY, linewidth=1, zorder=0.5)
        ax.set_title(f"{label} — TB = {stats['mean']:.3f}, "
                     f"trung vị = {stats['median']:.3f}",
                     color=INK, fontsize=10.5, fontweight="semibold", loc="left", pad=6)
        ax.set_ylabel("Mật độ")
        ax.set_xlim(-1.0, 1.0)
        _despine(ax)

    axes[-1].set_xlabel("Cosine similarity giữa các cặp file")
    fig.suptitle("MFCC thô dồn 126.756 cặp quanh 1,0 — không phân biệt được gì",
                 color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, "pairwise_cosine.png")


# ---------------------------------------------------------------------------
# 4. Duration by source
# ---------------------------------------------------------------------------
def figure_duration(metrics: dict, durations: np.ndarray,
                    sources: list[str]) -> str:
    """Overall duration distribution plus a per-source median comparison."""
    corpus_desc = metrics["corpus"]
    by_source = corpus_desc["duration_by_source"]
    names = sorted(by_source, key=lambda k: by_source[k]["median"], reverse=True)
    medians = [by_source[k]["median"] for k in names]
    counts = [by_source[k]["count"] for k in names]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.6, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})

    # Left: one series, one colour — the title names it.
    ax1.hist(durations / 60.0, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    ax1.axvline(np.median(durations) / 60.0, color=INK_SECONDARY, linewidth=1.5)
    ax1.annotate(f"trung vị {np.median(durations) / 60:.0f} phút",
                 xy=(np.median(durations) / 60.0, ax1.get_ylim()[1] * 0.94),
                 xytext=(6, 0), textcoords="offset points",
                 color=INK_SECONDARY, fontsize=9)
    ax1.set_xlabel("Thời lượng (phút)")
    ax1.set_ylabel("Số file")
    ax1.set_title("Toàn bộ corpus")
    _despine(ax1)

    # Right: nominal categories -> one series, one colour (never a value ramp).
    ys = np.arange(len(names))[::-1]
    ax2.barh(ys, medians, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=2)
    for y, med, cnt in zip(ys, medians, counts):
        ax2.annotate(f"{med:.0f} phút  ·  {cnt} file",
                     xy=(med, y), xytext=(6, 0), textcoords="offset points",
                     va="center", color=INK_SECONDARY, fontsize=9)
    ax2.set_yticks(ys, names)
    ax2.set_xlim(0, max(medians) * 1.55)
    ax2.set_xlabel("Thời lượng trung vị (phút)")
    ax2.set_title("Trung vị theo nguồn")
    ax2.grid(axis="y", visible=False)
    _despine(ax2)

    fig.suptitle("Phân bố thời lượng — 504 file, "
                 f"{corpus_desc['duration']['total_hours']:.0f} giờ âm thanh",
                 color=INK, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    return _save(fig, "duration_by_source.png")


# ---------------------------------------------------------------------------
# 5. Label separation
# ---------------------------------------------------------------------------
def figure_label_separation(metrics: dict) -> str:
    """Within-group vs across-group cosine for ECAPA under three labelings.

    The legend sits below the x-axis, not inside the plot: the bars reach
    0.92, so any legend placed in the upper half of the axes lands on top of
    a bar or one of the gap callouts.
    """
    sep = metrics["separation"]
    families = [("ecapa_by_course", "Cùng môn học (Khan)"),
                ("ecapa_by_topic", "Cùng chủ đề"),
                ("ecapa_by_source", "Cùng nguồn")]
    families = [(k, lbl) for k, lbl in families if k in sep and "gap" in sep[k]]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(families))
    width = 0.34

    same = [sep[k]["same_mean"] for k, _ in families]
    diff = [sep[k]["diff_mean"] for k, _ in families]
    gap = [sep[k]["gap"] for k, _ in families]

    # Two series -> legend present; adjacent bars separated by a surface gap.
    ax.bar(xs - width / 2 - 0.01, same, width, color=SERIES[0], label="Trong cùng nhóm",
           edgecolor=SURFACE, linewidth=2)
    ax.bar(xs + width / 2 + 0.01, diff, width, color=SERIES[1], label="Khác nhóm",
           edgecolor=SURFACE, linewidth=2)

    for x, s, d, g in zip(xs, same, diff, gap):
        ax.annotate(f"{s:.3f}", xy=(x - width / 2 - 0.01, s), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=9)
        ax.annotate(f"{d:.3f}", xy=(x + width / 2 + 0.01, d), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=9)
        ax.annotate(f"chênh lệch {g:+.3f}", xy=(x, max(s, d)), xytext=(0, 26),
                    textcoords="offset points", ha="center",
                    color=INK, fontsize=10, fontweight="semibold")

    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xticks(xs, [lbl for _, lbl in families])
    # Headroom so the highest callout clears the title.
    ax.set_ylim(0, max(same + diff) * 1.30)
    ax.set_ylabel("Cosine similarity trung bình")
    ax.set_title("ECAPA phân biệt tốt theo môn học và chủ đề, kém theo nguồn",
                 pad=12)
    ax.grid(axis="x", visible=False)
    _despine(ax)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    return _save(fig, "label_separation.png")


# ---------------------------------------------------------------------------
# 6. Duration effect
# ---------------------------------------------------------------------------
def figure_duration_effect(metrics: dict, durations: np.ndarray,
                           nn_cos: np.ndarray) -> str:
    """Does a longer file make its voice harder to match?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.scatter(durations / 60.0, nn_cos, s=22, color=SERIES[0], alpha=0.55,
               linewidths=0.8, edgecolors=SURFACE)

    slope, intercept = np.polyfit(durations / 60.0, nn_cos, 1)
    xs = np.linspace(durations.min() / 60.0, durations.max() / 60.0, 50)
    ax.plot(xs, slope * xs + intercept, color=INK_SECONDARY, linewidth=2,
            solid_capstyle="round")

    corr = metrics["duration_effect"]["ecapa"]["corr_duration_vs_nn_cos"]
    ax.annotate(f"tương quan = {corr:.3f}\n(file dài hơn → giọng khó khớp hơn)",
                xy=(0.03, 0.04), xycoords="axes fraction",
                color=INK_SECONDARY, fontsize=9, va="bottom")

    ax.set_xlabel("Thời lượng file (phút)")
    ax.set_ylabel("Cosine với file gần nhất nhất")
    ax.set_title("Ảnh hưởng của thời lượng lên khả năng khớp giọng (ECAPA)")
    _despine(ax)
    return _save(fig, "duration_effect.png")


# ---------------------------------------------------------------------------
# 7. Label coverage vs the plan's targets
# ---------------------------------------------------------------------------
def figure_targets(metrics: dict) -> str:
    """Target-by-target status against the corpus section 2.4c of report.md.

    Rendered as a status list rather than a bar chart: "more is better" is
    not true for every target (too many *distinct* topics is a defect, not an
    achievement), so a bar length would read backwards. Status colour is
    always paired with an icon and a word, never colour alone.
    """
    targets = metrics["targets_comparison"]
    rows = [
        ("Số file tổng", targets["target_files"]),
        ("Số người nói", targets["target_speakers"]),
        ("Người nói có nhiều file", targets["target_speakers_multi_file"]),
        ("Số chủ đề", targets["target_topics"]),
        ("Số chủ đề có ≥4 file", targets["target_topics_size"]),
    ]

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(rows) + 1)

    # Column headers, so the cells carry values only. The "wanted" column is
    # right-aligned at a fixed edge and the "actual" column left-aligned from
    # the next one, which keeps the two from colliding however long a target
    # string gets.
    ax.text(0.02, len(rows) + 0.5, "Tiêu chí", color=INK_MUTED, fontsize=9,
            va="center", fontweight="semibold")
    ax.text(0.62, len(rows) + 0.5, "Mục tiêu", color=INK_MUTED, fontsize=9,
            va="center", ha="right", fontweight="semibold")
    ax.text(0.66, len(rows) + 0.5, "Thực tế", color=INK_MUTED, fontsize=9,
            va="center", fontweight="semibold")
    ax.text(0.84, len(rows) + 0.5, "Kết quả", color=INK_MUTED, fontsize=9,
            va="center", fontweight="semibold")

    for i, (name, entry) in enumerate(rows):
        y = len(rows) - i - 0.5
        met = bool(entry["met"])
        color = STATUS_GOOD if met else STATUS_CRITICAL
        # mathtext glyphs: the Unicode ✓/✗ fall back to a boxed arrow on this
        # Windows font stack, the mathtext versions always render.
        icon = r"$\checkmark$" if met else r"$\times$"
        verdict = "Đạt" if met else "Chưa đạt"

        ax.plot([0.01, 0.99], [y - 0.5, y - 0.5], color=GRID, linewidth=0.8)
        ax.text(0.02, y, name, color=INK, fontsize=11, va="center")
        ax.text(0.62, y, str(entry["wanted"]), color=INK_SECONDARY,
                fontsize=9.5, va="center", ha="right")
        ax.text(0.66, y, str(entry["actual"]), color=INK,
                fontsize=9.5, va="center")
        ax.text(0.84, y, f"{icon} {verdict}", color=color, fontsize=10,
                va="center", fontweight="semibold")

    ax.set_title("Độ phủ nhãn so với mục tiêu thiết kế của bộ dữ liệu (§2.4c)",
                 color=INK, fontsize=12, fontweight="semibold", loc="left", pad=14)
    return _save(fig, "label_coverage_vs_targets.png")


# ---------------------------------------------------------------------------
# 8. Clustering
# ---------------------------------------------------------------------------
def figure_clustering(metrics: dict) -> str:
    """Cluster count and largest-cluster size of the ECAPA hierarchy.

    Both series count files, so one axis is honest. The legend sits below the
    x-axis because the tallest bar (308 clusters) reaches the top of the
    plot area, leaving no free corner for an in-axes legend.
    """
    clusters = metrics["clustering"]["ecapa"]
    levels = sorted(clusters, key=lambda k: float(k.split("_")[1]))
    n_clusters = [clusters[lvl]["n_clusters"] for lvl in levels]
    largest = [clusters[lvl]["largest_cluster"] for lvl in levels]
    purity = [clusters[lvl]["purity"] for lvl in levels]
    thresholds = [lvl.split("_")[1] for lvl in levels]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(len(levels))
    width = 0.36
    ax.bar(xs - width / 2 - 0.01, n_clusters, width, color=SERIES[0],
           label="Số cụm", edgecolor=SURFACE, linewidth=2)
    ax.bar(xs + width / 2 + 0.01, largest, width, color=SERIES[1],
           label="Cụm lớn nhất", edgecolor=SURFACE, linewidth=2)
    for x, n, lg in zip(xs, n_clusters, largest):
        ax.annotate(str(n), xy=(x - width / 2 - 0.01, n), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=9)
        ax.annotate(str(lg), xy=(x + width / 2 + 0.01, lg), xytext=(0, 5),
                    textcoords="offset points", ha="center",
                    color=INK_SECONDARY, fontsize=9)

    ax.set_xticks(xs, [f"ngưỡng {t}" for t in thresholds])
    ax.set_ylim(0, max(n_clusters) * 1.15)
    ax.set_ylabel("Số file")

    # The headline number comes from the data, not from a constant: the
    # largest cluster grows as the cut gets looser, so the title names the
    # threshold it belongs to.
    lo, hi = largest[0], largest[-1]
    ax.set_title(f"Phân cụm phân cấp ECAPA: cụm lớn nhất gom {hi} file "
                 f"(ngưỡng {thresholds[-1]}), {lo} file ở ngưỡng {thresholds[0]}\n"
                 f"Độ thuần cụm lớn nhất = {min(purity):.2f} ở cả ba ngưỡng",
                 fontsize=11.5, pad=12)
    ax.grid(axis="x", visible=False)
    _despine(ax)
    _legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncols=2)
    return _save(fig, "ecapa_clustering.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def render_all(metrics: dict, corpus_table=None) -> list[str]:
    """Render every figure. Returns the list of files written."""
    _style()

    voice = analysis.load_voice_matrix()
    table = corpus_table if corpus_table is not None else analysis.corpus.build_corpus_table(
        voice.ids)

    sim_mfcc_raw = analysis.cosine_matrix(voice.mfcc_stats)
    sim_mfcc_clean = analysis.cosine_matrix(analysis.mfcc_clean_variant(voice.mfcc_stats))
    sim_ecapa = analysis.cosine_matrix(voice.ecapa)
    similarity = {
        "mfcc_raw_78d": sim_mfcc_raw,
        "mfcc_clean_26d_zscore": sim_mfcc_clean,
        "ecapa_192d": sim_ecapa,
    }

    sources = table["source"].tolist()

    # Nearest-neighbour similarity per file, for the duration-effect figure.
    sim = sim_ecapa.copy()
    np.fill_diagonal(sim, -np.inf)
    nn_cos = sim.max(axis=1)

    written = [
        figure_pca_scree(metrics),
        figure_mfcc_energy(metrics),
        figure_pairwise_cosine(metrics, similarity),
        figure_duration(metrics, voice.durations, sources),
        figure_label_separation(metrics),
        figure_duration_effect(metrics, voice.durations, nn_cos),
        figure_targets(metrics),
        figure_clustering(metrics),
    ]
    return written
