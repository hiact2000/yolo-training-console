"""Render the Cam8 hold-out comparison figures and the report's metric tables.

Reads experiments/cam8_holdout_yolov8s/results/cam8_test_comparison.csv, writes:
  figures/cam8_test_precision_recall_comparison.png
  figures/cam8_test_map_comparison.png
  figures/cam8_val_vs_test_gap.png
  figures/expA|expB|expC_confusion_matrix.png   (copied from the Ultralytics test eval)
  figures/cam8_error_breakdown.png              (if the error analysis has run)
  results/metric_tables.md                      (markdown fragment for the report)

Palette: dataviz categorical slots 1-3 (blue / orange / aqua), validated all-pairs
in light mode. Aqua sits below 3:1 on the light surface, so every bar carries a
visible value label -- that is the required relief, not decoration.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

SERIES = {"expA": "#2a78d6", "expB": "#eb6834", "expC": "#1baf7a"}
LABELS = {
    "expA": "Exp A · original only",
    "expB": "Exp B · CCTV Cam9+24",
    "expC": "Exp C · original + CCTV",
}
RUN_NAMES = {
    "expA": "expA_original_yolov8s_cam8test",
    "expB": "expB_cctv_cam9_24_yolov8s_cam8test",
    "expC": "expC_original_cctv_cam9_24_yolov8s_cam8test",
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": BASELINE,
    }
)


def corner_geometry(fig, ax, radius_px: float = 4.0) -> tuple[float, float]:
    """Rounding size (x data units) + mutation_aspect for a visually circular corner.

    FancyBboxPatch applies one rounding_size in patch coordinates after dividing y by
    mutation_aspect. x and y here are on wildly different scales (score vs box count),
    so a single radius would distort into a lens; solving for both axes keeps the
    corner a true `radius_px` circle in display space.
    """
    fig.canvas.draw()
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    radius_x = radius_px * (x1 - x0) / max(box.width, 1e-9)
    radius_y = radius_px * (y1 - y0) / max(box.height, 1e-9)
    return radius_x, radius_y / max(radius_x, 1e-12)


def rounded_bar(ax, x, width, height, colour, corners) -> None:
    """A bar with a rounded data-end, square at the baseline.

    The patch is extended below zero by the corner radius and clipped at the axes
    floor, so only the top corners survive.
    """
    rounding, aspect = corners
    rounding = min(rounding, width * 0.35)
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, -rounding * aspect),
            width,
            height + rounding * aspect,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            linewidth=0,
            facecolor=colour,
            mutation_aspect=aspect,
            clip_on=True,
        )
    )


def style_axes(ax, tick_positions, tick_labels, xlim, ylim, ylabel, title, label_size=10) -> None:
    """Recessive chrome: horizontal grid only, no box, no tick marks."""
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=label_size, color=INK_2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, color=INK, pad=14, loc="left")
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)


def draw_groups(fig, ax, groups, values, order, label_fmt, label_pad_frac=0.018) -> None:
    """One group per entry in `groups`, one bar per experiment. Limits must be set already."""
    n = len(order)
    bar_width = 0.72 / n
    gap = bar_width * 0.06  # ~2px surface gap between adjacent bars
    corners = corner_geometry(fig, ax)
    pad = (ax.get_ylim()[1] - ax.get_ylim()[0]) * label_pad_frac
    for group_index, group in enumerate(groups):
        for series_index, key in enumerate(order):
            centre = group_index + (series_index - (n - 1) / 2) * bar_width
            height = values[key].get(group, 0.0)
            rounded_bar(ax, centre, bar_width - gap, height, SERIES[key], corners)
            ax.text(
                centre,
                height + pad,
                label_fmt(height),
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK_2,
            )


def legend(fig, order, y=0.0) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=SERIES[k], label=LABELS[k])
        for k in order
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=len(order),
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handletextpad=0.5,
        columnspacing=1.8,
    )


def read_comparison(path: Path):
    by_exp_split: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_exp_split[(row["experiment"], row["split"])] = {
                "Precision": float(row["precision"]),
                "Recall": float(row["recall"]),
                "F1": float(row["f1"]),
                "mAP50": float(row["mAP50"]),
                "mAP50-95": float(row["mAP50_95"]),
                "n_images": int(row["n_images"]),
                "n_labels": int(row["n_labels"]),
                "weights": row["weights"],
                "inference_ms": float(row["speed_inference_ms"]),
                "tp": row["confusion_matrix_tp"],
                "fp": row["confusion_matrix_fp"],
                "fn": row["confusion_matrix_fn"],
            }
    return by_exp_split


def _score_figure(data, order, metrics, title, out: Path, width: float) -> None:
    fig, ax = plt.subplots(figsize=(width, 4.4))
    values = {k: data[(k, "test")] for k in order}
    top = max(max(v.get(m, 0) for m in metrics) for v in values.values())
    style_axes(
        ax,
        range(len(metrics)),
        metrics,
        (-0.6, len(metrics) - 0.4),
        (0, min(1.06, top * 1.22) if top else 1.0),
        "score (Cam8 hold-out test)",
        title,
    )
    fig.subplots_adjust(bottom=0.24)
    draw_groups(fig, ax, metrics, values, order, lambda v: f"{v:.3f}")
    legend(fig, order, y=0.005)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_precision_recall(data, order, out: Path) -> None:
    _score_figure(
        data, order, ["Precision", "Recall", "F1"],
        "Cam8 hold-out test — precision, recall, F1", out, 7.6,
    )


def figure_map(data, order, out: Path) -> None:
    _score_figure(
        data, order, ["mAP50", "mAP50-95"],
        "Cam8 hold-out test — mAP50 and mAP50-95", out, 6.6,
    )


def figure_val_vs_test(data, order, out: Path) -> None:
    """The generalisation gap: each model's own val score next to its Cam8 score."""
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), sharey=True)
    fig.subplots_adjust(bottom=0.26, top=0.80)
    for ax, metric in zip(axes, ("mAP50", "Recall")):
        values = {
            key: {"own val split": data[(key, "val")][metric],
                  "Cam8 hold-out test": data[(key, "test")][metric]}
            for key in order
        }
        style_axes(
            ax, [0, 1], ["own val split", "Cam8 hold-out test"], (-0.6, 1.4), (0, 1.12),
            "score" if metric == "mAP50" else "", metric,
        )
        ax.title.set_fontsize(12)
        draw_groups(
            fig, ax, ["own val split", "Cam8 hold-out test"], values, order, lambda v: f"{v:.3f}"
        )
    fig.suptitle(
        "Validation score vs Cam8 hold-out score — the cross-camera drop",
        fontsize=13, color=INK, x=0.012, ha="left", y=0.97,
    )
    legend(fig, order, y=0.005)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_error_breakdown(order, out: Path) -> bool:
    path = EXP_ROOT / "results" / "cam8_error_summary.csv"
    if not path.exists():
        return False
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["bucket"].startswith(("FP:", "FN:")):
                counts[row["experiment"]][row["bucket"]] = int(row["count"])
    if not counts:
        return False
    # False negatives first, then false positives; both alphabetical inside their group.
    buckets = sorted(
        {b for exp in counts.values() for b in exp}, key=lambda b: (not b.startswith("FN:"), b)
    )
    values = {k: {b: counts.get(k, {}).get(b, 0) for b in buckets} for k in order}
    top = max(max(v.values()) for v in values.values()) or 1
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    fig.subplots_adjust(bottom=0.30)
    style_axes(
        ax,
        range(len(buckets)),
        [b.replace(":", "\n") for b in buckets],
        (-0.6, len(buckets) - 0.4),
        (0, top * 1.18),
        "boxes on the Cam8 hold-out set",
        "Cam8 error cases by category (conf 0.25, match IoU 0.5)",
        label_size=8.5,
    )
    draw_groups(fig, ax, buckets, values, order, lambda v: f"{v:.0f}")
    legend(fig, order, y=0.005)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def copy_confusion_matrices(project: Path, figures: Path) -> list[str]:
    copied = []
    for key, run_name in RUN_NAMES.items():
        source = project / f"{run_name}_testeval" / "confusion_matrix.png"
        if source.exists():
            shutil.copy2(source, figures / f"{key}_confusion_matrix.png")
            copied.append(key)
        normalised = project / f"{run_name}_testeval" / "confusion_matrix_normalized.png"
        if normalised.exists():
            shutil.copy2(normalised, figures / f"{key}_confusion_matrix_normalized.png")
    return copied


def markdown_tables(data, order) -> str:
    lines = ["### Cam8 hold-out test results", ""]
    lines.append(
        "| Experiment | Test images | Test labels | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms/img) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in order:
        row = data[(key, "test")]
        lines.append(
            f"| {LABELS[key]} | {row['n_images']} | {row['n_labels']} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['F1']:.4f} | {row['mAP50']:.4f} | {row['mAP50-95']:.4f} | "
            f"{row['inference_ms']:.2f} |"
        )
    lines += ["", "### Own-validation results (for reference, not the headline)", ""]
    lines.append("| Experiment | Val images | Val labels | Precision | Recall | mAP50 | mAP50-95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in order:
        row = data[(key, "val")]
        lines.append(
            f"| {LABELS[key]} | {row['n_images']} | {row['n_labels']} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['mAP50']:.4f} | {row['mAP50-95']:.4f} |"
        )
    lines += ["", "### Validation -> Cam8 generalisation gap", ""]
    lines.append("| Experiment | val mAP50 | Cam8 mAP50 | Δ mAP50 | val Recall | Cam8 Recall | Δ Recall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key in order:
        val, test = data[(key, "val")], data[(key, "test")]
        lines.append(
            f"| {LABELS[key]} | {val['mAP50']:.4f} | {test['mAP50']:.4f} | "
            f"{test['mAP50'] - val['mAP50']:+.4f} | {val['Recall']:.4f} | {test['Recall']:.4f} | "
            f"{test['Recall'] - val['Recall']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=DEFAULT_DATA_ROOT / "runs" / "cam8_holdout_yolov8s"
    )
    args = parser.parse_args()

    comparison = EXP_ROOT / "results" / "cam8_test_comparison.csv"
    if not comparison.exists():
        raise SystemExit(f"missing {comparison} -- run scripts/run_cam8_holdout.py first")

    data = read_comparison(comparison)
    order = [k for k in ("expA", "expB", "expC") if (k, "test") in data]
    figures = EXP_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    figure_precision_recall(data, order, figures / "cam8_test_precision_recall_comparison.png")
    figure_map(data, order, figures / "cam8_test_map_comparison.png")
    figure_val_vs_test(data, order, figures / "cam8_val_vs_test_gap.png")
    made_errors = figure_error_breakdown(order, figures / "cam8_error_breakdown.png")
    copied = copy_confusion_matrices(args.project, figures)

    (EXP_ROOT / "results" / "metric_tables.md").write_text(
        markdown_tables(data, order), encoding="utf-8"
    )

    print(f"figures -> {figures}")
    print(f"  confusion matrices copied for: {', '.join(copied) or 'none'}")
    print(f"  error breakdown: {'yes' if made_errors else 'skipped (run analyze_cam8_errors.py)'}")
    print(f"tables  -> {EXP_ROOT / 'results' / 'metric_tables.md'}")


if __name__ == "__main__":
    main()
