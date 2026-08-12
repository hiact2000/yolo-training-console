"""Figures comparing the previous experiment's models with the Cam8 hold-out runs.

Every model here is scored on the same 103 Cam8 images, so the bars are directly
comparable -- but the previous CCTV/merged models trained on Cam8, so their scores
are inflated by construction. Leaked bars are drawn in red *and* hatched *and*
labelled, so the distinction never rests on colour alone.

Reads results/prev_vs_new_cam8.csv, writes into figures/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"

CLEAN = "#2a78d6"
LEAKED = "#d03b3b"
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

SHORT = {
    "prev_original": "Prev\noriginal\n(2026-03)",
    "prev_original_aborted": "Prev\noriginal\n8-epoch",
    "prev_cctv_all": "Prev\nCCTV\n8+9+24",
    "prev_merged": "Prev\nmerged\n(incl. Cam8)",
    "expA": "New\noriginal\nonly",
    "expB": "New\nCCTV\n9+24",
    "expC": "New\noriginal\n+CCTV",
}
ORDER = [
    "prev_original",
    "prev_original_aborted",
    "expA",
    "prev_cctv_all",
    "expB",
    "prev_merged",
    "expC",
]


def corner_geometry(fig, ax, radius_px: float = 4.0) -> tuple[float, float]:
    fig.canvas.draw()
    box = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    radius_x = radius_px * (x1 - x0) / max(box.width, 1e-9)
    radius_y = radius_px * (y1 - y0) / max(box.height, 1e-9)
    return radius_x, radius_y / max(radius_x, 1e-12)


def rounded_bar(ax, x, width, height, colour, corners, hatch=None) -> None:
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
            hatch=hatch,
            edgecolor="#ffffff",
            mutation_aspect=aspect,
            clip_on=True,
        )
    )


def style_axes(ax, ticks, labels, xlim, ylim, ylabel, title, label_size=8.5) -> None:
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=label_size, color=INK_2)
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


def legend(fig, y=0.005) -> None:
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=10, color=CLEAN,
                   label="Cam8 held out — trustworthy"),
        plt.Line2D([], [], marker="X", linestyle="", markersize=10, color=LEAKED,
                   label="Cam8 was IN training — inflated, not a generalisation score"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=2,
        frameon=False, fontsize=9, labelcolor=INK_2, handletextpad=0.5, columnspacing=2.0,
    )


def read_rows(path: Path) -> dict[str, dict]:
    return {r["key"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}


def figure_cam8_landscape(rows, out: Path) -> None:
    """All seven models on the same Cam8 test set."""
    keys = [k for k in ORDER if k in rows]
    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    fig.subplots_adjust(bottom=0.30)
    style_axes(
        ax, range(len(keys)), [SHORT[k] for k in keys], (-0.6, len(keys) - 0.4), (0, 1.10),
        "mAP50 on the Cam8 hold-out set (103 images)",
        "Same 103 Cam8 images, seven models — old runs scored Cam8 they had trained on",
    )
    corners = corner_geometry(fig, ax)
    for i, key in enumerate(keys):
        row = rows[key]
        leaked = row["cam8_leaked"] == "YES"
        value = float(row["cam8_mAP50"])
        rounded_bar(ax, i, 0.62, value, LEAKED if leaked else CLEAN, corners,
                    hatch="////" if leaked else None)
        ax.text(i, value + 0.022, f"{value:.3f}", ha="center", va="bottom",
                fontsize=9.5, color=INK_2)
        if leaked:
            ax.text(i, value + 0.062, "leaked", ha="center", va="bottom",
                    fontsize=8, color=LEAKED, fontweight="bold")
    legend(fig)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_leakage_inflation(rows, out: Path) -> None:
    """Matched pairs: same training pool, Cam8 in vs out."""
    pairs = [
        ("prev_cctv_all", "expB", "CCTV-only training"),
        ("prev_merged", "expC", "original + CCTV training"),
    ]
    pairs = [(a, b, t) for a, b, t in pairs if a in rows and b in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    fig.subplots_adjust(bottom=0.30)
    style_axes(
        ax, range(len(pairs)), [t for _, _, t in pairs], (-0.6, len(pairs) - 0.4), (0, 1.14),
        "mAP50 on the Cam8 hold-out set",
        "How much does training on Cam8 inflate the Cam8 score?",
        label_size=10,
    )
    corners = corner_geometry(fig, ax)
    for i, (leak_key, clean_key, _) in enumerate(pairs):
        leaked_v = float(rows[leak_key]["cam8_mAP50"])
        clean_v = float(rows[clean_key]["cam8_mAP50"])
        rounded_bar(ax, i - 0.17, 0.30, leaked_v, LEAKED, corners, hatch="////")
        rounded_bar(ax, i + 0.17, 0.30, clean_v, CLEAN, corners)
        ax.text(i - 0.17, leaked_v + 0.02, f"{leaked_v:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK_2)
        ax.text(i + 0.17, clean_v + 0.02, f"{clean_v:.3f}", ha="center", va="bottom",
                fontsize=9, color=INK_2)
        ax.annotate(
            f"−{leaked_v - clean_v:.3f}",
            xy=(i, max(leaked_v, clean_v) + 0.075),
            ha="center", fontsize=10.5, color=LEAKED, fontweight="bold",
        )
    legend(fig)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure_val_vs_cam8(rows, out: Path) -> None:
    """Own validation score against the Cam8 score, for every model."""
    keys = [k for k in ORDER if k in rows]
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.plot([0.3, 1.0], [0.3, 1.0], linestyle="--", linewidth=1.2, color=BASELINE, zorder=1)
    ax.text(0.965, 0.985, "val = Cam8", fontsize=8.5, color=MUTED, ha="right", rotation=38)
    for key in keys:
        row = rows[key]
        leaked = row["cam8_leaked"] == "YES"
        x, y = float(row["own_val_mAP50"]), float(row["cam8_mAP50"])
        ax.scatter(
            x, y, s=170, zorder=3,
            marker="X" if leaked else "o",
            color=LEAKED if leaked else CLEAN,
            edgecolor=SURFACE, linewidth=2,
        )
        ax.annotate(
            SHORT[key].replace("\n", " "),
            xy=(x, y), xytext=(0, -20), textcoords="offset points",
            ha="center", fontsize=8.5, color=INK_2,
        )
    ax.set_xlim(0.72, 1.0)
    ax.set_ylim(0.35, 1.03)
    ax.set_xlabel("mAP50 on the model's own validation split", fontsize=10)
    ax.set_ylabel("mAP50 on the Cam8 hold-out set", fontsize=10)
    ax.set_title(
        "A high validation score does not predict the Cam8 score",
        fontsize=13, color=INK, pad=14, loc="left",
    )
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    fig.subplots_adjust(bottom=0.20)
    legend(fig, y=0.005)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXP_ROOT / "figures")
    args = parser.parse_args()

    source = EXP_ROOT / "results" / "prev_vs_new_cam8.csv"
    if not source.exists():
        raise SystemExit(f"missing {source} -- run scripts/eval_prev_on_cam8.py first")
    rows = read_rows(source)
    args.out.mkdir(parents=True, exist_ok=True)

    figure_cam8_landscape(rows, args.out / "prev_vs_new_cam8_landscape.png")
    figure_leakage_inflation(rows, args.out / "prev_vs_new_leakage_inflation.png")
    figure_val_vs_cam8(rows, args.out / "prev_vs_new_val_vs_cam8.png")
    for name in (
        "prev_vs_new_cam8_landscape.png",
        "prev_vs_new_leakage_inflation.png",
        "prev_vs_new_val_vs_cam8.png",
    ):
        size = (args.out / name).stat().st_size / 1024
        print(f"  {name}  {size:.0f} KB")
    print(f"figures -> {args.out}")


if __name__ == "__main__":
    main()
