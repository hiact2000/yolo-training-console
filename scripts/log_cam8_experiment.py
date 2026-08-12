"""Record the three Cam8 hold-out runs in the project's train_log.xlsx.

Uses train_ui.excel_logger so the rows keep the schema the existing entries use.
That logger takes its metrics from the *last* row of results.csv -- the final
epoch, not the best one -- so with patience=30 the logged mAP is ~30 epochs past
the checkpoint that best.pt actually holds. We keep that behaviour for
consistency with the two 2026-07-15 rows, then append extra columns carrying the
best-epoch numbers and, more importantly, the Cam8 hold-out scores, which the
stock schema has nowhere to put.

Idempotent: re-running replaces this experiment's rows instead of duplicating.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

RUNS = {
    "expA": ("expA_original_yolov8s_cam8test", "original dataset only (Cam8 removed at source)"),
    "expB": ("expB_cctv_cam9_24_yolov8s_cam8test", "CCTV Cam9 + Cam24 only"),
    "expC": ("expC_original_cctv_cam9_24_yolov8s_cam8test", "original + CCTV Cam9/24"),
}
EXPERIMENT_TAG = "cam8_holdout_yolov8s"

EXTRA_COLS = [
    "experiment_tag",
    "best_epoch",
    "best_mAP50",
    "best_mAP50_95",
    "cam8_test_images",
    "cam8_test_labels",
    "cam8_precision",
    "cam8_recall",
    "cam8_f1",
    "cam8_mAP50",
    "cam8_mAP50_95",
    "cam8_leaked",
    "split_seed",
]


def best_epoch_metrics(run_dir: Path) -> tuple[int, float, float]:
    path = run_dir / "results.csv"
    if not path.exists():
        return 0, float("nan"), float("nan")
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    if frame.empty:
        return 0, float("nan"), float("nan")
    best = frame.loc[frame["metrics/mAP50-95(B)"].idxmax()]
    return (
        int(best["epoch"]),
        float(best["metrics/mAP50(B)"]),
        float(best["metrics/mAP50-95(B)"]),
    )


def cam8_metrics() -> dict[str, dict[str, str]]:
    path = EXP_ROOT / "results" / "cam8_test_comparison.csv"
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] == "test":
                out[row["experiment"]] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--excel", type=Path, default=None, help="Defaults to <data-root>/train_log.xlsx"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT))
    from train_ui import excel_logger

    excel_path = args.excel or (args.data_root / "train_log.xlsx")
    project = args.data_root / "runs" / EXPERIMENT_TAG
    cam8 = cam8_metrics()

    # Drop any previous attempt at logging this experiment, so re-runs stay clean.
    if excel_path.exists():
        existing = pd.read_excel(excel_path)
        if "experiment_tag" in existing.columns:
            keep = existing[existing["experiment_tag"] != EXPERIMENT_TAG]
            if len(keep) != len(existing):
                keep.to_excel(excel_path, index=False)
                print(f"removed {len(existing) - len(keep)} earlier row(s) for {EXPERIMENT_TAG}")

    for key, (run_name, pool) in RUNS.items():
        run_dir = project / run_name
        if not (run_dir / "args.yaml").exists():
            print(f"[{key}] no args.yaml at {run_dir}, skipping")
            continue
        test = cam8[key]
        note = (
            f"Cam8 hold-out experiment ({key}). Training pool: {pool}. "
            f"Cam8 held out entirely -- 0 Cam8 images in train/val, verified by assertion. "
            f"Tested on 103 Cam8 images / 798 effective boxes: "
            f"P={float(test['precision']):.4f} R={float(test['recall']):.4f} "
            f"mAP50={float(test['mAP50']):.4f} mAP50-95={float(test['mAP50_95']):.4f}. "
            f"Split seed 42, grouped by source image (Roboflow triplicates kept on one side). "
            f"Augmentation = Ultralytics defaults (mosaic=1.0) copied from the previous "
            f"experiment -- this is NOT a no-augmentation run. "
            f"Branch feature/cam8-holdout-yolov8s-experiment."
        )
        written = excel_logger.log_to_excel(
            save_dir=str(run_dir), ui_config={}, notes=note, excel_path=str(excel_path)
        )
        print(f"[{key}] logged -> {written}")

    # Second pass: attach the columns the stock schema has no room for.
    frame = pd.read_excel(excel_path)
    for column in EXTRA_COLS:
        if column not in frame.columns:
            frame[column] = None
    for key, (run_name, _) in RUNS.items():
        mask = frame["run_tag"].astype(str).str.endswith(run_name)
        if not mask.any():
            continue
        epoch, map50, map95 = best_epoch_metrics(project / run_name)
        test = cam8[key]
        # log_to_excel builds run_tag from the raw project arg, which we passed as an
        # absolute path. Normalise it to the `runs/<project>/<name>` form the two
        # 2026-07-15 rows use, so the column stays sortable and readable.
        frame.loc[mask, "run_tag"] = f"runs/{EXPERIMENT_TAG}/{run_name}"
        frame.loc[mask, "experiment_tag"] = EXPERIMENT_TAG
        frame.loc[mask, "best_epoch"] = epoch
        frame.loc[mask, "best_mAP50"] = round(map50, 6)
        frame.loc[mask, "best_mAP50_95"] = round(map95, 6)
        frame.loc[mask, "cam8_test_images"] = 103
        frame.loc[mask, "cam8_test_labels"] = 798
        frame.loc[mask, "cam8_precision"] = float(test["precision"])
        frame.loc[mask, "cam8_recall"] = float(test["recall"])
        frame.loc[mask, "cam8_f1"] = float(test["f1"])
        frame.loc[mask, "cam8_mAP50"] = float(test["mAP50"])
        frame.loc[mask, "cam8_mAP50_95"] = float(test["mAP50_95"])
        frame.loc[mask, "cam8_leaked"] = "no"
        frame.loc[mask, "split_seed"] = 42
    frame.to_excel(excel_path, index=False)

    print(f"\n{excel_path}: {len(frame)} rows, {len(frame.columns)} columns")
    view = frame[["timestamp", "run_tag", "best_mAP50", "cam8_mAP50", "experiment_tag"]]
    print(view.to_string(max_colwidth=52))


if __name__ == "__main__":
    main()
