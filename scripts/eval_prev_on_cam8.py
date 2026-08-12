"""Evaluate the previous experiment's models on the Cam8 hold-out test set.

The previous three-way comparison (2026-07-15) never held Cam8 out: the CCTV and
merged datasets both contain Cam8 frames in train *and* val, and the original
Roboflow dataset hides 10 more. Scoring those models on the same 103 Cam8 images
the new models were tested on therefore does two things at once:

  * puts old and new runs on one axis instead of comparing incomparable val sets;
  * measures how much Cam8-in-training inflates a Cam8 score -- the old models'
    numbers are leaked *by construction* and are labelled that way everywhere.

Never present a `cam8_leaked=True` row as evidence of generalisation. It is the
control, not the result.

Writes results/prev_vs_new_cam8.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

# Cam8 counts below are what each run *actually saw*, which depends on when it ran:
# datasets/detect grew on 2026-01-28 (487 base), 2026-03-05 (+7), 2026-03-18 (+34)
# and 2026-04-02 (+19 = 10 CAM08 + 9 CAM09). So the 2026-03-05 run predates Cam8
# entering the original dataset at all and is a genuinely clean cross-camera point;
# everything from 2026-04-02 onward is leaked.
PREVIOUS_RUNS = [
    {
        "rel": "egg_project/run_test_block10",
        "key": "prev_original",
        "label": "Prev · original dataset (2026-03)",
        "pool": "original dataset, 494 imgs (pre-CAM08 snapshot)",
        "cam8_train": 0,
        "cam8_val": 0,
        "date": "2026-03-05",
    },
    {
        "rel": "exp",
        "key": "prev_original_aborted",
        "label": "Prev · original, 8-epoch abort",
        "pool": "original dataset, 547 imgs (incl. 10 CAM08)",
        "cam8_train": 9,
        "cam8_val": 1,
        "date": "2026-04-05",
    },
    {
        "rel": "0715data_yolov8s_ui",
        "key": "prev_cctv_all",
        "label": "Prev · CCTV Cam8+9+24",
        "pool": "CCTV incl. Cam8 (0715data)",
        "cam8_train": 74,
        "cam8_val": 19,
        "date": "2026-07-15",
    },
    {
        "rel": "detect_0715data_merged_yolov8s_ui",
        "key": "prev_merged",
        "label": "Prev · original + CCTV (incl. Cam8)",
        "pool": "original + CCTV incl. Cam8 (merged)",
        "cam8_train": 83,
        "cam8_val": 20,
        "date": "2026-07-15",
    },
]

FIELDS = [
    "key",
    "label",
    "generation",
    "training_pool",
    "cam8_in_train",
    "cam8_in_val",
    "cam8_leaked",
    "run_date",
    "epochs_run",
    "best_epoch",
    "own_val_mAP50",
    "own_val_mAP50_95",
    "cam8_precision",
    "cam8_recall",
    "cam8_f1",
    "cam8_mAP50",
    "cam8_mAP50_95",
    "cam8_tp",
    "cam8_fp",
    "cam8_fn",
    "weights",
]


def scalar(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0])
        except Exception:
            return float("nan")


def own_val_from_results_csv(run_dir: Path) -> tuple[int, int, float, float]:
    """Best-epoch validation numbers as recorded during that run's own training."""
    path = run_dir / "results.csv"
    if not path.exists():
        return 0, 0, float("nan"), float("nan")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))]
    if not rows:
        return 0, 0, float("nan"), float("nan")

    def key(row):
        return scalar(row.get("metrics/mAP50-95(B)"))

    best = max(rows, key=key)
    return (
        len(rows),
        int(scalar(best.get("epoch", 0))),
        scalar(best.get("metrics/mAP50(B)")),
        scalar(best.get("metrics/mAP50-95(B)")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--eval-project",
        type=Path,
        default=DEFAULT_DATA_ROOT / "runs" / "cam8_holdout_prev_eval",
        help="Where the Ultralytics eval runs for the previous models are written.",
    )
    args = parser.parse_args()

    from ultralytics import YOLO

    # Any of the three yamls works: all point at the same cam8_test.txt.
    data_yaml = EXP_ROOT / "datasets" / "expB_cctv_cam9_24_cam8test.yaml"
    prev_root = args.data_root / "runs" / "detect"

    rows: list[dict] = []
    for spec in PREVIOUS_RUNS:
        rel, key, label = spec["rel"], spec["key"], spec["label"]
        pool, cam8_train, cam8_val = spec["pool"], spec["cam8_train"], spec["cam8_val"]
        run_date = spec["date"]
        run_dir = prev_root / rel
        best = run_dir / "weights" / "best.pt"
        if not best.exists():
            print(f"[{key}] no weights at {best}, skipping")
            continue
        epochs, best_epoch, val_map50, val_map = own_val_from_results_csv(run_dir)
        print(f"[{key}] evaluating on Cam8 hold-out ({rel})")
        result = YOLO(str(best)).val(
            data=str(data_yaml),
            split="test",
            project=str(args.eval_project),
            name=f"{key}_cam8test",
            exist_ok=True,
            plots=True,
            imgsz=640,
            batch=16,
            workers=0,
        )
        matrix = getattr(getattr(result, "confusion_matrix", None), "matrix", None)
        tp = fp = fn = ""
        if matrix is not None and matrix.shape == (2, 2):
            tp, fp, fn = int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0])
        rows.append(
            {
                "key": key,
                "label": label,
                "generation": "previous",
                "training_pool": pool,
                "cam8_in_train": cam8_train,
                "cam8_in_val": cam8_val,
                "cam8_leaked": "YES" if (cam8_train or cam8_val) else "no",
                "run_date": run_date,
                "epochs_run": epochs,
                "best_epoch": best_epoch,
                "own_val_mAP50": round(val_map50, 6),
                "own_val_mAP50_95": round(val_map, 6),
                "cam8_precision": round(scalar(result.box.mp), 6),
                "cam8_recall": round(scalar(result.box.mr), 6),
                "cam8_f1": round(scalar(result.box.f1), 6),
                "cam8_mAP50": round(scalar(result.box.map50), 6),
                "cam8_mAP50_95": round(scalar(result.box.map), 6),
                "cam8_tp": tp,
                "cam8_fp": fp,
                "cam8_fn": fn,
                "weights": f"runs/detect/{rel}/weights/best.pt",
            }
        )

    # Fold in this experiment's three models, already measured on the same images.
    new_meta = {
        "expA": ("New · original only (Cam8 removed)", "original dataset, Cam8 stripped", 178, 148),
        "expB": ("New · CCTV Cam9+24 only", "CCTV Cam9 + Cam24", 87, 57),
        "expC": ("New · original + CCTV Cam9/24", "original + CCTV Cam9/24", 129, 99),
    }
    comparison = EXP_ROOT / "results" / "cam8_test_comparison.csv"
    by_key: dict[str, dict[str, dict]] = {}
    for row in csv.DictReader(comparison.open(encoding="utf-8")):
        by_key.setdefault(row["experiment"], {})[row["split"]] = row
    for key, (label, pool, epochs, best_epoch) in new_meta.items():
        val, test = by_key[key]["val"], by_key[key]["test"]
        rows.append(
            {
                "key": key,
                "label": label,
                "generation": "new (Cam8 hold-out)",
                "training_pool": pool,
                "cam8_in_train": 0,
                "cam8_in_val": 0,
                "cam8_leaked": "no",
                "run_date": "2026-08-12",
                "epochs_run": epochs,
                "best_epoch": best_epoch,
                "own_val_mAP50": float(val["mAP50"]),
                "own_val_mAP50_95": float(val["mAP50_95"]),
                "cam8_precision": float(test["precision"]),
                "cam8_recall": float(test["recall"]),
                "cam8_f1": float(test["f1"]),
                "cam8_mAP50": float(test["mAP50"]),
                "cam8_mAP50_95": float(test["mAP50_95"]),
                "cam8_tp": test["confusion_matrix_tp"],
                "cam8_fp": test["confusion_matrix_fp"],
                "cam8_fn": test["confusion_matrix_fn"],
                "weights": test["weights"],
            }
        )

    out = EXP_ROOT / "results" / "prev_vs_new_cam8.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'model':38s} {'leaked':7s} {'val mAP50':>10s} {'Cam8 mAP50':>11s}")
    for row in rows:
        print(
            f"{row['label']:38s} {row['cam8_leaked']:7s} "
            f"{row['own_val_mAP50']:10.4f} {row['cam8_mAP50']:11.4f}"
        )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
