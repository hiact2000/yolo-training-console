"""Train the three Cam8 hold-out YOLOv8s models and evaluate all of them on Cam8.

Hyper-parameters replicate the previous three-way experiment exactly
(runs/detect/exp, runs/detect/0715data_yolov8s_ui,
runs/detect/detect_0715data_merged_yolov8s_ui): yolov8s.pt, 300 epochs,
patience 30, batch 16, imgsz 640, seed 0, workers 0, and stock Ultralytics
augmentation (mosaic=1.0, fliplr=0.5, hsv/translate/scale/erasing defaults).
Only the data split differs, so Cam8 numbers stay comparable with that run.

Each experiment is trained once, then evaluated twice: on its own val split and
on the shared Cam8 hold-out test split. Completed stages are skipped on re-run.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

EXPERIMENTS = [
    ("expA", "expA_original_cam8test.yaml", "expA_original_yolov8s_cam8test"),
    ("expB", "expB_cctv_cam9_24_cam8test.yaml", "expB_cctv_cam9_24_yolov8s_cam8test"),
    (
        "expC",
        "expC_original_cctv_cam9_24_cam8test.yaml",
        "expC_original_cctv_cam9_24_yolov8s_cam8test",
    ),
]

# Copied verbatim from the previous experiment's args.yaml.
TRAIN_ARGS = dict(
    epochs=300,
    patience=30,
    batch=16,
    imgsz=640,
    seed=0,
    workers=0,
    deterministic=True,
    optimizer="auto",
    pretrained=True,
    val=True,
    plots=True,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.4,
    close_mosaic=10,
)

METRIC_FIELDS = [
    "experiment",
    "run_name",
    "split",
    "weights",
    "n_images",
    "n_labels",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP50_95",
    "fitness",
    "speed_preprocess_ms",
    "speed_inference_ms",
    "speed_postprocess_ms",
    "confusion_matrix_tp",
    "confusion_matrix_fp",
    "confusion_matrix_fn",
]


def count_list(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def count_labels(list_path: Path) -> int:
    total = 0
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label = Path(line.replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if label.exists():
            total += sum(1 for row in label.read_text(encoding="utf-8").splitlines() if row.strip())
    return total


def scalar(value) -> float:
    """Ultralytics returns numpy scalars/arrays depending on the field."""
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value[0])
        except Exception:
            return float("nan")


def metrics_row(experiment, run_name, split, weights, results, n_images, n_labels) -> dict:
    box = results.box
    matrix = getattr(getattr(results, "confusion_matrix", None), "matrix", None)
    tp = fp = fn = ""
    if matrix is not None and matrix.shape == (2, 2):
        # Single class: [[TP, FP], [FN, TN-ish]] in Ultralytics' background-augmented layout.
        tp, fp, fn = int(matrix[0][0]), int(matrix[0][1]), int(matrix[1][0])
    speed = getattr(results, "speed", {}) or {}
    return {
        "experiment": experiment,
        "run_name": run_name,
        "split": split,
        "weights": weights,
        "n_images": n_images,
        "n_labels": n_labels,
        "precision": round(scalar(box.mp), 6),
        "recall": round(scalar(box.mr), 6),
        "f1": round(scalar(box.f1), 6),
        "mAP50": round(scalar(box.map50), 6),
        "mAP50_95": round(scalar(box.map), 6),
        "fitness": round(scalar(getattr(results, "fitness", float("nan"))), 6),
        "speed_preprocess_ms": round(float(speed.get("preprocess", float("nan"))), 4),
        "speed_inference_ms": round(float(speed.get("inference", float("nan"))), 4),
        "speed_postprocess_ms": round(float(speed.get("postprocess", float("nan"))), 4),
        "confusion_matrix_tp": tp,
        "confusion_matrix_fp": fp,
        "confusion_matrix_fn": fn,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_DATA_ROOT / "runs" / "cam8_holdout_yolov8s",
        help="Ultralytics project dir. Kept in the primary checkout so weights outlive the worktree.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_DATA_ROOT / "yolov8s.pt")
    parser.add_argument("--only", nargs="*", help="Subset of experiment keys, e.g. --only expA expB")
    args = parser.parse_args()

    from ultralytics import YOLO

    results_dir = EXP_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    selected = [e for e in EXPERIMENTS if not args.only or e[0] in args.only]

    for key, yaml_name, run_name in selected:
        data_yaml = EXP_ROOT / "datasets" / yaml_name
        run_dir = args.project / run_name
        best = run_dir / "weights" / "best.pt"

        if best.exists():
            print(f"[{key}] weights already present, skipping training: {best}")
        else:
            print(f"[{key}] training -> {run_dir}")
            started = time.time()
            YOLO(str(args.model)).train(
                data=str(data_yaml),
                project=str(args.project),
                name=run_name,
                exist_ok=True,
                **TRAIN_ARGS,
            )
            print(f"[{key}] training done in {(time.time() - started) / 60:.1f} min")

        rows = []
        for split in ("val", "test"):
            list_path = EXP_ROOT / "lists" / (
                "cam8_test.txt" if split == "test" else f"{_full_key(key)}_{split}.txt"
            )
            print(f"[{key}] evaluating split={split} ({count_list(list_path)} images)")
            evaluated = YOLO(str(best)).val(
                data=str(data_yaml),
                split=split,
                project=str(args.project),
                name=f"{run_name}_{split}eval",
                exist_ok=True,
                plots=True,
                imgsz=TRAIN_ARGS["imgsz"],
                batch=TRAIN_ARGS["batch"],
                workers=0,
            )
            rows.append(
                metrics_row(
                    key,
                    run_name,
                    split,
                    str(best.relative_to(args.data_root)).replace("\\", "/"),
                    evaluated,
                    count_list(list_path),
                    count_labels(list_path),
                )
            )

        write_rows(results_dir / f"{key}_cam8_test_metrics.csv", rows)
        print(f"[{key}] metrics -> {results_dir / f'{key}_cam8_test_metrics.csv'}")

    # Combined comparison table across whichever experiments have results on disk.
    combined = []
    for key, _, _ in EXPERIMENTS:
        path = results_dir / f"{key}_cam8_test_metrics.csv"
        if path.exists():
            combined.extend(csv.DictReader(path.open(encoding="utf-8")))
    if combined:
        write_rows(results_dir / "cam8_test_comparison.csv", combined)
        print(f"comparison -> {results_dir / 'cam8_test_comparison.csv'}")
        (results_dir / "run_settings.json").write_text(
            json.dumps(
                {
                    "model": str(args.model.name),
                    "train_args": TRAIN_ARGS,
                    "split_seed": 42,
                    "note": "augmentation replicates the previous experiment (stock Ultralytics defaults)",
                },
                indent=2,
            ),
            encoding="utf-8",
        )


def _full_key(key: str) -> str:
    return {
        "expA": "expA_original",
        "expB": "expB_cctv_cam9_24",
        "expC": "expC_original_cctv_cam9_24",
    }[key]


if __name__ == "__main__":
    main()
