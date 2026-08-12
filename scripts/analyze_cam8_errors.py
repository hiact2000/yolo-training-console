"""Error-case analysis for the three Cam8 hold-out models.

For each model we predict on the Cam8 hold-out set, greedily match predictions to
ground truth at IoU >= 0.5, then bucket every unmatched box into a coarse failure
category. The categories are *image-measurable proxies*, not verified causes --
they tell a human where to look, they do not diagnose the model.

False negatives (missed combs):
  small_or_distant   box area below the small-object percentile of this test set
  occluded_crowded   overlaps another ground-truth comb (IoU > 0.1)
  low_light          mean V (HSV) inside the box below the dark percentile
  truncated_at_edge  box touches an image border
  other              none of the above

False positives (spurious detections):
  localization_error partially over a real comb (0 < IoU < 0.5)
  background_fp      no overlap with any ground truth at all

Outputs under experiments/cam8_holdout_yolov8s/:
  predictions/<exp>/            annotated Cam8 images (green = TP, red = FP, orange = FN)
  results/<exp>_cam8_error_cases.csv
  results/cam8_error_summary.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

RUN_NAMES = {
    "expA": "expA_original_yolov8s_cam8test",
    "expB": "expB_cctv_cam9_24_yolov8s_cam8test",
    "expC": "expC_original_cctv_cam9_24_yolov8s_cam8test",
}
CONF_THRESHOLD = 0.25
MATCH_IOU = 0.5
COLOUR = {"TP": (60, 200, 60), "FP": (60, 60, 235), "FN": (0, 165, 255)}  # BGR


def load_gt(image_path: Path) -> np.ndarray:
    """YOLO label file -> xyxy pixel boxes."""
    label = Path(str(image_path).replace(f"{chr(92)}images{chr(92)}", f"{chr(92)}labels{chr(92)}"))
    label = label.with_suffix(".txt")
    if not label.exists():
        return np.zeros((0, 4), dtype=np.float32)
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    # Ultralytics drops exact duplicate rows during training; mirror that here.
    unique = {tuple(np.round(b, 3)) for b in boxes}
    return np.array(sorted(unique), dtype=np.float32) if unique else np.zeros((0, 4), np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def greedy_match(pred: np.ndarray, gt: np.ndarray, scores: np.ndarray):
    """Highest-confidence-first matching, one prediction per ground-truth box."""
    ious = iou_matrix(pred, gt)
    matched_gt: dict[int, int] = {}
    matched_pred: dict[int, int] = {}
    for pred_idx in np.argsort(-scores):
        if ious.shape[1] == 0:
            break
        order = np.argsort(-ious[pred_idx])
        for gt_idx in order:
            if ious[pred_idx, gt_idx] < MATCH_IOU:
                break
            if gt_idx not in matched_gt:
                matched_gt[int(gt_idx)] = int(pred_idx)
                matched_pred[int(pred_idx)] = int(gt_idx)
                break
    return matched_pred, matched_gt, ious


def box_brightness(image: np.ndarray, box: np.ndarray) -> float:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, max(x1 + 1, x2)), min(h, max(y1 + 1, y2))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return float("nan")
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 2].mean())


def classify_fn(
    box: np.ndarray,
    gt: np.ndarray,
    gt_index: int,
    image_shape: tuple[int, int],
    area: float,
    brightness: float,
    small_cut: float,
    dark_cut: float,
) -> str:
    height, width = image_shape
    others = np.delete(gt, gt_index, axis=0)
    if len(others) and iou_matrix(box[None, :], others).max() > 0.1:
        return "occluded_crowded"
    if area <= small_cut:
        return "small_or_distant"
    if brightness == brightness and brightness <= dark_cut:
        return "low_light"
    margin = 2.0
    if (
        box[0] <= margin
        or box[1] <= margin
        or box[2] >= width - margin
        or box[3] >= height - margin
    ):
        return "truncated_at_edge"
    return "other"


def draw(image: np.ndarray, box: np.ndarray, kind: str, text: str) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), COLOUR[kind], 2)
    cv2.putText(
        image, text, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR[kind], 1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--project", type=Path, default=DEFAULT_DATA_ROOT / "runs" / "cam8_holdout_yolov8s"
    )
    parser.add_argument("--max-annotated", type=int, default=40, help="Annotated images per model.")
    args = parser.parse_args()

    from ultralytics import YOLO

    test_images = [
        Path(line)
        for line in (EXP_ROOT / "lists" / "cam8_test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Percentile cutoffs come from the Cam8 ground truth itself, so "small" and
    # "dark" mean small/dark *relative to this test set*.
    areas, brightnesses = [], []
    for image_path in test_images:
        image = cv2.imread(str(image_path))
        for box in load_gt(image_path):
            areas.append(float((box[2] - box[0]) * (box[3] - box[1])))
            brightnesses.append(box_brightness(image, box))
    small_cut = float(np.percentile(areas, 25)) if areas else 0.0
    dark_cut = float(np.nanpercentile(brightnesses, 25)) if brightnesses else 0.0
    print(f"Cam8 GT boxes={len(areas)}  small_cut(area px^2)={small_cut:.0f}  dark_cut(V)={dark_cut:.1f}")

    summary_rows = []
    for key, run_name in RUN_NAMES.items():
        best = args.project / run_name / "weights" / "best.pt"
        if not best.exists():
            print(f"[{key}] no weights at {best}, skipping")
            continue
        model = YOLO(str(best))
        out_dir = EXP_ROOT / "predictions" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        counts: Counter[str] = Counter()
        per_image_errors: list[tuple[int, Path]] = []

        for image_path in test_images:
            image = cv2.imread(str(image_path))
            gt = load_gt(image_path)
            result = model.predict(
                source=str(image_path), conf=CONF_THRESHOLD, verbose=False, imgsz=640
            )[0]
            pred = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
            scores = (
                result.boxes.conf.cpu().numpy() if result.boxes is not None else np.zeros((0,))
            )
            matched_pred, matched_gt, ious = greedy_match(pred, gt, scores)

            canvas = image.copy()
            n_errors = 0
            for pred_idx in range(len(pred)):
                if pred_idx in matched_pred:
                    counts["TP"] += 1
                    draw(canvas, pred[pred_idx], "TP", f"TP {scores[pred_idx]:.2f}")
                    continue
                best_iou = float(ious[pred_idx].max()) if ious.shape[1] else 0.0
                category = "localization_error" if best_iou > 0 else "background_fp"
                counts[f"FP:{category}"] += 1
                n_errors += 1
                draw(canvas, pred[pred_idx], "FP", f"FP {category[:4]} {scores[pred_idx]:.2f}")
                rows.append(
                    {
                        "experiment": key,
                        "image": image_path.name,
                        "error_type": "FP",
                        "category": category,
                        "confidence": round(float(scores[pred_idx]), 4),
                        "best_iou_with_gt": round(best_iou, 4),
                        "box_area_px2": round(
                            float(
                                (pred[pred_idx][2] - pred[pred_idx][0])
                                * (pred[pred_idx][3] - pred[pred_idx][1])
                            )
                        ),
                        "box_mean_v": round(box_brightness(image, pred[pred_idx]), 1),
                        "xyxy": " ".join(f"{v:.0f}" for v in pred[pred_idx]),
                    }
                )
            for gt_idx in range(len(gt)):
                if gt_idx in matched_gt:
                    continue
                box = gt[gt_idx]
                area = float((box[2] - box[0]) * (box[3] - box[1]))
                brightness = box_brightness(image, box)
                category = classify_fn(
                    box, gt, gt_idx, image.shape[:2], area, brightness, small_cut, dark_cut
                )
                counts[f"FN:{category}"] += 1
                n_errors += 1
                draw(canvas, box, "FN", f"FN {category[:5]}")
                rows.append(
                    {
                        "experiment": key,
                        "image": image_path.name,
                        "error_type": "FN",
                        "category": category,
                        "confidence": "",
                        "best_iou_with_gt": round(
                            float(ious[:, gt_idx].max()) if ious.shape[0] else 0.0, 4
                        ),
                        "box_area_px2": round(area),
                        "box_mean_v": round(brightness, 1),
                        "xyxy": " ".join(f"{v:.0f}" for v in box),
                    }
                )
            per_image_errors.append((n_errors, image_path))
            cv2.imwrite(str(out_dir / f"_all_{image_path.name}"), canvas)

        # Keep the --max-annotated frames with the most errors, prefixed by their error
        # count so the worst cases sort to the top of the folder. Drop the rest.
        per_image_errors.sort(key=lambda pair: -pair[0])
        keep = {path.name: n for n, path in per_image_errors[: args.max_annotated]}
        for rendered in out_dir.glob("_all_*.jpg"):
            original = rendered.name[len("_all_") :]
            if original in keep:
                rendered.replace(out_dir / f"err{keep[original]:03d}_{original}")
            else:
                rendered.unlink()

        error_csv = EXP_ROOT / "results" / f"{key}_cam8_error_cases.csv"
        error_csv.parent.mkdir(parents=True, exist_ok=True)
        with error_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "experiment",
                    "image",
                    "error_type",
                    "category",
                    "confidence",
                    "best_iou_with_gt",
                    "box_area_px2",
                    "box_mean_v",
                    "xyxy",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        total_fp = sum(v for k, v in counts.items() if k.startswith("FP:"))
        total_fn = sum(v for k, v in counts.items() if k.startswith("FN:"))
        print(
            f"[{key}] TP={counts['TP']} FP={total_fp} FN={total_fn} "
            f"-> {error_csv.name}, {len(keep)} annotated images"
        )
        for name, value in sorted(counts.items()):
            summary_rows.append(
                {
                    "experiment": key,
                    "bucket": name,
                    "count": value,
                    "conf_threshold": CONF_THRESHOLD,
                    "match_iou": MATCH_IOU,
                }
            )
        summary_rows.append(
            {
                "experiment": key,
                "bucket": "TOTAL_GT_BOXES",
                "count": len(areas),
                "conf_threshold": CONF_THRESHOLD,
                "match_iou": MATCH_IOU,
            }
        )

    if summary_rows:
        summary_path = EXP_ROOT / "results" / "cam8_error_summary.csv"
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
