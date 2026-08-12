"""Package everything the Cam8 hold-out work produced into one hand-off folder.

Collects, into a single dated directory: a unified metrics table across all seven
models, both reports, the comparison figures, every Ultralytics-native plot from
every training and evaluation run, the error-case analysis with its annotated
predictions, the split manifests, the scripts, and (optionally) the weights.

Everything is copied, never moved -- the source runs and the worktree are left
untouched, so this can be re-run at any time.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = REPO_ROOT / "experiments" / "cam8_holdout_yolov8s"
DEFAULT_DATA_ROOT = Path(r"C:\Users\hicat\PycharmProjects\PythonProject3")

NEW_RUNS = {
    "expA": ("expA_original_yolov8s_cam8test", "Exp A · original only (Cam8 removed)"),
    "expB": ("expB_cctv_cam9_24_yolov8s_cam8test", "Exp B · CCTV Cam9+24 only"),
    "expC": ("expC_original_cctv_cam9_24_yolov8s_cam8test", "Exp C · original + CCTV Cam9/24"),
}
PREV_EVALS = {
    "prev_original": "Prev · original dataset (2026-03)",
    "prev_original_aborted": "Prev · original, 8-epoch abort",
    "prev_cctv_all": "Prev · CCTV Cam8+9+24",
    "prev_merged": "Prev · original + CCTV (incl. Cam8)",
}

MASTER_FIELDS = [
    "generation",
    "model_key",
    "label",
    "training_pool",
    "cam8_in_train",
    "cam8_leaked",
    "split",
    "n_images",
    "n_labels",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP50_95",
    "tp",
    "fp",
    "fn",
    "epochs_run",
    "best_epoch",
    "weights",
]


def copy_tree(src: Path, dst: Path, patterns: tuple[str, ...] | None = None) -> int:
    """Copy files from src into dst. Returns how many landed."""
    if not src.is_dir():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for item in sorted(src.iterdir()):
        if item.is_dir():
            continue
        if patterns and item.suffix.lower() not in patterns:
            continue
        shutil.copy2(item, dst / item.name)
        count += 1
    return count


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_master_metrics(bundle: Path) -> list[dict]:
    """One table for every model x split we measured, old and new."""
    rows: list[dict] = []

    new_meta = {
        "expA": ("original dataset, Cam8 stripped", 178, 148),
        "expB": ("CCTV Cam9 + Cam24", 87, 57),
        "expC": ("original + CCTV Cam9/24", 129, 99),
    }
    for row in read_csv(EXP_ROOT / "results" / "cam8_test_comparison.csv"):
        key = row["experiment"]
        pool, epochs, best_epoch = new_meta[key]
        rows.append(
            {
                "generation": "new (Cam8 hold-out)",
                "model_key": key,
                "label": NEW_RUNS[key][1],
                "training_pool": pool,
                "cam8_in_train": 0,
                "cam8_leaked": "no",
                "split": "own val" if row["split"] == "val" else "Cam8 hold-out test",
                "n_images": row["n_images"],
                "n_labels": row["n_labels"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "mAP50": row["mAP50"],
                "mAP50_95": row["mAP50_95"],
                "tp": row["confusion_matrix_tp"],
                "fp": row["confusion_matrix_fp"],
                "fn": row["confusion_matrix_fn"],
                "epochs_run": epochs,
                "best_epoch": best_epoch,
                "weights": row["weights"],
            }
        )

    for row in read_csv(EXP_ROOT / "results" / "prev_vs_new_cam8.csv"):
        if row["generation"] != "previous":
            continue  # the new models are already in, with both splits
        rows.append(
            {
                "generation": "previous",
                "model_key": row["key"],
                "label": row["label"],
                "training_pool": row["training_pool"],
                "cam8_in_train": row["cam8_in_train"],
                "cam8_leaked": row["cam8_leaked"],
                "split": "Cam8 hold-out test",
                "n_images": 103,
                "n_labels": 798,
                "precision": row["cam8_precision"],
                "recall": row["cam8_recall"],
                "f1": row["cam8_f1"],
                "mAP50": row["cam8_mAP50"],
                "mAP50_95": row["cam8_mAP50_95"],
                "tp": row["cam8_tp"],
                "fp": row["cam8_fp"],
                "fn": row["cam8_fn"],
                "epochs_run": row["epochs_run"],
                "best_epoch": row["best_epoch"],
                "weights": row["weights"],
            }
        )
        rows.append(
            {
                "generation": "previous",
                "model_key": row["key"],
                "label": row["label"],
                "training_pool": row["training_pool"],
                "cam8_in_train": row["cam8_in_train"],
                "cam8_leaked": row["cam8_leaked"],
                "split": "own val (as trained)",
                "n_images": "",
                "n_labels": "",
                "precision": "",
                "recall": "",
                "f1": "",
                "mAP50": row["own_val_mAP50"],
                "mAP50_95": row["own_val_mAP50_95"],
                "tp": "",
                "fp": "",
                "fn": "",
                "epochs_run": row["epochs_run"],
                "best_epoch": row["best_epoch"],
                "weights": row["weights"],
            }
        )

    out = bundle / "00_metrics" / "master_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MASTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_fixed_threshold_table(bundle: Path) -> list[dict]:
    """Precision/recall at a single deployable confidence, from the error analysis."""
    summary = read_csv(EXP_ROOT / "results" / "cam8_error_summary.csv")
    if not summary:
        return []
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    total_gt = 0
    for row in summary:
        counts[row["experiment"]][row["bucket"]] = int(row["count"])
        if row["bucket"] == "TOTAL_GT_BOXES":
            total_gt = int(row["count"])

    rows = []
    for key in ("expA", "expB", "expC"):
        bucket = counts.get(key, {})
        tp = bucket.get("TP", 0)
        fp = sum(v for k, v in bucket.items() if k.startswith("FP:"))
        fn = sum(v for k, v in bucket.items() if k.startswith("FN:"))
        rows.append(
            {
                "model_key": key,
                "label": NEW_RUNS[key][1],
                "conf_threshold": 0.25,
                "match_iou": 0.5,
                "gt_boxes": total_gt,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(tp / (tp + fp), 4) if tp + fp else "",
                "recall": round(tp / total_gt, 4) if total_gt else "",
                "f1": round(2 * tp / (2 * tp + fp + fn), 4) if tp else "",
                **{k: v for k, v in sorted(bucket.items()) if k.startswith(("FP:", "FN:"))},
            }
        )
    fields = list(dict.fromkeys(k for row in rows for k in row))
    out = bundle / "00_metrics" / "cam8_fixed_threshold_operating_point.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_metrics_summary(bundle: Path, master: list[dict], fixed: list[dict]) -> None:
    lines = [
        "# 指標總表",
        "",
        "所有數字皆為 YOLOv8s，單一類別 `comb`。Cam8 hold-out 測試集固定為 103 張影像 / 798 個有效標註框。",
        "",
        "> 標記 `cam8_leaked = YES` 的模型訓練時就看過 Cam8，其 Cam8 分數依定義即為灌水，",
        "> **不可作為泛化能力的證據**。",
        "",
        "## Cam8 hold-out 測試集（同一組影像，全部模型）",
        "",
        "| 模型 | 洩漏 | Precision | Recall | F1 | mAP50 | mAP50-95 | TP | FP | FN |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cam8_rows = [r for r in master if r["split"] == "Cam8 hold-out test"]
    for row in sorted(cam8_rows, key=lambda r: float(r["mAP50"])):
        leaked = "⚠️ 是" if row["cam8_leaked"] == "YES" else "否"
        lines.append(
            f"| {row['label']} | {leaked} | {float(row['precision']):.4f} | "
            f"{float(row['recall']):.4f} | {float(row['f1']):.4f} | {float(row['mAP50']):.4f} | "
            f"{float(row['mAP50_95']):.4f} | {row['tp']} | {row['fp']} | {row['fn']} |"
        )

    lines += [
        "",
        "## 各模型自己的 validation（僅供對照落差，彼此不可比較）",
        "",
        "| 模型 | Val 影像 | Val 標註 | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [r for r in master if r["split"].startswith("own val")]:
        def fmt(value):
            return f"{float(value):.4f}" if value not in ("", None) else "—"

        lines.append(
            f"| {row['label']} | {row['n_images'] or '—'} | {row['n_labels'] or '—'} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} | {fmt(row['mAP50'])} | "
            f"{fmt(row['mAP50_95'])} |"
        )

    if fixed:
        lines += [
            "",
            "## 固定操作點（conf 0.25、match IoU 0.5）—— 實務部署看這組",
            "",
            "上面的 Precision/Recall 是 Ultralytics 在各模型自己的最佳 F1 信心值取的，三組操作點不同。",
            "下表把三個新模型放在同一個門檻上重新配對。",
            "",
            "| 模型 | TP | FP | FN | Precision | Recall | F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in fixed:
            lines.append(
                f"| {row['label']} | {row['tp']} | {row['fp']} | {row['fn']} | "
                f"{row['precision']} | {row['recall']} | {row['f1']} |"
            )

    (bundle / "00_metrics" / "metrics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


README = """# Cam8 Hold-out 實驗 — 完整結果打包

**建立日期** {today}　**模型** YOLOv8s　**類別** 單一 `comb`（nc=1）
**Cam8 hold-out 測試集** 103 張影像 / 798 個有效標註框，全部模型共用

> **範圍聲明**：本資料夾只涉及雞冠**位置偵測**。資料集沒有正常／異常之分，
> 任何數字都不能解讀為異常雞冠比例、OACP 或疾病相關。

> **⚠️ 讀數字前必看**：前次實驗的 CCTV 與 merged 模型訓練時就看過 Cam8，
> 其 Cam8 分數依定義即為灌水。所有表格與圖表都以 `leaked` 標示，
> **這些數字不可作為泛化能力的證據**。

## 從哪裡開始

1. `01_reports/EXPERIMENT_LOG.md` — 實驗日誌：做了什麼、得到什麼、哪裡不能信
2. `00_metrics/metrics_summary.md` — 一頁看完所有 precision / recall / mAP
3. `01_reports/cam8_holdout_experiment_report.html` — 本次實驗完整報告
4. `01_reports/prev_vs_cam8_holdout_comparison.html` — 與前次實驗的對照
5. `01_reports/slide_summary.md` — 8 張投影片大綱

## 資料夾結構

```text
00_metrics/          統一指標表（CSV + 可讀 Markdown）
01_reports/          兩份報告的 Markdown 與自帶圖片的 HTML、投影片大綱
02_figures/          比較圖表（本次自製）
03_yolo_native/      Ultralytics 原生輸出，依 run 分開
   <run>/train/        訓練期輸出：results.png/csv、labels.jpg、train_batch*、args.yaml
   <run>/val_eval/     自己 validation 的評估：confusion matrix、PR/F1/P/R 曲線、val_batch*
   <run>/test_eval/    Cam8 測試的評估：同上
   prev_eval/<model>/  前次模型在 Cam8 上的評估輸出
04_error_analysis/   FP/FN 逐框明細 CSV + 標註圖（綠 TP / 紅 FP / 橘 FN）
05_manifests/        每張影像的 split 歸屬、來源、分組鍵、MD5、leakage 標記
06_scripts/          重建全部結果所需的腳本
07_weights/          三個新模型的 best.pt（若打包時包含）
```

## 指標檔案對照

| 檔案 | 內容 |
|---|---|
| `00_metrics/master_metrics.csv` | 全部模型 × 全部 split 的統一大表 |
| `00_metrics/metrics_summary.md` | 同上，可讀版本 |
| `00_metrics/cam8_test_comparison.csv` | 本次三組的 val + Cam8 test |
| `00_metrics/prev_vs_new_cam8.csv` | 七個模型在同一組 Cam8 上的對照 |
| `00_metrics/cam8_fixed_threshold_operating_point.csv` | conf 0.25 固定門檻的 TP/FP/FN |
| `00_metrics/cam8_error_summary.csv` | 錯誤案例分類統計 |
| `00_metrics/run_settings.json` | 訓練超參數 |

## 重現方式

```bash
python 06_scripts/build_cam8_holdout.py --seed 42 --data-root <含 datasets/ 的 checkout>
python 06_scripts/run_cam8_holdout.py
python 06_scripts/analyze_cam8_errors.py
python 06_scripts/make_cam8_figures.py
python 06_scripts/eval_prev_on_cam8.py
python 06_scripts/make_prev_vs_new_figures.py
python 06_scripts/render_cam8_report.py
python 06_scripts/build_cam8_bundle.py
```

切分固定 `seed=42`，訓練固定 `seed=0`，重跑會得到相同的 split。
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Bundle directory (default: runs/cam8_holdout_yolov8s/_bundle_<today>).",
    )
    parser.add_argument("--no-weights", action="store_true", help="Skip the 66 MB of best.pt files.")
    parser.add_argument("--zip", action="store_true", help="Also produce a .zip beside the folder.")
    args = parser.parse_args()

    today = date.today().isoformat().replace("-", "")
    bundle = args.out or (
        args.data_root / "runs" / "cam8_holdout_yolov8s" / f"_bundle_{today}"
    )
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    new_project = args.data_root / "runs" / "cam8_holdout_yolov8s"
    prev_project = args.data_root / "runs" / "cam8_holdout_prev_eval"
    tally: dict[str, int] = {}

    # ---- 00_metrics ----
    master = build_master_metrics(bundle)
    fixed = build_fixed_threshold_table(bundle)
    write_metrics_summary(bundle, master, fixed)
    for name in (
        "cam8_test_comparison.csv",
        "prev_vs_new_cam8.csv",
        "cam8_error_summary.csv",
        "metric_tables.md",
        "run_settings.json",
    ):
        source = EXP_ROOT / "results" / name
        if source.exists():
            shutil.copy2(source, bundle / "00_metrics" / name)

    # Snapshot of the project's local Excel training log (itself gitignored).
    train_log = args.data_root / "train_log.xlsx"
    if train_log.exists():
        shutil.copy2(train_log, bundle / "00_metrics" / "train_log_snapshot.xlsx")
    tally["00_metrics"] = len(list((bundle / "00_metrics").iterdir()))

    # ---- 01_reports ----
    reports = bundle / "01_reports"
    reports.mkdir()
    for name in (
        "EXPERIMENT_LOG.md",
        "cam8_holdout_experiment_report.md",
        "cam8_holdout_experiment_report.html",
        "prev_vs_cam8_holdout_comparison.md",
        "prev_vs_cam8_holdout_comparison.html",
        "slide_summary.md",
    ):
        source = EXP_ROOT / name
        if source.exists():
            shutil.copy2(source, reports / name)
    tally["01_reports"] = len(list(reports.iterdir()))

    # ---- 02_figures ----
    tally["02_figures"] = copy_tree(EXP_ROOT / "figures", bundle / "02_figures", (".png",))

    # ---- 03_yolo_native ----
    native = bundle / "03_yolo_native"
    total_native = 0
    for key, (run_name, _) in NEW_RUNS.items():
        total_native += copy_tree(new_project / run_name, native / key / "train")
        total_native += copy_tree(new_project / f"{run_name}_valeval", native / key / "val_eval")
        total_native += copy_tree(new_project / f"{run_name}_testeval", native / key / "test_eval")
    for key in PREV_EVALS:
        total_native += copy_tree(
            prev_project / f"{key}_cam8test", native / "prev_eval" / key
        )
    tally["03_yolo_native"] = total_native

    # ---- 04_error_analysis ----
    errors = bundle / "04_error_analysis"
    errors.mkdir()
    count = 0
    for name in ("expA", "expB", "expC"):
        source = EXP_ROOT / "results" / f"{name}_cam8_error_cases.csv"
        if source.exists():
            shutil.copy2(source, errors / source.name)
            count += 1
        count += copy_tree(
            EXP_ROOT / "predictions" / name, errors / "predictions" / name, (".jpg",)
        )
    tally["04_error_analysis"] = count

    # ---- 05_manifests ----
    tally["05_manifests"] = copy_tree(EXP_ROOT / "manifests", bundle / "05_manifests", (".csv",))

    # ---- 06_scripts ----
    scripts = bundle / "06_scripts"
    scripts.mkdir()
    count = 0
    for name in (
        "build_cam8_holdout.py",
        "run_cam8_holdout.py",
        "analyze_cam8_errors.py",
        "make_cam8_figures.py",
        "eval_prev_on_cam8.py",
        "make_prev_vs_new_figures.py",
        "render_cam8_report.py",
        "log_cam8_experiment.py",
        "build_cam8_bundle.py",
    ):
        source = REPO_ROOT / "scripts" / name
        if source.exists():
            shutil.copy2(source, scripts / name)
            count += 1
    tally["06_scripts"] = count

    # ---- 07_weights ----
    if not args.no_weights:
        weights = bundle / "07_weights"
        weights.mkdir()
        count = 0
        for key, (run_name, _) in NEW_RUNS.items():
            best = new_project / run_name / "weights" / "best.pt"
            if best.exists():
                shutil.copy2(best, weights / f"{key}_best.pt")
                count += 1
        tally["07_weights"] = count

    (bundle / "README.md").write_text(
        README.format(today=date.today().isoformat()), encoding="utf-8"
    )
    (bundle / "00_metrics" / "bundle_inventory.json").write_text(
        json.dumps(tally, indent=2), encoding="utf-8"
    )

    total_files = sum(1 for _ in bundle.rglob("*") if _.is_file())
    total_mb = sum(f.stat().st_size for f in bundle.rglob("*") if f.is_file()) / 1024 / 1024
    for name, n in tally.items():
        print(f"  {name:20s} {n:4d} files")
    print(f"\nbundle -> {bundle}")
    print(f"{total_files} files, {total_mb:.1f} MB")

    if args.zip:
        archive = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
        print(f"zip    -> {archive} ({Path(archive).stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
