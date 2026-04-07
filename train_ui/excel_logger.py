"""
excel_logger.py
---------------
訓練結束後，將本次訓練資訊 append 到 Excel 記錄檔。
邏輯延用 scripts/train_log.py，改為函式介面供 UI 呼叫。
"""
import datetime
import os
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_EXCEL_PATH = PROJECT_ROOT / "train_log.xlsx"

# 要從 args.yaml 擷取的訓練參數欄位
_ARG_COLS = [
    "task", "model", "data",
    "epochs", "batch", "imgsz", "patience", "device", "workers", "seed",
    "optimizer", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "mosaic", "mixup", "copy_paste", "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "dropout", "freeze", "pretrained", "amp", "cos_lr", "resume", "cache",
    "project", "name", "save_period", "save_dir",
]

# 要從 results.csv 最後一列擷取的指標欄位
_METRIC_COLS = [
    # Detection
    "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    # Classification
    "metrics/accuracy_top1", "metrics/accuracy_top5",
    "train/loss", "val/loss",
]

# Excel 欄位順序：固定欄位 → 參數 → 指標
_FIXED_COLS = [
    "timestamp", "run_tag", "notes",
    "save_dir", "args_yaml", "results_csv", "best_pt", "last_pt",
]


def _read_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_last_csv_row(save_dir: str) -> tuple[str | None, dict]:
    """讀取 results.csv 的最後一列（最終 epoch 結果）"""
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None, {}
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()     # 去除欄位名稱空白
        if df.empty:
            return csv_path, {}
        return csv_path, df.iloc[-1].to_dict()
    except Exception:
        return csv_path, {}


def _append_row(excel_path: str, row: dict) -> None:
    """Append 一列到 Excel；若檔案被占用則另存備份"""
    df_new = pd.DataFrame([row])

    if os.path.exists(excel_path):
        try:
            df_old = pd.read_excel(excel_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    # 補齊所有欄位並固定排序
    ordered = _FIXED_COLS + _ARG_COLS + _METRIC_COLS
    extra = [c for c in df_all.columns if c not in ordered]
    for c in ordered:
        if c not in df_all.columns:
            df_all[c] = None
    df_all = df_all[ordered + extra]

    try:
        df_all.to_excel(excel_path, index=False)
    except PermissionError:
        # Excel 開著時改存備份
        ts = datetime.datetime.now().strftime("%H%M%S")
        backup = excel_path.replace(".xlsx", f"_backup_{ts}.xlsx")
        df_all.to_excel(backup, index=False)
        raise PermissionError(f"Excel 被占用，已另存: {backup}")


def log_to_excel(
    save_dir: str,
    ui_config: dict,
    notes: str = "",
    excel_path: str | None = None,
) -> str:
    """
    訓練完成後呼叫此函式，將本次訓練資訊寫入 Excel。

    Args:
        save_dir:   YOLO 輸出目錄（含 args.yaml / results.csv / weights/）
        ui_config:  UI 收集的設定 dict（用於補充 args.yaml 缺失的欄位）
        notes:      使用者備註文字
        excel_path: 自訂 Excel 路徑；None 則用預設 train_log.xlsx

    Returns:
        實際寫入的 Excel 檔案路徑
    """
    target = excel_path or str(DEFAULT_EXCEL_PATH)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── 從 args.yaml 讀取 YOLO 實際生效的參數 ──────────────
    args_yaml_path = os.path.join(save_dir, "args.yaml") if save_dir else ""
    args = _read_yaml(args_yaml_path)

    # UI 設定為基礎，args.yaml 的值優先（args.yaml 是實際執行的）
    merged = {**ui_config, **args}

    row: dict = {
        "timestamp": now,
        "notes":     notes,
        "run_tag":   f"{merged.get('project', '')}/{merged.get('name', '')}",
        "save_dir":  save_dir or "",
        "args_yaml": args_yaml_path if os.path.exists(args_yaml_path) else None,
    }

    # 訓練參數
    for k in _ARG_COLS:
        row[k] = merged.get(k)

    # results.csv 最終指標
    results_csv, last_row = _read_last_csv_row(save_dir) if save_dir else (None, {})
    row["results_csv"] = results_csv
    for k in _METRIC_COLS:
        if k in last_row:
            row[k] = last_row[k]

    # best / last weights
    if save_dir:
        best_pt = os.path.join(save_dir, "weights", "best.pt")
        last_pt = os.path.join(save_dir, "weights", "last.pt")
        row["best_pt"] = best_pt if os.path.exists(best_pt) else None
        row["last_pt"] = last_pt if os.path.exists(last_pt) else None

    _append_row(target, row)
    return target
