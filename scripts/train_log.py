import os
import sys
import subprocess
import datetime
import yaml
import pandas as pd

# ====== 你要存的 Excel 檔名 ======
EXCEL_PATH = "train_log.xlsx"

# 你覺得重要的參數 key（會從 save_dir/args.yaml 擷取）
IMPORTANT_ARGS = [
    "task", "mode", "model", "data",
    "imgsz", "epochs", "batch", "patience", "device",
    "optimizer", "confidence", "lr0", "lrf", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "mosaic", "mixup", "copy_paste", "fliplr", "flipud",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "project", "name", "save_dir", "seed"
]

# 常見 results.csv 欄位（不同版本可能有差，程式會自動找最接近的）
IMPORTANT_METRICS = [
    "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50-95", "metrics/mAP50", "metrics/precision", "metrics/recall",
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "val/box_loss", "val/cls_loss", "val/dfl_loss"
]


def safe_read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick_keys(d, keys):
    out = {}
    for k in keys:
        out[k] = d.get(k, None)
    return out


def find_latest_save_dir(project, name):
    # Ultralytics 預設 save_dir: project/name 或 project/name2 ...
    # 我們先用你傳進來的 project/name 當主路徑；如果不存在再 fallback 搜尋
    base = os.path.join(project, name)
    if os.path.exists(base):
        return base

    # fallback: 找 project 底下最接近 name 的資料夾
    if os.path.exists(project):
        candidates = [os.path.join(project, d) for d in os.listdir(project) if d.startswith(name)]
        candidates = [d for d in candidates if os.path.isdir(d)]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return candidates[0]
    return None


def read_results_csv(save_dir):
    """
    Ultralytics 會輸出 results.csv（通常在 save_dir 內）。
    取最後一列作為最終結果（也可改成取 best 的 epoch，但最終通常夠用）。
    """
    csv_path = os.path.join(save_dir, "results.csv")
    if not os.path.exists(csv_path):
        return None, {}

    df = pd.read_csv(csv_path)
    if df.empty:
        return csv_path, {}

    last = df.iloc[-1].to_dict()
    return csv_path, last


def get_metric(last_row, possible_keys):
    for k in possible_keys:
        if k in last_row:
            return last_row[k]
    return None


def append_to_excel(excel_path, row_dict):
    df_new = pd.DataFrame([row_dict])

    if os.path.exists(excel_path):
        df_old = pd.read_excel(excel_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    # 讓欄位順序穩定：先放固定欄位，再放 args，再放 metrics
    fixed_cols = ["timestamp", "run_tag", "save_dir", "args_yaml", "results_csv", "best_pt", "last_pt"]
    arg_cols = [c for c in IMPORTANT_ARGS if c not in fixed_cols]
    metric_cols = [c for c in IMPORTANT_METRICS if c not in fixed_cols]

    # 補齊可能出現的新欄位
    for c in fixed_cols + arg_cols + metric_cols:
        if c not in df_all.columns:
            df_all[c] = None

    df_all = df_all[fixed_cols + arg_cols + metric_cols + [c for c in df_all.columns if c not in (fixed_cols + arg_cols + metric_cols)]]
    df_all.to_excel(excel_path, index=False)


def main():
    """
    用法：
    python train_and_log.py detect train model=yolov8s.pt data=0121data.yaml imgsz=640 epochs=300 batch=16 patience=30 project=egg_project name=run_test_block
    你原本的 yolo 命令列參數原封不動接在後面即可
    """
    if len(sys.argv) < 2:
        print("Usage: python train_and_log.py <yolo args...>")
        sys.exit(1)

    yolo_args = sys.argv[1:]

    # 從命令列粗略抓 project/name（若沒給，沿用 Ultralytics 預設 runs）
    project = "runs"
    name = "train"
    for s in yolo_args:
        if s.startswith("project="):
            project = s.split("=", 1)[1].strip("\"'")
        elif s.startswith("name="):
            name = s.split("=", 1)[1].strip("\"'")

    # 1) 執行訓練（同步等待結束）
    cmd = ["yolo"] + yolo_args
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd)

    # 2) 找輸出資料夾
    save_dir = find_latest_save_dir(project, name)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "timestamp": timestamp,
        "run_tag": f"{project}/{name}",
        "save_dir": save_dir,
    }

    if save_dir is None:
        row["args_yaml"] = None
        row["results_csv"] = None
        row["best_pt"] = None
        row["last_pt"] = None
        # 即便失敗也記錄一次（方便你回頭追）
        append_to_excel(EXCEL_PATH, row)
        print(f"[WARN] save_dir not found. Logged to {EXCEL_PATH}")
        sys.exit(proc.returncode)

    # 3) 讀 args.yaml（Ultralytics 會在 save_dir/args.yaml）
    args_yaml = os.path.join(save_dir, "args.yaml")
    args = safe_read_yaml(args_yaml)

    row["args_yaml"] = args_yaml if os.path.exists(args_yaml) else None

    # 重要參數
    row.update(pick_keys(args, IMPORTANT_ARGS))

    # 4) 讀 results.csv
    results_csv, last_row = read_results_csv(save_dir)
    row["results_csv"] = results_csv

    # 重要 metrics（抓得到就填）
    for k in IMPORTANT_METRICS:
        if k in last_row:
            row[k] = last_row[k]

    # 5) best/last
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")
    row["best_pt"] = best_pt if os.path.exists(best_pt) else None
    row["last_pt"] = last_pt if os.path.exists(last_pt) else None

    # 6) 寫入 Excel（append）
    append_to_excel(EXCEL_PATH, row)
    print(f"Logged to: {EXCEL_PATH}")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()