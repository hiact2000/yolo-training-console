"""
config_manager.py
-----------------
JSON 設定的讀取、儲存、預設值管理。
"""
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# 上次設定的自動儲存路徑
LAST_CONFIG_PATH = PROJECT_ROOT / "train_ui" / "last_config.json"

# 預設訓練用 Python（conda 20250831 環境，含 CUDA torch）
DEFAULT_TRAIN_PYTHON = r"C:\Users\hicat\anaconda3\envs\20250831\python.exe"

# 預設訓練參數（對應 YOLOv8 常用設定）
DEFAULT_CONFIG: dict = {
    # ── 資料集 ──────────────────────────────────────
    "task":  "detect",
    "data":  str(PROJECT_ROOT / "configs" / "0121data.yaml"),

    # ── 模型 ────────────────────────────────────────
    "model":      "yolov8s.pt",
    "pretrained": True,
    "freeze":     "",       # 空字串 = None（不凍結）
    "dropout":    0.0,

    # ── 訓練核心 ────────────────────────────────────
    "epochs":   300,
    "batch":    16,
    "imgsz":    640,
    "patience": 30,
    "device":   "",         # 空字串 = auto
    "workers":  8,
    "seed":     0,
    "resume":   False,
    "cache":    False,
    "cos_lr":   False,
    "amp":      True,

    # ── 優化器 ──────────────────────────────────────
    "optimizer":       "auto",
    "lr0":             0.01,
    "lrf":             0.01,
    "momentum":        0.937,
    "weight_decay":    0.0005,
    "warmup_epochs":   3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr":  0.1,

    # ── Augmentation ────────────────────────────────
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "degrees":      0.0,
    "translate":    0.1,
    "scale":        0.5,
    "fliplr":       0.5,
    "flipud":       0.0,
    "mosaic":       1.0,
    "mixup":        0.0,
    "erasing":      0.4,
    "auto_augment": "randaugment",
    "copy_paste":   0.0,

    # ── 輸出 ────────────────────────────────────────
    "project":     "runs/detect",
    "name":        "exp",
    "save_period": -1,

    # ── 備註 ────────────────────────────────────────
    "notes": "",

    # ── 訓練用 Python 路徑（必須是含 CUDA torch 的環境）────
    "train_python": DEFAULT_TRAIN_PYTHON,
}

# task → 預設 project 路徑對應
TASK_DEFAULT_PROJECT = {
    "detect":   "runs/detect",
    "classify": "runs/classify",
    "segment":  "runs/segment",
}


def load_config(path: str) -> dict:
    """從指定 JSON 檔案載入設定，缺少的 key 以預設值補齊"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    config = DEFAULT_CONFIG.copy()
    config.update(data)
    return config


def save_config(path: str, config: dict) -> None:
    """將設定儲存為 JSON 檔案"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_last_config() -> dict:
    """載入上次使用的設定；若不存在則回傳預設值"""
    if LAST_CONFIG_PATH.exists():
        try:
            return load_config(str(LAST_CONFIG_PATH))
        except Exception:
            pass
    return DEFAULT_CONFIG.copy()


def save_last_config(config: dict) -> None:
    """將目前設定自動儲存（每次啟動訓練前呼叫）"""
    save_config(str(LAST_CONFIG_PATH), config)


def get_local_pt_files() -> list[str]:
    """掃描 project 根目錄及 runs/ 底下所有 best.pt，回傳路徑列表"""
    files: list[str] = []
    # 根目錄的 .pt
    for f in sorted(PROJECT_ROOT.glob("*.pt")):
        files.append(str(f))
    # runs/ 底下的 best.pt
    for f in sorted(PROJECT_ROOT.glob("runs/**/weights/best.pt")):
        files.append(str(f))
    return files
