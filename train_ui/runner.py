"""
runner.py
---------
由 UI 以 subprocess 呼叫的獨立訓練腳本。
從 JSON 設定檔讀取參數，直接執行 YOLO 訓練。
效能與在終端機直接執行完全相同。

用法（由 main_window.py 自動呼叫，無需手動執行）：
    python train_ui/runner.py <config_json_path>
"""
import json
import sys
from pathlib import Path


def _build_params(raw: dict) -> tuple[str, dict]:
    """從 UI 設定 dict 轉換為 model_path + train() 參數"""
    p = raw.copy()

    model_path = p.pop("model", "yolov8s.pt")
    p.pop("notes", None)          # UI 專用欄位
    p.pop("train_python", None)   # UI 專用欄位

    # device：空字串 → None
    if p.get("device") == "":
        p["device"] = None

    # freeze：空字串/None → None；否則轉 int
    freeze = p.get("freeze", "")
    p["freeze"] = int(freeze) if str(freeze).strip().lstrip("-").isdigit() else None

    # auto_augment："(none)" → None
    if p.get("auto_augment") == "(none)":
        p["auto_augment"] = None

    return model_path, p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python runner.py <config.json>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    if not config_path.exists():
        print(f"[runner] 找不到設定檔: {config_path}")
        sys.exit(1)

    params = json.loads(config_path.read_text(encoding="utf-8"))
    model_path, train_params = _build_params(params)

    from ultralytics import YOLO

    model = YOLO(model_path)
    model.train(**train_params)
