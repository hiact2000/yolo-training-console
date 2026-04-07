"""
app.py
------
YOLO 訓練控制台進入點。

【執行方式】
    conda activate <你的環境名稱>
    cd /path/to/PythonProject3
    python main.py
    # 或
    python -m train_ui.app
"""
import sys
from pathlib import Path

# 確保 project root 在 sys.path，讓 `import train_ui` 可運作
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── 關鍵：在主執行緒預先載入 torch / ultralytics ──────────────
# Windows 上 PyTorch DLL（c10.dll 等）必須由主執行緒第一次初始化。
# 若延遲到 QThread 內才 import，會觸發 WinError 1114。
try:
    import torch          # noqa: F401  載入 c10.dll、torch_cpu.dll
    from ultralytics import YOLO as _YOLO  # noqa: F401  完成所有 DLL 綁定
    del _YOLO
except Exception as _e:
    print(f"[WARNING] 預載入 torch/ultralytics 失敗: {_e}")
# ─────────────────────────────────────────────────────────────

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

from train_ui.main_window import MainWindow


def main():
    # 高 DPI 支援（Windows 縮放 > 100% 時仍清晰）
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")          # 跨平台一致外觀
    app.setApplicationName("YOLO 訓練控制台")

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
