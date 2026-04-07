"""
training_worker.py
------------------
在 QThread 中執行 YOLO 訓練，透過 Qt Signals 回傳 log 與結果。
訓練過程不阻塞主視窗。

【設計說明】
不攔截 stdout/stderr，讓 ultralytics 直接輸出到終端機（PyCharm console）。
UI 只透過 ultralytics callbacks 接收 epoch 進度，零額外開銷。
"""
import sys
import traceback

from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    """
    YOLO 訓練執行緒。

    Signals:
        log_signal(str):          每行 log 文字
        progress_signal(int,int): (目前 epoch, 總 epoch)
        finished_signal(str):     訓練完成，帶 save_dir 路徑
        error_signal(str):        訓練失敗，帶錯誤訊息
    """

    log_signal      = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal    = pyqtSignal(str)

    def __init__(self, params: dict):
        """
        Args:
            params: 訓練參數 dict。
                    必須含 'model' key（用於 YOLO() 建構子）。
                    其餘 key 傳給 model.train()。
        """
        super().__init__()
        # 複製一份避免修改原始 dict
        self._params = params.copy()
        self._stop_requested = False
        self._trainer = None          # 儲存 ultralytics Trainer 參考

    # ── 公開介面 ─────────────────────────────────────────────

    def request_stop(self):
        """從主執行緒呼叫，在下一個 epoch 開始時中止訓練"""
        self._stop_requested = True
        self.log_signal.emit("[UI] 已發送停止訊號，訓練將在本 epoch 結束後中止...")

    # ── 私有工具 ─────────────────────────────────────────────

    def _build_train_params(self) -> dict:
        """從 UI 設定 dict 建立傳給 model.train() 的參數"""
        p = self._params.copy()

        # device：空字串 → None（讓 ultralytics 自動選）
        if p.get("device") == "":
            p["device"] = None

        # freeze：空字串 → None；數字字串 → int
        freeze_raw = p.get("freeze", "")
        if freeze_raw == "" or freeze_raw is None:
            p["freeze"] = None
        else:
            try:
                p["freeze"] = int(freeze_raw)
            except (ValueError, TypeError):
                p["freeze"] = None

        # auto_augment："(none)" → None
        if p.get("auto_augment") == "(none)":
            p["auto_augment"] = None

        # notes 是 UI 專用欄位，不傳給 YOLO
        p.pop("notes", None)

        return p

    # ── QThread.run() ────────────────────────────────────────

    def run(self):
        # stdout/stderr 不攔截，ultralytics 直接輸出到終端機（零額外開銷）
        save_dir = ""
        try:
            from ultralytics import YOLO

            params = self._build_train_params()
            model_path = params.pop("model", "yolov8s.pt")

            self.log_signal.emit(f"[UI] 載入模型: {model_path}")
            model = YOLO(model_path)

            # ── ultralytics callbacks（僅用於進度與停止控制）────
            worker = self

            def _on_train_start(trainer):
                worker._trainer = trainer

            def _on_epoch_start(trainer):
                if worker._stop_requested:
                    raise RuntimeError("[UI] 使用者請求停止訓練")

            def _on_epoch_end(trainer):
                epoch = int(trainer.epoch) + 1
                total = int(trainer.epochs)
                worker.progress_signal.emit(epoch, total)
                worker.log_signal.emit(f"[Epoch {epoch}/{total}] 完成")

            model.add_callback("on_train_start",       _on_train_start)
            model.add_callback("on_train_epoch_start", _on_epoch_start)
            model.add_callback("on_train_epoch_end",   _on_epoch_end)
            # ─────────────────────────────────────────────────

            self.log_signal.emit("[UI] 開始訓練，詳細 log 請看 PyCharm console...")
            results = model.train(**params)

            if results is not None and hasattr(results, "save_dir"):
                save_dir = str(results.save_dir)
            elif self._trainer is not None and hasattr(self._trainer, "save_dir"):
                save_dir = str(self._trainer.save_dir)

            self.log_signal.emit(f"[UI] 訓練完成！save_dir = {save_dir}")
            self.finished_signal.emit(save_dir)

        except RuntimeError as e:
            msg = str(e)
            self.log_signal.emit(f"[UI] 訓練中斷: {msg}")
            if save_dir:
                self.finished_signal.emit(save_dir)
            else:
                self.error_signal.emit(msg)

        except Exception:
            tb = traceback.format_exc()
            self.log_signal.emit(f"[UI] 訓練錯誤: {tb.splitlines()[-1]}")
            self.error_signal.emit(tb.splitlines()[-1])
