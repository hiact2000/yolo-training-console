"""
main_window.py
--------------
YOLO 訓練控制台主視窗（PyQt5）。

分區：
  左側 (QScrollArea) ─ 參數設定（資料集、模型、訓練、優化器、Augmentation、輸出）
  右側               ─ 訓練控制 + 即時 Log + 結果摘要
"""
import os
import sys
import subprocess
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QLineEdit, QTextEdit,
    QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox,
    QProgressBar, QFileDialog, QMessageBox, QAction, QToolBar,
    QSizePolicy,
)

from train_ui import config_manager as cfg_mgr
from train_ui import excel_logger

PROJECT_ROOT = Path(__file__).parent.parent

# 預設模型選項（下拉選單）
_MODEL_PRESETS = [
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n-cls.pt", "yolov8s-cls.pt", "yolov8m-cls.pt",
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt",
]

# Optimizer 選項
_OPTIMIZERS = ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"]

# auto_augment 選項（"(none)" 在程式內轉為 None）
_AUTO_AUGMENTS = ["randaugment", "autoaugment", "augmix", "(none)"]

# device 選項
_DEVICES = ["", "0", "1", "0,1", "cpu"]


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 訓練控制台")
        self.resize(1440, 900)

        # 訓練 subprocess（獨立 process，效能等同終端機直接執行）
        self._proc: subprocess.Popen | None = None
        self._last_save_dir: str = ""
        self._expected_epochs: int = 0
        self._train_project: str = ""
        self._train_name: str = ""

        # 輪詢 timer：每秒檢查 subprocess 是否結束、更新進度
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_training)

        # log buffer (備用，現在主要 log 在終端機)
        self._log_buffer: list[str] = []
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(200)
        self._log_flush_timer.timeout.connect(self._flush_log_buffer)
        self._log_flush_timer.start()

        self._build_ui()
        self._load_last_config()

    # ═══════════════════════════════════════════════════════════
    #  UI 建構
    # ═══════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── Toolbar ──────────────────────────────────────────
        tb = QToolBar("工具列")
        tb.setMovable(False)
        self.addToolBar(tb)

        def _act(text, slot, shortcut=None):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if shortcut:
                a.setShortcut(shortcut)
            tb.addAction(a)

        _act("📂 載入設定",   self._on_load_config)
        _act("💾 儲存設定",   self._on_save_config)
        tb.addSeparator()
        _act("📤 匯出 JSON", self._on_export_json)
        _act("📥 從 JSON 載入", self._on_import_json)
        tb.addSeparator()
        _act("🔄 重設為預設值", self._on_reset_defaults)

        # ── 主佈局：左右分割 ──────────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        # 左：參數 ScrollArea
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(500)
        scroll.setMaximumWidth(580)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(6)
        left_layout.setContentsMargins(8, 8, 8, 8)

        left_layout.addWidget(self._build_dataset_group())
        left_layout.addWidget(self._build_model_group())
        left_layout.addWidget(self._build_training_group())
        left_layout.addWidget(self._build_optimizer_group())
        left_layout.addWidget(self._build_augmentation_group())
        left_layout.addWidget(self._build_output_group())
        left_layout.addStretch()

        scroll.setWidget(left_widget)
        splitter.addWidget(scroll)

        # 右：控制 + Log
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([520, 900])

    # ─── 左側各區塊 ─────────────────────────────────────────

    def _make_group(self, title: str) -> tuple[QGroupBox, QFormLayout]:
        """建立 QGroupBox + QFormLayout，回傳兩者"""
        grp = QGroupBox(title)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)
        grp.setLayout(form)
        return grp, form

    def _build_dataset_group(self) -> QGroupBox:
        grp, form = self._make_group("📁 資料集設定")

        # task
        self.cb_task = QComboBox()
        self.cb_task.addItems(["detect", "classify", "segment"])
        self.cb_task.currentTextChanged.connect(self._on_task_changed)
        form.addRow("task:", self.cb_task)

        # data（yaml / 資料夾）+ browse 按鈕
        data_row = QHBoxLayout()
        self.le_data = QLineEdit()
        self.le_data.setPlaceholderText("yaml 路徑 (detect/segment) 或 資料夾 (classify)")
        self.btn_browse_data = QPushButton("瀏覽")
        self.btn_browse_data.setFixedWidth(52)
        self.btn_browse_data.clicked.connect(self._on_browse_data)
        data_row.addWidget(self.le_data)
        data_row.addWidget(self.btn_browse_data)
        form.addRow("data:", data_row)

        return grp

    def _build_model_group(self) -> QGroupBox:
        grp, form = self._make_group("🤖 模型設定")

        # model（可編輯下拉 + 瀏覽按鈕）
        model_row = QHBoxLayout()
        self.cb_model = QComboBox()
        self.cb_model.setEditable(True)
        self._refresh_model_list()
        self.btn_browse_model = QPushButton("瀏覽")
        self.btn_browse_model.setFixedWidth(52)
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        model_row.addWidget(self.cb_model)
        model_row.addWidget(self.btn_browse_model)
        form.addRow("model:", model_row)

        # pretrained
        self.ck_pretrained = QCheckBox("使用預訓練權重")
        self.ck_pretrained.setChecked(True)
        form.addRow("pretrained:", self.ck_pretrained)

        # freeze（None 或整數）
        self.le_freeze = QLineEdit()
        self.le_freeze.setPlaceholderText("空白 = 不凍結，整數 = 凍結前 N 層")
        self.le_freeze.setFixedWidth(200)
        form.addRow("freeze:", self.le_freeze)

        # dropout
        self.dsb_dropout = self._make_dsb(0.0, 0.0, 1.0, 0.1, 2)
        form.addRow("dropout:", self.dsb_dropout)

        return grp

    def _build_training_group(self) -> QGroupBox:
        grp, form = self._make_group("⚙️ 訓練核心參數")

        self.sb_epochs  = self._make_sb(300, 1, 9999)
        self.sb_batch   = self._make_sb(16, -1, 1024)
        self.sb_imgsz   = self._make_sb(640, 32, 4096, step=32)
        self.sb_patience= self._make_sb(30, 0, 9999)
        self.sb_workers = self._make_sb(8, 0, 64)
        self.sb_seed    = self._make_sb(0, 0, 99999)

        # device（可編輯下拉）
        self.cb_device = QComboBox()
        self.cb_device.setEditable(True)
        self.cb_device.addItems(_DEVICES)
        self.cb_device.setItemData(0, "自動選擇（GPU 優先）", Qt.ToolTipRole)
        self.cb_device.setToolTip("空白=auto, '0'=CUDA:0, 'cpu'=CPU")

        self.ck_resume = QCheckBox("從 last.pt 繼續訓練")
        self.ck_cache  = QCheckBox("快取資料集到記憶體")
        self.ck_cos_lr = QCheckBox("Cosine LR Scheduler")
        self.ck_amp    = QCheckBox("自動混合精度 (AMP)")
        self.ck_amp.setChecked(True)

        form.addRow("epochs:",   self.sb_epochs)
        form.addRow("batch:",    self.sb_batch)
        form.addRow("imgsz:",    self.sb_imgsz)
        form.addRow("patience:", self.sb_patience)
        form.addRow("device:",   self.cb_device)
        form.addRow("workers:",  self.sb_workers)
        form.addRow("seed:",     self.sb_seed)

        # 勾選項目用 grid 排列
        chk_grid = QGridLayout()
        chk_grid.addWidget(self.ck_resume, 0, 0)
        chk_grid.addWidget(self.ck_cache,  0, 1)
        chk_grid.addWidget(self.ck_cos_lr, 1, 0)
        chk_grid.addWidget(self.ck_amp,    1, 1)
        form.addRow("選項:", chk_grid)

        return grp

    def _build_optimizer_group(self) -> QGroupBox:
        grp, form = self._make_group("📈 優化器參數")

        self.cb_optimizer = QComboBox()
        self.cb_optimizer.addItems(_OPTIMIZERS)

        self.dsb_lr0             = self._make_dsb(0.01,  0.0001, 1.0,  0.001, 4)
        self.dsb_lrf             = self._make_dsb(0.01,  0.0001, 1.0,  0.001, 4)
        self.dsb_momentum        = self._make_dsb(0.937, 0.0,    1.0,  0.01,  3)
        self.dsb_weight_decay    = self._make_dsb(0.0005,0.0,    0.1,  0.0001,5)
        self.dsb_warmup_epochs   = self._make_dsb(3.0,   0.0,    20.0, 0.5,   1)
        self.dsb_warmup_momentum = self._make_dsb(0.8,   0.0,    1.0,  0.05,  2)
        self.dsb_warmup_bias_lr  = self._make_dsb(0.1,   0.0,    1.0,  0.01,  3)

        form.addRow("optimizer:",       self.cb_optimizer)
        form.addRow("lr0:",             self.dsb_lr0)
        form.addRow("lrf:",             self.dsb_lrf)
        form.addRow("momentum:",        self.dsb_momentum)
        form.addRow("weight_decay:",    self.dsb_weight_decay)
        form.addRow("warmup_epochs:",   self.dsb_warmup_epochs)
        form.addRow("warmup_momentum:", self.dsb_warmup_momentum)
        form.addRow("warmup_bias_lr:",  self.dsb_warmup_bias_lr)

        return grp

    def _build_augmentation_group(self) -> QGroupBox:
        grp, form = self._make_group("🎨 Augmentation 參數")

        self.dsb_hsv_h      = self._make_dsb(0.015, 0.0, 1.0, 0.01, 3)
        self.dsb_hsv_s      = self._make_dsb(0.7,   0.0, 1.0, 0.1,  2)
        self.dsb_hsv_v      = self._make_dsb(0.4,   0.0, 1.0, 0.1,  2)
        self.dsb_degrees    = self._make_dsb(0.0,   0.0, 180.0, 5.0, 1)
        self.dsb_translate  = self._make_dsb(0.1,   0.0, 1.0, 0.05, 2)
        self.dsb_scale      = self._make_dsb(0.5,   0.0, 10.0, 0.1, 2)
        self.dsb_fliplr     = self._make_dsb(0.5,   0.0, 1.0, 0.1,  2)
        self.dsb_flipud     = self._make_dsb(0.0,   0.0, 1.0, 0.1,  2)
        self.dsb_mosaic     = self._make_dsb(1.0,   0.0, 1.0, 0.1,  2)
        self.dsb_mixup      = self._make_dsb(0.0,   0.0, 1.0, 0.1,  2)
        self.dsb_erasing    = self._make_dsb(0.4,   0.0, 1.0, 0.1,  2)
        self.dsb_copy_paste = self._make_dsb(0.0,   0.0, 1.0, 0.1,  2)

        self.cb_auto_augment = QComboBox()
        self.cb_auto_augment.addItems(_AUTO_AUGMENTS)

        form.addRow("hsv_h:",        self.dsb_hsv_h)
        form.addRow("hsv_s:",        self.dsb_hsv_s)
        form.addRow("hsv_v:",        self.dsb_hsv_v)
        form.addRow("degrees:",      self.dsb_degrees)
        form.addRow("translate:",    self.dsb_translate)
        form.addRow("scale:",        self.dsb_scale)
        form.addRow("fliplr:",       self.dsb_fliplr)
        form.addRow("flipud:",       self.dsb_flipud)
        form.addRow("mosaic:",       self.dsb_mosaic)
        form.addRow("mixup:",        self.dsb_mixup)
        form.addRow("erasing:",      self.dsb_erasing)
        form.addRow("copy_paste:",   self.dsb_copy_paste)
        form.addRow("auto_augment:", self.cb_auto_augment)

        return grp

    def _build_output_group(self) -> QGroupBox:
        grp, form = self._make_group("📂 輸出設定")

        # project（含瀏覽按鈕）
        proj_row = QHBoxLayout()
        self.le_project = QLineEdit("runs/detect")
        self.btn_browse_proj = QPushButton("瀏覽")
        self.btn_browse_proj.setFixedWidth(52)
        self.btn_browse_proj.clicked.connect(self._on_browse_project)
        proj_row.addWidget(self.le_project)
        proj_row.addWidget(self.btn_browse_proj)
        form.addRow("project:", proj_row)

        self.le_name = QLineEdit("exp")
        form.addRow("name:", self.le_name)

        self.sb_save_period = self._make_sb(-1, -1, 9999)
        self.sb_save_period.setToolTip("-1 = 不定期儲存，0 = 每 epoch 儲存")
        form.addRow("save_period:", self.sb_save_period)

        # 訓練用 Python（必須是含 CUDA torch 的 conda 環境）
        py_row = QHBoxLayout()
        self.le_train_python = QLineEdit(cfg_mgr.DEFAULT_TRAIN_PYTHON)
        self.le_train_python.setToolTip("用於執行訓練的 python.exe 路徑（需含 CUDA torch）")
        btn_browse_py = QPushButton("瀏覽")
        btn_browse_py.setFixedWidth(52)
        btn_browse_py.clicked.connect(self._on_browse_train_python)
        py_row.addWidget(self.le_train_python)
        py_row.addWidget(btn_browse_py)
        form.addRow("訓練 Python:", py_row)

        # GPU 狀態顯示
        self.lbl_gpu_status = QLabel("⏳ 偵測中...")
        form.addRow("GPU 狀態:", self.lbl_gpu_status)
        QTimer.singleShot(500, self._check_gpu_status)

        # 備註
        self.le_notes = QLineEdit()
        self.le_notes.setPlaceholderText("本次訓練備註（寫入 Excel）")
        form.addRow("備註:", self.le_notes)

        # Excel 路徑
        excel_row = QHBoxLayout()
        self.le_excel = QLineEdit(str(excel_logger.DEFAULT_EXCEL_PATH))
        self.btn_browse_excel = QPushButton("瀏覽")
        self.btn_browse_excel.setFixedWidth(52)
        self.btn_browse_excel.clicked.connect(self._on_browse_excel)
        excel_row.addWidget(self.le_excel)
        excel_row.addWidget(self.btn_browse_excel)
        form.addRow("Excel 輸出:", excel_row)

        return grp

    # ─── 右側面板 ─────────────────────────────────────────────

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── 控制按鈕 ──────────────────────────────────────────
        btn_row = QHBoxLayout()

        self.btn_start = QPushButton("▶  開始訓練")
        self.btn_start.setFixedHeight(40)
        self.btn_start.setStyleSheet(
            "background-color:#4CAF50;color:white;font-size:14px;font-weight:bold;"
        )
        self.btn_start.clicked.connect(self._on_start_training)

        self.btn_stop = QPushButton("■  停止訓練")
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet(
            "background-color:#f44336;color:white;font-size:14px;font-weight:bold;"
        )
        self.btn_stop.clicked.connect(self._on_stop_training)

        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        layout.addLayout(btn_row)

        # ── 進度條 ────────────────────────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Epoch %v / %m")
        self.progress_bar.setFixedHeight(20)
        layout.addWidget(self.progress_bar)

        # ── 狀態標籤 ──────────────────────────────────────────
        self.lbl_status = QLabel("就緒")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_status)

        # ── Log 區域 ──────────────────────────────────────────
        log_label = QLabel("訓練 Log：")
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet(
            "background-color:#1e1e1e;color:#d4d4d4;"
        )
        layout.addWidget(self.log_text, stretch=1)

        # ── 結果摘要 ──────────────────────────────────────────
        result_grp = QGroupBox("訓練結果")
        result_layout = QFormLayout()
        result_layout.setSpacing(4)
        result_grp.setLayout(result_layout)

        self.lbl_save_dir  = QLabel("-")
        self.lbl_save_dir.setWordWrap(True)
        self.lbl_best_pt   = QLabel("-")
        self.lbl_best_pt.setWordWrap(True)
        self.lbl_map50     = QLabel("-")
        self.lbl_map5095   = QLabel("-")
        self.lbl_precision = QLabel("-")
        self.lbl_recall    = QLabel("-")
        self.lbl_excel_out = QLabel("-")
        self.lbl_excel_out.setWordWrap(True)

        result_layout.addRow("save_dir:", self.lbl_save_dir)
        result_layout.addRow("best.pt:",  self.lbl_best_pt)
        result_layout.addRow("mAP50:",    self.lbl_map50)
        result_layout.addRow("mAP50-95:", self.lbl_map5095)
        result_layout.addRow("Precision:",self.lbl_precision)
        result_layout.addRow("Recall:",   self.lbl_recall)
        result_layout.addRow("Excel:",    self.lbl_excel_out)

        layout.addWidget(result_grp)

        return panel

    # ═══════════════════════════════════════════════════════════
    #  Widget 輔助函式
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _make_sb(default: int, min_: int, max_: int, step: int = 1) -> QSpinBox:
        sb = QSpinBox()
        sb.setRange(min_, max_)
        sb.setValue(default)
        sb.setSingleStep(step)
        return sb

    @staticmethod
    def _make_dsb(default: float, min_: float, max_: float,
                  step: float, decimals: int) -> QDoubleSpinBox:
        dsb = QDoubleSpinBox()
        dsb.setRange(min_, max_)
        dsb.setValue(default)
        dsb.setSingleStep(step)
        dsb.setDecimals(decimals)
        return dsb

    def _refresh_model_list(self):
        """重新掃描本機 .pt 檔案，填入 model 下拉選單"""
        current = self.cb_model.currentText() if self.cb_model.count() > 0 else ""
        self.cb_model.blockSignals(True)
        self.cb_model.clear()
        local_files = cfg_mgr.get_local_pt_files()
        items = list(dict.fromkeys(_MODEL_PRESETS + local_files))  # 去重
        self.cb_model.addItems(items)
        if current:
            idx = self.cb_model.findText(current)
            if idx >= 0:
                self.cb_model.setCurrentIndex(idx)
        self.cb_model.blockSignals(False)

    # ═══════════════════════════════════════════════════════════
    #  設定 get / set
    # ═══════════════════════════════════════════════════════════

    def get_config(self) -> dict:
        """從所有 Widget 收集目前設定，回傳 dict"""
        return {
            # 資料集
            "task":  self.cb_task.currentText(),
            "data":  self.le_data.text().strip(),
            # 模型
            "model":      self.cb_model.currentText().strip(),
            "pretrained": self.ck_pretrained.isChecked(),
            "freeze":     self.le_freeze.text().strip(),
            "dropout":    self.dsb_dropout.value(),
            # 訓練核心
            "epochs":   self.sb_epochs.value(),
            "batch":    self.sb_batch.value(),
            "imgsz":    self.sb_imgsz.value(),
            "patience": self.sb_patience.value(),
            "device":   self.cb_device.currentText().strip(),
            "workers":  self.sb_workers.value(),
            "seed":     self.sb_seed.value(),
            "resume":   self.ck_resume.isChecked(),
            "cache":    self.ck_cache.isChecked(),
            "cos_lr":   self.ck_cos_lr.isChecked(),
            "amp":      self.ck_amp.isChecked(),
            # 優化器
            "optimizer":       self.cb_optimizer.currentText(),
            "lr0":             self.dsb_lr0.value(),
            "lrf":             self.dsb_lrf.value(),
            "momentum":        self.dsb_momentum.value(),
            "weight_decay":    self.dsb_weight_decay.value(),
            "warmup_epochs":   self.dsb_warmup_epochs.value(),
            "warmup_momentum": self.dsb_warmup_momentum.value(),
            "warmup_bias_lr":  self.dsb_warmup_bias_lr.value(),
            # Augmentation
            "hsv_h":        self.dsb_hsv_h.value(),
            "hsv_s":        self.dsb_hsv_s.value(),
            "hsv_v":        self.dsb_hsv_v.value(),
            "degrees":      self.dsb_degrees.value(),
            "translate":    self.dsb_translate.value(),
            "scale":        self.dsb_scale.value(),
            "fliplr":       self.dsb_fliplr.value(),
            "flipud":       self.dsb_flipud.value(),
            "mosaic":       self.dsb_mosaic.value(),
            "mixup":        self.dsb_mixup.value(),
            "erasing":      self.dsb_erasing.value(),
            "copy_paste":   self.dsb_copy_paste.value(),
            "auto_augment": self.cb_auto_augment.currentText(),
            # 輸出
            "project":     self.le_project.text().strip(),
            "name":        self.le_name.text().strip(),
            "save_period": self.sb_save_period.value(),
            # 備註（UI 專用，不傳給 YOLO）
            "notes": self.le_notes.text().strip(),
            # 訓練用 Python 路徑（UI 專用）
            "train_python": self.le_train_python.text().strip(),
        }

    def set_config(self, config: dict):
        """從 dict 更新所有 Widget"""
        def _set_cb(cb: QComboBox, val: str):
            idx = cb.findText(str(val))
            if idx >= 0:
                cb.setCurrentIndex(idx)
            elif cb.isEditable():
                cb.setCurrentText(str(val))

        _set_cb(self.cb_task,         config.get("task", "detect"))
        self.le_data.setText(          config.get("data", ""))
        _set_cb(self.cb_model,         config.get("model", "yolov8s.pt"))
        self.ck_pretrained.setChecked( config.get("pretrained", True))
        self.le_freeze.setText(        str(config.get("freeze", "") or ""))
        self.dsb_dropout.setValue(     float(config.get("dropout", 0.0)))

        self.sb_epochs.setValue(       int(config.get("epochs", 300)))
        self.sb_batch.setValue(        int(config.get("batch", 16)))
        self.sb_imgsz.setValue(        int(config.get("imgsz", 640)))
        self.sb_patience.setValue(     int(config.get("patience", 30)))
        _set_cb(self.cb_device,        config.get("device", ""))
        self.sb_workers.setValue(      int(config.get("workers", 8)))
        self.sb_seed.setValue(         int(config.get("seed", 0)))
        self.ck_resume.setChecked(     bool(config.get("resume", False)))
        self.ck_cache.setChecked(      bool(config.get("cache", False)))
        self.ck_cos_lr.setChecked(     bool(config.get("cos_lr", False)))
        self.ck_amp.setChecked(        bool(config.get("amp", True)))

        _set_cb(self.cb_optimizer,     config.get("optimizer", "auto"))
        self.dsb_lr0.setValue(         float(config.get("lr0", 0.01)))
        self.dsb_lrf.setValue(         float(config.get("lrf", 0.01)))
        self.dsb_momentum.setValue(    float(config.get("momentum", 0.937)))
        self.dsb_weight_decay.setValue(float(config.get("weight_decay", 0.0005)))
        self.dsb_warmup_epochs.setValue(float(config.get("warmup_epochs", 3.0)))
        self.dsb_warmup_momentum.setValue(float(config.get("warmup_momentum", 0.8)))
        self.dsb_warmup_bias_lr.setValue(float(config.get("warmup_bias_lr", 0.1)))

        self.dsb_hsv_h.setValue(     float(config.get("hsv_h", 0.015)))
        self.dsb_hsv_s.setValue(     float(config.get("hsv_s", 0.7)))
        self.dsb_hsv_v.setValue(     float(config.get("hsv_v", 0.4)))
        self.dsb_degrees.setValue(   float(config.get("degrees", 0.0)))
        self.dsb_translate.setValue( float(config.get("translate", 0.1)))
        self.dsb_scale.setValue(     float(config.get("scale", 0.5)))
        self.dsb_fliplr.setValue(    float(config.get("fliplr", 0.5)))
        self.dsb_flipud.setValue(    float(config.get("flipud", 0.0)))
        self.dsb_mosaic.setValue(    float(config.get("mosaic", 1.0)))
        self.dsb_mixup.setValue(     float(config.get("mixup", 0.0)))
        self.dsb_erasing.setValue(   float(config.get("erasing", 0.4)))
        self.dsb_copy_paste.setValue(float(config.get("copy_paste", 0.0)))
        _set_cb(self.cb_auto_augment, config.get("auto_augment", "randaugment") or "(none)")

        self.le_project.setText(    config.get("project", "runs/detect"))
        self.le_name.setText(       config.get("name", "exp"))
        self.sb_save_period.setValue(int(config.get("save_period", -1)))
        self.le_notes.setText(      config.get("notes", ""))
        self.le_train_python.setText(
            config.get("train_python", cfg_mgr.DEFAULT_TRAIN_PYTHON)
        )

    # ═══════════════════════════════════════════════════════════
    #  槽（事件處理）
    # ═══════════════════════════════════════════════════════════

    def _on_task_changed(self, task: str):
        """task 改變時更新 project 預設路徑、data 提示"""
        default_proj = cfg_mgr.TASK_DEFAULT_PROJECT.get(task, "runs/detect")
        current_proj = self.le_project.text()
        # 只在還是預設值時才自動更新
        if current_proj in cfg_mgr.TASK_DEFAULT_PROJECT.values():
            self.le_project.setText(default_proj)

        hint = "yaml 路徑（detect/segment）" if task != "classify" \
               else "資料夾路徑（classify），例如 datasets/classify"
        self.le_data.setPlaceholderText(hint)

    def _on_browse_data(self):
        task = self.cb_task.currentText()
        if task == "classify":
            path = QFileDialog.getExistingDirectory(self, "選擇分類資料集資料夾",
                                                    str(PROJECT_ROOT))
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "選擇 data yaml", str(PROJECT_ROOT / "configs"),
                "YAML Files (*.yaml *.yml)"
            )
        if path:
            self.le_data.setText(path)

    def _on_browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型權重", str(PROJECT_ROOT),
            "PyTorch Weights (*.pt)"
        )
        if path:
            self.cb_model.setCurrentText(path)

    def _on_browse_project(self):
        path = QFileDialog.getExistingDirectory(self, "選擇 project 輸出資料夾",
                                                str(PROJECT_ROOT))
        if path:
            self.le_project.setText(path)

    def _on_browse_train_python(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇訓練用 python.exe",
            str(Path(cfg_mgr.DEFAULT_TRAIN_PYTHON).parent),
            "Python Executable (python.exe python3.exe)"
        )
        if path:
            self.le_train_python.setText(path)
            QTimer.singleShot(100, self._check_gpu_status)

    def _check_gpu_status(self):
        """用設定的 Python 執行 CUDA 偵測，結果顯示在 GPU 狀態標籤"""
        python = self.le_train_python.text().strip()
        if not os.path.exists(python):
            self.lbl_gpu_status.setText("❌ python.exe 路徑不存在")
            return
        try:
            result = subprocess.run(
                [python, "-c",
                 "import torch; "
                 "print('OK', torch.__version__, "
                 "'GPU:'+torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                 "else 'CPU only')"],
                capture_output=True, text=True, timeout=15
            )
            out = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            if "OK" in out:
                self.lbl_gpu_status.setText(f"✅ {out.replace('OK ', '')}")
            else:
                self.lbl_gpu_status.setText(f"⚠️ {result.stderr.strip()[:80]}")
        except Exception as e:
            self.lbl_gpu_status.setText(f"❌ 偵測失敗: {e}")

    def _on_browse_excel(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "選擇 Excel 記錄檔", str(excel_logger.DEFAULT_EXCEL_PATH),
            "Excel Files (*.xlsx)"
        )
        if path:
            self.le_excel.setText(path)

    def _on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "載入設定檔", str(PROJECT_ROOT / "train_ui"),
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            config = cfg_mgr.load_config(path)
            self.set_config(config)
            self._append_log(f"[UI] 已載入設定: {path}")
        except Exception as e:
            QMessageBox.critical(self, "載入失敗", str(e))

    def _on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "儲存設定檔", str(PROJECT_ROOT / "train_ui" / "my_config.json"),
            "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            cfg_mgr.save_config(path, self.get_config())
            self._append_log(f"[UI] 設定已儲存: {path}")
        except Exception as e:
            QMessageBox.critical(self, "儲存失敗", str(e))

    def _on_export_json(self):
        self._on_save_config()   # 同 save，只是換個名稱讓工具列更清楚

    def _on_import_json(self):
        self._on_load_config()

    def _on_reset_defaults(self):
        reply = QMessageBox.question(self, "確認", "重設為預設值？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.set_config(cfg_mgr.DEFAULT_CONFIG.copy())

    def _load_last_config(self):
        """啟動時自動載入上次設定"""
        config = cfg_mgr.load_last_config()
        self.set_config(config)

    # ── 訓練控制 ──────────────────────────────────────────────

    def _on_start_training(self):
        config = self.get_config()

        # ── 基本驗證 ──────────────────────────────────────────
        data = config.get("data", "").strip()
        if not data:
            QMessageBox.warning(self, "缺少參數", "請設定 data 路徑")
            return
        if not os.path.exists(data):
            QMessageBox.warning(self, "路徑不存在", f"data 路徑不存在:\n{data}")
            return
        model = config.get("model", "").strip()
        if not model:
            QMessageBox.warning(self, "缺少參數", "請選擇模型")
            return
        if os.path.isabs(model) and not os.path.exists(model):
            QMessageBox.warning(self, "權重不存在", f"模型權重檔不存在:\n{model}")
            return

        # ── 儲存設定 & 寫入暫存 config JSON ───────────────────
        cfg_mgr.save_last_config(config)
        config_json = PROJECT_ROOT / "train_ui" / ".current_run.json"
        cfg_mgr.save_config(str(config_json), config)

        # ── 記錄本次訓練預期的輸出路徑（用於輪詢進度）─────────
        self._expected_epochs = config["epochs"]
        self._train_project   = config.get("project", "runs/detect")
        self._train_name      = config.get("name", "exp")

        # ── 清空 UI ───────────────────────────────────────────
        self.log_text.clear()
        self._reset_result_labels()
        self.progress_bar.setMaximum(self._expected_epochs)
        self.progress_bar.setValue(0)

        # ── 以獨立 subprocess 啟動訓練 ────────────────────────
        # 使用設定的 train_python（確保是 conda 20250831 的 CUDA 環境）
        train_python = config.get("train_python", "").strip() or sys.executable
        if not os.path.exists(train_python):
            QMessageBox.critical(self, "Python 路徑錯誤",
                                 f"訓練 Python 不存在:\n{train_python}\n\n"
                                 "請在輸出設定區確認「訓練 Python」路徑。")
            return
        runner = str(PROJECT_ROOT / "train_ui" / "runner.py")
        self._proc = subprocess.Popen(
            [train_python, runner, str(config_json)],
            cwd=str(PROJECT_ROOT),
        )
        self._append_log(f"[UI] Python: {train_python}")

        self._poll_timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_status.setText("⏳ 訓練中（詳細 log 請看 PyCharm console）...")
        self._append_log(f"[UI] 已啟動訓練 subprocess（PID {self._proc.pid}）")
        self._append_log(f"[UI] project={self._train_project}  name={self._train_name}")

    def _on_stop_training(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("⚠️ 正在停止...")
            self._append_log("[UI] 已發送終止訊號給訓練 subprocess")

    # ── subprocess 輪詢（每秒一次） ──────────────────────────

    def _poll_training(self):
        """每秒檢查 subprocess 狀態，並從 results.csv 更新進度"""
        if self._proc is None:
            self._poll_timer.stop()
            return

        # 更新進度條（讀 results.csv 行數）
        self._update_progress_from_csv()

        # 檢查是否結束
        rc = self._proc.poll()
        if rc is None:
            return  # 還在跑

        self._poll_timer.stop()
        self._proc = None

        if rc == 0:
            save_dir = self._find_save_dir()
            self._on_training_finished(save_dir)
        else:
            self._on_training_error(f"訓練 subprocess 結束，return code = {rc}")

    def _update_progress_from_csv(self):
        """從 results.csv 的行數推算目前 epoch"""
        save_dir = self._find_save_dir()
        if not save_dir:
            return
        csv_path = os.path.join(save_dir, "results.csv")
        if not os.path.exists(csv_path):
            return
        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                epoch = max(0, sum(1 for _ in f) - 1)   # 減去 header 列
            if epoch > 0:
                self.progress_bar.setValue(min(epoch, self._expected_epochs))
                self.lbl_status.setText(f"⏳ Epoch {epoch} / {self._expected_epochs}")
        except Exception:
            pass

    def _find_save_dir(self) -> str:
        """找出 ultralytics 實際建立的 save_dir（會自動在 name 後加數字）"""
        base = PROJECT_ROOT / self._train_project
        if not base.exists():
            return ""
        # 找所有以 name 開頭的子資料夾，取最新修改時間的那個
        candidates = [
            d for d in base.iterdir()
            if d.is_dir() and d.name.startswith(self._train_name)
        ]
        if not candidates:
            return ""
        return str(max(candidates, key=lambda d: d.stat().st_mtime))

    def _on_training_finished(self, save_dir: str):
        self._last_save_dir = save_dir
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("✅ 訓練完成！")
        self._append_log(f"[UI] 訓練完成！save_dir = {save_dir}")

        self.lbl_save_dir.setText(save_dir or "-")
        best_pt = os.path.join(save_dir, "weights", "best.pt") if save_dir else ""
        self.lbl_best_pt.setText(best_pt if os.path.exists(best_pt) else "-")

        self._fill_metrics_from_csv(save_dir)
        self._write_excel(save_dir)

    def _on_training_error(self, msg: str):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("❌ 訓練失敗 / 中止")
        self._append_log(f"[UI] {msg}")
        QMessageBox.critical(self, "訓練結束", msg)

    # ── 結果摘要 & Excel ─────────────────────────────────────

    def _fill_metrics_from_csv(self, save_dir: str):
        """從 results.csv 最後一列填入結果標籤"""
        if not save_dir:
            return
        csv_path = os.path.join(save_dir, "results.csv")
        if not os.path.exists(csv_path):
            return
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if df.empty:
                return
            last = df.iloc[-1].to_dict()

            def _get(*keys) -> str:
                for k in keys:
                    if k in last:
                        return f"{last[k]:.4f}"
                return "-"

            self.lbl_map50.setText(_get("metrics/mAP50(B)", "metrics/mAP50"))
            self.lbl_map5095.setText(_get("metrics/mAP50-95(B)", "metrics/mAP50-95"))
            self.lbl_precision.setText(_get("metrics/precision(B)", "metrics/precision"))
            self.lbl_recall.setText(_get("metrics/recall(B)", "metrics/recall"))
        except Exception:
            pass

    def _write_excel(self, save_dir: str):
        config = self.get_config()
        notes  = config.get("notes", "")
        excel  = self.le_excel.text().strip() or str(excel_logger.DEFAULT_EXCEL_PATH)
        try:
            out = excel_logger.log_to_excel(save_dir, config, notes, excel)
            self.lbl_excel_out.setText(out)
            self._append_log(f"[UI] Excel 記錄已寫入: {out}")
        except PermissionError as e:
            self.lbl_excel_out.setText(str(e))
            QMessageBox.warning(self, "Excel 寫入警告", str(e))
        except Exception as e:
            self.lbl_excel_out.setText(f"失敗: {e}")
            self._append_log(f"[UI] Excel 寫入失敗: {e}")

    # ── UI 輔助 ──────────────────────────────────────────────

    def _append_log(self, text: str):
        """將一行 log 放入緩衝區（由 QTimer 批次寫入，不即時 repaint）"""
        self._log_buffer.append(text)

    def _flush_log_buffer(self):
        """每 150ms 由 QTimer 呼叫，將緩衝的 log 一次寫入 QTextEdit"""
        if not self._log_buffer:
            return
        # 一次 append 所有行，只觸發一次 repaint
        self.log_text.append("\n".join(self._log_buffer))
        self._log_buffer.clear()
        self.log_text.moveCursor(QTextCursor.End)

    def _reset_result_labels(self):
        for lbl in (self.lbl_save_dir, self.lbl_best_pt,
                    self.lbl_map50, self.lbl_map5095,
                    self.lbl_precision, self.lbl_recall, self.lbl_excel_out):
            lbl.setText("-")

    def closeEvent(self, event):
        """關閉視窗前停止訓練並儲存設定"""
        if self._proc and self._proc.poll() is None:
            reply = QMessageBox.question(
                self, "訓練進行中",
                "訓練 subprocess 仍在執行，確定要關閉並終止訓練嗎？",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()

        self._poll_timer.stop()
        cfg_mgr.save_last_config(self.get_config())
        event.accept()
