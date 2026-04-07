# 雞冠辨識系統 — YOLO 實作入門指南 for VT Lab / V.1

By 蘇柏瑋 2026.04.05

> 這份 README 會帶你從零開始，了解這個專案的目的、架構、環境設定，以及如何跑訓練。

---

## 目錄

1. [專案目標](#1-專案目標)
2. [系統架構概覽](#2-系統架構概覽)
3. [資料夾結構說明](#3-資料夾結構說明)
4. [環境設定](#4-環境設定)
5. [資料集說明（Roboflow）](#5-資料集說明roboflow)
6. [資料集準備流程](#6-資料集準備流程)
7. [訓練流程](#7-訓練流程)
8. [推論（測試模型效果）](#8-推論測試模型效果)
9. [腳本功能說明](#9-腳本功能說明)
10. [設定檔說明](#10-設定檔說明)
11. [常見問題](#11-常見問題)
12. [YOLO 基礎觀念快速入門](#12-yolo-基礎觀念快速入門)

---

## 1. 專案目標

這個專案的目的是用電腦視覺自動評估**蛋雞雞冠的健康等級（Score 1~5）**，取代人工肉眼判斷。

整個流程分兩個階段：

```
輸入影像
   │
   ▼
[第一階段] 偵測 (Detection)
   └─ 找出影像中雞冠的位置（bounding box）
   │
   ▼
[第二階段] 分類 (Classification)
   └─ 把偵測到的雞冠裁切出來，判斷是哪個等級（1~5）
   │
   ▼
輸出結果（標示等級的影像或影片）
```

---

## 2. 系統架構概覽

| 模組 | 說明 |
|------|------|
| **Detection 模型** | YOLOv8s，找出雞冠在影像中的位置 |
| **Classification 模型** | YOLOv8n-cls，判斷雞冠品質等級（1~5） |
| **訓練 GUI** | PyQt5 視窗介面，方便調整訓練參數 |
| **資料準備腳本** | 把偵測標註轉換成分類用的裁切圖片 |
| **Excel 記錄** | 自動把每次訓練結果記錄到 Excel 方便比較 |

---

## 3. 資料夾結構說明

```
PythonProject3/
│
├── configs/                  # 設定檔
│   ├── data.yaml             # Detection 資料集設定（Score1 標籤）
│   └── 0121data.yaml         # Detection 資料集設定（comb 標籤）
│
├── datasets/                 # 資料集（訓練/驗證/測試用的圖片）
│   ├── detect/               # Detection 用（YOLO txt 格式標註）
│   │   ├── train/
│   │   │   ├── images/       # 訓練圖片
│   │   │   └── labels/       # YOLO 格式標註（.txt）
│   │   ├── valid/            # 驗證集
│   │   └── test/             # 測試集
│   │
│   └── classify/             # Classification 用（資料夾即類別）
│       ├── train/
│       │   ├── score1/
│       │   ├── score2/
│       │   ├── score3/
│       │   ├── score4/
│       │   └── score5/
│       └── valid/
│           └── score1/ ~ score5/
│
├── scripts/                  # 工具腳本
│   ├── train_cls.py          # 分類訓練（簡易版，參數寫死）
│   ├── predict.py            # 兩階段推論（偵測+分類）
│   ├── prepare_data.py       # 把偵測標註轉成分類裁切圖
│   ├── balance.py            # 平衡各類別數量
│   ├── check_labels.py       # 驗證標註檔是否正確
│   └── test_color.py         # 用顏色演算法（非 ML）評分
│
├── train_ui/                 # 訓練 GUI 介面
│   ├── app.py                # 啟動入口
│   ├── main_window.py        # 主視窗（PyQt5）
│   ├── config_manager.py     # 設定存取
│   ├── runner.py             # 真正呼叫訓練的腳本
│   ├── training_worker.py    # 背景執行訓練的執行緒
│   └── excel_logger.py       # 訓練結果記錄到 Excel
│
├── runs/                     # 訓練輸出（YOLO 自動產生）
│   ├── detect/               # Detection 訓練結果
│   └── classify/             # Classification 訓練結果
│
├── egg_project/              # 另一組 Detection 訓練結果（舊版）
│
├── yolov8n-cls.pt            # 預訓練模型（分類，nano）
├── yolov8s-cls.pt            # 預訓練模型（分類，small）
└── yolov8s.pt                # 預訓練模型（偵測，small）
```

---

## 4. 環境設定

### 4.1 必要軟體

- **Anaconda** (Python 環境管理)
- **CUDA** (GPU 加速，建議 11.8 或 12.x)
- **PyCharm** (IDE，可選)

### 4.2 建立 Conda 環境

這個專案使用的 conda 環境名稱是 `20250831`。(可使用自己喜歡的名子)

```bash
# 建立環境（Python 3.10 建議）
conda create -n 20250831 python=3.10

# 啟動環境
conda activate 20250831
```

### 4.3 安裝套件

```bash
# PyTorch（依你的 CUDA 版本調整，以下是 CUDA 11.8 範例）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# YOLO 核心套件
pip install ultralytics

# GUI 相關
pip install PyQt5

# 其他工具
pip install pandas openpyxl pyyaml opencv-python tqdm
```

### 4.4 確認 GPU 可用

```python
import torch
print(torch.cuda.is_available())    # 應該輸出 True
print(torch.cuda.get_device_name()) # 顯示你的 GPU 名稱
```

---

## 5. 資料集說明（Roboflow）

**本專案的標注資料集統一存放在 Roboflow 上。**  
`datasets/` 資料夾不包含在這個 repo 中，請從以下連結下載。

### 可用資料集

| 資料集 | 用途 | 類別 | Roboflow 連結 |
|--------|------|------|---------------|
| comb_block v3 | Detection（找雞冠位置） | `comb`（1 類） | [下載連結](https://universe.roboflow.com/yu-chuan-liang/comb_block/dataset/3) |
| layercombv2 v1 | Detection（Score1 標籤） | `Score1`（1 類） | [下載連結](https://universe.roboflow.com/yu-chuan-liang/layercombv2/dataset/1) |

> **授權：** CC BY 4.0（可自由使用，需標明來源）

### 下載方式

**方法一：從瀏覽器下載**

1. 點上方連結進入 Roboflow 頁面
2. 點 **Download Dataset**
3. 選格式：**YOLOv8**（for Detection）
4. 下載後解壓縮，放到對應資料夾

**方法二：用 Roboflow Python 套件下載（需帳號）**

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")  # 登入 Roboflow 取得 API Key

# 下載 Detection 資料集
project = rf.workspace("yu-chuan-liang").project("comb_block")
version = project.version(3)
dataset = version.download("yolov8", location="datasets/detect")
```

### Classification 資料集的來源

Classification 用的裁切圖片（`datasets/classify/`）不在 Roboflow 上，而是由 Detection 資料集裁切而來：

```
Detection 資料集（Roboflow 下載）
    └─► scripts/prepare_data.py 裁切
        └─► datasets/classify/  （分類用）
```

請按照第 6 節的步驟從偵測標注自動產生分類資料集。

---

## 6. 資料集準備流程

### 6.1 Detection 資料集（偵測用）

偵測資料集使用 **YOLO 格式**，每張圖對應一個 `.txt` 標註檔。

**標註格式：**
```
<class_id> <x_center> <y_center> <width> <height>
```
- 所有數值都是相對比例（0~1），不是像素座標
- `class_id`：類別編號，這個專案只有 1 類（雞冠），所以都是 `0`

**範例：**
```
0 0.512 0.438 0.213 0.185
```
表示：類別 0，bbox 中心在 (51.2%, 43.8%)，寬 21.3%，高 18.5%

資料放在：
```
datasets/detect/
├── train/images/   <-- 訓練圖片（.jpg）
├── train/labels/   <-- 對應標註（.txt，同檔名）
├── valid/images/
├── valid/labels/
├── test/images/
└── test/labels/
```

### 6.2 從偵測標註製作分類資料集

如果你已有偵測標註，可以用 `scripts/prepare_data.py` 自動裁切出雞冠區域：

```bash
python scripts/prepare_data.py
```

這個腳本會：
1. 讀取 `datasets/detect/train/labels/` 的標註
2. 依 bbox 裁切圖片中的雞冠區域
3. 依類別（score1~score5）存到 `datasets/classify/train/`

**修改腳本中的路徑設定（第一次使用需確認）：**

```python
# 在 prepare_data.py 中找到這些變數
detect_img_dir = "datasets/detect/train/images"
detect_label_dir = "datasets/detect/train/labels"
output_dir = "datasets/classify/train"
class_map = {0: "score1", 1: "score2", ...}  # 依你的標註對應
```

### 6.3 平衡資料集

各等級圖片數量可能不均（例如 score1 有 500 張但 score4 只有 80 張），這會讓模型偏向多數類別。用 `balance.py` 自動平衡：

```bash
python scripts/balance.py
```

預設目標是每類 200 張：
- 超過的類別：隨機刪除多餘圖片
- 不足的類別：自動做輕量增量（水平翻轉、微小旋轉、亮度調整）

---

## 7. 訓練流程

### 方法一：使用 GUI（推薦新手）

```bash
conda activate <你的環境名稱>
cd /path/to/PythonProject3
python main.py
```

開啟後你會看到一個視窗，左側是參數設定，右側是訓練日誌。

**GUI 主要參數說明：**

| 欄位 | 說明 | 建議值 |
|------|------|--------|
| Task | 任務類型 | `classify` 或 `detect` |
| Data | 資料集路徑 | 選擇對應的 data.yaml |
| Model | 預訓練模型 | `yolov8n-cls.pt`（分類）/ `yolov8s.pt`（偵測） |
| Epochs | 訓練幾輪 | 50~100（分類），100~300（偵測） |
| Batch | 每批幾張圖 | 16（視 GPU 記憶體調整） |
| Imgsz | 輸入圖片大小 | 224（分類），640（偵測） |
| Patience | 幾輪沒進步就停止 | 10~30 |
| Project / Name | 輸出資料夾名稱 | 自訂，方便辨識 |

點 **Start Training** 後訓練開始，右側會即時顯示訓練 log。結束後自動記錄到 Excel。

### 方法二：直接執行訓練腳本

如果不想開 GUI，可以直接改 `scripts/train_cls.py` 裡的參數後執行：

```bash
conda activate 20250831
python scripts/train_cls.py
```

**train_cls.py 重要設定：**

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # 載入預訓練模型

model.train(
    data="datasets/classify",   # 資料集路徑
    epochs=50,                   # 訓練輪數
    batch=16,                    # batch size
    imgsz=224,                   # 輸入尺寸
    freeze=6,                    # 凍結前 N 層（遷移學習）
    dropout=0.5,                 # Dropout 比率（防過擬合）
    patience=5,                  # Early stopping
    project="comb_score_project",
    name="run_nano_freeze",
)
```

### 方法三：命令列（進階）

YOLO 本身也支援直接用命令列訓練：

```bash
# 分類訓練
yolo classify train data=datasets/classify model=yolov8n-cls.pt epochs=50 imgsz=224

# 偵測訓練
yolo detect train data=configs/0121data.yaml model=yolov8s.pt epochs=100 imgsz=640
```

### 7.1 訓練輸出在哪裡？

訓練完成後，結果會存在 `runs/` 資料夾：

```
runs/classify/comb_score_project/run_nano_freeze/
├── weights/
│   ├── best.pt     # 驗證集上表現最好的模型
│   └── last.pt     # 最後一個 epoch 的模型
├── results.csv     # 每個 epoch 的 loss、accuracy 數據
├── results.png     # 訓練曲線圖
└── args.yaml       # 這次訓練用的參數（方便日後重現）
```

**使用 `best.pt`** 進行推論，不要用 `last.pt`（除非你知道自己在做什麼）。

---

## 8. 推論（測試模型效果）

### 方法一：顏色演算法推論（`scripts/test_color.py`）

不需要分類模型，直接用 HSV 顏色比對判斷等級：

```bash
python scripts/test_color.py
```

**修改腳本裡的設定：**

```python
# 偵測模型路徑（改成你訓練好的 best.pt）
detect_model = YOLO("runs/detect/egg_project/run_test_block/weights/best.pt")

# 輸入影片
source = "test_video.mp4"  # 換成你的影片檔
```

輸出會在視窗中顯示，每個偵測到的雞冠依平均顏色判斷等級並標注。

**原理：** 根據雞冠紅色深淺（HSV 色彩空間）與標準色比對歐式距離，不需額外訓練分類模型，可作為 ML 模型的比較基準。

### 方法二：YOLO 兩階段推論

如需用 YOLO 分類模型推論，可參考 `scripts/train_cls.py` 的模型載入方式，自行撰寫推論腳本，或直接使用 Comb_DetectionV2 專案的 GUI（已整合完整偵測+分類流程）。

---

## 9. 腳本功能說明

| 腳本 | 功能 | 何時使用 |
|------|------|----------|
| `main.py` | 啟動訓練 GUI | 每次訓練（推薦入口） |
| `scripts/train_cls.py` | 簡易分類訓練腳本（參數寫在程式裡） | 快速測試單次訓練 |
| `scripts/prepare_data.py` | 偵測標注轉分類裁切圖 | 首次準備分類資料集 |
| `scripts/balance.py` | 平衡各類別數量（欠缺則增強，過多則刪除） | 資料不均時 |
| `scripts/check_labels.py` | 檢查標注是否有誤（class_id 是否全為 0） | 訓練前驗證資料 |
| `scripts/test_color.py` | 顏色演算法推論（HSV 色彩比對，非 ML） | 對照基準測試 |

---

## 10. 設定檔說明

### data.yaml（Detection 資料集設定）

```yaml
# path 為相對路徑基準（相對於 configs/ 資料夾位置）
path: ../datasets/detect
train: train/images     # 訓練圖片
val:   valid/images     # 驗證圖片
test:  test/images      # 測試圖片
nc: 1                   # 類別數量
names: ['comb']         # 類別名稱
```

路徑已改為相對路徑，換電腦不需修改（只要資料集放在 `datasets/detect/` 下即可）。

### last_config.json（GUI 上次設定）

GUI 會自動存上次的設定到 `train_ui/last_config.json`，下次開啟會自動載入。如果想重置，刪掉這個檔案即可。

---

## 11. 常見問題

### Q: `CUDA is not available` 怎麼辦？

1. 確認已安裝支援 CUDA 的 PyTorch（`pip install torch --index-url https://download.pytorch.org/whl/cu118`）
2. 確認驅動程式版本夠新（`nvidia-smi` 檢查）
3. 如果沒有 GPU，在訓練參數中設 `device=cpu`（速度會很慢）

### Q: 訓練 loss 一直不降？

- **batch size 太小**：試試增加到 32
- **學習率太高/太低**：預設 `lr0=0.01`，可以試 `0.001`
- **資料集有問題**：先跑 `check_labels.py` 確認標註正確
- **Epoch 太少**：分類至少 50，偵測至少 100

### Q: 資料不夠怎麼辦？

1. 先用 `balance.py` 做輕量資料增量
2. 在訓練參數中開啟 YOLO 內建增量（`hsv_h`, `fliplr`, `mosaic` 等）
3. 到 Roboflow 上標注更多圖片，再重新下載（參考第 5 節）

### Q: 訓練很久才完成，可以提早停止嗎？

在 GUI 中按 **Stop**，或設定 `patience` 讓 Early Stopping 自動處理。

### Q: `best.pt` 和 `last.pt` 差在哪？

- `best.pt`：訓練過程中，在驗證集表現**最好的那個 epoch** 存的模型
- `last.pt`：最後一個 epoch 的模型

**推論時請用 `best.pt`。**

### Q: 訓練結果要怎麼比較不同實驗？

每次訓練結束後，GUI 會自動把結果記錄到 `train_log.xlsx`，打開 Excel 就能看到每次實驗的 loss、accuracy 等指標。

---

## 12. YOLO 基礎觀念快速入門

### 12.1 什麼是 YOLO？

YOLO（You Only Look Once）是一種即時物件偵測（Object Detection）的神經網路架構，特點是**快速且準確**。這個專案使用 Ultralytics 開發的 **YOLOv8 / YOLO11**。

### 12.2 Detection vs Classification

| 任務 | 說明 | 輸出 |
|------|------|------|
| **Detection** | 找出物體在哪、框出位置 | Bounding Box + 類別 + 信心值 |
| **Classification** | 判斷圖片屬於哪個類別 | 類別 + 信心值 |
| **Segmentation** | 找出物體輪廓（像素級別） | Mask |

這個專案用 Detection 找到雞冠位置，再用 Classification 判斷等級。

### 12.3 模型大小選擇

YOLO 依模型大小分為幾個版本（以 YOLOv8 為例）：

| 模型 | 速度 | 精度 | 適合 |
|------|------|------|------|
| yolov8n | 最快 | 最低 | 測試、資源有限 |
| yolov8s | 快 | 中低 | 一般用途（本專案） |
| yolov8m | 中 | 中 | 平衡 |
| yolov8l | 慢 | 高 | 高精度需求 |
| yolov8x | 最慢 | 最高 | 研究、高性能 GPU |

**分類**用 `n`（nano）就夠，**偵測**這個專案用 `s`（small）。

### 12.4 遷移學習（Transfer Learning）

不要從頭訓練，要從**預訓練的模型**開始。預訓練模型已經學會辨識很多特徵（邊緣、顏色、形狀），遷移學習就是把這些知識應用到新任務上。

```python
model = YOLO("yolov8n-cls.pt")  # 載入預訓練的 ImageNet 模型
model.train(data="...", freeze=6, ...)  # 凍結前幾層，只訓練後段
```

`freeze=6` 代表凍結前 6 層不更新，這樣訓練更快、不容易過擬合。

### 12.5 主要超參數

| 參數 | 說明 | 影響 |
|------|------|------|
| `epochs` | 訓練幾個完整輪次 | 太少欠擬合，太多過擬合 |
| `batch` | 每次更新用幾張圖 | 越大越穩定，但需要更多 GPU 記憶體 |
| `lr0` | 初始學習率 | 太大不收斂，太小訓練慢 |
| `imgsz` | 輸入圖片大小 | 越大越精確，但更慢更耗記憶體 |
| `patience` | 幾輪沒改善就提早停止 | 避免浪費訓練時間 |
| `freeze` | 凍結前 N 層 | 遷移學習加速 |
| `dropout` | 隨機丟棄神經元比率 | 防止過擬合 |

### 12.6 判斷訓練好不好

看 `results.csv` 或 `results.png` 的曲線：

- **Loss（損失）** 應該要**持續下降**並趨於平穩
- **Accuracy（準確率）** 應該要**持續上升**並趨於平穩
- 如果訓練 loss 繼續降但驗證 loss 開始上升 → **過擬合（Overfitting）**，需要更多資料或增加 dropout

---

## 快速上手流程總結

```
1. 安裝環境
   conda activate <你的環境名稱>
   pip install -r requirements.txt

2. 下載資料集（從 Roboflow，見第 5 節）
   → Detection 資料放到  datasets/detect/
   → 執行 scripts/prepare_data.py 產生分類用裁切圖
   → 執行 scripts/balance.py 平衡各類別數量

3. 確認資料集結構
   datasets/classify/train/score1/ ~ score5/
   datasets/classify/valid/score1/ ~ score5/

4. 開始訓練
   python main.py
   → 選擇模型、資料、設定參數、按 Start

5. 訓練完成後測試
   python scripts/test_color.py
   → 修改腳本中的模型路徑與影片路徑後執行

6. 查看結果
   runs/classify/{project}/{name}/weights/best.pt
   train_log.xlsx（每次訓練自動記錄）
```

---

*如有問題請詢問學長姐，或參考 [Ultralytics 官方文件](https://docs.ultralytics.com/)。*
