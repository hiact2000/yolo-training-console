# yolo-training-console

**快速學會如何訓練 YOLO model**，並知道訓練完成後要去哪裡找模型結果。

By 蘇柏瑋 Update 2026.04.23

---

## 1. 目前專案結構

根據目前 repo 結構，主要資料夾如下：

```bash
yolo-training-console/
├─ configs/          # 設定檔，例如 data.yaml
├─ scripts/          # 訓練或資料處理腳本
├─ train_ui/         # GUI 介面相關程式
├─ main.py           # 專案入口
├─ requirements.txt  # 套件需求
├─ README.md         # 專案說明
└─ .gitignore        # Git 忽略規則
```

如果你是第一次接手，先記住三個重點：

* `main.py`：啟動程式
* `configs/`：放資料集設定檔
* `scripts/`：放訓練或資料處理腳本

---

## 2. 這個專案在做什麼？

這個專案是 YOLO 訓練控制台，重點是幫助使用者：

1. 準備資料集
2. 設定訓練參數
3. 執行 YOLO 訓練
4. 找到訓練好的模型 `best.pt`

簡單來說，這是一個給學弟妹學習 **train model** 的專案。

---

## 3. 需要先安裝什麼？

建議先安裝：

* Python 3.10
* Anaconda
* PyCharm
* Git
* PyTorch

---

## 4. 建立環境

### Step 1：建立 conda 虛擬環境 (請在Terminal中操作，並確認是否進入venv)

```bash
conda create -n yolo_train python=3.10
conda activate yolo_train
```

### Step 2：安裝 PyTorch

如果是 CUDA 11.8，可先用 (你的電腦要有顯卡才能用GPU!!!!!!!)：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

如果只是先測試，也可以先安裝 CPU 版：

```bash
pip install torch torchvision
```

### Step 3：安裝其他套件

```bash
pip install -r requirements.txt
```

---

## 5. 下載專案

```bash
git clone <repo-url>
cd yolo-training-console
```

---

## 6. 如何啟動專案？

```bash
python main.py
```

如果 GUI 能正常打開，就代表環境大致沒問題。

---

## 7. 資料集要去哪裡下載？

本專案的資料集主要放在 **Roboflow**。

可用資料集：

- comb_block v3（Detection / comb）  
  https://universe.roboflow.com/yu-chuan-liang/comb_block/dataset/3

- layercombv2 v1（Detection / Score1）  
  https://universe.roboflow.com/yu-chuan-liang/layercombv2/dataset/1

下載方式：

1. 進入資料集頁面
2. 點選 **Download Dataset**
3. 選擇 **YOLOv8**
4. 解壓縮後放到專案資料夾
5. 確認 `data.yaml` 路徑正確

---

## 8. Train model 的最基本流程

### Step 1：準備資料集

你需要先有：

* 圖片
* 標註
* `data.yaml`

通常設定檔會放在：

```bash
configs/
```

例如(可以去學習何為相對路徑)：

```yaml
path: ./datasets
train: train/images
val: valid/images
test: test/images
nc: 1
names: ['comb']
```

---

### Step 2：執行訓練 (執行前請確認資料夾擺放的位置是否正確，以及指向.yaml的名稱，以及是否已進入venv)

範例指令：

```bash
yolo detect train model=yolov8s.pt data=configs/0121data.yaml imgsz=640 epochs=300 batch=16 patience=30 project=egg_project name=run_test_block
```

這些參數的意思：

* `model=yolov8s.pt`：初始模型
* `data=configs/0121data.yaml`：資料集設定檔 (務必確認路徑與名稱)
* `imgsz=640`：輸入圖片大小
* `epochs=300`：最多訓練 300 輪
* `batch=16`：每批 16 張
* `patience=30`：太久沒進步就提早停止
* `project=egg_project`：輸出資料夾
* `name=run_test_block`：這次實驗名稱

---

## 9. 訓練完後要看哪裡？

訓練完成後，通常會產生：

```bash
egg_project/run_test_block/
```

或：

```bash
runs/detect/
```

最重要的是：

```bash
weights/best.pt
```

這就是你訓練好的模型。

一般來說：

* `best.pt`：最重要，正式推論通常用這個
* `last.pt`：最後一輪的模型，不一定最好

---

## 10. 學弟妹最短上手流程

如果你是第一次接觸coding，請照這個順序：

### 第一步：建環境

```bash
conda create -n yolo_train python=3.10
conda activate yolo_train
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 第二步：下載專案

```bash
git clone <repo-url>
cd yolo-training-console
```

### 第三步：開啟程式

```bash
python main.py
```

### 第四步：執行 train 指令

```bash
yolo detect train model=yolov8s.pt data=configs/0121data.yaml imgsz=640 epochs=300 batch=16 patience=30 project=egg_project name=run_test_block
```

### 第五步：找到訓練好的模型

```bash
weights/best.pt
```

---

## 11. 常見問題

### Q1. `python main.py` 跑不起來

請先確認：

* 有沒有進入正確的 conda 環境
* 有沒有安裝 `requirements.txt`
* Python 版本是否正確

---

### Q2. 找不到 `best.pt`

請先確認訓練有沒有真的跑完。

一般來說，訓練成功後才會出現：

```bash
weights/best.pt
```

---

### Q3. `best.pt` 和 `last.pt` 差在哪？

* `best.pt`：驗證表現最好的模型
* `last.pt`：最後一輪的模型

一般優先使用 `best.pt`。

---



