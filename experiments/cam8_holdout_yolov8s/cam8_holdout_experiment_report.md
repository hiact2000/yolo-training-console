# Cam8 Hold-out Testing — YOLOv8s 雞冠偵測跨監視器泛化實驗

**日期**：2026-08-12 　**模型**：YOLOv8s 　**類別**：單一類別 `comb`（nc=1）
**分支**：`feature/cam8-holdout-yolov8s-experiment`
**產出根目錄**：`experiments/cam8_holdout_yolov8s/`

> **範圍聲明**：本實驗只做「雞冠位置偵測」。資料集只有一個類別 `comb`，**沒有 normal / abnormal 之分**，因此本報告的任何數字都**不能**被解讀為異常雞冠比例、OACP、疾病盛行率或臨床診斷。這裡量測的只有一件事：模型能不能在沒看過的監視器視角上把雞冠框出來。

---

## 1. 實驗目的

前一次的三組實驗（原始資料集 / 純 CCTV / 原始＋CCTV）都是用**同分布**的 validation 來評估，因此無法回答「換一台監視器還能不能用」。本次改成嚴格的 hold-out camera 設計：**Cam8 完全不參與訓練與驗證，只當測試集**，用來量測跨 camera view 的泛化能力。

具體要回答：

1. 只用原始 dataset 訓練，能否在 Cam8 上維持偵測效果？
2. 只用 Cam9/Cam24 的 CCTV 影像訓練，能否泛化到沒看過的 Cam8？
3. 原始 dataset ＋ Cam9/Cam24 混合訓練，能否提升 Cam8 表現？
4. Cam8 當獨立測試集時，三種訓練策略的 precision / recall / mAP 差異有多大？

## 2. 為什麼用 Cam8 當 hold-out test set

- **Cam8 是一個獨立的實體視角**，不是隨機切出來的一批影像。把它整台保留，測到的才是「換視角」的損失，而不是「同一支影片換幾張影格」的損失。
- 隨機切分在這批資料上會**嚴重高估**表現：CCTV 影像是同一支連續影片抽格，相鄰影格幾乎相同；原始 dataset 又有 Roboflow 三重複製（見 §6）。以 camera 為單位切開，是唯一能避開這兩種污染的切法。
- Cam8 的畫面條件與訓練資料明顯不同（長走道俯視、雞冠隔著籠具鐵絲網、右側大面積逆光過曝、目標尺寸小），構成一個誠實的壓力測試。

## 3. 三組實驗設計

| 實驗 | 訓練資料 | Validation | Testing |
|---|---|---|---|
| **Exp A** | 原始 dataset（已移除全部 Cam8） | 原始 dataset 內部 8:2 | Cam8 only |
| **Exp B** | CCTV Cam9 + Cam24 | Cam9+Cam24 內部 8:2 | Cam8 only |
| **Exp C** | 原始 dataset + CCTV Cam9 + Cam24 | 混合資料內部 8:2 | Cam8 only |

三組使用**完全相同的一組 Cam8 測試集**（同一份 `lists/cam8_test.txt`，103 張）。Exp C 的切分刻意設計成 **Exp A 與 Exp B 切分的聯集**，因此三組之間唯一的變因是訓練資料池，切分方式本身不構成差異。

## 4. 資料來源與數量

**來源一：原始 dataset** — `datasets/detect/{train,valid,test}`（Roboflow `comb_block` v3，對應 `configs/0121data.yaml`）
共 547 個影像檔。組成：手機拍攝 `P_2025*` 479、`video_frame_*` 34、`S__*` 8、`-N_JPG` 8、**CAM08 10 張**、**CAM09 9 張**（後兩者為 2026-03-03 拍攝）。

> ⚠️ **重要發現**：所謂的「原始 dataset」裡其實混了 10 張 Cam8 影像。若照前次做法直接使用，Cam8 就會進入 Exp A / Exp C 的訓練集，hold-out 設計當場失效。本實驗已在資料來源層把這 10 張全部移出訓練與驗證。

**來源二：CCTV 0715data** — `datasets/detect/0715data/{train,valid}`（對應 `configs/0715data.yaml`）
cam8 93、cam9 86、cam24 101，共 280 張，全部抽自 2026-06-12 14:52 的連續影片。

**類別**：全部資料只有 class 0 = `comb`，共 826 個 label 檔、4295 個標註框、1 個空 label 檔。

**排除**：`datasets/detect/train/images/20260303120000_CAM09_fixed_t0010-00s_00001_*.jpg` 沒有對應 label 檔，全實驗排除。

### 實際切分結果

| 實驗 | Split | 影像數 | 標註框數 | 不重複原圖組數 |
|---|---|---:|---:|---:|
| Exp A | train | 436 | 1642 | 200 |
| Exp A | val | 100 | 350 | 50 |
| Exp B | train | 150 | 1208 | 150 |
| Exp B | val | 37 | 289 | 37 |
| Exp C | train | 586 | 2850 | 350 |
| Exp C | val | 137 | 639 | 87 |
| **全部三組** | **test (Cam8)** | **103** | **806** | **103** |

Cam8 測試集組成：0715data cam8 93 張（2026-06-12）＋ 原始 dataset 內的 CAM08 10 張（2026-03-03），共 103 張，涵蓋兩個拍攝時段。

> **標註數注意**：Cam8 測試集的 806 列標註中，`cam8__20260612145200_CAM08_00001.jpg` 有 **8 列完全重複**。Ultralytics 載入時會自動去重，因此評估時實際使用的是 **798 個框**。其餘所有 split 皆無重複列。

## 5. Split 規則

1. Train : Val = 8 : 2，固定 `seed=42`。
2. **以「原圖」為單位分組切分（GroupShuffle）**，不是以檔案為單位。同一張原圖的所有 Roboflow 副本強制落在同一側。
3. 依來源子類型分層（`original_phone` / `original_videoframe` / `original_misc` / `original_cam9` / `cctv_cam9` / `cctv_cam24`），維持各層的 8:2 比例。
4. Cam8 在來源層即被抽離，不進入任何 train/val 池。
5. Cam9/Cam24 與原始 dataset 可進入 train/val。
6. 完整 manifest 記錄每張影像的 experiment / split / 來源 / camera / 分組鍵 / 標註數 / MD5 / leakage 標記。

> **8:2 是在「原圖分組」層級成立的**。因為各組大小不同（Roboflow 副本 1 份或 3 份），換算到影像層級的 val 比例是 Exp A 18.7%、Exp B 19.8%、Exp C 18.9%，略低於 20%。這是分組切分的必然結果，不是設定錯誤。

切分由 `scripts/build_cam8_holdout.py` 產生，可重現：

```bash
python scripts/build_cam8_holdout.py --seed 42
```

## 6. 防止 data leakage 的檢查

腳本內建三個**硬性 assertion**，任一失敗即中止，三組全部通過：

| 檢查 | 結果 |
|---|---|
| train/val 中是否有 Cam8 影像 | **0 張**（三組皆是） |
| train/val 是否有與 Cam8 test **byte 完全相同**的檔案（MD5 比對） | **0 筆** |
| 是否有原圖分組跨越 train/val | **0 筆** |

另外標記（只標記、不刪除）三類風險，記錄在 manifest 的 `leakage_risk` 欄：

| 標記 | 筆數 | 說明 |
|---|---:|---|
| `duplicate_content_across_splits` | **0** | 沒有任何 split 之間有完全相同的影像 |
| `roboflow_multicopy_grouped_ok` | 430 | 原始 dataset 的 547 個檔案其實只有 **261 張不重複原圖**，144 張原圖各被複製成 3 份。分組切分已化解，未跨界 |
| `adjacent_video_frame_other_split` | 93 | CCTV 連續影格：train 56 筆、val 37 筆 |

> 🔴 **必須誠實揭露的一點**：`adjacent_video_frame_other_split` 顯示 **Exp B 的 37 張 validation 影像全部**在 train 中有相鄰影格。CCTV 抽格是同一支連續影片，相鄰影格幾乎相同，因此 **Exp B 與 Exp C 的 validation 分數偏樂觀**，不應被當成獨立評估。
>
> **這不影響本報告的主結論**：Cam8 是整台保留的獨立 camera，與 train/val 無任何影像重疊、無任何 byte 相同檔案、亦非同一支影片。所有 Cam8 testing 數字都是乾淨的。
>
> 順帶一提，Roboflow 原本的 train/valid 切分本身就有 1 張原圖同時出現在 train 與 valid——這也是本次改用分組切分的直接原因。

## 7. 訓練設定

先檢查前次三組實驗的實際 `args.yaml`（`runs/detect/exp`、`runs/detect/0715data_yolov8s_ui`、`runs/detect/detect_0715data_merged_yolov8s_ui`），確認真正使用的參數後**完整複製**：

```text
model      = yolov8s.pt        epochs   = 300      patience = 30
batch      = 16                imgsz    = 640      seed     = 0
workers    = 0                 optimizer= auto     lr0      = 0.01 / lrf = 0.01
momentum   = 0.937             weight_decay = 0.0005          warmup_epochs = 3.0
box/cls/dfl= 7.5 / 0.5 / 1.5   deterministic = true           pretrained = true
```

Augmentation（**沿用前次的 Ultralytics 預設，全部開啟**）：

```text
mosaic = 1.0    close_mosaic = 10    fliplr = 0.5    flipud = 0.0
hsv_h/s/v = 0.015 / 0.7 / 0.4        translate = 0.1   scale = 0.5
degrees = shear = perspective = 0.0  mixup = copy_paste = 0.0   erasing = 0.4
```

> ⚠️ **與原始需求的差異，必須標明**：本次任務描述中寫的是「No Augmentation, No Mosaic」，但前次三組實驗的 `args.yaml` 顯示實際使用的是 **Ultralytics 預設 augmentation 全開（mosaic=1.0）**。兩者矛盾。經確認後採「完整複製前次設定」，理由是這樣 Cam8 的結果才能與前次三組實驗直接對照，唯一變因是資料切分。**因此本報告不得描述為 "no augmentation" 實驗。** 若要做真正的 no-aug 對照，需另跑一組。

環境：NVIDIA RTX 5060 Ti、torch 2.8.0+cu129、ultralytics 8.3.189。

### 執行結果

| 實驗 | Run name | 完成 epoch | 最佳 epoch | 訓練時間 |
|---|---|---:|---:|---:|
| Exp A | `expA_original_yolov8s_cam8test` | 178（早停） | 148 | 23.5 min |
| Exp B | `expB_cctv_cam9_24_yolov8s_cam8test` | 87（早停） | 57 | 4.7 min |
| Exp C | `expC_original_cctv_cam9_24_yolov8s_cam8test` | 129（早停） | 99 | 24.5 min |

三組皆因 patience=30 早停，未達 300 epoch 上限。

執行指令：

```bash
python scripts/run_cam8_holdout.py          # 訓練 + val 評估 + Cam8 test 評估
python scripts/analyze_cam8_errors.py       # FP/FN 分類 + 標註圖
python scripts/make_cam8_figures.py         # 比較圖表 + 指標表
python scripts/render_cam8_report.py        # Markdown -> HTML
```

## 8. 三組模型的 validation 結果

| 實驗 | Val 影像 | Val 標註 | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp A · 原始 only | 100 | 350 | 0.9548 | 0.8686 | 0.9096 | 0.9440 | 0.7074 |
| Exp B · CCTV Cam9+24 | 37 | 289 | 0.7717 | 0.7251 | 0.7477 | 0.7944 | 0.4120 |
| Exp C · 原始+CCTV | 137 | 639 | 0.8723 | 0.8337 | 0.8525 | 0.8980 | 0.6005 |

**這三個數字不可互相比較，也不是本實驗的結論**：三組的 validation 集內容完全不同（A 幾乎全是近距離手機照，B 全是 CCTV 遠景，C 是兩者混合），難度不同。而且如 §6 所述，B 與 C 的 val 受相鄰影格污染而偏高。列在這裡只是為了對照 §9 的落差。

## 9. 三組模型的 Cam8 testing 結果 —— 本報告的主結果

**同一組 Cam8 測試集，103 張影像、798 個有效標註框。**

| 實驗 | Precision | Recall | F1 | mAP50 | mAP50-95 | 推論速度 |
|---|---:|---:|---:|---:|---:|---:|
| Exp A · 原始 only | 0.6690 | 0.6165 | 0.6417 | 0.6558 | 0.3087 | 2.44 ms/img |
| Exp B · CCTV Cam9+24 | 0.7920 | **0.7707** | 0.7812 | **0.8373** | 0.4100 | 3.27 ms/img |
| Exp C · 原始+CCTV | **0.8243** | 0.7431 | **0.7816** | 0.8319 | **0.4460** | 2.51 ms/img |

Ultralytics 內建 confusion matrix（conf 0.25）：

| 實驗 | TP | FP | FN |
|---|---:|---:|---:|
| Exp A | 301 | 59 | 497 |
| Exp B | 442 | 48 | 356 |
| Exp C | **586** | 101 | **212** |

![Cam8 hold-out test — precision, recall, F1](figures/cam8_test_precision_recall_comparison.png)

![Cam8 hold-out test — mAP50 and mAP50-95](figures/cam8_test_map_comparison.png)

### Validation → Cam8 的泛化落差

| 實驗 | val mAP50 | Cam8 mAP50 | Δ mAP50 | val Recall | Cam8 Recall | Δ Recall |
|---|---:|---:|---:|---:|---:|---:|
| Exp A · 原始 only | 0.9440 | 0.6558 | **−0.2882** | 0.8686 | 0.6165 | −0.2521 |
| Exp B · CCTV Cam9+24 | 0.7944 | 0.8373 | **+0.0429** | 0.7251 | 0.7707 | +0.0456 |
| Exp C · 原始+CCTV | 0.8980 | 0.8319 | −0.0661 | 0.8337 | 0.7431 | −0.0906 |

![Validation vs Cam8 hold-out — the cross-camera drop](figures/cam8_val_vs_test_gap.png)

這張圖是整個實驗最重要的一張：**Exp A 的 validation 分數最高（0.944），Cam8 分數卻最低（0.656）**。如果只看 validation，會得到完全相反的模型排名。

### Confusion matrix

![Exp A confusion matrix](figures/expA_confusion_matrix.png)

![Exp B confusion matrix](figures/expB_confusion_matrix.png)

![Exp C confusion matrix](figures/expC_confusion_matrix.png)

## 10. 指標比較表（含固定操作點）

上表的 Precision / Recall 是 Ultralytics 在**各模型自己的最佳 F1 信心值**取的，三組的操作點不同。為了在同一個操作點上比較，另外以 **conf = 0.25、match IoU = 0.5** 重新配對：

| 實驗 | TP | FP | FN | 該操作點 Precision | 該操作點 Recall |
|---|---:|---:|---:|---:|---:|
| Exp A | 293 | 60 | 505 | 0.830 | 0.367 |
| Exp B | 445 | 53 | 353 | 0.894 | 0.558 |
| Exp C | **591** | 117 | **207** | 0.835 | **0.740** |

**兩種算法沒有互相矛盾，只是操作點不同。** 在固定 conf=0.25 這個實務上會採用的門檻下，Exp C 抓到的雞冠數量（591）遠多於 Exp B（445）與 Exp A（293）；Exp B 之所以在 Ultralytics 的 recall 欄勝出，是因為它的最佳 F1 點落在更低的信心值。做投影片或實際部署時應以固定門檻的這組數字為準。

## 11. 圖表

| 檔案 | 內容 |
|---|---|
| `figures/cam8_test_precision_recall_comparison.png` | 三組在 Cam8 的 P / R / F1 |
| `figures/cam8_test_map_comparison.png` | 三組在 Cam8 的 mAP50 / mAP50-95 |
| `figures/cam8_val_vs_test_gap.png` | validation 與 Cam8 的落差（核心圖） |
| `figures/cam8_error_breakdown.png` | Cam8 錯誤案例分類 |
| `figures/exp{A,B,C}_confusion_matrix.png` | 各組 Cam8 confusion matrix |
| `figures/exp{A,B,C}_confusion_matrix_normalized.png` | 正規化版本 |

## 12. Cam8 錯誤案例

分類標準為**影像可量測的代理指標，不是已驗證的成因**——用途是指引人工檢查方向。小目標門檻取 Cam8 標註面積的 25 百分位（937 px²），低光門檻取框內 HSV-V 平均的 25 百分位（138）。

![Cam8 error cases by category](figures/cam8_error_breakdown.png)

| 類別 | Exp A | Exp B | Exp C |
|---|---:|---:|---:|
| FN 小雞冠 / 遠距離 | 164 | 143 | **90** |
| FN 光照不足 | 62 | 55 | **36** |
| FN 遮蔽 / 重疊 | 2 | 2 | **0** |
| FN 邊緣截斷 | 8 | 2 | 3 |
| FN 其他 | 269 | 151 | **78** |
| FP 背景誤判 | 37 | 37 | 75 |
| FP 定位偏移 | 23 | 16 | 42 |

觀察：

- **漏檢（FN）是主要問題，不是誤報**。三組的 FN 都遠多於 FP。
- **小目標與遠距離是最大的可歸因失敗類型**。Cam8 是長走道俯視，畫面深處的雞冠只有幾十個像素。Exp C 把這類漏檢從 164 降到 90（−45%）。
- **Exp C 的 FP 反而變多**（背景誤判 37→75、定位偏移 16→42）。它偵測得更積極，用一部分 precision 換到大量 recall。這是可接受的取捨，但若要做比例估計，FP 上升會直接影響分母。
- **遮蔽在本次分類下幾乎不出現**（0–2 筆），但這是分類定義造成的：判定條件是「與另一個標註框 IoU > 0.1」，而 Cam8 真正的遮蔽來自**籠具鐵絲網**，網格不是標註物件，因此被歸入「其他」。這是本次錯誤分類法的已知盲點。
- 「其他」類別在 Exp A 高達 269 筆，代表原始 dataset 訓練出的模型在 Cam8 上是**全面性失效**，而非集中在某個可辨識的困難情境。

人工檢查用的標註圖（綠 = TP、紅 = FP、橘 = FN，檔名前綴 `errNNN_` 為該張的錯誤數，數字越大越該先看）：

```text
experiments/cam8_holdout_yolov8s/predictions/expA/   40 張
experiments/cam8_holdout_yolov8s/predictions/expB/   40 張
experiments/cam8_holdout_yolov8s/predictions/expC/   40 張
```

逐框明細：`results/exp{A,B,C}_cam8_error_cases.csv`（含 error_type、category、confidence、與 GT 的最佳 IoU、框面積、框內亮度、座標）。

## 13. 初步解讀 —— 直接回答七個問題

**Q1. 哪一組在 Cam8 testing 上 precision 最高？**
**Exp C（原始＋CCTV），0.8243。** 依序為 Exp B 0.7920、Exp A 0.6690。

**Q2. 哪一組 recall 最高？**
以 Ultralytics 最佳 F1 操作點：**Exp B，0.7707**（C 0.7431、A 0.6165）。
但在固定 conf=0.25 的實務操作點：**Exp C 明顯最高，0.740**（B 0.558、A 0.367）。兩個答案都要講，否則會誤導。

**Q3. 哪一組 mAP50 最高？**
**Exp B，0.8373**；Exp C 0.8319。**差距只有 0.005，實質上是平手**，單一 seed 的實驗不足以宣稱誰勝出。Exp A 0.6558 則明顯落後。
若看更嚴格的 mAP50-95，**Exp C 最高（0.4460）**，領先 Exp B 0.036——代表 C 的框位置更準。

**Q4. 混合資料集是否真的提升了 Cam8 表現？**
**部分提升，不是全面提升。** 相對於 Exp B：

- 提升：precision +0.032、mAP50-95 +0.036、固定門檻下 TP 從 445 增至 591、FN 從 353 降至 207。
- 未提升：mAP50 −0.005（平手）、Ultralytics recall −0.028。
- 代價：FP 從 53 增至 117。

**相對於 Exp A 則是全面且大幅提升**（mAP50 +0.176、mAP50-95 +0.137）。結論：加入 CCTV 資料是關鍵；在已有 CCTV 資料之後，再加原始手機照帶來的是「定位更準、抓得更多、誤報也更多」，而非 mAP50 的提升。

**Q5. 只用 Cam9/Cam24 訓練能否泛化到 Cam8？**
**能，而且相當好。** Exp B 在 Cam8 上 mAP50 = 0.8373，**甚至高於它自己的 validation（0.7944，+0.043）**。這說明同屬 CCTV 的視角之間存在可轉移的共同特徵（相似的安裝高度、俯視角度、影像品質、雞隻密度）。
（其 val 分數偏低有兩個原因：val 只有 37 張，且 Cam9/Cam24 的畫面比 Cam8 更擁擠；這也提醒不要用這麼小的 val 來下判斷。）

**Q6. 原始 dataset 對 Cam8 是否不足？**
**明顯不足。** Exp A 的 validation mAP50 高達 0.9440，Cam8 卻只有 0.6558，**落差 −0.2882**；recall 從 0.8686 掉到 0.6165。固定門檻下它漏掉 798 個框中的 505 個。原始 dataset 以近距離手機照為主，與 Cam8 的遠距離、鐵絲網遮蔽、逆光條件差距太大。
**這正是本實驗最重要的發現：validation 分數最高的模型，在真實新視角上表現最差。**

**Q7. 是否需要更多 camera view 或更嚴謹的 cross-camera validation？**
**需要，而且是本實驗最明確的建議。** 目前只有三台 CCTV，hold-out 只做了一台，等於 n=1 的泛化估計，無法區分「Exp B/C 真的泛化好」與「Cam8 剛好接近 Cam9/Cam24」。應改做 leave-one-camera-out 輪流測試。

## 14. 方法限制

1. **單一 seed、單次訓練**。三組各跑一次，沒有重複實驗，因此 mAP50 上 0.005 這種差距完全落在雜訊範圍內，不應解讀為排名。
2. **只 hold out 一台 camera**。n=1 的跨視角估計，結論不能外推到任意新監視器。
3. **Exp B / Exp C 的 validation 受相鄰影格污染**（§6），其 val 數字偏樂觀。Cam8 testing 不受影響。
4. **三組 validation 集內容不同**，彼此不可比較，只能各自與自己的 Cam8 分數比落差。
5. **Augmentation 全開**，與任務描述中的「No Augmentation」不符（已於 §7 說明並取得確認）。沒有 no-aug 對照組，因此無法判斷 augmentation 對跨視角泛化的貢獻。
6. **錯誤分類是影像可量測的代理指標**，非驗證過的因果歸因；且鐵絲網遮蔽無法被目前的遮蔽判準捕捉，被歸入「其他」。
7. **Cam8 測試集混合兩個拍攝時段**（2026-03-03 的 10 張與 2026-06-12 的 93 張），未分開報告；兩段的難度可能不同。
8. **測試集內部有相鄰影格**（93 張抽自同一支連續影片），因此 103 張的有效獨立樣本數低於 103，信賴區間比表面上寬。
9. **單一類別偵測**。本實驗完全沒有涉及雞冠正常／異常判別，不能延伸到任何健康或疾病結論。
10. **Ultralytics 會在原始資料夾寫入 `labels.cache`**。內容以檔案清單雜湊驗證，不影響正確性，但確實在資料夾產生了新檔案。原始影像與標註本身未被修改。

## 15. 下一步建議

**優先（直接回應本實驗暴露的問題）**

1. **Leave-one-camera-out 交叉驗證**：依序 hold out Cam8、Cam9、Cam24，得到三個跨視角估計與變異範圍，取代目前 n=1 的結論。
2. **多 seed 重複**：每組至少跑 3 個 seed，回報平均 ± 標準差。目前 B 與 C 的 0.005 差距沒有意義。
3. **蒐集更多 camera view**：三台是泛化研究的下限。新增視角時優先挑選安裝高度、角度、光照條件與現有三台不同者。

**次要（改善絕對表現）**

4. **針對小目標調整**：Cam8 最大的可歸因失敗是小/遠雞冠。可嘗試 `imgsz=960/1280`、或對走道深處做切片推論（SAHI 式 tiling）。
5. **處理 Exp C 的 FP 上升**：以 Cam8 的 PR 曲線挑選信心門檻，而非沿用 0.25；或在混合訓練時對原始手機照降權。
6. **補一組真正的 no-augmentation 對照**：目前無法回答 augmentation 對跨視角泛化的貢獻。
7. **修掉資料集本身的問題**：補上那 1 張缺失的 CAM09 label；清掉 `cam8__...00001.jpg` 的 8 列重複標註；考慮把 Roboflow 三重複製攤平為單張原圖再重新增強。

**方法學**

8. 後續若要把偵測結果接到比例估計，必須先量化 FP/FN 對分母的影響——目前 Exp C 用 precision 換 recall 的取捨，會直接偏移任何以「有效雞冠實例」為分母的指標。

---

## 附錄：產出檔案

```text
experiments/cam8_holdout_yolov8s/
├─ manifests/          10 個 CSV（三組 train/val/test + all_split_summary）
├─ datasets/           expA/expB/expC 三個 dataset yaml
├─ lists/              Ultralytics 影像清單（gitignore，由腳本重建）
├─ results/            metrics CSV、comparison、error cases、run_settings.json
├─ figures/            6+ 張 PNG（gitignore，已內嵌於 HTML 報告）
├─ predictions/        三組各 40 張標註圖（gitignore）
├─ cam8_holdout_experiment_report.md / .html
└─ slide_summary.md

scripts/
├─ build_cam8_holdout.py     切分 + manifest + yaml（含硬性 assertion）
├─ run_cam8_holdout.py       訓練 + val/test 評估
├─ analyze_cam8_errors.py    FP/FN 分類 + 標註圖
├─ make_cam8_figures.py      圖表 + 指標表
└─ render_cam8_report.py     Markdown -> 自帶圖片的 HTML
```

模型權重（不進 git）：

```text
runs/cam8_holdout_yolov8s/expA_original_yolov8s_cam8test/weights/best.pt
runs/cam8_holdout_yolov8s/expB_cctv_cam9_24_yolov8s_cam8test/weights/best.pt
runs/cam8_holdout_yolov8s/expC_original_cctv_cam9_24_yolov8s_cam8test/weights/best.pt
```
