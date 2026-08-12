# Cam8 Hold-out 實驗 — 投影片大綱（8 張）

適用場合：實驗室週會 / 進度報告。全部數字來自 `results/cam8_test_comparison.csv`。

> **講者提醒**：全程只講「雞冠偵測」。資料集只有一個 `comb` 類別，沒有正常／異常之分，任何場合都不要把這些數字說成異常比例、OACP 或疾病相關。

---

## Slide 1 — 研究問題：validation 分數靠得住嗎？

**Key message**：同分布的 validation 無法回答「換一台監視器還能不能用」，所以把 Cam8 整台保留當測試集。

**建議圖表**：無圖，或放一張 Cam8 原始畫面（長走道、鐵絲網、右側逆光）當視覺錨點。
建議用 `predictions/expC/` 任一張，講的時候先不要顯示標註框。

**Bullets**
- 前一次三組實驗（原始 / 純 CCTV / 混合）都用同分布 validation 評估。
- 同分布 validation 只能說「這批資料學會了」，不能說「換視角還能用」。
- 本次改為 hold-out camera：**Cam8 完全不進 train、不進 val，只當 test**。
- 三種訓練策略，用**完全相同的一組 Cam8 影像**測試。

**Speaker note**：先問聽眾一個問題——如果一個模型 validation mAP50 有 0.944，你會覺得它可以部署嗎？留這個懸念到 Slide 6 揭曉。

---

## Slide 2 — 實驗設計：三種訓練資料，固定 Cam8 testing

**Key message**：唯一的變因是訓練資料池，測試集三組完全相同。

**建議圖表**：三列的設計表（可直接用報告 §3 的表格）。

**Bullets**
- **Exp A**：原始 dataset（手機近照為主）→ 436 train / 100 val。
- **Exp B**：CCTV Cam9 + Cam24 → 150 train / 37 val。
- **Exp C**：原始 + CCTV Cam9/24 → 586 train / 137 val。
- **共同 test**：Cam8 103 張 / 798 個有效標註框。
- Exp C 的切分＝Exp A 與 Exp B 切分的聯集，確保三組只差在資料池。

**Speaker note**：強調 Exp C 不是重新亂切一次，是 A 和 B 的聯集——這樣三組之間才沒有第二個變因。模型與超參數完全沿用前次實驗的 `args.yaml`。

---

## Slide 3 — 資料切分與防止 leakage

**Key message**：這批資料有兩個隱藏的污染源，都必須先處理掉，hold-out 才有意義。

**建議圖表**：leakage 檢查表（報告 §6 的兩張表），或一張「547 個檔案 → 261 張原圖」的示意圖。

**Bullets**
- **陷阱一**：所謂「原始 dataset」裡混了 **10 張 Cam8 影像**，照舊用就會直接破壞 hold-out。已在來源層移除。
- **陷阱二**：原始 dataset 的 547 個檔案其實只有 **261 張不重複原圖**（144 張各被 Roboflow 複製 3 份）。改用**以原圖分組**的切分，副本強制同側。
- 三個硬性 assertion 全數通過：train/val 中 Cam8 = **0 張**、與 test 的 MD5 重複 = **0 筆**、分組跨界 = **0 筆**。
- 誠實揭露：CCTV 連續影格使 **Exp B/C 的 validation 偏樂觀**，但 **Cam8 testing 不受影響**。

**Speaker note**：如果被問「你怎麼知道沒有 leakage」，就講這三個 assertion 寫在 `build_cam8_holdout.py` 裡，跑不過會直接中止。順帶提 Roboflow 原本的切分本身就有 1 張原圖同時在 train 和 valid。

---

## Slide 4 — 三組模型的 Cam8 testing 結果

**Key message**：Exp B 與 Exp C 明顯優於 Exp A；B 與 C 之間互有勝負。

**建議圖表**：`figures/cam8_test_precision_recall_comparison.png` ＋ `figures/cam8_test_map_comparison.png`（左右並排）。

**Bullets**
- **Precision 最高：Exp C 0.824**（B 0.792、A 0.669）。
- **Recall 最高：Exp B 0.771**（C 0.743、A 0.617）。
- **mAP50 最高：Exp B 0.837**，C 0.832 —— 差 0.005，**實質平手**。
- **mAP50-95 最高：Exp C 0.446**（B 0.410、A 0.309），代表 C 的框位置更準。
- Exp A 在每個指標上都墊底。

**Speaker note**：一定要主動說 B 和 C 的 mAP50 差 0.005 是雜訊，單一 seed 不能宣稱誰贏。如果有人追問，就導向 Slide 8 的多 seed 建議。

---

## Slide 5 — 指標比較圖：validation 與 Cam8 的落差

**Key message**：validation 分數最高的模型，在真實新視角上表現最差。

**建議圖表**：`figures/cam8_val_vs_test_gap.png`（**本場最重要的一張，值得單獨一頁**）。

**Bullets**
- **Exp A**：val mAP50 0.944 → Cam8 0.656，**掉了 0.288**。
- **Exp C**：val 0.898 → Cam8 0.832，只掉 0.066。
- **Exp B**：val 0.794 → Cam8 0.837，**反而上升 0.043**。
- 只看 validation 會得到**完全相反**的模型排名。
- Exp B 在 Cam8 比在自己的 val 還好 → CCTV 視角之間存在可轉移的共同特徵。

**Speaker note**：這裡回收 Slide 1 的懸念：那個 0.944 的模型就是 Exp A，換到 Cam8 只剩 0.656。這是整場報告的核心訊息——同分布 validation 會系統性高估跨視角表現。

---

## Slide 6 — 初步結論

**Key message**：加入 CCTV 資料是關鍵；原始手機照單獨用不足以支撐新視角。

**建議圖表**：沿用 Slide 5 的圖，或改放固定門檻（conf 0.25）的 TP/FN 對照表。

**Bullets**
- **原始 dataset 對 Cam8 明顯不足**：固定門檻下漏掉 798 個框中的 505 個。
- **只用 Cam9/Cam24 就能泛化到 Cam8**（mAP50 0.837），跨 CCTV 視角轉移是可行的。
- **混合是部分提升**：相對 Exp B，precision +0.032、mAP50-95 +0.036、漏檢從 353 降到 207；但 mAP50 平手、FP 從 53 升到 117。
- 實務部署建議看固定門檻：conf 0.25 下 **Exp C 抓到 591 個框，B 445、A 293**。
- 三組漏檢都遠多於誤報 → 現階段瓶頸在 recall，不在 precision。

**Speaker note**：如果被問「那到底該用哪個模型」，答案取決於下游用途：要少漏就用 C，要少誤報就用 B。目前資料量還不足以做這個決定。

---

## Slide 7 — 限制與錯誤案例

**Key message**：主要失敗模式是小/遠雞冠與低光，而且本實驗只 hold out 了一台 camera。

**建議圖表**：`figures/cam8_error_breakdown.png`，搭配 1–2 張 `predictions/expC/err007_*.jpg` 標註圖（綠 TP / 紅 FP / 橘 FN）。

**Bullets**
- **小雞冠 / 遠距離是最大可歸因失敗**：Exp A 漏 164，Exp C 降到 90（−45%）。
- **低光其次**：A 62 → C 36。
- **Exp C 用 precision 換 recall**：背景誤報 37 → 75、定位偏移 16 → 42。
- **限制**：單一 seed、只 hold out 一台 camera（n=1）、Exp B/C 的 val 受相鄰影格污染。
- **限制**：augmentation 沿用前次設定（Ultralytics 預設全開），**不是** no-augmentation 實驗。
- **分類盲點**：Cam8 真正的遮蔽來自籠具鐵絲網，不是雞隻互相重疊，因此未被遮蔽判準捕捉。

**Speaker note**：錯誤分類是影像可量測的代理指標，不是驗證過的成因，講的時候要說清楚。標註圖檔名前綴 `errNNN_` 是該張的錯誤數，可以直接挑最大的秀。

---

## Slide 8 — 下一步規劃

**Key message**：把 n=1 的泛化估計升級成有變異範圍的估計。

**建議圖表**：無圖，或一張 leave-one-camera-out 的示意圖（三輪，每輪 hold out 一台）。

**Bullets**
- **Leave-one-camera-out**：輪流 hold out Cam8 / Cam9 / Cam24，取得三個估計與變異範圍。
- **多 seed 重複**：每組至少 3 個 seed，回報平均 ± 標準差，才能判斷 B 與 C 的 0.005 是否有意義。
- **蒐集更多 camera view**：三台是泛化研究的下限，優先挑選高度／角度／光照條件不同者。
- **針對小目標**：提高 `imgsz` 或對走道深處做切片推論（tiling）。
- **補一組真正的 no-augmentation 對照**，才能回答 augmentation 對跨視角泛化的貢獻。

**Speaker note**：如果時間只剩一分鐘，就講第一點和第二點——這兩件事做完，這個實驗才算有統計上站得住的結論。
