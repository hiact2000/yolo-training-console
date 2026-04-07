import os
import cv2
import shutil
from tqdm import tqdm

# ================= 設定區 (請修改這裡) =================
# 1. Roboflow 資料集的根目錄 (裡面應該要有 train/images, train/labels)
dataset_root = r"C:\Users\hicat\PycharmProjects\PythonProject3\comb_score_data"

# 2. 輸出目標路徑 (程式會自動建立)
output_root = "dataset_cls"

# 3. 定義類別 ID 對應到分數資料夾名稱
# 請根據你的 data.yaml 修改這裡！
# 假設 Roboflow 標註時: Class 0 -> Score 1, Class 1 -> Score 2...
class_map = {
    0: "score1",
    1: "score2",
    2: "score3",
    3: "score4",
    4: "score5"
}


# =======================================================

def convert_yolo_to_xyxy(x_center, y_center, w, h, img_w, img_h):
    # 將 YOLO 正規化座標 (0-1) 轉為 像素座標
    x_center *= img_w
    y_center *= img_h
    w *= img_w
    h *= img_h

    x1 = int(x_center - w / 2)
    y1 = int(y_center - h / 2)
    x2 = int(x_center + w / 2)
    y2 = int(y_center + h / 2)

    # 防止切超過圖片邊界
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def process_split(split_name):
    # 處理 train 或 valid 資料夾
    images_dir = os.path.join(dataset_root, split_name, "images")
    labels_dir = os.path.join(dataset_root, split_name, "labels")

    if not os.path.exists(images_dir):
        print(f"⚠️ 找不到 {split_name} 資料夾，跳過。")
        return

    print(f"🚀 正在處理 {split_name} 資料集...")

    image_files = os.listdir(images_dir)

    for img_file in tqdm(image_files):
        if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        # 1. 讀取圖片
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None: continue
        img_h, img_w = img.shape[:2]

        # 2. 找對應的 label 檔 (將副檔名換成 .txt)
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            continue  # 有圖沒標籤就跳過

        # 3. 讀取標籤內容
        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) < 5: continue

            class_id = int(parts[0])
            coords = list(map(float, parts[1:5]))

            # 檢查這個類別我們是否需要 (例如只處理 score1-5)
            if class_id not in class_map:
                continue

            folder_name = class_map[class_id]

            # 4. 座標轉換與裁切
            x1, y1, x2, y2 = convert_yolo_to_xyxy(*coords, img_w, img_h)
            crop_img = img[y1:y2, x1:x2]

            if crop_img.size == 0: continue

            # 5. 存檔
            # 目標資料夾: dataset_cls/train/score1/
            save_dir = os.path.join(output_root, split_name, folder_name)
            os.makedirs(save_dir, exist_ok=True)

            # 檔名加上索引防止同一張圖有多個標註時檔名衝突
            save_name = f"{os.path.splitext(img_file)[0]}_crop_{i}.jpg"
            cv2.imwrite(os.path.join(save_dir, save_name), crop_img)


# --- 主程式執行 ---
# 刪除舊的輸出 (如果有的話)，確保乾淨
if os.path.exists(output_root):
    shutil.rmtree(output_root)

# 分別處理 train 和 valid
process_split("train")
process_split("valid")
# 如果你的資料夾叫 test 或 val，也可以自己加一行 process_split("test")

print("\n✅ 資料準備完成！")
print(f"📁 分類資料集已建立於: {os.path.abspath(output_root)}")
print("現在你可以直接執行 Step 2 的訓練程式了。")