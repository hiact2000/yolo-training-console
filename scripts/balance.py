import os
import random
import cv2
import numpy as np
from tqdm import tqdm

# ================= 設定區 =================
dataset_root = "dataset_cls/train"  # 指向你的訓練資料夾
target_count = 200  # 目標：每個類別都變成 200 張


# =========================================

def augment_image_safe(image):
    """
    僅進行「不裁切、不縮放」的安全增強
    """
    h, w = image.shape[:2]

    # 1. 隨機水平翻轉 (50% 機率)
    # 這完全不會損失邊緣資訊
    if random.random() < 0.5:
        image = cv2.flip(image, 1)

    # 2. 極微幅旋轉 (-5 ~ 5 度)
    # 我們不縮放 (scale=1.0)，並使用 BORDER_REPLICATE 填補旋轉產生的空隙
    # 這樣可以避免出現黑邊，也不會因為放大而切到雞冠
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 3. 隨機亮度調整 (0.9 ~ 1.1 倍)
    # 調整 HSV 的 V 通道，不影響形狀
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    brightness_factor = random.uniform(0.9, 1.1)  # 變動幅度減小，避免過曝
    v = cv2.multiply(v, brightness_factor)
    v = np.clip(v, 0, 255).astype(np.uint8)

    hsv_new = cv2.merge([h, s, v])
    image = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    return image


def balance_dataset(root_dir, target_n):
    print(f"🚀 開始「無裁切」資料平衡: {root_dir}")
    print(f"🎯 目標每類: {target_n} 張")

    categories = os.listdir(root_dir)

    for cat in categories:
        cat_dir = os.path.join(root_dir, cat)
        if not os.path.isdir(cat_dir): continue

        # 只讀取圖片檔
        images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        count = len(images)

        print(f"  📂 {cat}: 目前 {count} 張", end=" ")

        if count > target_n:
            # === 太多的：隨機刪除 ===
            files_to_remove = random.sample(images, count - target_n)
            for f in files_to_remove:
                os.remove(os.path.join(cat_dir, f))
            print(f"-> 刪除至 {target_n} 張 ✅")

        elif count < target_n:
            # === 太少的：增強並補充 ===
            needed = target_n - count
            print(f"-> 需補充 {needed} 張", end="")

            pbar = tqdm(total=needed, desc="Augmenting", leave=False)

            for i in range(needed):
                # 隨機挑一張原圖
                src_img_name = random.choice(images)
                src_path = os.path.join(cat_dir, src_img_name)

                img = cv2.imread(src_path)
                if img is None: continue

                # --- 執行安全增強 ---
                aug_img = augment_image_safe(img)

                # 存檔
                base, ext = os.path.splitext(src_img_name)
                dst_name = f"{base}_aug_{i}{ext}"
                dst_path = os.path.join(cat_dir, dst_name)

                cv2.imwrite(dst_path, aug_img)
                pbar.update(1)

            pbar.close()
            print(" ✅")
        else:
            print("-> 數量剛好")


if __name__ == "__main__":
    balance_dataset(dataset_root, target_count)