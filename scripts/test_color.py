import cv2
import numpy as np
from ultralytics import YOLO

# --- 設定區 ---
# 1. 載入你的「偵測模型」(不是分類模型喔！)
# 請修改為你的 Detection 模型路徑（runs/ 底下訓練好的 best.pt）
detect_model = YOLO("runs/detect/egg_project/run_test_block/weights/best.pt")

# 2. 定義標準色 (RGB 格式)
score_standards = {
    "Score 5": (187, 127, 127),
    "Score 4": (194, 122, 123),
    "Score 3": (203, 103, 105),
    "Score 2": (173, 84, 90),
    "Score 1": (157, 48, 50)
}

# 將標準色轉為 BGR (因為 OpenCV 是 BGR)
standards_bgr = {k: (v[2], v[1], v[0]) for k, v in score_standards.items()}


# -------------

def get_color_score(crop_img):
    """
    計算雞冠的平均顏色，並找出最接近的分數
    """
    # 1. 轉為 HSV 以過濾背景
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # 2. 建立紅色遮罩 (只計算紅色區域，忽略背景鐵籠、羽毛)
    # 紅色在 HSV 有兩個區間
    lower1 = np.array([0, 40, 40])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 40, 40])
    upper2 = np.array([180, 255, 255])

    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

    # 如果遮罩內沒有像素 (沒抓到紅色)，回傳 Unknown
    if cv2.countNonZero(mask) == 0:
        return "Unknown", (0, 0, 0)

    # 3. 計算遮罩區域內的平均顏色 (BGR)
    mean_color = cv2.mean(crop_img, mask=mask)[:3]

    # 4. 比對距離 (歐式距離)
    min_dist = float('inf')
    best_score = "Unknown"

    for score_name, std_color in standards_bgr.items():
        # 計算兩色之間的距離
        dist = np.linalg.norm(np.array(mean_color) - np.array(std_color))

        if dist < min_dist:
            min_dist = dist
            best_score = score_name

    return best_score, mean_color


# --- 主程式 (測試影片或鏡頭) ---
# source = 0 # 網路攝影機
source = "test_video.mp4"  # 或圖片路徑
cap = cv2.VideoCapture(source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # YOLO 偵測
    results = detect_model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # 確保不超出邊界
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

            # 裁切
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0: continue

            # === 核心演算法判斷 ===
            score_text, avg_color = get_color_score(crop)

            # 顯示結果
            # 畫框 (顏色用算出來的平均色顯示)
            display_color = (int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), display_color, 2)
            cv2.putText(frame, score_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, display_color, 2)

    cv2.imshow("Comb Color Scoring", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()