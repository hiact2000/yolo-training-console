import cv2
from ultralytics import YOLO

# --- 載入兩個模型 ---
print("正在載入模型...")
# 1. 偵測模型 (你之前練好的)
detect_model = YOLO('egg_project/run_test_block6/weights/best.pt')
# 2. 分類模型 (步驟二練好的)
cls_model = YOLO('comb_score_project/run_score_v1/weights/best.pt')

# 設定影片或圖片來源
source = 'test_video.mp4'  # 或 'image.jpg' 或 0 (webcam)
cap = cv2.VideoCapture(source)

# 顏色對應 (BGR) - 用來畫框框顏色區分等級
color_map = {
    "score1": (147, 20, 255),  # Pinkish
    "score2": (173, 84, 90),
    "score3": (105, 103, 203),
    "score4": (123, 122, 194),
    "score5": (127, 127, 187)  # Deep Red
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. 先進行偵測 (Detect)
    results = detect_model(frame, verbose=False)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # 安全檢查：確保座標沒有超出圖片範圍
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # 2. 裁切雞冠 (Crop)
            crop_img = frame[y1:y2, x1:x2]

            if crop_img.size == 0: continue

            # 3. 送進分類模型 (Classify)
            # verbose=False 讓它不要一直印 log
            cls_result = cls_model(crop_img, verbose=False)

            # 取得分數最高的類別
            probs = cls_result[0].probs
            top1_index = probs.top1
            score_label = cls_result[0].names[top1_index]  # 得到 "score5"
            conf = probs.top1conf.item()  # 信心度

            # 4. 畫圖 (Draw)
            color = color_map.get(score_label, (0, 255, 0))

            # 畫框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 寫字: Score 5 (98%)
            label_text = f"{score_label.capitalize()} ({conf:.2f})"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 顯示結果
    cv2.imshow('Comb Scoring System', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()