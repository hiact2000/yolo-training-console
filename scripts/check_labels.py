import os
import glob

# 請修改這裡為你的 labels 資料夾路徑（相對或絕對路徑皆可）
train_dir = 'datasets/detect/train/labels'
valid_dir = 'datasets/detect/valid/labels'


def check_labels(directory):
    print(f"正在檢查資料夾: {directory}")
    txt_files = glob.glob(os.path.join(directory, '*.txt'))

    if not txt_files:
        print("❌ 警告：找不到任何 .txt 檔案！請檢查路徑。")
        return

    error_count = 0
    for file_path in txt_files:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) > 0:
                    class_id = parts[0]
                    # 如果類別不是 '0'，就抓出來
                    if class_id != '0':
                        print(f"⚠️ 抓到了！檔案: {os.path.basename(file_path)}")
                        print(f"   -> 第 {i + 1} 行發現類別: {class_id}")
                        error_count += 1

    if error_count == 0:
        print("✅ 恭喜！此資料夾非常乾淨，所有標籤都是 0。")
    else:
        print(f"❌ 總共發現 {error_count} 個錯誤標籤。")


# 開始檢查
check_labels(train_dir)
print("-" * 30)
check_labels(valid_dir)