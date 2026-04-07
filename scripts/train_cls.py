from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n-cls.pt')

    model.train(
        data='dataset_cls',
        epochs=50,
        imgsz=224,
        batch=16,
        project='comb_score_project',
        name='run_nano_freeze',

        freeze=6,

        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        erasing=0.0,
        auto_augment=None,

        dropout=0.5,  # 50% Dropout
        patience=5
    )
