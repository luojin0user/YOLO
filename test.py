from ultralytics import YOLO
import cv2
import os

def main():
    # 1. 加载训练好的模型
    model = YOLO("runs/detect/runs/polar/yolov8_polar/weights/best.pt")

    # 2. 推理源（图片 / 文件夹 / 视频 / 摄像头）
    source = "test.jpg"   # 可换成 0 / video.mp4 / images/

    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.25,
        iou=0.5,
        device=0,
        save=True,          # 保存结果图
        show=False          # 是否实时显示
    )

    # 3. 解析结果（可选）
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id]
            print(f"Detected: {label}, confidence: {conf:.2f}")

if __name__ == "__main__":
    main()
