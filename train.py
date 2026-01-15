import os
import shutil
import xml.etree.ElementTree as ET
import torch
from ultralytics import YOLO

# ======================
# 1. 类别映射（POLAR）
# ======================
CLASS_MAP = {
    "stand": 0,
    "sit": 1,
    "squat": 2,
    "run": 3,
    "walk": 4,
    "jump": 5,
    "bendover": 6,
    "stretch": 7,
    "lying": 8
}



CLASS_NAMES = list(CLASS_MAP.keys())

# ======================
# 2. VOC → YOLO 预处理
# ======================
def preprocess_polar_json_to_yolo(polar_root, yolo_root):
    import os
    import json
    import shutil

    ann_dir = os.path.join(polar_root, "Annotations")
    img_dir = os.path.join(polar_root, "JPEGImages")
    split_dir = os.path.join(polar_root, "ImageSets")

    total, used = 0, 0

    for split in ["train", "val", "test"]:
        img_out = os.path.join(yolo_root, "images", split)
        lbl_out = os.path.join(yolo_root, "labels", split)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        split_file = os.path.join(split_dir, f"{split}.txt")
        if not os.path.exists(split_file):
            print(f"⚠️ 缺少 {split_file}")
            continue

        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]

        for img_id in image_ids:
            total += 1

            img_path = os.path.join(img_dir, img_id + ".jpg")
            json_path = os.path.join(ann_dir, img_id + ".json")

            if not os.path.exists(img_path) or not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            w = float(data["width"])
            h = float(data["height"])

            if not data.get("persons"):
                continue

            person = data["persons"][0]

            # 1️⃣ bbox
            box = person["bndbox"]
            xmin = float(box["xmin"])
            ymin = float(box["ymin"])
            xmax = float(box["xmax"])
            ymax = float(box["ymax"])

            # 2️⃣ 动作（actions 中值为 1 的那个）
            actions = person["actions"]
            cls_name = None
            for k, v in actions.items():
                if v == 1 and k in CLASS_MAP:
                    cls_name = k
                    break

            if cls_name is None:
                continue

            cls_id = CLASS_MAP[cls_name]

            # 3️⃣ YOLO 格式
            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            yolo_line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

            shutil.copy(img_path, os.path.join(img_out, img_id + ".jpg"))
            with open(os.path.join(lbl_out, img_id + ".txt"), "w") as f:
                f.write(yolo_line)

            used += 1

    print(f"✅ 预处理完成：{used}/{total} 张样本成功转换")

# ======================
# 3. 生成 polar.yaml
# ======================
def write_polar_yaml(yolo_root):
    yaml_path = os.path.join(yolo_root, "polar.yaml")
    with open(yaml_path, "w") as f:
        f.write(
            f"""path: {yolo_root}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
        )
    return yaml_path

# ======================
# 4. main：预处理 + 训练
# ======================
def main():
    polar_root = "POLAR"        # 原始 POLAR 数据
    yolo_root = "POLAR_YOLO"    # 转换后的 YOLO 数据

    # Step 1: 预处理
    # preprocess_polar_json_to_yolo(polar_root, yolo_root)

    # Step 2: 写 yaml
    # yaml_path = write_polar_yaml(yolo_root)
    yaml_path = os.path.join(yolo_root, "polar.yaml")


    # Step 3: 训练 YOLOv8
    model = YOLO("yolov8n.pt")

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=2,
        project="runs/polar",
        name="yolov8_polar",
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        patience=20
    )

if __name__ == "__main__":
    main()
