import random
from pathlib import Path
import cv2

CLASS_NAMES = {
    0: "human",
    1: "car",
}


def draw_yolo_boxes(image_path, label_path, save_path):
    image = cv2.imread(str(image_path))
    h, w, _ = image.shape

    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            if not line.strip():
                continue

            class_id, x_center, y_center, box_w, box_h = map(float, line.split())
            class_id = int(class_id)

            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            if class_id not in CLASS_NAMES:
                continue
            else:
                label = CLASS_NAMES[class_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

    cv2.imwrite(str(save_path), image)


def main():
    image_dir = Path("dataset/images/train")
    label_dir = Path("dataset/labels/train")
    save_dir = Path("outputs/dataset_samples")
    save_dir.mkdir(parents=True, exist_ok=True)

    images = list(image_dir.glob("*.jpg"))
    sample_images = random.sample(images, min(10, len(images)))

    for image_path in sample_images:
        label_path = label_dir / f"{image_path.stem}.txt"
        save_path = save_dir / image_path.name
        draw_yolo_boxes(image_path, label_path, save_path)

    print(f"Saved samples to {save_dir}")


if __name__ == "__main__":
    main()