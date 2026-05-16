"""
Run YOLO inference, draw human/car boxes, and overlay counts (Humans | Cars).

Example:
    python src/infer_count.py --model runs/detect/visdrone_yolo3/weights/best.pt \\
        --source dataset/images/val --conf 0.25 --save-dir outputs/counting_results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]

# BGR
COLOR_HUMAN = (0, 255, 0)
COLOR_CAR = (0, 165, 255)


def find_default_weights() -> Path | None:
    """Pick the most recently modified best.pt under runs/detect/, if any."""
    base = REPO_ROOT / "runs" / "detect"
    if not base.is_dir():
        return None
    candidates = list(base.glob("*/weights/best.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def process_image(model: YOLO, image_path: Path, save_path: Path, conf: float) -> tuple[int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    results = model.predict(image, conf=conf, verbose=False)

    human_count = 0
    car_count = 0

    for result in results:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # Use tensor indexing (avoids type-checker issues with iterating Boxes).
        xyxy = boxes.xyxy.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int).reshape(-1)
        confs = boxes.conf.cpu().numpy().reshape(-1)

        for i, class_id in enumerate(clss):
            confidence = float(confs[i])
            x1, y1, x2, y2 = map(int, xyxy[i])

            if class_id == 0:
                human_count += 1
                label = f"human {confidence:.2f}"
                color = COLOR_HUMAN
            elif class_id == 1:
                car_count += 1
                label = f"car {confidence:.2f}"
                color = COLOR_CAR
            else:
                # Model should only emit 0/1 for this project; skip anything else.
                continue

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    count_text = f"Humans: {human_count} | Cars: {car_count}"

    cv2.rectangle(image, (10, 10), (420, 55), (0, 0, 0), -1)
    cv2.putText(
        image,
        count_text,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), image)

    return human_count, car_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YOLO inference with human/car counting overlay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Required unless defaults exist:
  --model   path to weights (.pt), e.g. runs/detect/<run>/weights/best.pt
  --source  image file, or folder of images

If you run the script with no arguments, it tries the newest runs/detect/*/weights/best.pt
and dataset/images/val (if that folder exists).""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Weights path. If omitted, uses newest runs/detect/*/weights/best.pt when found.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Image file or directory. If omitted, uses dataset/images/val when it exists.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-dir", type=str, default="outputs/counting_results")

    args = parser.parse_args()

    model_path = args.model
    if not model_path:
        w = find_default_weights()
        if w is None:
            print(
                "[ERROR] Pass --model path/to/best.pt (no weights found under runs/detect/).",
                file=sys.stderr,
            )
            return 2
        model_path = str(w)
        print(f"[INFO] Using weights: {model_path}")

    source_str = args.source
    if not source_str:
        default_dir = REPO_ROOT / "dataset" / "images" / "val"
        if default_dir.is_dir() and any(default_dir.glob("*.jpg")):
            source_str = str(default_dir)
            print(f"[INFO] Using source: {source_str}")
        else:
            print(
                "[ERROR] Pass --source path/to/image.jpg or folder (dataset/images/val empty or missing).",
                file=sys.stderr,
            )
            return 2

    source = Path(source_str)
    if not source.is_absolute():
        source = (REPO_ROOT / source).resolve()
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = (REPO_ROOT / save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    model = YOLO(model_path)

    if source.is_file():
        save_path = save_dir / f"pred_{source.name}"
        humans, cars = process_image(model, source, save_path, args.conf)
        print(f"{source.name}: Humans={humans}, Cars={cars}")

    elif source.is_dir():
        image_files = [p for p in source.iterdir() if p.suffix.lower() in image_extensions]
        if not image_files:
            print(f"[ERROR] No images found in {source}", file=sys.stderr)
            return 2
        for image_path in sorted(image_files):
            save_path = save_dir / f"pred_{image_path.name}"
            humans, cars = process_image(model, image_path, save_path, args.conf)
            print(f"{image_path.name}: Humans={humans}, Cars={cars}")

    else:
        print(f"[ERROR] Source not found: {source}", file=sys.stderr)
        return 2

    print(f"Results saved to {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
