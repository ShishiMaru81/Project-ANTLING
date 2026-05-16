from pathlib import Path
import shutil
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "VisDrone_Dataset"

# Source VisDrone YOLO classes -> Target project classes
# Target:
# 0 = human
# 1 = car
SOURCE_TO_TARGET = {
    0: 0,  # pedestrian -> human
    1: 0,  # people -> human

    3: 1,  # car -> car
    4: 1,  # van -> car
    5: 1,  # truck -> car
    8: 1,  # bus -> car
}

SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}


def get_image_files(image_dir: Path):
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(image_dir.glob(ext))
    return image_files


def filter_and_remap_label(src_label: Path, dst_label: Path):
    filtered_lines = []

    if src_label.exists():
        with open(src_label, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                source_class = int(float(parts[0]))

                if source_class not in SOURCE_TO_TARGET:
                    continue

                target_class = SOURCE_TO_TARGET[source_class]
                parts[0] = str(target_class)

                filtered_lines.append(" ".join(parts))

    dst_label.write_text("\n".join(filtered_lines), encoding="utf-8")


def copy_split(raw_split_name: str, yolo_split_name: str):
    raw_split_dir = RAW_ROOT / raw_split_name
    image_dir = raw_split_dir / "images"
    label_dir = raw_split_dir / "labels"

    out_image_dir = REPO_ROOT / "dataset" / "images" / yolo_split_name
    out_label_dir = REPO_ROOT / "dataset" / "labels" / yolo_split_name

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        print(f"[WARN] Missing image folder: {image_dir}")
        return

    if not label_dir.exists():
        print(f"[WARN] Missing label folder: {label_dir}")

    images = get_image_files(image_dir)

    print(f"\nProcessing {raw_split_name}")
    print(f"Images found: {len(images)}")
    print(f"Source label folder: {label_dir}")
    print(f"Output image folder: {out_image_dir}")
    print(f"Output label folder: {out_label_dir}")

    for image_path in tqdm(images, desc=f"Copying {yolo_split_name}"):
        shutil.copy2(image_path, out_image_dir / image_path.name)

        src_label = label_dir / f"{image_path.stem}.txt"
        dst_label = out_label_dir / f"{image_path.stem}.txt"

        filter_and_remap_label(src_label, dst_label)


def main():
    print("Starting VisDrone YOLO label filtering/remapping...")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Raw dataset root: {RAW_ROOT}")

    if not RAW_ROOT.exists():
        print(f"[ERROR] Dataset not found: {RAW_ROOT}")
        return

    for raw_split, yolo_split in SPLITS.items():
        copy_split(raw_split, yolo_split)

    print("\nConversion complete.")
    print("Target classes:")
    print("0 = human")
    print("1 = car")
    print("\nNext check labels with:")
    print("Get-Content dataset\\labels\\train\\<some_file>.txt -TotalCount 5")


if __name__ == "__main__":
    main()