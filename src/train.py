from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / "dataset"
RUNTIME_DATA_YAML = DATASET_ROOT / ".ultralytics_data.yaml"


def write_runtime_data_yaml() -> Path:
    root = DATASET_ROOT.resolve()

    text = "\n".join(
        [
            f"path: {root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            "  0: human",
            "  1: car",
            "",
        ]
    )

    RUNTIME_DATA_YAML.write_text(text, encoding="utf-8")

    return RUNTIME_DATA_YAML


def main() -> None:
    yaml_path = write_runtime_data_yaml()

    print(f"[INFO] Using data config: {yaml_path}")

    model = YOLO("yolov8s.pt")

    model.train(
        data=str(yaml_path),
        epochs=30,
        imgsz=960,
        batch=8,
        name="visdrone_yolo",
    )


if __name__ == "__main__":
    main()