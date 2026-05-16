"""
Summarize YOLO label files: count instances per class id in the first column.

Works whether labels are already {0: human, 1: car} or still use VisDrone ids (0–11).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# If your labels came from raw VisDrone YOLO export (first column = VisDrone category)
VISDRONE_CLASS_NAMES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

LABEL_DIRS = [
    REPO_ROOT / "dataset" / "labels" / "train",
    REPO_ROOT / "dataset" / "labels" / "val",
    REPO_ROOT / "dataset" / "labels" / "test",
]


def main() -> None:
    counter: Counter[int] = Counter()

    for label_dir in LABEL_DIRS:
        if not label_dir.is_dir():
            print(f"[skip] missing folder: {label_dir}")
            continue
        for label_file in label_dir.glob("*.txt"):
            try:
                text = label_file.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                print(f"[warn] could not read {label_file}: {e}")
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                try:
                    class_id = int(float(parts[0]))
                except (ValueError, IndexError):
                    print(f"[warn] bad line in {label_file.name}: {line[:80]!r}")
                    continue
                counter[class_id] += 1

    if not counter:
        print("No label lines found. Check paths and that .txt files exist.")
        return

    print("Dataset class distribution (first column of each YOLO line):\n")

    for class_id in sorted(counter.keys()):
        n = counter[class_id]
        name = VISDRONE_CLASS_NAMES.get(class_id, f"custom / unknown id")
        print(f"  class {class_id}: {n:8d}  ({name})")

    # Training expects only YOLO class 0 = human, 1 = car (see dataset/data.yaml).
    if len(counter) > 2 or (counter.keys() & {2, 3, 4, 5, 6, 7, 8, 9, 10, 11}):
        print(
            "\n[NOTE] These counts follow VisDrone original category ids in the label files. "
            "For this internship project, remap to **only** class 0 (human) and 1 (car): "
            "pedestrian+people -> 0; car+van+truck+bus -> 1; skip other ids (and usually skip id 0)."
        )


if __name__ == "__main__":
    main()
