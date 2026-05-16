#!/usr/bin/env python3
"""
Create the drone-human-car-detection folder and file layout (empty placeholders).

Run from anywhere:
    python setup_project.py

Creates the structure under the directory containing this script (project root).
"""

from __future__ import annotations

from pathlib import Path

# Project root = directory where this script lives
ROOT = Path(__file__).resolve().parent

DIRS = [
    ROOT / "dataset" / "images" / "train",
    ROOT / "dataset" / "images" / "val",
    ROOT / "dataset" / "images" / "test",
    ROOT / "dataset" / "labels" / "train",
    ROOT / "dataset" / "labels" / "val",
    ROOT / "dataset" / "labels" / "test",
    ROOT / "raw_data" / "VisDrone",
    ROOT / "notebooks",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "outputs" / "dataset_samples",
    ROOT / "outputs" / "sample_predictions",
    ROOT / "outputs" / "counting_results",
    ROOT / "outputs" / "tracking_results",
    ROOT / "reports",
    ROOT / "runs" / "detect",
]

FILES = [
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / ".gitignore",
    ROOT / "dataset" / "data.yaml",
    ROOT / "notebooks" / "01_dataset_analysis.ipynb",
    ROOT / "scripts" / "convert_visdrone_to_yolo.py",
    ROOT / "scripts" / "visualize_dataset.py",
    ROOT / "scripts" / "check_dataset.py",
    ROOT / "src" / "train.py",
    ROOT / "src" / "infer_count.py",
    ROOT / "src" / "evaluate.py",
    ROOT / "src" / "track_count.py",
    ROOT / "reports" / "training_results.md",
]

# Keep empty dirs tracked in git via placeholder
GITKEEP_DIRS = [
    ROOT / "dataset" / "images" / "train",
    ROOT / "dataset" / "images" / "val",
    ROOT / "dataset" / "images" / "test",
    ROOT / "dataset" / "labels" / "train",
    ROOT / "dataset" / "labels" / "val",
    ROOT / "dataset" / "labels" / "test",
    ROOT / "raw_data" / "VisDrone",
    ROOT / "outputs" / "dataset_samples",
    ROOT / "outputs" / "sample_predictions",
    ROOT / "outputs" / "counting_results",
    ROOT / "outputs" / "tracking_results",
    ROOT / "runs" / "detect",
]

EMPTY_NOTEBOOK = """{
  "cells": [],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.12.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
"""


def main() -> None:
    for d in DIRS:
        d.mkdir(parents=True, exist_ok=True)
        print(f"mkdir: {d.relative_to(ROOT)}")

    for f in FILES:
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.name == "01_dataset_analysis.ipynb":
            if not f.exists() or f.read_text(encoding="utf-8").strip() == "":
                f.write_text(EMPTY_NOTEBOOK.strip() + "\n", encoding="utf-8")
                print(f"write: {f.relative_to(ROOT)} (empty notebook)")
        else:
            if not f.exists():
                f.write_bytes(b"")
                print(f"touch: {f.relative_to(ROOT)}")
            else:
                print(f"skip (exists): {f.relative_to(ROOT)}")

    for d in GITKEEP_DIRS:
        gk = d / ".gitkeep"
        if not gk.exists():
            gk.write_bytes(b"")
            print(f"touch: {gk.relative_to(ROOT)}")

    print("\nDone. Project root:", ROOT)


if __name__ == "__main__":
    main()
