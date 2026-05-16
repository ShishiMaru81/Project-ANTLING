# Drone Human & Car Detection and Counting System using YOLOv8

This project is an AI/ML internship assessment project for building a computer vision pipeline that detects humans and cars from drone/aerial images and counts the total number of humans.

The system uses the VisDrone dataset and fine-tunes YOLOv8 for aerial human and vehicle detection.

---

## Project Overview

Drone images are challenging because objects are often small, crowded, partially occluded, and captured from unusual angles. This project focuses on detecting two important object groups:

- Human
- Car / vehicle

The final system can:

- Detect humans from drone images
- Detect cars/vehicles from drone images
- Count total humans
- Draw bounding boxes
- Show confidence scores
- Save prediction outputs
- Evaluate the trained model

---

## Dataset

Dataset used:

**VisDrone Dataset**

The original dataset contains multiple object categories such as pedestrians, people, cars, vans, trucks, buses, bicycles, and others.

For this project, only human and vehicle-related classes were selected.

---

## Class Mapping

The dataset originally contains several object classes. For this project, they were remapped into two classes:

| Source Class | Target Class |
|---|---|
| pedestrian | human |
| people | human |
| car | car |
| van | car |
| truck | car |
| bus | car |

Final class format:

```txt
0 = human
1 = car
