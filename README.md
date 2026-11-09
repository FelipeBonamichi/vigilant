# VIGILANT – YOLO Models for Vehicle and License Plate Detection

Trained YOLO weights that detect **vehicles** and **Brazilian license plates** (legacy and Mercosur standards) in images from access-control gates. The models are ready to use with the [Ultralytics](https://github.com/ultralytics/ultralytics) library.

| | |
|---|---|
| **Task** | Object detection |
| **Classes** | `0: carro` (any vehicle) · `1: placa` (license plate) |
| **Input** | 320×320 pixels |
| **Architectures** | YOLOv8, YOLOv10, YOLO11, YOLOv12, YOLO26 |
| **Scales** | nano (n), small (s), medium (m): 15 models in total |
| **Training data** | [car-detection 2,1 – Roboflow Universe](https://universe.roboflow.com/dataset-final-x1wfw/car-detection-2-1) (CC BY 4.0) |

---

## Files

```
vigilante/
├── Modelos_yoloN/   # nano:   YOLOv8n, YOLOv10n, YOLO11n, YOLOv12n, YOLO26n
├── modelo_yolos/    # small:  YOLOv8s, YOLOv10s, YOLO11s, YOLOv12s, YOLO26s
├── modelo_yolom/    # medium: YOLOv8m, YOLOv10m, YOLO11m, YOLOv12m, YOLO26m
└── código.py        # inference script
```

### Which model should I use?

| Need | Suggested model |
|---|---|
| Low-cost hardware (CPU, Raspberry Pi, Jetson) | **YOLO26n** or **YOLO11n**: nano size, F1 = 0.987 |
| Best balance between size and accuracy | **YOLO26s**: highest F1 (0.990) |
| Fewest missed plates | **YOLO26m**: highest recall (0.988; plate recall 0.986) |
| Fewest false detections | **YOLOv10m** or **YOLOv8n**: highest precision (0.996) |

---

## Quick start

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO

model = YOLO("modelo_yolos/<weights>.pt")          # e.g., the YOLO26s weights
results = model.predict("image.jpg", imgsz=320, conf=0.25, save=True)

for box in results[0].boxes:
    cls = model.names[int(box.cls)]                  # 'carro' or 'placa'
    print(cls, round(float(box.conf), 2), box.xyxy[0].tolist())
```

Annotated images are saved in `runs/detect/predict/`.

**Cropping the plate for OCR:**

```python
import cv2

img = cv2.imread("image.jpg")
r = model.predict(img, imgsz=320, conf=0.25)[0]
for i, (c, b) in enumerate(zip(r.boxes.cls, r.boxes.xyxy)):
    if model.names[int(c)] == "placa":
        x1, y1, x2, y2 = map(int, b)
        cv2.imwrite(f"plate_{i}.jpg", img[y1:y2, x1:x2])
```

**Recommended settings:** use `imgsz=320`, the same resolution as training. `conf=0.25` is a good starting point; raise it to reduce false plates, or lower it to miss fewer plates.

---

## Performance

Evaluated on the dataset's test set: 364 images, 375 vehicles, and 369 plates, never used in training. The best value of each column is in bold.

| Scale | Model | Acc. | P | R | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| Nano | YOLOv8n | 0.967 | **0.996** | 0.971 | 0.983 | 0.989 | 0.767 |
| | YOLOv10n | 0.962 | 0.987 | 0.974 | 0.981 | 0.987 | 0.763 |
| | YOLO11n | 0.975 | 0.994 | 0.980 | 0.987 | 0.989 | 0.766 |
| | YOLOv12n | 0.965 | 0.983 | 0.981 | 0.982 | 0.989 | 0.764 |
| | YOLO26n | 0.975 | 0.994 | 0.981 | 0.987 | 0.985 | **0.769** |
| Small | YOLOv8s | **0.979** | 0.994 | 0.984 | 0.989 | **0.990** | 0.760 |
| | YOLOv10s | 0.976 | 0.989 | 0.986 | 0.988 | **0.990** | 0.766 |
| | YOLO11s | 0.974 | 0.991 | 0.982 | 0.987 | 0.984 | 0.762 |
| | YOLOv12s | 0.974 | 0.992 | 0.981 | 0.987 | 0.989 | 0.765 |
| | YOLO26s | **0.979** | 0.994 | 0.985 | **0.990** | 0.989 | 0.766 |
| Medium | YOLOv8m | 0.973 | 0.993 | 0.979 | 0.986 | **0.990** | 0.763 |
| | YOLOv10m | **0.979** | **0.996** | 0.983 | 0.989 | 0.985 | 0.761 |
| | YOLO11m | 0.974 | 0.994 | 0.980 | 0.987 | 0.989 | 0.759 |
| | YOLOv12m | 0.975 | 0.993 | 0.981 | 0.987 | 0.989 | 0.763 |
| | YOLO26m | **0.979** | 0.991 | **0.988** | 0.989 | **0.990** | 0.766 |

Accuracy = TP / (TP + FP + FN). The **vehicle** class is detected with recall between 0.981 and 0.995; the **plate** class is harder, with recall between 0.956 and 0.986.

---

## Training

| Parameter | Value |
|---|---|
| Initial weights | COCO pre-trained (transfer learning, full fine-tuning) |
| Input resolution | 320×320 |
| Batch size | 16 |
| Epochs | up to 300 (early stopping, patience 100) |
| Optimizer | Ultralytics default (`auto`) |
| Augmentation | saturation ±25% (Roboflow, 2×) + Ultralytics defaults (HSV, translation, scale, horizontal flip, mosaic) |
| Hardware | Google Colab, NVIDIA Tesla T4 |

Training data: 2,049 original images (4,098 after augmentation), 440 for validation, and 364 for testing. They include cars, vans, trucks, minibuses, and buses seen from the front or the rear.

---

## Limitations

- **Plates are the weak class:** most errors are missed plates under difficult lighting (reflections, shadows, low light) and false plates on stickers or text on the vehicle body.
- **Low resolution:** at 320×320, distant plates occupy only a few tens of pixels.
- **Scenarios not covered:** the training images include no rain, fog, or night scenes, and no images from the final gate cameras.
- **Vehicle type:** the `carro` class covers every vehicle type; the models do not tell cars from buses or trucks.

---

## Context

Developed in the **VIGILANT** project at the National Institute of Telecommunications (Inatel), Brazil, by the CTIoT Lab and WAI Lab. The detector is the first stage of an access-control system: the cropped plates are read by OCR, and the records are sent via MQTT to a dashboard that monitors campus occupancy.

The paper describing these models is under review; the citation will be added after publication.

## License

The training dataset is distributed under CC BY 4.0. <!-- Add the license chosen for the weights and code (e.g., AGPL-3.0, as Ultralytics models are AGPL). -->