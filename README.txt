CCEP N1/N2 Detection with YOLOv10
=================================

Background
----------
Cortico-cortical evoked potentials (CCEPs), elicited via single-pulse electrical stimulation,
are used to map brain networks. These responses include:
- N1: Early component (direct cortical connectivity)
- N2: Late component (indirect cortical connectivity)

Identifying N1 and N2 peaks is challenging due to variability in amplitude, phase, and timing.
Traditional methods relying on fixed time windows and manual review suffer from inter-rater variability
and often miss subject-specific differences.

New Method
----------
We developed a deep learning framework using YOLO v10 to:
1. Convert CCEP time-series into 2-D images using matplotlib.
2. Detect and classify N1, N2, and artifact regions directly in the image domain.
3. Map YOLO-predicted bounding boxes back to original time-series indices for clinical interpretation.

Performance:
- Trained on intracranial EEG from 9 patients with drug-resistant epilepsy (UAB)
- mAP@0.5 = 0.928 on held-out test data
- Tested on >4,000 unannotated epochs from 15 independent patients (UAB + University of Pennsylvania)

Installation
------------
1. Clone the repository and install dependencies:

    git clone https://github.com/yourusername/ccep-yolo.git
    cd ccep-yolo

2. (Optional) Create and activate conda environment:

    conda create -n ccep-yolo python=3.10 -y
    conda activate ccep-yolo

3. Install dependencies:

    pip install -r requirements.txt

requirements.txt
-----------------
numpy
matplotlib
pandas
ultralytics

Workflow
--------
Step 1: Convert CCEP signal to image
------------------------------------
import numpy as np
import matplotlib.pyplot as plt

def CCEP_signal_to_png(CCEP_signal: np.ndarray, out_path: str, zero_line=True):
    CCEP_signal = np.asarray(CCEP_signal).astype(float)
    if CCEP_signal.ndim != 1:
        raise ValueError("CCEP_signal must be 1-D")
    if np.isnan(CCEP_signal).any() or np.isinf(CCEP_signal).any():
        raise ValueError("CCEP_signal contains NaN/Inf")

    plt.figure(figsize=(6.022, 4.4), dpi=300)
    if zero_line:
        plt.axhline(0, linewidth=1, color='r')
    plt.plot(CCEP_signal, linewidth=1)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()

if __name__ == "__main__":
    x = np.linspace(0, 4*np.pi, 2000)
    CCEP_signal = 0.6*np.sin(2*x) + 0.3*np.sin(7*x)
    CCEP_signal_to_png(CCEP_signal, "CCEP_signal_example.png")
    print("Saved: CCEP_signal_example.png")

Step 2: Run YOLO model on the image
-----------------------------------
import os
import pandas as pd
from ultralytics import YOLO

def predict_CCEP_signal_images(model_path, source_path, imgsz=(1024, 1024), conf=0.5, out_csv="predictions.csv"):
    model = YOLO(model_path)
    results = model.predict(source=source_path, imgsz=imgsz, conf=conf, save=False)

    rows = []
    for r in results:
        img_name = os.path.basename(r.path)
        boxes = getattr(r, "boxes", None)
        if not boxes:
            rows.append([img_name, None, None, None, None, None])
            continue

        for b in boxes:
            cls = int(b.cls[0].item()) if b.cls is not None else -1
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
            rows.append([img_name, cls, x1, y1, x2, y2])

    pd.DataFrame(rows, columns=["image", "class", "x1", "y1", "x2", "y2"]).to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    predict_CCEP_signal_images(
        model_path="yolo_model.pt",
        source_path="CCEP_signal_example.png",
        imgsz=(1024, 1024),
        conf=0.5,
        out_csv="predictions.csv"
    )

Example Usage
-------------
# Step 1: Convert signal to PNG
python scripts/ccep_signal_to_image.py

# Step 2: Run YOLO model on generated image
python scripts/predict_ccep_signal_image.py

Output
------
- Images: Clean PNG plots of CCEP epochs with zero-line reference
- CSV: Predictions with bounding box coordinates, class labels, and file names

Example CSV:
image,class,x1,y1,x2,y2
CCEP_signal_example.png,0,123.0,456.0,200.0,500.0

Classes
-------
0 -> N1
1 -> N2
2 -> Artifact

License
-------
MIT License
