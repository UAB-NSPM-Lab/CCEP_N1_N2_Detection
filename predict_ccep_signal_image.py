# predict_ccep_signal_image.py
import os
import pandas as pd
from ultralytics import YOLO

def predict_CCEP_signal_images(model_path: str, source_path: str, imgsz=(1024, 1024), conf=0.5, out_csv="predictions.csv"):
    """
    Load a YOLO model and run predictions on one or more CCEP signal images.
    
    Parameters:
        model_path (str): Path to YOLO model weights (.pt file).
        source_path (str): Path to image(s) - can be a single PNG or a folder/glob (e.g., "signals/*.png").
        imgsz (tuple): Image size (width, height) for YOLO model input.
        conf (float): Confidence threshold for predictions.
        out_csv (str): Output CSV file to store predictions.
    """
    # Load the YOLO model
    model = YOLO(model_path)
    
    # Run YOLO predictions
    results = model.predict(source=source_path, imgsz=imgsz, conf=conf, save=False)

    # Store detection results
    rows = []
    for r in results:
        img_name = os.path.basename(r.path)  # Get image filename
        boxes = getattr(r, "boxes", None)    # Extract bounding boxes
        
        # If no detections found, log as None
        if boxes is None or len(boxes) == 0:
            rows.append([img_name, None, None, None, None, None])
            continue

        # Loop through each detected box
        for b in boxes:
            cls = int(b.cls[0].item()) if b.cls is not None else -1  # Class label
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()       # Coordinates
            rows.append([img_name, cls, x1, y1, x2, y2])

    # Save all detections to CSV
    pd.DataFrame(rows, columns=["image", "class", "x1", "y1", "x2", "y2"]).to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    # Example usage: run YOLO on the saved example image
    predict_CCEP_signal_images(
        model_path="yolo_model.pt",           # Replace with your YOLO weights path
        source_path="CCEP_signal_example.png",# Path to single image or folder
        imgsz=(1024, 1024),
        conf=0.5,
        out_csv="predictions.csv"
    )
