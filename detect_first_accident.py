import os
import cv2
import time
import torch
import pandas as pd
from ultralytics import YOLO  # For YOLOv8 / YOLOv12
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# --- CONFIGURATION ---
VIDEO_FOLDER = "dataset/video"       # Folder containing your 65 videos
OUTPUT_IMAGE_FOLDER = "detection_captures"
OUTPUT_CSV = "first_detection_results.csv"

# Models to test (Name: Path to weights)
MODELS = {
    "yolov5n": "saved/accident/yolov5n/weights/best.pt", 
    "yolov5s": "saved/accident/yolov5s/weights/best.pt", 
    "yolov5m": "saved/accident/yolov5m/weights/best.pt", 
    "yolov8n": "saved/accident/yolov8n/weights/best.pt", 
    "yolov8s": "saved/accident/yolov8s/weights/best.pt", 
    "yolov8m": "saved/accident/yolov8m/weights/best.pt", 
    "yolo12n": "saved/accident/yolo12n/weights/best.pt", 
    "yolo12s": "saved/accident/yolo12s/weights/best.pt", 
    "yolo12m": "saved/accident/yolo12m/weights/best.pt", 
}

# Settings
TARGET_CLASS_ID = 0      # Assuming Class 0 is 'Accident'
THRESHOLDS = [0.3, 0.5, 0.7]  # Thresholds to test
MAX_WORKERS = 2          # Limit parallel threads to avoid GPU OOM (Increase if GPU is strong)

def load_model(model_name, weight_path):
    """
    Loads YOLO models. Differentiates between v5 (torch.hub) and v8/v12 (ultralytics).
    """
    try:
        model = YOLO(weight_path)
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None

def process_video(video_file, model_name, weight_path, threshold):
    """
    Scans a video frame-by-frame. Stops at first detection > threshold.
    Returns dict of result.
    """
    video_path = os.path.join(VIDEO_FOLDER, video_file)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    detected = False
    result_data = None
    
    model = load_model(model_name, weight_path)
    if not model:
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Variables to store detection info for drawing
        xyxy = None 
        conf = 0
        has_detection = False

        # Inference
        try:
            results = model.predict(frame, verbose=False, conf=threshold)
            
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == TARGET_CLASS_ID:
                        has_detection = True
                        conf = float(box.conf)
                        # Get coordinates
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        xyxy = [coords[0], coords[1], coords[2], coords[3]]
                        break # Take the first valid detection
                if has_detection: break


            
            if has_detection:
                detected = True
                
                # --- DRAW RED BOUNDING BOX ---
                # cv2 uses BGR, so Red is (0, 0, 255)
                if xyxy:
                    x1, y1, x2, y2 = xyxy
                    # Draw rectangle: frame, (x1, y1), (x2, y2), color, thickness
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    
                    # Optional: Add label text
                    label = f"Accident {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # --- SAVE IMAGE ---
                base_name = os.path.splitext(video_file)[0]
                img_name = f"{base_name}_{model_name}_{threshold}.jpg"
                save_path = os.path.join(OUTPUT_IMAGE_FOLDER, img_name)
                
                cv2.imwrite(save_path, frame)
                
                # Prepare CSV Data
                result_data = {
                    "Video Name": video_file,
                    "Model": model_name,
                    "Threshold": threshold,
                    "Detected": True,
                    "Frame Number": frame_count,
                    "Time (sec)": round(frame_count / fps, 2) if fps > 0 else 0,
                    "Confidence": round(conf, 4),
                    "Image File": img_name,
                    "Valid (Manual Check)": "" 
                }
                break 
                
        except Exception as e:
            print(f"Inference error on {video_file}: {e}")
            break

    cap.release()
    
    if not detected:
        result_data = {
            "Video Name": video_file,
            "Model": model_name,
            "Threshold": threshold,
            "Detected": False,
            "Frame Number": -1,
            "Time (sec)": -1,
            "Confidence": 0,
            "Image File": "N/A",
            "Valid (Manual Check)": "N/A"
        }
        
    return result_data

def main():
    # Setup directories
    if not os.path.exists(OUTPUT_IMAGE_FOLDER):
        os.makedirs(OUTPUT_IMAGE_FOLDER)
        
    all_results = []
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"Found {len(video_files)} videos. Starting processing...")

    # We iterate by Model -> Threshold -> Videos to minimize model reloading overhead
    # But since user asked for parallel, we use ThreadPool. 
    # WARNING: GPU memory is shared. MAX_WORKERS > 1 might cause OOM.
    
    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model_name, weight_path in MODELS.items():
            for threshold in THRESHOLDS:
                print(f"Queueing tasks for {model_name} @ conf={threshold}...")
                for video in video_files:
                    tasks.append(
                        executor.submit(process_video, video, model_name, weight_path, threshold)
                    )

        # Progress bar
        for future in tqdm(as_completed(tasks), total=len(tasks), desc="Processing"):
            res = future.result()
            if res:
                all_results.append(res)

    # Save to CSV
    df = pd.DataFrame(all_results)
    # Sort for easier reading
    df = df.sort_values(by=["Model", "Threshold", "Video Name"])
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")
    print(f"Captured images saved to {OUTPUT_IMAGE_FOLDER}/")

if __name__ == "__main__":
    # Fix for some multiprocessing issues on Windows/macOS
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()