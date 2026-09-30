import cv2
import time
import os
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
import glob

def run_video_inference():
    # --- Configuration ---
    # List of models to test on the video set
    models_to_test = ["yolov5n", "yolov5s", "yolov5m", "yolov8n", "yolov8s", "yolov8m", "yolo12n", "yolo12s", "yolo12m"] 
    # List of your 10 video files (update paths as needed)
    #video_files = #[f"dataset/test_interface/video_{i}.mp4" for i in range(1, 12)] 
    video_files = glob.glob('dataset/video/*.mp4') #os.listdir('dataset/video/')
    model_path_template = "saved/accident/{model}/weights/best.pt"
    output_csv = "video_test_results.csv"
    img_size = 640
    conf_threshold = 0.5 # Threshold to trigger an "accident" detection
    
    results_list = []

    # Outer loop for models with tqdm as requested
    model_pbar = tqdm(models_to_test, desc="Testing Models on Videos")
    for model_name in model_pbar:
        model_pbar.set_description(f"Processing {model_name}")
        model_path = model_path_template.format(model=model_name)
        
        if not os.path.exists(model_path):
            print(f"\n[SKIP] Model not found: {model_path}")
            continue
            
        model = YOLO(model_path)
        
        # Inner loop for videos
        for video_path in video_files:
            if not os.path.exists(video_path):
                continue
                
            cap = cv2.VideoCapture(video_path)
            fps_original = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            first_accident_frame = None
            total_inference_time = 0
            processed_frames = 0
            
            # tqdm for frame processing
            frame_pbar = tqdm(total=total_frames, desc=f"  -> {video_path}", leave=False)
            
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                # Inference
                start_time = time.time()
                results = model.predict(frame, imgsz=img_size, conf=conf_threshold, verbose=False)
                end_time = time.time()
                
                total_inference_time += (end_time - start_time)
                processed_frames += 1
                
                # Check for "accident" class in detections
                # Assuming class '0' or the specific index for 'accident' from your yaml
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        label = model.names[class_id].lower()
                        
                        if "accident" in label and first_accident_frame is None:
                            first_accident_frame = processed_frames
                
                frame_pbar.update(1)
            
            cap.release()
            frame_pbar.close()
            
            # Calculate metrics
            avg_fps = processed_frames / total_inference_time if total_inference_time > 0 else 0
            
            results_list.append({
                "model_name": model_name,
                "video_file": video_path,
                "avg_fps": round(avg_fps, 2),
                "avg_latency_ms": round((total_inference_time / processed_frames) * 1000, 2) if processed_frames > 0 else 0,
                "detected_first_frame": first_accident_frame if first_accident_frame else "Not Detected",
                "total_frames": total_frames
            })

    # Save to CSV
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_csv, index=False)
        print(f"\n[INFO] Video testing completed. Results saved to {output_csv}")
        print("Note: Precision, Recall, and F1 must be calculated manually by comparing 'detected_first_frame' with your ground truth labels.")

if __name__ == "__main__":
    run_video_inference()