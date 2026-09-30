# evaluator.py
from ultralytics import YOLO
from tqdm import tqdm
import pandas as pd
import os




MODEL_NAMES = [
    "yolov5n", "yolov5s", "yolov5m", 
    "yolov8n", "yolov8s", "yolov8m", 
    "yolo12n", "yolo12s", "yolo12m"
]

# Path pattern for your saved models
MODEL_PATH_TEMPLATE = "saved/accident/{model}/weights/best.pt"

# Dataset configuration
DATA_YAML = "accident.yaml"

# Output settings
RESULTS_CSV = "experiment_results.csv"
IMG_SIZE = 640

def save_results_to_csv(data_list, output_path):
    """
    Saves or appends a list of result dictionaries to a CSV file.
    """
    df = pd.DataFrame(data_list)
    
    # If file exists, append without header; otherwise create new
    if not os.path.isfile(output_path):
        df.to_csv(output_path, index=False)
    else:
        df.to_csv(output_path, mode='a', header=False, index=False)
        
    print(f"\n[INFO] Results saved to {output_path}")
    
def run_evaluation():
    all_results = []
    
    # Using tqdm for the model loop as per instructions
    for model_name in tqdm(MODEL_NAMES, desc="Evaluating Models"):
        model_path = MODEL_PATH_TEMPLATE.format(model=model_name)
        
        try:
            # Load the best weights
            model = YOLO(model_path)
            
            # Run validation
            # We use the validation set defined in accident.yaml
            metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, plots=False)
            
            # Extracting metrics from the results object
            # mAP50 is used as the 'accuracy' metric for object detection
            mAP50 = metrics.results_dict['metrics/mAP50(B)']
            precision = metrics.results_dict['metrics/precision(B)']
            recall = metrics.results_dict['metrics/recall(B)']
            
            # Calculate F1-Score (Harmonic mean of precision and recall)
            if (precision + recall) > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            # Prepare the row for CSV
            result_entry = {
                "modelname": model_name,
                "mAP50": round(mAP50, 4),
                "mAP50-95": round(metrics.results_dict['metrics/mAP50-95(B)'], 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "inference_speed_ms": round(metrics.speed['inference'], 2)
            }
            
            all_results.append(result_entry)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to evaluate {model_name}: {e}")

    # Save all gathered results to CSV
    if all_results:
        save_results_to_csv(all_results, RESULTS_CSV)

if __name__ == "__main__":
    run_evaluation()