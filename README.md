# Real-Time Traffic Accident Detection via AIoT and Edge-Based YOLOv12

This repository contains the official code, configuration files, and reproducibility package for the paper: **"Real-Time Traffic Accident Detection via AIoT and Edge-Based YOLOv12"**.

This project benchmarks standardized YOLO architectures (YOLOv5, YOLOv8, YOLOv12) for edge-native deployment in smart cities. It introduces the **"Initial Trigger Fidelity"** metric to evaluate how reliably a model issues its *first* alert in a continuous video stream without triggering premature false alarms.

## 📊 Dataset Availability & Data Splitting
To ensure reproducibility without hosting large files on GitHub, the custom dual-source accident dataset (images and videos) is hosted publicly on Zenodo.

*   **Dataset Link (Zenodo):** [https://doi.org/10.5281/zenodo.21819374](https://doi.org/10.5281/zenodo.21819374)

### Setup Instructions:
1. Download the dataset from Zenodo.
2. Extract the contents into the `dataset/` folder in this repository. 
3. **Data Splitting:** To prove our "event-level separation" (ensuring no data leakage of the same physical crash between subsets), the exact static lists of images used for training and validation are provided in this repository as `train.txt` and `val.txt`.

The expected directory structure should look like this:
```text
├── dataset/
│   ├── images/
│   ├── labels/
│   ├── video/
│   └── accident.yaml
├── train.txt      <-- (List of training images)
├── val.txt        <-- (List of validation images)
├── train.py
├── eval.py
└── ... (other scripts)
```

## ⚙️ Installation

This project utilizes the `ultralytics` framework. To install the required dependencies, run:

```bash
pip install ultralytics pandas opencv-python tqdm
```

## 🚀 Usage & Scripts

This repository contains several scripts to train models, evaluate static accuracy, and benchmark temporal/video performance.

### 1. Training (`train.py`)
Trains YOLOv5, YOLOv8, and YOLOv12 (Nano, Small, Medium) using the standardized configuration defined in `accident.yaml`.
```bash
python train.py
```

### 2. Static Evaluation (`eval.py`)
Evaluates the trained models on the validation set to generate static metrics (mAP@0.5, Precision, Recall, F1-Score). Results are saved to `experiment_results.csv`.
```bash
python eval.py
```

### 3. Temporal Evaluation & Threshold Sensitivity (`detect_first_accident.py`)
This is the core script used to calculate the **Initial Trigger Fidelity** and **Detection Latency** metrics described in the paper. It scans the 65 continuous accident videos frame-by-frame across multiple confidence thresholds (0.3, 0.5, 0.7) and logs the exact frame of the first triggered alert.
```bash
python detect_first_accident.py
```
*Outputs: `first_detection_results.csv` and bounding box captures in the `detection_captures/` folder.*

### 4. Verification Filtering (`detected_first_accident_filter.py`)
A utility script used after manual verification to separate True Positives (Valid Triggers) from False Positives (Premature False Starts), generating the final `verified_detection_results.csv` used for Table 4 in the paper.
```bash
python detected_first_accident_filter.py
```

### 5. AIoT Speed Benchmarking (`interface.py`)
Simulates edge deployment by measuring the continuous video inference throughput (Average FPS and Latency in milliseconds) for each architecture.
```bash
python interface.py
```

### Utility Scripts
*   `extract_frame.py`: Extracts sparse frames from raw dashcam/CCTV video files at a specified interval.
*   `labelcount.py`: Audits the dataset to provide class distribution statistics (Accident vs. Car vs. Motorcycle).

## 📝 Citation
If you find this code or our dataset useful in your research, please consider citing our paper:
*(Citation details will be updated upon publication)*