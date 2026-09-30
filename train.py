from ultralytics import YOLO

def main():
    for modelname in ["yolo12n", "yolo12s", "yolo12m", "yolov8n", "yolov8s", "yolov8m", "yolov5n", "yolov5s", "yolov5m"]:
        model = YOLO(f"{modelname}.pt")  # load a pretrained model (recommended for training)
        model.train(data="accident.yaml", epochs=300, imgsz=640, device=0, workers=2, patience=50, project="saved/accident", name=modelname, exist_ok=True)
    
if __name__ == '__main__':
    main()