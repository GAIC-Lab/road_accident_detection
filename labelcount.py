import os
from collections import Counter
from tqdm import tqdm

def count_classes(label_path, class_names):
    """
    Counts instances of each class in YOLO label files (.txt).
    """
    stats = Counter()
    
    # List all label files
    files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    
    for file in tqdm(files, desc=f"Processing {os.path.basename(label_path)}"):
        with open(os.path.join(label_path, file), 'r') as f:
            for line in f:
                # YOLO format: <class_id> <x_center> <y_center> <width> <height>
                class_id = line.split()[0]
                stats[class_id] += 1
                
    print(f"\nDistribution for {label_path}:")
    for cid, count in stats.items():
        name = class_names.get(int(cid), f"Unknown({cid})")
        print(f" - {name}: {count}")
    return stats

# Configuration
classes = {0: "Accident", 1: "Car", 2: "Motorcycle"}
train_labels = "dataset/labels/train"
val_labels = "dataset/labels/val"

# Execute
train_stats = count_classes(train_labels, classes)
val_stats = count_classes(val_labels, classes)