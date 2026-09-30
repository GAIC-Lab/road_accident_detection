import pandas as pd
import os

# --- CONFIGURATION ---
INPUT_CSV = "first_detection_results.csv"
OUTPUT_CSV = "verified_detection_results.csv"

# Folder Paths
FOLDER_ALL_CAPTURES = "detection_captures"           # Original folder with all images
FOLDER_VALID_CAPTURES = "detection_captures_removed" # Filtered folder (True Positives only)

def verify_detections():
    # 1. Read the CSV
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    df = pd.read_csv(INPUT_CSV)
    
    # Create lists to store new column data
    verification_status = []
    
    print(f"Processing {len(df)} rows...")

    for index, row in df.iterrows():
        image_file = row['Image File']
        detected = row['Detected']

        # Skip rows where no detection occurred
        if pd.isna(image_file) or not detected:
            verification_status.append("No Detection")
            continue
            
        # Construct full file paths
        path_in_all = os.path.join(FOLDER_ALL_CAPTURES, str(image_file))
        path_in_valid = os.path.join(FOLDER_VALID_CAPTURES, str(image_file))
        
        # Check existence
        exists_in_all = os.path.exists(path_in_all)
        exists_in_valid = os.path.exists(path_in_valid)
        
        # --- LOGIC ---
        if exists_in_valid:
            # If it's in the filtered folder, you manually verified it as correct
            verification_status.append("True Positive")
        elif exists_in_all and not exists_in_valid:
            # If it's in the original folder but NOT in the filtered one, you removed it (False Alarm)
            verification_status.append("False Positive")
        else:
            # If it's in neither (but CSV says detected), the file might be missing or renamed
            verification_status.append("File Missing")

    # 2. Add the new column
    df['First_Frame_Accuracy'] = verification_status

    # 3. Save to new CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("-" * 30)
    print("Verification Complete.")
    print(f"Results saved to: {OUTPUT_CSV}")
    print("-" * 30)
    
    # 4. Show Summary Stats
    print("\nSummary of Results:")
    print(df['First_Frame_Accuracy'].value_counts())

if __name__ == "__main__":
    # Create dummy folders for testing if they don't exist (optional safety check)
    if not os.path.exists(FOLDER_ALL_CAPTURES):
        print(f"Warning: Folder '{FOLDER_ALL_CAPTURES}' does not exist.")
    if not os.path.exists(FOLDER_VALID_CAPTURES):
        print(f"Warning: Folder '{FOLDER_VALID_CAPTURES}' does not exist.")
        
    verify_detections()