import cv2
import os
from tqdm import tqdm

def create_output_directory(directory_path):
    """Creates the output directory if it does not exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def get_video_properties(cap):
    """Extracts total frame count and fps from the video capture object."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return total_frames, fps

def save_frame(frame, output_folder, frame_number):
    """Saves a single frame to the output folder."""
    filename = os.path.join(output_folder, f"frame_{frame_number:06d}.jpg")
    cv2.imwrite(filename, frame)

def extract_frames_from_video(video_path, output_folder, interval=10):
    """
    Reads an MP4 file and saves a JPG image every 'interval' frames.
    """
    # Verify input
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    create_output_directory(output_folder)

    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get properties for progress bar
    total_frames, fps = get_video_properties(cap)
    print(f"Processing '{video_path}' ({total_frames} total frames, {fps} FPS)...")

    # Main Loop with tqdm
    current_frame = 0
    saved_count = 0
    
    with tqdm(total=total_frames, unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            # Check interval
            if current_frame % interval == 0:
                save_frame(frame, output_folder, current_frame)
                saved_count += 1

            current_frame += 1
            pbar.update(1)

    cap.release()
    print(f"\nExtraction complete. Saved {saved_count} images to '{output_folder}'.")

if __name__ == "__main__":
    # Configuration
    VIDEO_FILE = "dataset/video/1534.mp4"  # Change this to your video path
    OUTPUT_DIR = "extracted_frames"
    FRAME_INTERVAL = 10

    extract_frames_from_video(VIDEO_FILE, OUTPUT_DIR, FRAME_INTERVAL)