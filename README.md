# Real-Time Traffic Accident Detection via AIoT and Edge-Based YOLOv12

This repository contains the code, configuration files, and reproducibility materials for the paper **"Real-Time Traffic Accident Detection via AIoT and Edge-Based YOLOv12"**.

The project compares YOLOv5, YOLOv8, and YOLOv12 models as candidate detectors for edge nodes in a tiered AIoT traffic-monitoring architecture. Besides static detection metrics, it evaluates the **Initial Trigger Fidelity**: whether the *first* alert a model raises in a continuous video stream occurs during an actual hazard or prematurely during normal traffic.

> **Note on hardware.** All speed measurements were obtained on an NVIDIA RTX 3090 workstation. They compare the models relative to each other and do not represent absolute performance on edge devices.

## Data

### Training and validation images (Zenodo)

The static image set (frames extracted from public accident videos) and our annotations are available on Zenodo:

* **DOI:** [https://doi.org/10.5281/zenodo.21819374](https://doi.org/10.5281/zenodo.21819374)

| File | Content |
|------|---------|
| `train.txt`, `val.txt` | Exact list of images in each subset |
| `split_mapping.csv` | Source clip, source platform, and collision event of every image |
| `check_split.py` | Verifies that no collision event appears in both subsets |

Images are split at the level of collision events: all frames of one collision are in the same subset. Non-accident frames from the same source clip or camera may appear in both subsets, so validation results reflect **in-domain** performance (see Section 3.2.1 of the paper).

### Source videos of the training images

The training and validation frames were extracted from publicly available accident videos on the following YouTube channels:

* WoWtchout: [https://www.youtube.com/@WoWtchout](https://www.youtube.com/@WoWtchout)
* Choques de Autos Mx: [https://www.youtube.com/@ChoquesdeAutosMx](https://www.youtube.com/@ChoquesdeAutosMx)

The source clip of each image is listed in `split_mapping.csv`.

### Test videos (not redistributed)

The temporal evaluation uses 64 videos from the public **Road Traffic Accidents** dataset (Sabry, 2021). We do not redistribute these videos. To reproduce the evaluation:

1. Download the dataset from its original source: `[ADD LINK]`.
2. Copy the videos listed in `test_videos.csv` into `dataset/video/`.
3. Verify the files against the MD5 checksums in `test_videos.csv`.

### License

* **Annotations** (label files, `split_mapping.csv`): CC BY 4.0.
* **Extracted frames:** remain the property of the owners of the original videos (the WoWtchout and Choques de Autos Mx channels linked above, and the YouTube clips listed in `split_mapping.csv`). They are provided for non-commercial research use only.
* **Code:** `[ADD LICENSE, e.g. MIT]`.

If you own footage included in this dataset and want it removed, please open an issue or contact the authors.

## Directory structure

```text
├── accident.yaml              # dataset configuration used by all scripts
├── dataset/
│   ├── images/{train,val}/
│   ├── labels/{train,val}/
│   └── video/                 # place the 64 test videos here
├── train.txt / val.txt        # image lists per subset
├── split_mapping.csv          # image → clip → collision event
├── test_videos.csv            # test video list with MD5 checksums
├── runs/<model>/args.yaml     # argument file recorded for each training run
├── results/                   # result files behind Tables 2–5
├── requirements.txt
└── *.py                       # scripts described below
```

## Installation

```bash
pip install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Scripts

| Step | Script | Output | Paper |
|------|--------|--------|-------|
| 1. Train all models | `train.py` | `saved/accident/<model>/` | Section 3.3 |
| 2. Static evaluation | `eval.py` | `experiment_results.csv` | Table 2 |
| 3. Inference speed | `interface.py` | `video_test_results.csv` | Table 3 |
| 4. First detection per video and threshold | `detect_first_accident.py` | `first_detection_results.csv`, `detection_captures/` | Tables 4–5 |
| 5. Merge manual verification | `detected_first_accident_filter.py` | `verified_detection_results.csv` | Table 4 |
| Split check | `check_split.py` | `outputs/split_check/` | Section 3.2.1 |

**Notes**

* `train.py` uses the Ultralytics default hyperparameters; only the number of epochs (300), early-stopping patience (50), and input size (640) are set. The recorded settings for each run are in `runs/<model>/args.yaml`.
* Small (`s`) variants are also trained and evaluated by the scripts, but the paper reports only the Nano (`n`) and Medium (`m`) variants.
* `interface.py` measures throughput on the local GPU at batch size 1. Timing covers each detector call, including pre- and post-processing.
* **Manual verification (between steps 4 and 5).** See the next section.

## Evaluating the first detections (Valid First Trigger vs. False Start)

The temporal results in Tables 4–5 are obtained by manually verifying the first alert of each model in each video:

1. **Run the detection.** `detect_first_accident.py` processes every video at each threshold (0.3, 0.5, 0.7). For each model, threshold, and video, the frame of the first alert is saved to `detection_captures/` as `<video>_<model>_<threshold>.jpg`, with the detected Accident box drawn in red. The frame number and confidence are written to `first_detection_results.csv`.
2. **Duplicate the captures.** Copy all images from `detection_captures/` to a new folder `detection_captures_removed/`. Keep `detection_captures/` unchanged as the complete record.
3. **Remove non-accidents manually.** Inspect each image in `detection_captures_removed/` and delete it if the first alert is not a valid accident detection, i.e., if it occurs during normal traffic before the hazard, or if the box marks vehicles not involved in the event. The images that remain are Valid First Triggers.
4. **Check that the results are up to date.** Before running the filter script, make sure that `first_detection_results.csv` and both image folders come from the same, latest run of `detect_first_accident.py`. The filter script matches rows by image file name, so captures left over from an earlier run would be counted with the wrong labels.
5. **Label the results.** Run `detected_first_accident_filter.py`. It compares the two folders and writes `verified_detection_results.csv`, labelling each row as `True Positive` (Valid First Trigger, image kept), `False Positive` (False Start, image removed), or `No Detection` (Missed Detection).

The verified results used in the paper are provided in `results/verified_detection_results.csv`.

## Citation

*(Citation details will be added upon publication.)*
