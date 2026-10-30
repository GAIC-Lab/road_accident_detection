"""Check whether train/val split lists share identical images or accident events.

Outputs (in --out_dir):
  args.json                  all argparse parameters of this run
  name_overlap.csv           filenames present in both lists, with MD5 comparison
  clip_accident_overlap.csv  clips with frames in both subsets, and whether class 0 is labelled in each
  summary.txt                counts printed at the end of the run
"""
import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser(description="Train/val split leakage check")
    p.add_argument("--dataset_root", default="dataset", help="dataset root containing images/ and labels/")
    p.add_argument("--train_list", default="train.txt")
    p.add_argument("--val_list", default="val.txt")
    p.add_argument("--train_subdir", default="train", help="subfolder name under images/ and labels/")
    p.add_argument("--val_subdir", default="val")
    p.add_argument("--accident_class", type=int, default=0)
    p.add_argument("--out_dir", default="outputs/split_check")
    return p.parse_args()


def read_list(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def md5(path):
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_classes(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {int(float(line.split()[0])) for line in f if line.strip()}


def clip_id(name):
    return name.split("_frame_")[0]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train, val = read_list(args.train_list), read_list(args.val_list)
    img = lambda sub, n: os.path.join(args.dataset_root, "images", sub, n)
    lbl = lambda sub, n: os.path.join(args.dataset_root, "labels", sub, os.path.splitext(n)[0] + ".txt")

    # 1) identical filenames: same file or a name collision?
    shared_names = sorted(set(train) & set(val))
    name_rows, n_identical, n_missing = [], 0, 0
    for n in shared_names:
        h_t, h_v = md5(img(args.train_subdir, n)), md5(img(args.val_subdir, n))
        status = "missing" if h_t is None or h_v is None else ("identical" if h_t == h_v else "different")
        n_identical += status == "identical"
        n_missing += status == "missing"
        name_rows.append({"filename": n, "status": status, "md5_train": h_t, "md5_val": h_v})
    with open(os.path.join(args.out_dir, "name_overlap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "status", "md5_train", "md5_val"])
        w.writeheader()
        w.writerows(name_rows)

    # 2) clips present in both subsets: is the accident class labelled in both?
    by_clip = defaultdict(lambda: {"train": [], "val": []})
    for n in train:
        by_clip[clip_id(n)]["train"].append(n)
    for n in val:
        by_clip[clip_id(n)]["val"].append(n)

    def has_accident(sub, names):
        return any(args.accident_class in (label_classes(lbl(sub, n)) or set()) for n in names)

    clip_rows, n_conflict = [], 0
    for c, d in sorted(by_clip.items()):
        if not d["train"] or not d["val"]:
            continue
        acc_t, acc_v = has_accident(args.train_subdir, d["train"]), has_accident(args.val_subdir, d["val"])
        n_conflict += acc_t and acc_v
        clip_rows.append({"clip": c, "n_train": len(d["train"]), "n_val": len(d["val"]),
                          "accident_in_train": acc_t, "accident_in_val": acc_v,
                          "accident_in_both": acc_t and acc_v})
    with open(os.path.join(args.out_dir, "clip_accident_overlap.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(clip_rows[0].keys()) if clip_rows else ["clip"])
        w.writeheader()
        w.writerows(clip_rows)

    summary = (
        f"train images: {len(train)}, val images: {len(val)}\n"
        f"filenames in both lists: {len(shared_names)}\n"
        f"  byte-identical images: {n_identical}\n"
        f"  different images (name collision): {len(shared_names) - n_identical - n_missing}\n"
        f"  file missing in one subset: {n_missing}\n"
        f"clips with frames in both subsets: {len(clip_rows)}\n"
        f"  clips with accident labelled in both subsets: {n_conflict}\n"
    )
    print(summary)
    with open(os.path.join(args.out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
