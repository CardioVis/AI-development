"""
Chuẩn bị data CardioVis cho FixMatch:
- Frames: D:\\Khanh\\CardioVis\\frames (frame_01044, frame_01045, ... — bắt đầu từ 01044)
- Masks: D:\\Khanh\\CardioVis\\label (task-1-annotation-...-tag-..., ...)
- Tự động matching: task K -> frame 5 chữ số (FRAME_OFFSET + K), task 1 -> frame_01044.
- Mỗi task có 3 mask (Epicardial, Pericardium, Phrenic nerve) -> gộp thành 1 mask 4 lớp (0=nền, 1=Epicardial, 2=Pericardium, 3=Phrenic nerve).
"""
import os
import re
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

FRAMES_DIR = r"D:\Khanh\CardioVis\frames"
# Thử thêm frames_for_label nếu frames trống hoặc không tìm thấy ảnh
FRAMES_FALLBACK = os.path.join(os.path.dirname(FRAMES_DIR) or ".", "frames_for_label")
LABEL_DIR = r"D:\Khanh\CardioVis\label"
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "data_processed", "cardio", "train")
# Frame bắt đầu từ 01044: task 1 -> 01044, task 2 -> 01045 (số = 1043 + task_id, format 5 chữ số)
FRAME_OFFSET = 1043
FRAME_ID_DIGITS = 5  # frame_01044, frame_01045, ...

# 4 lớp: 0=nền, 1=Epicardial, 2=Pericardium, 3=Phrenic nerve
NUM_CLASSES = 4
TAG_TO_CLASS = {
    "epicardial adipose tissue": 1,
    "pericardium": 2,
    "phrenic nerve": 3,
}

def parse_task_and_tag(filename):
    """
    Parse từ tên dạng: task-1-annotation-1-by-1-tag-Epicardial adipose tissue-0.png
    Trả về (task_id, class_idx) hoặc None nếu không map được tag.
    """
    stem = Path(filename).stem
    m = re.match(r"task-(\d+)-annotation-[^-]+-by-1-tag-(.+)-0", stem, re.I)
    if not m:
        return None
    task_id = int(m.group(1))
    tag = m.group(2).strip().lower()
    class_idx = TAG_TO_CLASS.get(tag)
    if class_idx is None:
        for key in TAG_TO_CLASS:
            if key in tag or tag in key:
                return task_id, TAG_TO_CLASS[key]
    return (task_id, class_idx) if class_idx is not None else None


def get_frame_id_from_filename(name):
    """Lấy frame id từ tên file (bỏ extension, bỏ prefix 'frame_' nếu có)."""
    stem = Path(name).stem
    if stem.startswith("frame_"):
        return stem[6:]  # frame_01044 -> 01044
    return stem


def find_file_by_id(folder, file_id, extensions=(".jpg", ".jpeg", ".png", ".bmp")):
    """Tìm file trong folder có stem trùng file_id hoặc 'frame_'+file_id."""
    for ext in extensions:
        for name in (f"{file_id}{ext}", f"frame_{file_id}{ext}"):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path
    return None


def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    labeled_data_dir = os.path.join(OUTPUT_BASE, "labeled_data")
    unlabeled_data_dir = os.path.join(OUTPUT_BASE, "unlabeled_data")
    os.makedirs(os.path.join(labeled_data_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(labeled_data_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(unlabeled_data_dir, "images"), exist_ok=True)

    # Gom mask theo task_id. Mỗi task có 3 file: Epicardial, Pericardium, Phrenic nerve.
    # task_id -> [(class_idx, path), ...]
    by_task = defaultdict(list)
    for f in os.listdir(LABEL_DIR):
        path = os.path.join(LABEL_DIR, f)
        if not os.path.isfile(path) or not f.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
            continue
        parsed = parse_task_and_tag(f)
        if parsed is None:
            continue
        task_id, class_idx = parsed
        by_task[task_id].append((class_idx, path))

    # Danh sách frame files: thử frames/ rồi frames_for_label/
    frame_files = {}
    for folder in [FRAMES_DIR, FRAMES_FALLBACK]:
        if not os.path.isdir(folder):
            continue
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                fid = get_frame_id_from_filename(f)
                if fid not in frame_files:
                    frame_files[fid] = path
    if not frame_files:
        print("Không tìm thấy ảnh trong", FRAMES_DIR, "hoặc", FRAMES_FALLBACK)
        return
    print("Đã quét", len(frame_files), "frame từ", FRAMES_DIR, "và/hoặc", FRAMES_FALLBACK)

    labeled_names = []
    for task_id in sorted(by_task.keys()):
        masks_for_task = by_task[task_id]
        if not masks_for_task:
            continue
        # Task K -> frame 01044, 01045, ... (5 chữ số)
        frame_number = FRAME_OFFSET + task_id
        image_id = str(frame_number).zfill(FRAME_ID_DIGITS)  # 01044, 01045, ...
        img_path = frame_files.get(image_id) or frame_files.get(str(frame_number))
        if not img_path:
            for folder in [FRAMES_DIR, FRAMES_FALLBACK]:
                img_path = find_file_by_id(folder, image_id)
                if img_path:
                    break
        if not img_path:
            print("Không tìm thấy ảnh cho task", task_id, "-> frame", image_id)
            continue

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print("Đọc ảnh lỗi:", img_path)
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        # Gộp 3 mask thành 1: 0=nền, 1=Epicardial, 2=Pericardium, 3=Phrenic nerve (4 lớp)
        merged = np.zeros((h, w), dtype=np.uint8)
        for class_idx, mask_path in masks_for_task:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            merged[m > 0] = class_idx

        sample_name = f"{image_id}.npy"
        np.save(os.path.join(labeled_data_dir, "images", sample_name), img)
        np.save(os.path.join(labeled_data_dir, "labels", sample_name), merged)
        labeled_names.append(sample_name)

    labeled_names = sorted(labeled_names)
    if not labeled_names:
        print("Không có cặp labeled nào. Kiểm tra đường dẫn và FRAME_OFFSET (task K -> frame {}+K).".format(FRAME_OFFSET))
        return

    # Chia 5 fold cho labeled
    n = len(labeled_names)
    fold_size = max(1, n // 5)
    for k in range(1, 6):
        start = (k - 1) * fold_size
        end = n if k == 5 else k * fold_size
        fold = labeled_names[start:end]
        with open(os.path.join(labeled_data_dir, f"fold_label_{k}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(fold))

    # Unlabeled: tất cả frame chưa nằm trong labeled
    labeled_ids = {Path(s).stem for s in labeled_names}
    unlabeled_names = []
    for fid, path in sorted(frame_files.items()):
        if fid in labeled_ids:
            continue
        sample_name = f"{fid}.npy"
        img = cv2.imread(path)
        if img is None:
            continue
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        np.save(os.path.join(unlabeled_data_dir, "images", sample_name), img)
        unlabeled_names.append(sample_name)

    unlabeled_names = sorted(unlabeled_names)
    with open(os.path.join(unlabeled_data_dir, "unlabeled.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(unlabeled_names))

    print("Done.")
    print("Labeled:", len(labeled_names), "samples, 5 folds")
    print("Unlabeled:", len(unlabeled_names), "samples")
    print("Labeled data:", labeled_data_dir)
    print("Unlabeled data:", unlabeled_data_dir)


if __name__ == "__main__":
    main()
