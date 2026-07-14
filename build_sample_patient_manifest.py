#!/usr/bin/env python3
"""Map frame_samples_1_per_20 images to patients and assign train/test split."""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

EXTRACT_FPS = 20
SAMPLE_EVERY_N = 20
TRAIN_RATIO = 0.8
RANDOM_SEED = 42
STAGES = ("dat_kim_goc", "rach_nhi")

# Label Studio task ID ranges (upload order == filename sort order).
# Cardioplegia needle placement -> dat_kim_goc
# atrial incision              -> rach_nhi
LS_TASK_ID_RANGE: dict[str, tuple[int, int]] = {
    "dat_kim_goc": (5087, 5432),
    "rach_nhi": (5433, 5731),
}

ROOT = Path(__file__).resolve().parent
TASK_RESULTS = ROOT / "tmp" / "task_results.csv"
FRAME_EXTRACT_REPORT = ROOT / "tmp" / "frame_extract_report.csv"
SAMPLE_REPORT = ROOT / "tmp" / "sample_report.csv"
OUTPUT_DIR = ROOT / "tmp"
DEFAULT_SAMPLES_ROOT = Path(
    os.getenv(
        "SAMPLES_ROOT",
        "/content/drive/MyDrive/Cardiovis-related/stage_outputs/frame_samples_1_per_20",
    )
)

# Clip merge order from Drive (sorted alphabetically, same as pipeline).
CLIP_ORDER: dict[str, list[str]] = {
    "dat_kim_goc": [
        "patient_01_merged_dat_kim_goc_36_51-37_06.mp4",
        "patient_02_merged_dat_kim_goc_16_41-16_56.mp4",
        "patient_04_merged_dat_kim_goc_12_45-13_00.mp4",
        "patient_05_merged_dat_kim_goc_7_46-8_01.mp4",
        "patient_06_merged_dat_kim_goc_10_30-10_43.mp4",
        "patient_07_goc_v002_dat_kim_goc_0_02-0_12.mp4",
        "patient_07_goc_v002_dat_kim_goc_0_02-0_15.mp4",
        "patient_08_merged_dat_kim_goc_11_33-11_48.mp4",
        "patient_09_goc_v003_dat_kim_goc_0-0_13.mp4",
        "patient_10_merged_dat_kim_goc_10_35-10_41.mp4",
        "patient_11_merged_dat_kim_goc_23_15-23_30.mp4",
        "patient_12_merged_dat_kim_goc_9_16-9_30.mp4",
        "patient_13_goc_v001_dat_kim_goc_0_05-0_21.mp4",
        "patient_14_merged_dat_kim_goc_14_06-14_21.mp4",
        "patient_16_merged_dat_kim_goc_19_45-19_58.mp4",
        "patient_17_goc_v014_dat_kim_goc_0_27-0_38.mp4",
        "patient_18_merged_dat_kim_goc_12_25-12_37.mp4",
        "patient_19_merged_dat_kim_goc_0_17-0_32.mp4",
        "patient_20_goc_v002_dat_kim_goc_0-0_10.mp4",
        "patient_21_merged_dat_kim_goc_11_05-11_20.mp4",
        "patient_22_goc_v006_dat_kim_goc_0-0_10.mp4",
        "patient_23_merged_dat_kim_goc_10_56-11_06.mp4",
        "patient_24_merged_dat_kim_goc_13_09-13_16.mp4",
        "patient_26_merged_dat_kim_goc_13_53-14_08.mp4",
        "patient_27_merged_dat_kim_goc_29_08-29_20.mp4",
        "patient_29_merged_dat_kim_goc_1_11-1_26.mp4",
        "patient_30_goc_v004_dat_kim_goc_4_50-5_00.mp4",
    ],
    "rach_nhi": [
        "patient_01_merged_rach_nhi_56_38-56_53.mp4",
        "patient_02_merged_rach_nhi_23_22-23_37.mp4",
        "patient_04_merged_rach_nhi_20_02-20_12.mp4",
        "patient_05_merged_rach_nhi_14_33-14_43.mp4",
        "patient_06_merged_rach_nhi_19_07-19_22.mp4",
        "patient_07_goc_v002_rach_nhi_6_26-6_36.mp4",
        "patient_08_merged_rach_nhi_19_57-20_11.mp4",
        "patient_09_goc_v003_rach_nhi_9_06-9_13.mp4",
        "patient_10_merged_rach_nhi_17_40-17_48.mp4",
        "patient_11_merged_rach_nhi_46_08-46_18.mp4",
        "patient_12_merged_rach_nhi_22_36-22_46.mp4",
        "patient_13_goc_v002_rach_nhi_0_15-0_24.mp4",
        "patient_14_merged_rach_nhi_21_15-21_25.mp4",
        "patient_16_merged_rach_nhi_29_48-30_01.mp4",
        "patient_18_merged_rach_nhi_27_08-27_23.mp4",
        "patient_19_merged_rach_nhi_12_50-13_04.mp4",
        "patient_20_goc_v002_rach_nhi_4_50-5_00.mp4",
        "patient_21_merged_rach_nhi_23_45-24_00.mp4",
        "patient_23_merged_rach_nhi_22_09-22_18.mp4",
        "patient_24_merged_rach_nhi_22_03-22_11.mp4",
        "patient_26_merged_rach_nhi_23_07-23_22.mp4",
        "patient_27_merged_rach_nhi_40_47-41_02.mp4",
        "patient_29_merged_rach_nhi_11_14-11_24.mp4",
        "patient_30_goc_v004_rach_nhi_6_22-6_33.mp4",
        "patient_33_goc_v001_rach_nhi_5_00-5_09.mp4",
    ],
}


def parse_doc_time(token: str) -> int:
    token = token.strip().replace(" ", "").replace(",", ".")
    if token.count(".") == 2:
        h, m, s = map(int, token.split("."))
        return h * 3600 + m * 60 + s
    if token.count(".") == 1:
        a, b = token.split(".")
        return int(a) * 60 + int(b)
    return int(token)


def parse_patient_from_clip(name: str) -> int:
    match = re.search(r"patient_(\d+)", name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot parse patient from clip name: {name}")
    return int(match.group(1))


def parse_time_range_from_clip(name: str) -> tuple[int, int]:
    match = re.search(r"_(\d+(?:_\d+)*)-(\d+(?:_\d+)*)\.mp4$", name)
    if not match:
        raise ValueError(f"Cannot parse time range from clip name: {name}")
    start_token = match.group(1).replace("_", ".")
    end_token = match.group(2).replace("_", ".")
    return parse_doc_time(start_token), parse_doc_time(end_token)


def load_task_durations() -> dict[str, float]:
    durations: dict[str, float] = {}
    with TASK_RESULTS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clip_name = Path(row["clip_file_path"]).name
            durations[clip_name] = float(int(row["end_sec"]) - int(row["start_sec"]))
    return durations


def load_extracted_frame_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with FRAME_EXTRACT_REPORT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["stage"]] = int(row["frame_count"])
    return counts


def load_expected_sample_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with SAMPLE_REPORT.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            counts[row["stage"]] = int(row["sample_count"])
    return counts


def clip_duration(clip_name: str, task_durations: dict[str, float]) -> float:
    if clip_name in task_durations:
        return task_durations[clip_name]
    start_sec, end_sec = parse_time_range_from_clip(clip_name)
    return float(end_sec - start_sec)


def build_boundaries(stage: str, task_durations: dict[str, float]) -> list[dict[str, object]]:
    boundaries: list[dict[str, object]] = []
    offset = 0.0
    for clip_name in CLIP_ORDER[stage]:
        duration = clip_duration(clip_name, task_durations)
        boundaries.append(
            {
                "patient_id": parse_patient_from_clip(clip_name),
                "clip_name": clip_name,
                "t_start_sec": offset,
                "t_end_sec": offset + duration,
                "duration_sec": duration,
            }
        )
        offset += duration
    return boundaries


def scale_boundaries_to_extracted_timeline(
    boundaries: list[dict[str, object]],
    extracted_frame_count: int,
) -> list[dict[str, object]]:
    if not boundaries:
        return boundaries
    doc_total = float(boundaries[-1]["t_end_sec"])
    actual_total = extracted_frame_count / EXTRACT_FPS
    if doc_total <= 0:
        return boundaries
    scale = actual_total / doc_total
    scaled: list[dict[str, object]] = []
    for row in boundaries:
        scaled.append(
            {
                **row,
                "t_start_sec": float(row["t_start_sec"]) * scale,
                "t_end_sec": float(row["t_end_sec"]) * scale,
                "duration_sec": float(row["duration_sec"]) * scale,
            }
        )
    return scaled


def lookup_patient(time_sec: float, boundaries: list[dict[str, object]]) -> dict[str, object]:
    for row in boundaries:
        if row["t_start_sec"] <= time_sec < row["t_end_sec"]:
            return row
    if boundaries and time_sec >= float(boundaries[-1]["t_start_sec"]):
        return boundaries[-1]
    raise ValueError(f"Time {time_sec}s is outside merged video boundaries")


def parse_frame_index(filename: str) -> int:
    match = re.search(r"_frame_(\d+)\.jpg$", filename)
    if not match:
        raise ValueError(f"Cannot parse frame index from filename: {filename}")
    return int(match.group(1))


def generate_sample_filenames(stage: str, extracted_frame_count: int) -> list[str]:
    # Mirrors sample_every_n(): sorted(full_frames)[::20] with 1-based ffmpeg indices.
    prefix = f"{stage}_full"
    frame_indices = list(range(1, extracted_frame_count + 1, SAMPLE_EVERY_N))
    return [f"{prefix}_frame_{idx:08d}.jpg" for idx in frame_indices]


def discover_sample_filenames(stage: str, extracted_frame_count: int, samples_root: Path) -> list[str]:
    local_dir = samples_root / stage
    if local_dir.is_dir():
        files = sorted(path.name for path in local_dir.glob("*.jpg"))
        if files:
            return files
    return generate_sample_filenames(stage, extracted_frame_count)


def split_patients(patients: list[int]) -> dict[int, str]:
    patients = sorted(set(patients))
    rng = random.Random(RANDOM_SEED)
    shuffled = patients[:]
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * TRAIN_RATIO))
    if train_count == len(shuffled) and len(shuffled) > 1:
        train_count -= 1
    train_ids = set(shuffled[:train_count])
    return {patient: ("train" if patient in train_ids else "test") for patient in patients}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def ls_task_ids_for_stage(stage: str, n_samples: int) -> list[int]:
    start_id, end_id = LS_TASK_ID_RANGE[stage]
    expected = end_id - start_id + 1
    if expected != n_samples:
        raise RuntimeError(
            f"{stage}: Label Studio ID range {start_id}-{end_id} has {expected} IDs, "
            f"but sample count is {n_samples}"
        )
    return list(range(start_id, end_id + 1))


def build_stage_manifest(
    stage: str,
    patient_splits: dict[int, str],
    extracted_frame_count: int,
    samples_root: Path,
) -> list[dict[str, object]]:
    task_durations = load_task_durations()
    boundaries = scale_boundaries_to_extracted_timeline(
        build_boundaries(stage, task_durations),
        extracted_frame_count,
    )
    filenames = discover_sample_filenames(stage, extracted_frame_count, samples_root)
    ls_ids = ls_task_ids_for_stage(stage, len(filenames))
    rows: list[dict[str, object]] = []

    for filename, label_studio_id in zip(filenames, ls_ids):
        frame_idx = parse_frame_index(filename)
        time_sec = (frame_idx - 1) / EXTRACT_FPS
        hit = lookup_patient(time_sec, boundaries)
        patient_id = int(hit["patient_id"])
        rows.append(
            {
                "label_studio_id": label_studio_id,
                "filename": filename,
                "stage": stage,
                "frame_idx": frame_idx,
                "time_sec": round(time_sec, 3),
                "patient_id": patient_id,
                "clip_name": hit["clip_name"],
                "clip_t_start_sec": round(float(hit["t_start_sec"]), 3),
                "clip_t_end_sec": round(float(hit["t_end_sec"]), 3),
                "split": patient_splits[patient_id],
                "relative_path": f"stage_outputs/frame_samples_1_per_20/{stage}/{filename}",
            }
        )
    return rows


def main() -> None:
    extracted_counts = load_extracted_frame_counts()
    expected_sample_counts = load_expected_sample_counts()
    task_durations = load_task_durations()
    samples_root = DEFAULT_SAMPLES_ROOT

    all_patients = sorted(
        {
            parse_patient_from_clip(clip_name)
            for stage in STAGES
            for clip_name in CLIP_ORDER[stage]
        }
    )
    patient_splits = split_patients(all_patients)

    split_rows = [
        {
            "patient_id": patient_id,
            "split": patient_splits[patient_id],
            "in_dat_kim_goc": int(any(parse_patient_from_clip(c) == patient_id for c in CLIP_ORDER["dat_kim_goc"])),
            "in_rach_nhi": int(any(parse_patient_from_clip(c) == patient_id for c in CLIP_ORDER["rach_nhi"])),
            "random_seed": RANDOM_SEED,
            "train_ratio": TRAIN_RATIO,
        }
        for patient_id in all_patients
    ]
    write_csv(
        OUTPUT_DIR / "patient_train_test_split.csv",
        split_rows,
        ["patient_id", "split", "in_dat_kim_goc", "in_rach_nhi", "random_seed", "train_ratio"],
    )

    boundary_rows: list[dict[str, object]] = []
    for stage in STAGES:
        scaled = scale_boundaries_to_extracted_timeline(
            build_boundaries(stage, task_durations),
            extracted_counts[stage],
        )
        for row in scaled:
            boundary_rows.append({"stage": stage, **row})
    write_csv(
        OUTPUT_DIR / "clip_boundaries_for_mapping.csv",
        boundary_rows,
        ["stage", "patient_id", "clip_name", "t_start_sec", "t_end_sec", "duration_sec"],
    )

    for stage in STAGES:
        rows = build_stage_manifest(
            stage,
            patient_splits,
            extracted_counts[stage],
            samples_root,
        )
        expected = expected_sample_counts[stage]
        if len(rows) != expected:
            raise RuntimeError(
                f"{stage}: generated {len(rows)} sample rows, expected {expected} from sample_report.csv"
            )
        write_csv(
            OUTPUT_DIR / f"sample_patient_manifest_{stage}.csv",
            rows,
            [
                "label_studio_id",
                "filename",
                "stage",
                "frame_idx",
                "time_sec",
                "patient_id",
                "clip_name",
                "clip_t_start_sec",
                "clip_t_end_sec",
                "split",
                "relative_path",
            ],
        )

    train_patients = [p for p, s in patient_splits.items() if s == "train"]
    test_patients = [p for p, s in patient_splits.items() if s == "test"]
    print(f"Patients total: {len(all_patients)}")
    print(f"Train patients ({len(train_patients)}): {train_patients}")
    print(f"Test patients ({len(test_patients)}): {test_patients}")
    for stage in STAGES:
        rows = build_stage_manifest(
            stage,
            patient_splits,
            extracted_counts[stage],
            samples_root,
        )
        patients = sorted({int(r["patient_id"]) for r in rows})
        train_n = sum(1 for r in rows if r["split"] == "train")
        test_n = sum(1 for r in rows if r["split"] == "test")
        ls_start = int(rows[0]["label_studio_id"])
        ls_end = int(rows[-1]["label_studio_id"])
        print(
            f"{stage}: {len(rows)} samples, LS IDs {ls_start}-{ls_end}, "
            f"{len(patients)} patients covered -> train={train_n}, test={test_n}"
        )
    print(f"Wrote CSVs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
