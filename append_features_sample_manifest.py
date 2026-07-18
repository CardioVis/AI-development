#!/usr/bin/env python3
"""Append Features_timestamps sample rows to existing patient manifests.

Uses:
  - Prefer stage_outputs/reports/features_frame_index_map.csv if present (after Colab)
  - Else assume exact 10s clips at EXTRACT_FPS=20 => 200 frames/clip

Keeps existing patient train/test splits frozen; assigns 80/20 (seed=42) only for new patients.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

EXTRACT_FPS = 20
SAMPLE_EVERY_N = 20
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parent
TMP = ROOT / "tmp"
SAMPLES_LOCAL = ROOT / "stage_outputs" / "frame_samples_1_per_20"
REPORTS = ROOT / "stage_outputs" / "reports"
# Also accept Drive-synced reports under Cardiovis path if present
REPORTS_CANDIDATES = [
    REPORTS,
    TMP,
]

# Assumed continuation of Label Studio IDs
LS_NEXT = {
    # Actual LS project IDs after upload.
    "dat_kim_goc": 5956,  # frame 6921
    "rach_nhi": 6006,  # frame 5981
}

# Fallback last full-frame index before Features batch
LAST_FRAME_BEFORE = {
    "dat_kim_goc": 6917,
    "rach_nhi": 5980,
}

# Planned clips from Features_timestamps.xlsx (MM:SS reinterpret)
# Ordered by patient (stable); pipeline processes Excel row order which matches this.
FEATURE_CLIPS = {
    "dat_kim_goc": [
        {"patient": 34, "start_sec": 255, "end_sec": 265, "video": "2026-02-27_163832_VID002.mp4", "vid": 2},
        {"patient": 35, "start_sec": 308, "end_sec": 318, "video": "2026-03-02_093147_VID002.mp4", "vid": 2},
        {"patient": 39, "start_sec": 306, "end_sec": 316, "video": "2026-05-30_125700_VID001.mp4", "vid": 1},
        {"patient": 40, "start_sec": 451, "end_sec": 461, "video": "2026-04-09_160139_VID002.mp4", "vid": 2},
        {"patient": 43, "start_sec": 97, "end_sec": 107, "video": "2025-12-23_092850_VID004.mp4", "vid": 4},
    ],
    "rach_nhi": [
        {"patient": 33, "start_sec": 5, "end_sec": 15, "video": "2026-07-01_090716_VID001.mp4", "vid": 1},
        {"patient": 34, "start_sec": 165, "end_sec": 175, "video": "2026-02-27_163832_VID003.mp4", "vid": 3},
        {"patient": 35, "start_sec": 153, "end_sec": 163, "video": "2026-03-02_093147_VID003.mp4", "vid": 3},
        {"patient": 36, "start_sec": 6, "end_sec": 16, "video": "2026-07-13_095711_VID001.mp4", "vid": 1},
        {"patient": 37, "start_sec": 537, "end_sec": 547, "video": "2026-05-19_160735_VID002.mp4", "vid": 2},
        {"patient": 38, "start_sec": 0, "end_sec": 10, "video": "2026-07-01_090716_VID001.mp4", "vid": 1},
        {"patient": 39, "start_sec": 230, "end_sec": 240, "video": "2026-05-30_125700_VID002.mp4", "vid": 2},
        {"patient": 40, "start_sec": 447, "end_sec": 457, "video": "2026-04-09_160139_VID003.mp4", "vid": 3},
        {"patient": 43, "start_sec": 283, "end_sec": 293, "video": "2025-12-23_092850_VID004.mp4", "vid": 4},
    ],
}


def seconds_slug(sec: int) -> str:
    minutes, seconds = divmod(int(sec), 60)
    if minutes == 0:
        return str(seconds)
    return f"{minutes}_{seconds:02d}"


def clip_name(stage: str, clip: dict) -> str:
    return (
        f"patient_{clip['patient']:02d}_goc_v{clip['vid']:03d}_{stage}_"
        f"{seconds_slug(clip['start_sec'])}-{seconds_slug(clip['end_sec'])}.mp4"
    )


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def find_frame_map() -> Path | None:
    for base in REPORTS_CANDIDATES:
        p = base / "features_frame_index_map.csv"
        if p.exists():
            return p
    return None


def planned_frame_map() -> list[dict]:
    rows = []
    for stage, clips in FEATURE_CLIPS.items():
        next_idx = LAST_FRAME_BEFORE[stage]
        for clip in clips:
            duration = clip["end_sec"] - clip["start_sec"]
            frame_count = duration * EXTRACT_FPS
            f_start = next_idx + 1
            f_end = next_idx + frame_count
            next_idx = f_end
            sample_idxs = [i for i in range(f_start, f_end + 1) if (i - 1) % SAMPLE_EVERY_N == 0]
            rows.append(
                {
                    "patient": clip["patient"],
                    "stage": stage,
                    "clip_name": clip_name(stage, clip),
                    "start_sec": clip["start_sec"],
                    "end_sec": clip["end_sec"],
                    "frame_start_idx": f_start,
                    "frame_end_idx": f_end,
                    "frame_count": frame_count,
                    "sample_count": len(sample_idxs),
                }
            )
    return rows


def split_new_patients(existing: dict[int, str], new_patients: list[int]) -> dict[int, str]:
    only_new = sorted(set(new_patients) - set(existing))
    if not only_new:
        return {}
    rng = random.Random(RANDOM_SEED)
    shuffled = only_new[:]
    rng.shuffle(shuffled)
    train_count = int(round(len(shuffled) * TRAIN_RATIO))
    if train_count == len(shuffled) and len(shuffled) > 1:
        train_count -= 1
    train_ids = set(shuffled[:train_count])
    return {p: ("train" if p in train_ids else "test") for p in only_new}


def build_sample_rows(frame_map: list[dict], patient_splits: dict[int, str]) -> dict[str, list[dict]]:
    by_stage: dict[str, list[dict]] = {"dat_kim_goc": [], "rach_nhi": []}
    next_ls = dict(LS_NEXT)
    for row in frame_map:
        stage = row["stage"]
        patient = int(row["patient"])
        f_start = int(row["frame_start_idx"])
        f_end = int(row["frame_end_idx"])
        for idx in range(f_start, f_end + 1):
            if (idx - 1) % SAMPLE_EVERY_N != 0:
                continue
            filename = f"{stage}_full_frame_{idx:08d}.jpg"
            ls_id = next_ls[stage]
            next_ls[stage] = ls_id + 1
            # Approximate time within the Features batch timeline is not needed for patient lookup;
            # use clip-local mapping via clip boundaries already known.
            by_stage[stage].append(
                {
                    "label_studio_id": ls_id,
                    "filename": filename,
                    "stage": stage,
                    "frame_idx": idx,
                    "time_sec": round((idx - 1) / EXTRACT_FPS, 3),  # global timeline-ish for full extract history
                    "patient_id": patient,
                    "clip_name": row["clip_name"],
                    "clip_t_start_sec": "",
                    "clip_t_end_sec": "",
                    "split": patient_splits[patient],
                    "relative_path": f"stage_outputs/frame_samples_1_per_20/{stage}/{filename}",
                }
            )
    return by_stage


def main() -> None:
    split_path = TMP / "patient_train_test_split.csv"
    existing_split_rows = load_csv(split_path)
    existing = {int(r["patient_id"]): r["split"] for r in existing_split_rows}

    new_patients = sorted(
        {
            c["patient"]
            for clips in FEATURE_CLIPS.values()
            for c in clips
        }
    )
    new_splits = split_new_patients(existing, new_patients)
    patient_splits = {**existing, **new_splits}

    # Update patient_train_test_split.csv (freeze old rows)
    out_split_rows = existing_split_rows[:]
    existing_ids = {int(r["patient_id"]) for r in existing_split_rows}
    for pid, split in sorted(new_splits.items()):
        if pid in existing_ids:
            continue
        out_split_rows.append(
            {
                "patient_id": pid,
                "split": split,
                "in_dat_kim_goc": int(any(c["patient"] == pid for c in FEATURE_CLIPS["dat_kim_goc"])),
                "in_rach_nhi": int(any(c["patient"] == pid for c in FEATURE_CLIPS["rach_nhi"])),
                "random_seed": RANDOM_SEED,
                "train_ratio": TRAIN_RATIO,
            }
        )
    write_csv(
        split_path,
        out_split_rows,
        ["patient_id", "split", "in_dat_kim_goc", "in_rach_nhi", "random_seed", "train_ratio"],
    )

    map_path = find_frame_map()
    if map_path:
        print(f"Using frame map from Colab run: {map_path}")
        frame_map = load_csv(map_path)
    else:
        print("No features_frame_index_map.csv yet — using planned 10s*20fps assumption")
        frame_map = planned_frame_map()
        write_csv(
            TMP / "features_frame_index_map_planned.csv",
            frame_map,
            [
                "patient",
                "stage",
                "clip_name",
                "start_sec",
                "end_sec",
                "frame_start_idx",
                "frame_end_idx",
                "frame_count",
                "sample_count",
            ],
        )

    sample_rows = build_sample_rows(frame_map, patient_splits)
    fieldnames = [
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
    ]

    for stage in ("dat_kim_goc", "rach_nhi"):
        manifest_path = TMP / f"sample_patient_manifest_{stage}.csv"
        old_rows = load_csv(manifest_path)
        feature_names = {r["filename"] for r in sample_rows[stage]}
        # Drop previous Features append (if re-run), keep original batch rows
        kept = [r for r in old_rows if r["filename"] not in feature_names]
        final_rows = kept + sample_rows[stage]
        write_csv(manifest_path, final_rows, fieldnames)

        # Ensure local sample dir exists
        (SAMPLES_LOCAL / stage).mkdir(parents=True, exist_ok=True)
        if sample_rows[stage]:
            print(
                f"{stage}: old={len(kept)} + new={len(sample_rows[stage])} => {len(final_rows)} "
                f"LS {sample_rows[stage][0]['label_studio_id']}-{sample_rows[stage][-1]['label_studio_id']}"
            )
        else:
            print(f"{stage}: no new samples")

    print("New patient splits:", new_splits)
    print("Updated", split_path)


if __name__ == "__main__":
    main()
