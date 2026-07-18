"""
Prepare dat_kim_goc (Label Studio project-74) for FixMatch.

Outputs under training_related/dat_kim_goc/:
  labeled_data/          — train split with annotations (images/labels .npy + fold_label_*.txt)
  test_data/             — test split with annotations
  unlabeled_data/        — baseline unlabeled (all matching train-patient frames)
  unlabeled_data_dense/  — densified unlabeled (every Nth frame; default stride=5)

Example:
  python prepare_cardio_data.py
  python prepare_cardio_data.py --unlabeled-only --dense-stride 5
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from label_studio_converter import brush

ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "training_related" / "project-74-at-2026-07-16-21-44-d9e1b0b9.json"
DEFAULT_MANIFEST = ROOT / "training_related" / "sample_patient_manifest_dat_kim_goc.csv"
DEFAULT_SAMPLES = ROOT / "stage_outputs" / "frame_samples_1_per_20" / "dat_kim_goc"
DEFAULT_FRAMES = ROOT / "stage_outputs" / "frames" / "dat_kim_goc"
DEFAULT_FRAME_MAP = ROOT / "stage_outputs" / "reports" / "features_frame_index_map.csv"
DEFAULT_OUT = ROOT / "training_related" / "dat_kim_goc"
EXCLUDE_IDS_JSON = ROOT / "training_related" / "excluded_label_studio_ids.json"

FRAME_RE = re.compile(r"(\d+)\.(?:jpg|jpeg|png|bmp)$", re.I)

# Patient 1 is the only source of Ascending aorta → keep in train so the model can learn it.
FORCE_TRAIN_PATIENTS = {1}


def expand_excluded_ids(path: Path = EXCLUDE_IDS_JSON) -> set[int]:
    """Build flat set of Label Studio task IDs to drop."""
    ids: set[int] = set()
    if not path.is_file():
        return ids
    data = json.loads(path.read_text(encoding="utf-8"))
    for _group, spec in data.items():
        if _group.startswith("_") or not isinstance(spec, dict):
            continue
        for x in spec.get("ids") or []:
            ids.add(int(x))
        for lo, hi in spec.get("ranges_inclusive") or []:
            ids.update(range(int(lo), int(hi) + 1))
    return ids


def resplit_patients_for_coverage(
    patient_classes: dict[int, set[int]],
    patient_counts: dict[int, int],
    force_train: set[int] | None = None,
    target_test_frac: float = 0.22,
) -> tuple[set[int], set[int]]:
    """Patient-level split: no leakage; maximize train class coverage; grow test coverage.

    - Every FG class that appears in ≥1 patient stays in train when possible.
    - Patients with singleton classes (only that patient has the class) stay in train
      if that class would otherwise vanish from train (e.g. patient 1 / Ascending aorta).
    - Test receives patients that add multi-patient classes until ~target_test_frac frames.
    """
    force_train = set(force_train or FORCE_TRAIN_PATIENTS)
    patients = sorted(patient_classes.keys())
    if not patients:
        return set(), set()

    class_to_patients: dict[int, set[int]] = defaultdict(set)
    all_classes: set[int] = set()
    for pid, cls in patient_classes.items():
        all_classes |= cls
        for c in cls:
            class_to_patients[c].add(pid)

    train = set(patients)
    test: set[int] = set()
    total_frames = sum(patient_counts[p] for p in patients)
    target_test_frames = max(1, int(round(total_frames * target_test_frac)))

    def coverage(pids: set[int]) -> set[int]:
        cov: set[int] = set()
        for p in pids:
            cov |= patient_classes.get(p, set())
        return cov

    def can_move_to_test(pid: int) -> bool:
        if pid in force_train:
            return False
        remaining = train - {pid}
        if not remaining:
            return False
        # Never remove any FG class from train coverage
        return all_classes.issubset(coverage(remaining))

    # Classes that can appear in both splits (≥2 patients)
    uncovered_test = {c for c, ps in class_to_patients.items() if len(ps) >= 2}
    test_frames = 0

    while True:
        best = None
        best_key = None
        for pid in list(train):
            if not can_move_to_test(pid):
                continue
            new_cover = patient_classes[pid] & uncovered_test
            # Prefer covering new test classes, then prefer larger patients
            key = (len(new_cover), patient_counts[pid])
            if best_key is None or key > best_key:
                best_key = key
                best = pid

        if best is None:
            break

        gains_coverage = bool(patient_classes[best] & uncovered_test)
        if test_frames >= target_test_frames and not gains_coverage:
            break

        train.remove(best)
        test.add(best)
        test_frames += patient_counts[best]
        uncovered_test -= patient_classes[best]

        if test_frames >= target_test_frames and not uncovered_test:
            break

    # Ensure force_train
    for pid in force_train:
        if pid in test:
            test.remove(pid)
            train.add(pid)

    assert train.isdisjoint(test)
    return train, test


def stem_from_upload(file_upload: str) -> str:
    """e29a242c-dat_kim_goc_full_frame_00000001.jpg -> dat_kim_goc_full_frame_00000001.jpg"""
    name = Path(file_upload).name
    if "-" in name and name.lower().endswith((".jpg", ".jpeg", ".png")):
        # Label Studio prefixes a uuid before the original filename
        _, rest = name.split("-", 1)
        if rest.startswith("dat_kim_goc") or rest.startswith("rach_nhi"):
            return rest
    return name


def collect_label_names(tasks: list) -> list[str]:
    labels: set[str] = set()
    for task in tasks:
        for ann in task.get("annotations") or []:
            for res in ann.get("result") or []:
                if res.get("type") != "brushlabels":
                    continue
                value = res.get("value") or {}
                for lab in value.get("brushlabels") or []:
                    labels.add(lab.replace("/", "_"))
    return sorted(labels)


def build_class_map(label_names: list[str]) -> dict:
    mapping = {name: i + 1 for i, name in enumerate(label_names)}
    return {
        "background": 0,
        **mapping,
        "_num_classes": len(label_names) + 1,
    }


def decode_brush_mask(res: dict) -> tuple[np.ndarray, str] | None:
    value = res.get("value") or {}
    if value.get("format") != "rle" or "rle" not in value:
        return None
    h = int(value.get("original_height") or res.get("original_height"))
    w = int(value.get("original_width") or res.get("original_width"))
    labels = value.get("brushlabels") or ["mask"]
    label = labels[0].replace("/", "_")
    decoded = brush.decode_rle(value["rle"])
    rgba = np.array(decoded, dtype=np.uint8).reshape((h, w, 4))
    mask = (rgba[:, :, 3] > 0).astype(np.uint8)
    return mask, label


def merge_task_masks(task: dict, class_map: dict) -> np.ndarray | None:
    regions: list[tuple[str, np.ndarray]] = []
    h = w = None
    for ann in task.get("annotations") or []:
        for res in ann.get("result") or []:
            if res.get("type") != "brushlabels":
                continue
            decoded = decode_brush_mask(res)
            if decoded is None:
                continue
            mask, label = decoded
            h, w = mask.shape
            regions.append((label, mask))
    if not regions or h is None:
        return None
    merged = np.zeros((h, w), dtype=np.uint8)
    for label, mask in sorted(regions, key=lambda x: class_map.get(x[0], 0)):
        cid = class_map.get(label)
        if cid is None:
            continue
        merged[mask > 0] = cid
    return merged


def patient_stratified_folds(
    rows: list[tuple[str, int]], n_folds: int = 5
) -> list[list[str]]:
    """rows: (sample_name, patient_id). Distribute patients across folds."""
    by_patient: dict[int, list[str]] = defaultdict(list)
    for name, pid in rows:
        by_patient[int(pid)].append(name)
    patients = sorted(by_patient.keys())
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # Greedy: assign next patient to currently smallest fold
    fold_sizes = [0] * n_folds
    for pid in patients:
        names = sorted(by_patient[pid])
        k = int(np.argmin(fold_sizes))
        folds[k].extend(names)
        fold_sizes[k] += len(names)
    return folds


def train_patient_frame_ranges(
    manifest: pd.DataFrame, frame_map_path: Path
) -> list[tuple[int, int]]:
    """Inclusive (start, end) frame index ranges for train patients only."""
    train_pids = set(manifest.loc[manifest["split"] == "train", "patient_id"].astype(int))
    ranges: list[tuple[int, int]] = []

    if frame_map_path.is_file():
        fmap = pd.read_csv(frame_map_path)
        if "stage" in fmap.columns:
            fmap = fmap[fmap["stage"] == "dat_kim_goc"]
        for _, r in fmap.iterrows():
            pid = int(r["patient"])
            if pid not in train_pids:
                continue
            ranges.append((int(r["frame_start_idx"]), int(r["frame_end_idx"])))

    # Also cover sample-based train clips (frame_idx in manifest) as contiguous spans
    # between min/max frame_idx per train patient (dense frames may use same numbering).
    for pid, g in manifest[manifest["split"] == "train"].groupby("patient_id"):
        lo, hi = int(g["frame_idx"].min()), int(g["frame_idx"].max())
        ranges.append((lo, hi))

    # Merge overlapping
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for a, b in ranges[1:]:
        la, lb = merged[-1]
        if a <= lb + 1:
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a, b))
    return merged


def frame_in_ranges(frame_num: int, ranges: list[tuple[int, int]]) -> bool:
    for a, b in ranges:
        if a <= frame_num <= b:
            return True
    return False


def write_list(path: Path, names: list[str]) -> None:
    path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")


def build_unlabeled_from_frames(
    frames_dir: Path,
    ranges: list[tuple[int, int]],
    out_dir: Path,
    stride: int | None = None,
) -> list[str]:
    """Copy matching train-patient frames to out_dir/images as .npy.

    stride=None → keep every matching frame (baseline).
    stride=N → keep frames whose index satisfies frame_num % N == 0.
    """
    import shutil

    images_dir = out_dir / "images"
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    unlabeled_names: list[str] = []
    if not frames_dir.is_dir():
        print(f"[warn] frames dir missing: {frames_dir}")
        write_list(out_dir / "unlabeled.txt", [])
        return []
    if not ranges:
        print("[warn] no train-patient frame ranges; skipping unlabeled to avoid test leakage")
        write_list(out_dir / "unlabeled.txt", [])
        return []

    for p in sorted(frames_dir.iterdir()):
        if not p.is_file():
            continue
        m = FRAME_RE.search(p.name)
        if not m:
            continue
        fnum = int(m.group(1))
        if not frame_in_ranges(fnum, ranges):
            continue
        if stride is not None and stride > 1 and (fnum % stride) != 0:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        sample_name = p.with_suffix(".npy").name
        np.save(images_dir / sample_name, img.astype(np.uint8))
        unlabeled_names.append(sample_name)

    unlabeled_names = sorted(unlabeled_names)
    write_list(out_dir / "unlabeled.txt", unlabeled_names)
    stride_msg = f"stride={stride}" if stride else "stride=all"
    print(f"Unlabeled → {out_dir.name}: {len(unlabeled_names)} ({stride_msg}; ranges={len(ranges)})")
    return unlabeled_names


def write_class_weights_json(labeled_dir: Path, num_classes: int) -> Path:
    """FG-only 1/sqrt(freq) weights next to labeled_data (for ablation M2)."""
    labels_dir = labeled_dir / "labels"
    counts = np.zeros(num_classes, dtype=np.float64)
    for p in sorted(labels_dir.glob("*.npy")):
        mask = np.load(p)
        if mask.ndim > 2:
            mask = np.argmax(mask, axis=0)
        flat = mask.astype(np.int64).ravel()
        bc = np.bincount(flat, minlength=num_classes)
        counts += bc[:num_classes]
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(1, num_classes):
        weights[c] = 1.0 / np.sqrt(max(counts[c], 1.0))
    fg_sum = weights[1:].sum()
    if fg_sum > 0:
        weights[1:] = weights[1:] / fg_sum * (num_classes - 1)
    weights[0] = 0.0
    out = labeled_dir / "class_weights.json"
    out.write_text(
        json.dumps(
            {
                "weights": weights.tolist(),
                "pixel_counts": counts.tolist(),
                "formula": "1/sqrt(freq_fg); bg=0; renormalized over FG",
                "num_classes": num_classes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return out


def prepare_unlabeled_only(args: argparse.Namespace) -> None:
    """Rebuild dense unlabeled (+ optional baseline) without touching labeled/test."""
    out = Path(args.output)
    manifest = pd.read_csv(args.manifest)
    ranges = train_patient_frame_ranges(manifest, Path(args.frame_map))
    frames_dir = Path(args.frames_dir)
    if args.rebuild_baseline:
        build_unlabeled_from_frames(frames_dir, ranges, out / "unlabeled_data", stride=None)
    else:
        baseline = out / "unlabeled_data" / "unlabeled.txt"
        n = len(baseline.read_text().splitlines()) if baseline.is_file() else 0
        print(f"Keeping existing unlabeled_data ({n} entries)")
    dense_stride = int(args.dense_stride)
    build_unlabeled_from_frames(
        frames_dir, ranges, out / "unlabeled_data_dense", stride=dense_stride
    )
    labeled_dir = out / "labeled_data"
    class_map_path = labeled_dir / "class_map.json"
    if class_map_path.is_file():
        class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        num_classes = int(class_map.get("_num_classes", 18))
        write_class_weights_json(labeled_dir, num_classes)


def prepare(args: argparse.Namespace) -> None:
    import shutil

    out = Path(args.output)
    labeled_dir = out / "labeled_data"
    test_dir = out / "test_data"
    unlabeled_dir = out / "unlabeled_data"

    # Clean previous npy trees so old split files cannot linger
    for d in (labeled_dir / "images", labeled_dir / "labels", test_dir / "images", test_dir / "labels", unlabeled_dir / "images"):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_dir.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(Path(args.json).read_text(encoding="utf-8"))
    manifest = pd.read_csv(args.manifest)
    by_id = {int(r.label_studio_id): r for _, r in manifest.iterrows()}
    by_fn = {str(r.filename): r for _, r in manifest.iterrows()}

    excluded = expand_excluded_ids(Path(args.exclude_ids))
    print(f"Excluded Label Studio IDs: {len(excluded)}")

    label_names = collect_label_names(tasks)
    class_map = build_class_map(label_names)
    (labeled_dir / "class_map.json").write_text(
        json.dumps(class_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (test_dir / "class_map.json").write_text(
        json.dumps(class_map, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Classes: {class_map['_num_classes']} (bg + {len(label_names)} FG)")

    samples_dir = Path(args.samples_dir)
    records: list[dict] = []
    skipped = 0
    skipped_excluded = 0

    for task in tasks:
        tid = int(task["id"])
        if tid in excluded:
            skipped_excluded += 1
            continue

        row = by_id.get(tid)
        if row is None:
            fn = stem_from_upload(task.get("file_upload") or "")
            row = by_fn.get(fn)
        if row is None:
            print(f"  [skip] task {tid}: not in manifest")
            skipped += 1
            continue

        merged = merge_task_masks(task, class_map)
        if merged is None:
            print(f"  [skip] task {tid}: no brush masks")
            skipped += 1
            continue

        filename = str(row.filename)
        img_path = samples_dir / filename
        if not img_path.is_file():
            rel = row.get("relative_path")
            if isinstance(rel, str) and rel:
                img_path = ROOT / rel
        if not img_path.is_file():
            print(f"  [skip] task {tid}: missing image {filename}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [skip] task {tid}: unreadable {img_path}")
            skipped += 1
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        if merged.shape[:2] != (h, w):
            merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_NEAREST)

        sample_name = Path(filename).with_suffix(".npy").name
        pid = int(row.patient_id)
        fg = {int(x) for x in np.unique(merged) if int(x) > 0}
        records.append(
            {
                "task_id": tid,
                "patient_id": pid,
                "filename": filename,
                "sample_name": sample_name,
                "img": img.astype(np.uint8),
                "mask": merged.astype(np.uint8),
                "fg_classes": fg,
            }
        )

    print(f"Decoded labeled frames: {len(records)} (excluded_in_json={skipped_excluded}, other_skip={skipped})")

    # Patient → classes / counts for resplit
    patient_classes: dict[int, set[int]] = defaultdict(set)
    patient_counts: dict[int, int] = defaultdict(int)
    for r in records:
        patient_classes[r["patient_id"]] |= r["fg_classes"]
        patient_counts[r["patient_id"]] += 1

    train_pids, test_pids = resplit_patients_for_coverage(
        dict(patient_classes),
        dict(patient_counts),
        force_train=FORCE_TRAIN_PATIENTS,
        target_test_frac=float(args.test_frac),
    )
    print(f"Resplit patients: train={sorted(train_pids)} test={sorted(test_pids)}")
    print(f"Train frames≈{sum(patient_counts[p] for p in train_pids)}  test frames≈{sum(patient_counts[p] for p in test_pids)}")

    train_cov = set().union(*(patient_classes[p] for p in train_pids)) if train_pids else set()
    test_cov = set().union(*(patient_classes[p] for p in test_pids)) if test_pids else set()
    id_to_name = {v: k for k, v in class_map.items() if isinstance(v, int)}
    print("Train FG classes:", sorted(id_to_name.get(c, str(c)) for c in train_cov))
    print("Test FG classes:", sorted(id_to_name.get(c, str(c)) for c in test_cov))
    print("Train-only classes:", sorted(id_to_name.get(c, str(c)) for c in (train_cov - test_cov)))
    print("Test-only classes:", sorted(id_to_name.get(c, str(c)) for c in (test_cov - train_cov)))

    # Persist split meta + update manifest CSV (backup once)
    split_meta = {
        "force_train_patients": sorted(FORCE_TRAIN_PATIENTS),
        "train_patients": sorted(train_pids),
        "test_patients": sorted(test_pids),
        "excluded_ids_file": str(Path(args.exclude_ids)),
        "n_excluded_ids": len(excluded),
        "n_labeled_kept": len(records),
    }
    (out / "split_patients.json").write_text(
        json.dumps(split_meta, indent=2) + "\n", encoding="utf-8"
    )

    manifest_path = Path(args.manifest)
    backup = manifest_path.with_suffix(".csv.bak_before_resplit")
    if not backup.is_file():
        backup.write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    # Update split for all rows of affected patients (including unlabeled-only manifest rows)
    new_split = []
    for _, row in manifest.iterrows():
        pid = int(row.patient_id)
        tid = int(row.label_studio_id)
        if tid in excluded:
            new_split.append("exclude")
        elif pid in test_pids:
            new_split.append("test")
        elif pid in train_pids:
            new_split.append("train")
        else:
            # patients with no labeled frames after filter: keep prior or mark train if was train
            new_split.append(str(row.split) if str(row.split) in ("train", "test") else "train")
    manifest = manifest.copy()
    manifest["split"] = new_split
    manifest.to_csv(manifest_path, index=False)
    print(f"Updated manifest splits → {manifest_path} (backup: {backup.name})")

    train_rows: list[tuple[str, int]] = []
    test_names: list[str] = []
    for r in records:
        pid = r["patient_id"]
        if pid in test_pids:
            dest = test_dir
            test_names.append(r["sample_name"])
        else:
            dest = labeled_dir
            train_rows.append((r["sample_name"], pid))
        np.save(dest / "images" / r["sample_name"], r["img"])
        np.save(dest / "labels" / r["sample_name"], r["mask"])

    train_rows = sorted(train_rows, key=lambda x: x[0])
    test_names = sorted(set(test_names))
    folds = patient_stratified_folds(train_rows, n_folds=5)
    for i, fold in enumerate(folds, start=1):
        write_list(labeled_dir / f"fold_label_{i}.txt", sorted(fold))
    write_list(test_dir / "test.txt", test_names)

    print(f"Labeled train: {len(train_rows)}  test: {len(test_names)}")

    # Unlabeled: baseline (all matching) + densified (stride) for train patients only
    ranges = train_patient_frame_ranges(manifest, Path(args.frame_map))
    frames_dir = Path(args.frames_dir)
    build_unlabeled_from_frames(frames_dir, ranges, unlabeled_dir, stride=None)
    dense_dir = out / "unlabeled_data_dense"
    build_unlabeled_from_frames(
        frames_dir, ranges, dense_dir, stride=int(args.dense_stride)
    )
    write_class_weights_json(labeled_dir, int(class_map["_num_classes"]))
    print(f"Wrote: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare dat_kim_goc FixMatch dataset")
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--samples-dir", default=str(DEFAULT_SAMPLES))
    ap.add_argument("--frames-dir", default=str(DEFAULT_FRAMES))
    ap.add_argument("--frame-map", default=str(DEFAULT_FRAME_MAP))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--exclude-ids", default=str(EXCLUDE_IDS_JSON))
    ap.add_argument("--test-frac", type=float, default=0.22, help="Target fraction of labeled frames in test")
    ap.add_argument(
        "--dense-stride",
        type=int,
        default=5,
        help="Stride for unlabeled_data_dense (every Nth frame index)",
    )
    ap.add_argument(
        "--unlabeled-only",
        action="store_true",
        help="Only rebuild unlabeled_data_dense (+ class_weights.json); keep baseline unless --rebuild-baseline",
    )
    ap.add_argument(
        "--rebuild-baseline",
        action="store_true",
        help="With --unlabeled-only, also rebuild unlabeled_data from all matching frames",
    )
    args = ap.parse_args()
    if args.unlabeled_only:
        prepare_unlabeled_only(args)
    else:
        prepare(args)


if __name__ == "__main__":
    main()
