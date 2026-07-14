#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 180
DEFAULT_PIXEL_THRESHOLD = 8
DEFAULT_THRESHOLDS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0)


@dataclass
class DiffMetrics:
    source: str
    n: int
    means: np.ndarray
    fracs: np.ndarray
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD


@dataclass
class DedupResult:
    manifest_path: Path
    kept_count: int
    dropped_count: int


def _preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _compute_pair_stats(prev_gray: np.ndarray, cur_gray: np.ndarray, pixel_threshold: int) -> tuple[float, float]:
    diff = cv2.absdiff(cur_gray, prev_gray)
    mean_abs_diff = float(diff.mean())
    changed_frac = float((diff >= pixel_threshold).mean())
    return mean_abs_diff, changed_frac


def compute_diffs_from_video(video_path: Path, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD) -> DiffMetrics:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    means: list[float] = []
    fracs: list[float] = []
    n = 0
    prev_gray = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            gray = _preprocess(frame, width=width, height=height)
            if prev_gray is not None:
                m, f = _compute_pair_stats(prev_gray, gray, pixel_threshold=pixel_threshold)
                means.append(m)
                fracs.append(f)
            prev_gray = gray
    finally:
        cap.release()

    return DiffMetrics(
        source=str(video_path),
        n=n,
        means=np.asarray(means, dtype=np.float32),
        fracs=np.asarray(fracs, dtype=np.float32),
        width=width,
        height=height,
        pixel_threshold=pixel_threshold,
    )


def compute_diffs_from_frames(frames_dir: Path, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT, pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD) -> DiffMetrics:
    frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    means: list[float] = []
    fracs: list[float] = []
    prev_gray = None
    n = 0
    for path in frame_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        n += 1
        gray = _preprocess(frame, width=width, height=height)
        if prev_gray is not None:
            m, f = _compute_pair_stats(prev_gray, gray, pixel_threshold=pixel_threshold)
            means.append(m)
            fracs.append(f)
        prev_gray = gray

    return DiffMetrics(
        source=str(frames_dir),
        n=n,
        means=np.asarray(means, dtype=np.float32),
        fracs=np.asarray(fracs, dtype=np.float32),
        width=width,
        height=height,
        pixel_threshold=pixel_threshold,
    )


def save_metrics(path: Path, metrics: DiffMetrics) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        str(path),
        {
            "source": metrics.source,
            "n": metrics.n,
            "means": metrics.means,
            "fracs": metrics.fracs,
            "width": metrics.width,
            "height": metrics.height,
            "pixel_threshold": metrics.pixel_threshold,
        },
        allow_pickle=True,
    )


def load_metrics(path: Path) -> DiffMetrics:
    payload = np.load(str(path), allow_pickle=True).item()
    return DiffMetrics(
        source=str(payload.get("source", "")),
        n=int(payload.get("n", 0)),
        means=np.asarray(payload.get("means", []), dtype=np.float32),
        fracs=np.asarray(payload.get("fracs", []), dtype=np.float32),
        width=int(payload.get("width", DEFAULT_WIDTH)),
        height=int(payload.get("height", DEFAULT_HEIGHT)),
        pixel_threshold=int(payload.get("pixel_threshold", DEFAULT_PIXEL_THRESHOLD)),
    )


def print_diff_histogram(means: np.ndarray) -> None:
    if len(means) == 0:
        print("hist: empty")
        return
    bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, 5, 8, 12, 20, 40, 80, 160, 255]
    hist, edges = np.histogram(means, bins=bins)
    print("histogram mean_abs_diff (pair count):")
    for i, count in enumerate(hist):
        print(f"  [{edges[i]:6.2f}, {edges[i+1]:6.2f}): {int(count)}")


def print_threshold_table(means: np.ndarray, thresholds: tuple[float, ...], fps: float | None = None) -> None:
    if len(means) == 0:
        print("threshold stats: empty")
        return
    print("threshold table (drop if mean_abs_diff < T):")
    for t in thresholds:
        drops = int((means < t).sum())
        pct = (drops / len(means)) * 100.0
        if fps and fps > 0:
            sec = drops / fps
            print(f"  T={t:6.2f} | drops={drops:7d}/{len(means)} ({pct:6.2f}%) | est removed={sec:8.2f}s")
        else:
            print(f"  T={t:6.2f} | drops={drops:7d}/{len(means)} ({pct:6.2f}%)")


def apply_dedup_to_frames_dir(
    frames_dir: Path,
    output_dir: Path,
    metrics: DiffMetrics,
    threshold: float,
    renumber: bool = True,
    use_symlink: bool = False,
    manifest_path: Path | None = None,
) -> DedupResult:
    frame_paths = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if len(frame_paths) != metrics.n:
        # Best effort alignment on available frames.
        pass
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_flags = [True] * len(frame_paths)
    for i in range(1, min(len(frame_paths), len(metrics.means) + 1)):
        if float(metrics.means[i - 1]) < threshold:
            keep_flags[i] = False

    kept = 0
    dropped = 0
    manifest_rows = []
    for idx, src in enumerate(frame_paths):
        keep = keep_flags[idx]
        if keep:
            kept += 1
            if renumber:
                dst = output_dir / f"frame_{kept:05d}{src.suffix.lower()}"
            else:
                dst = output_dir / src.name
            if use_symlink:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
        else:
            dropped += 1
        manifest_rows.append({"index": idx, "frame": src.name, "keep": keep})

    out_manifest = manifest_path or (output_dir / "dedup_manifest.json")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(
        json.dumps(
            {
                "source": str(frames_dir),
                "threshold": threshold,
                "total": len(frame_paths),
                "kept": kept,
                "dropped": dropped,
                "rows": manifest_rows,
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    return DedupResult(manifest_path=out_manifest, kept_count=kept, dropped_count=dropped)
