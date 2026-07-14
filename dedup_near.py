#!/usr/bin/env python3
"""
Dedup near-identical frames (keep-first) into a new exports subfolder.

Frame i is dropped when mean_abs_diff(i, i-1) < threshold.
Uses cached metrics from diff_stats.py when available.

Examples:
  python dedup_near.py --frames-dir exports/video3 -T 2.0 --output-dir exports/video3_deduped
  python dedup_near.py --frames-dir exports/video3 --metrics exports/video3/frame_diffs.npy -T 1.0 --output-dir exports/video3_deduped
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_frame_dedup():
    path = ROOT / "inference" / "frame_dedup.py"
    spec = importlib.util.spec_from_file_location("frame_dedup", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["frame_dedup"] = mod
    spec.loader.exec_module(mod)
    return mod


fd = _load_frame_dedup()
DEFAULT_HEIGHT = fd.DEFAULT_HEIGHT
DEFAULT_PIXEL_THRESHOLD = fd.DEFAULT_PIXEL_THRESHOLD
DEFAULT_WIDTH = fd.DEFAULT_WIDTH
apply_dedup_to_frames_dir = fd.apply_dedup_to_frames_dir
compute_diffs_from_frames = fd.compute_diffs_from_frames
load_metrics = fd.load_metrics
save_metrics = fd.save_metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Remove near-duplicate frames into a new output folder."
    )
    ap.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Source frames directory under exports/.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination folder for kept frames.",
    )
    ap.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Cached metrics .npy (default: <frames-dir>/frame_diffs.npy if present).",
    )
    ap.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=2.0,
        help="Drop frame i when mean_abs_diff(i, i-1) < T (default: 2.0).",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help="Analysis width when computing metrics on the fly.",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help="Analysis height when computing metrics on the fly.",
    )
    ap.add_argument(
        "--pixel-threshold",
        type=int,
        default=DEFAULT_PIXEL_THRESHOLD,
        help="Pixel change threshold for frac metric.",
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink kept frames instead of copying.",
    )
    ap.add_argument(
        "--no-renumber",
        action="store_true",
        help="Keep original filenames instead of frame_00000.jpg, ...",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Manifest JSON path (default: <output-dir>/dedup_manifest.json).",
    )
    ap.add_argument(
        "--save-metrics",
        type=Path,
        default=None,
        help="Save metrics after on-the-fly compute (default: <frames-dir>/frame_diffs.npy).",
    )
    args = ap.parse_args(argv)

    if not args.frames_dir.is_dir():
        raise SystemExit(f"Frames directory not found: {args.frames_dir}")

    metrics = None
    metrics_path = args.metrics or (args.frames_dir / "frame_diffs.npy")
    if metrics_path.is_file():
        metrics = load_metrics(metrics_path)
        print(f"[load] {metrics_path}  n={metrics.n}")

    if metrics is None:
        print(f"[compute] {args.frames_dir}")
        metrics = compute_diffs_from_frames(
            args.frames_dir,
            width=args.width,
            height=args.height,
            pixel_threshold=args.pixel_threshold,
        )
        save_path = args.save_metrics or (args.frames_dir / "frame_diffs.npy")
        save_metrics(save_path, metrics)
        print(f"[saved] {save_path}  n={metrics.n}")

    result = apply_dedup_to_frames_dir(
        args.frames_dir,
        args.output_dir,
        metrics=metrics,
        threshold=args.threshold,
        renumber=not args.no_renumber,
        use_symlink=args.symlink,
        manifest_path=args.manifest,
    )
    print(f"manifest: {result.manifest_path}")
    print("DONE")


if __name__ == "__main__":
    main()
