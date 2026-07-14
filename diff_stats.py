#!/usr/bin/env python3
"""
Compute frame-to-frame diff metrics and print threshold statistics.

Caches metrics to .npy for reuse by dedup_near.py.

Examples:
  python diff_stats.py --frames-dir exports/video3 --fps 20
  python diff_stats.py --frames-dir exports/video3 --metrics-out exports/video3/frame_diffs.npy
  python diff_stats.py --metrics exports/video3/frame_diffs.npy --fps 21
  python diff_stats.py --video local_videos/2025-04-01_163954_VID003.mp4 --metrics-out exports/vid3/frame_diffs.npy
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
DEFAULT_THRESHOLDS = fd.DEFAULT_THRESHOLDS
DEFAULT_WIDTH = fd.DEFAULT_WIDTH
compute_diffs_from_frames = fd.compute_diffs_from_frames
compute_diffs_from_video = fd.compute_diffs_from_video
load_metrics = fd.load_metrics
print_diff_histogram = fd.print_diff_histogram
print_threshold_table = fd.print_threshold_table
save_metrics = fd.save_metrics


def _parse_thresholds(text: str | None) -> tuple[float, ...]:
    if not text:
        return DEFAULT_THRESHOLDS
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Compute or load frame diff metrics and print dedup statistics."
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--frames-dir",
        type=Path,
        help="Directory of frame_*.jpg|.png to analyze.",
    )
    src.add_argument(
        "--video",
        type=Path,
        help="Input video to decode and analyze.",
    )
    ap.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Load existing metrics .npy (skip decode unless --frames-dir/--video given).",
    )
    ap.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Save computed metrics here (default: <frames-dir>/frame_diffs.npy).",
    )
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--pixel-threshold", type=int, default=DEFAULT_PIXEL_THRESHOLD)
    ap.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS for estimating seconds removed per threshold.",
    )
    ap.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds (default: built-in list).",
    )
    args = ap.parse_args(argv)

    if args.frames_dir is None and args.video is None and args.metrics is None:
        raise SystemExit("Provide --frames-dir, --video, or --metrics.")

    metrics = None
    if args.metrics is not None and args.metrics.is_file():
        metrics = load_metrics(args.metrics)
        print(f"[load] {args.metrics}  n={metrics.n}  source={metrics.source}")

    if args.frames_dir is not None:
        if not args.frames_dir.is_dir():
            raise SystemExit(f"Frames directory not found: {args.frames_dir}")
        print(f"[decode] {args.frames_dir}")
        metrics = compute_diffs_from_frames(
            args.frames_dir,
            width=args.width,
            height=args.height,
            pixel_threshold=args.pixel_threshold,
        )
        out = args.metrics_out or (args.frames_dir / "frame_diffs.npy")
        save_metrics(out, metrics)
        print(f"[saved] {out}  n={metrics.n}")
    elif args.video is not None:
        if not args.video.is_file():
            raise SystemExit(f"Video not found: {args.video}")
        print(f"[decode] {args.video}")
        metrics = compute_diffs_from_video(
            args.video,
            width=args.width,
            height=args.height,
            pixel_threshold=args.pixel_threshold,
        )
        out = args.metrics_out or (args.video.parent / f"{args.video.stem}_frame_diffs.npy")
        save_metrics(out, metrics)
        print(f"[saved] {out}  n={metrics.n}")

    if metrics is None:
        raise SystemExit("No metrics available.")

    print(f"total frames: {metrics.n} | pairwise diffs: {len(metrics.means)}")
    print_diff_histogram(metrics.means)
    print_threshold_table(
        metrics.means,
        _parse_thresholds(args.thresholds),
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
