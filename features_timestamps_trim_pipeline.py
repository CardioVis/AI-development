#!/usr/bin/env python3
"""Trim clips from Features_timestamps.xlsx and append continuous frame samples.

Designed for Colab with Google Drive mounted, but also runnable locally if paths exist.

Excel time quirk: displayed HH:MM:SS is actually MM:SS misread by Excel.
Convert with: seconds = hour * 60 + minute.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas/openpyxl required: pip install pandas openpyxl") from exc


# ------------------------- USER CONFIG -------------------------
CARDIOVIS_RELATED_DIR = Path(
    os.getenv("CARDIOVIS_RELATED_DIR", "/content/drive/MyDrive/Cardiovis-related")
)
ROOT_VIDEOS_DIR = Path(os.getenv("ROOT_VIDEOS_DIR", "/content/drive/MyDrive/3) Videos"))
XLSX_PATH = Path(
    os.getenv(
        "FEATURES_TIMESTAMPS_XLSX",
        str(CARDIOVIS_RELATED_DIR / "Features_timestamps.xlsx"),
    )
)

OUTPUT_ROOT = CARDIOVIS_RELATED_DIR / "stage_outputs"
CLIPS_DIR = OUTPUT_ROOT / "clips"
FRAMES_DIR = OUTPUT_ROOT / "frames"
SAMPLES_DIR = OUTPUT_ROOT / "frame_samples_1_per_20"
REPORTS_DIR = OUTPUT_ROOT / "reports"
TEMP_DIR = OUTPUT_ROOT / "tmp" / "features_timestamps"

EXTRACT_FPS = int(os.getenv("EXTRACT_FPS", "20"))
SAMPLE_EVERY_N_FRAMES = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "20"))
DRY_RUN = os.getenv("DRY_RUN", "0").strip().lower() in {"1", "true", "yes", "y", "on"}
OVERWRITE = os.getenv("OVERWRITE", "0").strip().lower() in {"1", "true", "yes", "y", "on"}

STAGE_SHEETS = {
    "Đặt kim gốc": "dat_kim_goc",
    "Rạch vách liên nhĩ": "rach_nhi",
}

# Continue numbering after previous batch full-frame extract.
# From stage_outputs/reports/frame_extract_report.csv:
#   dat_kim_goc: 6917 frames, rach_nhi: 5980 frames
# Sample cadence remains indices where (idx - 1) % 20 == 0 (…, 6901, then 6921, …).
FALLBACK_LAST_FRAME_IDX = {
    "dat_kim_goc": 6917,
    "rach_nhi": 5980,
}
# --------------------------------------------------------------


@dataclass
class FeatureTask:
    patient: int
    stage: str
    start_sec: int
    end_sec: int
    video_name: str
    source_path: Optional[Path] = None
    clip_path: Optional[Path] = None
    status: str = "pending"
    message: str = ""
    frame_start_idx: int = 0
    frame_end_idx: int = 0
    frame_count: int = 0
    sample_count: int = 0


def run(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    run(["apt-get", "update", "-qq"], check=False)
    run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        raise RuntimeError("ffmpeg/ffprobe not available")


def ensure_dirs() -> None:
    for stage in STAGE_SHEETS.values():
        (CLIPS_DIR / stage).mkdir(parents=True, exist_ok=True)
        (FRAMES_DIR / stage).mkdir(parents=True, exist_ok=True)
        (SAMPLES_DIR / stage).mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


def excel_time_to_seconds(value) -> Optional[int]:
    """Convert Excel-misread MM:SS (shown as HH:MM:SS) to seconds."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return int(value.hour) * 60 + int(value.minute)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    # Accept HH:MM:SS or MM:SS
    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = parts
        # Prefer reinterpret as MM:SS when seconds component is 00
        if s in {"00", "0"}:
            return int(h) * 60 + int(m)
        return int(h) * 3600 + int(m) * 60 + int(float(s))
    if len(parts) == 2:
        return int(parts[0]) * 60 + int(float(parts[1]))
    if text.replace(".", "", 1).isdigit():
        return int(float(text))
    return None


def parse_tasks(xlsx_path: Path) -> List[FeatureTask]:
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel not found: {xlsx_path}")
    tasks: List[FeatureTask] = []
    xl = pd.ExcelFile(xlsx_path)
    for sheet_name, stage in STAGE_SHEETS.items():
        if sheet_name not in xl.sheet_names:
            print(f"[warn] missing sheet: {sheet_name}", flush=True)
            continue
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        for _, row in df.iterrows():
            patient_raw = row.get("Patient")
            if pd.isna(patient_raw):
                continue
            start_sec = excel_time_to_seconds(row.get("Start time"))
            end_sec = excel_time_to_seconds(row.get("End time"))
            video_name = row.get("video")
            if start_sec is None or end_sec is None or pd.isna(video_name):
                continue
            video_name = str(video_name).strip()
            if not video_name:
                continue
            if end_sec <= start_sec:
                continue
            tasks.append(
                FeatureTask(
                    patient=int(patient_raw),
                    stage=stage,
                    start_sec=int(start_sec),
                    end_sec=int(end_sec),
                    video_name=video_name,
                )
            )
    return tasks


def extract_video_index(name: str) -> int:
    m = re.search(r"VID0*([0-9]+)", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0


def seconds_slug(sec: int) -> str:
    # Match existing clip naming style: minutes_seconds with underscores
    minutes, seconds = divmod(int(sec), 60)
    if minutes == 0:
        return str(seconds)
    return f"{minutes}_{seconds:02d}"


def clip_filename(task: FeatureTask) -> str:
    vid = extract_video_index(task.video_name)
    return (
        f"patient_{task.patient:02d}_goc_v{vid:03d}_{task.stage}_"
        f"{seconds_slug(task.start_sec)}-{seconds_slug(task.end_sec)}.mp4"
    )


def resolve_video(task: FeatureTask, video_index: Dict[str, List[Path]]) -> Tuple[Optional[Path], str]:
    candidates = video_index.get(task.video_name.lower(), [])
    if not candidates:
        # fallback: loose search by basename
        matches = list(ROOT_VIDEOS_DIR.rglob(task.video_name))
        candidates = matches
    if not candidates:
        return None, "source_not_found"
    if len(candidates) == 1:
        return candidates[0], "unique_filename"

    patient_token = f"patient {task.patient}"
    preferred = [
        p
        for p in candidates
        if patient_token in str(p).lower() or f"patient_{task.patient:02d}" in str(p).lower()
        or f"patient_{task.patient}" in str(p).lower()
    ]
    if len(preferred) == 1:
        return preferred[0], "patient_folder_match"
    if len(preferred) > 1:
        return preferred[0], "patient_folder_ambiguous_first"
    return candidates[0], "ambiguous_filename_first"


def build_video_index(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".mov", ".m4v", ".avi", ".mkv"}:
            continue
        index.setdefault(path.name.lower(), []).append(path)
    return index


def ffmpeg_trim(src: Path, dst: Path, start_sec: int, end_sec: int) -> Tuple[bool, str]:
    duration = end_sec - start_sec
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not OVERWRITE:
        return True, "existing"
    cmd = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-ss",
        str(start_sec),
        "-i",
        str(src),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-an",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    if DRY_RUN:
        return True, "dry_run"
    proc = run(cmd, check=False)
    if proc.returncode == 0 and dst.exists():
        return True, "created"
    return False, (proc.stderr or "")[-500:]


def max_existing_frame_idx(stage: str) -> int:
    stage_dir = FRAMES_DIR / stage
    if not stage_dir.exists():
        return FALLBACK_LAST_FRAME_IDX.get(stage, 0)
    pattern = re.compile(rf"^{re.escape(stage)}_full_frame_(\d+)\.jpg$", re.IGNORECASE)
    max_idx = 0
    for path in stage_dir.glob(f"{stage}_full_frame_*.jpg"):
        m = pattern.match(path.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    if max_idx == 0:
        return FALLBACK_LAST_FRAME_IDX.get(stage, 0)
    return max_idx


def extract_and_renumber_frames(task: FeatureTask, next_idx: int) -> Tuple[bool, str, int, int, int]:
    """Extract fps=EXTRACT_FPS frames and rename into continuous {stage}_full_frame_*.jpg.

    Returns: ok, message, frame_start_idx, frame_end_idx, frame_count
    """
    assert task.clip_path is not None
    stage = task.stage
    tmp_dir = TEMP_DIR / f"extract_{stage}_p{task.patient:02d}_{task.start_sec}_{task.end_sec}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    pattern = tmp_dir / "tmp_frame_%08d.jpg"

    if DRY_RUN:
        # Approximate count for dry-run reporting
        approx = max(1, (task.end_sec - task.start_sec) * EXTRACT_FPS)
        start_idx = next_idx + 1
        end_idx = next_idx + approx
        return True, "dry_run", start_idx, end_idx, approx

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(task.clip_path),
        "-vf",
        f"fps={EXTRACT_FPS}",
        "-q:v",
        "2",
        str(pattern),
    ]
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        return False, (proc.stderr or "")[-500:], 0, 0, 0

    extracted = sorted(tmp_dir.glob("tmp_frame_*.jpg"))
    if not extracted:
        return False, "no_frames_extracted", 0, 0, 0

    frame_dir = FRAMES_DIR / stage
    frame_dir.mkdir(parents=True, exist_ok=True)
    start_idx = next_idx + 1
    cur = start_idx
    for src in extracted:
        dst = frame_dir / f"{stage}_full_frame_{cur:08d}.jpg"
        if dst.exists() and not OVERWRITE:
            cur += 1
            continue
        shutil.move(str(src), str(dst))
        cur += 1
    end_idx = cur - 1
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return True, "ok", start_idx, end_idx, len(extracted)


def sample_new_frames(stage: str, frame_start: int, frame_end: int) -> int:
    """Copy frames whose index matches historical 1,21,41,... cadence into samples dir."""
    if DRY_RUN:
        return sum(1 for i in range(frame_start, frame_end + 1) if (i - 1) % SAMPLE_EVERY_N_FRAMES == 0)

    sample_dir = SAMPLES_DIR / stage
    sample_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for idx in range(frame_start, frame_end + 1):
        if (idx - 1) % SAMPLE_EVERY_N_FRAMES != 0:
            continue
        src = FRAMES_DIR / stage / f"{stage}_full_frame_{idx:08d}.jpg"
        if not src.exists():
            continue
        dst = sample_dir / src.name
        if dst.exists() and not OVERWRITE:
            count += 1
            continue
        shutil.copy2(src, dst)
        count += 1
    return count


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    print("=" * 78)
    print("CardioVis Features_timestamps trim + continuous frames")
    print(f"XLSX       : {XLSX_PATH}")
    print(f"Videos     : {ROOT_VIDEOS_DIR}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"DRY_RUN    : {DRY_RUN}")
    print("=" * 78)

    ensure_ffmpeg()
    ensure_dirs()

    tasks = parse_tasks(XLSX_PATH)
    print(f"Parsed {len(tasks)} valid tasks from Excel", flush=True)

    print("Indexing videos under ROOT_VIDEOS_DIR...", flush=True)
    video_index = build_video_index(ROOT_VIDEOS_DIR)
    print(f"Indexed {sum(len(v) for v in video_index.values())} video files", flush=True)

    next_idx_by_stage = {
        stage: max_existing_frame_idx(stage) for stage in STAGE_SHEETS.values()
    }
    print(f"Starting frame indices (after existing): {next_idx_by_stage}", flush=True)

    task_rows: List[dict] = []
    frame_map_rows: List[dict] = []

    for i, task in enumerate(tasks, start=1):
        src, resolve_msg = resolve_video(task, video_index)
        task.source_path = src
        clip_name = clip_filename(task)
        out_clip = CLIPS_DIR / task.stage / clip_name
        task.clip_path = out_clip

        print(
            f"[{i}/{len(tasks)}] P{task.patient:02d} {task.stage} "
            f"{task.start_sec}-{task.end_sec}s video={task.video_name}",
            flush=True,
        )

        if src is None:
            task.status = "skipped"
            task.message = resolve_msg
            task_rows.append(task_to_row(task, resolve_msg))
            continue

        ok, msg = ffmpeg_trim(src, out_clip, task.start_sec, task.end_sec)
        if not ok:
            task.status = "failed_trim"
            task.message = msg
            task_rows.append(task_to_row(task, resolve_msg))
            continue

        start_next = next_idx_by_stage[task.stage]
        ok2, msg2, f_start, f_end, f_count = extract_and_renumber_frames(task, start_next)
        if not ok2:
            task.status = "failed_extract"
            task.message = msg2
            task_rows.append(task_to_row(task, resolve_msg))
            continue

        samples = sample_new_frames(task.stage, f_start, f_end)
        next_idx_by_stage[task.stage] = f_end
        task.status = "ok"
        task.message = f"{resolve_msg}; trim={msg}; extract={msg2}"
        task.frame_start_idx = f_start
        task.frame_end_idx = f_end
        task.frame_count = f_count
        task.sample_count = samples

        task_rows.append(task_to_row(task, resolve_msg))
        frame_map_rows.append(
            {
                "patient": task.patient,
                "stage": task.stage,
                "clip_name": clip_name,
                "source_path": str(src),
                "start_sec": task.start_sec,
                "end_sec": task.end_sec,
                "frame_start_idx": f_start,
                "frame_end_idx": f_end,
                "frame_count": f_count,
                "sample_count": samples,
            }
        )
        print(
            f"  -> clip={clip_name} frames={f_start}-{f_end} (n={f_count}) samples={samples}",
            flush=True,
        )

    write_csv(
        REPORTS_DIR / "features_timestamps_task_results.csv",
        task_rows,
        [
            "patient",
            "stage",
            "start_sec",
            "end_sec",
            "video_name",
            "source_path",
            "clip_path",
            "status",
            "message",
            "frame_start_idx",
            "frame_end_idx",
            "frame_count",
            "sample_count",
        ],
    )
    write_csv(
        REPORTS_DIR / "features_frame_index_map.csv",
        frame_map_rows,
        [
            "patient",
            "stage",
            "clip_name",
            "source_path",
            "start_sec",
            "end_sec",
            "frame_start_idx",
            "frame_end_idx",
            "frame_count",
            "sample_count",
        ],
    )

    ok_n = sum(1 for r in task_rows if r["status"] == "ok")
    print("=" * 78)
    print(f"Done. ok={ok_n}/{len(task_rows)}")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Final next indices: {next_idx_by_stage}")


def task_to_row(task: FeatureTask, resolve_msg: str) -> dict:
    return {
        "patient": task.patient,
        "stage": task.stage,
        "start_sec": task.start_sec,
        "end_sec": task.end_sec,
        "video_name": task.video_name,
        "source_path": str(task.source_path) if task.source_path else "",
        "clip_path": str(task.clip_path) if task.clip_path else "",
        "status": task.status,
        "message": task.message or resolve_msg,
        "frame_start_idx": task.frame_start_idx,
        "frame_end_idx": task.frame_end_idx,
        "frame_count": task.frame_count,
        "sample_count": task.sample_count,
    }


if __name__ == "__main__":
    main()
