#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# ------------------------- USER CONFIG -------------------------
DOCX_PATH = Path("/content/drive/MyDrive/passio-lab-1.docx")
MERGED_DIR = Path("/content/drive/MyDrive/CardioVis-merged-videos")
ROOT_VIDEOS_DIR = Path("/content/drive/MyDrive/3) Videos")
CARDIOVIS_RELATED_DIR = Path("/content/drive/MyDrive/Cardiovis-related")

OUTPUT_ROOT = CARDIOVIS_RELATED_DIR / "stage_outputs"
CLIPS_DIR = OUTPUT_ROOT / "clips"
FINAL_DIR = OUTPUT_ROOT / "final"
FRAMES_DIR = OUTPUT_ROOT / "frames"
SAMPLES_DIR = OUTPUT_ROOT / "frame_samples_1_per_20"
REPORTS_DIR = OUTPUT_ROOT / "reports"
TEMP_DIR = OUTPUT_ROOT / "tmp"
PROGRESS_FILE = REPORTS_DIR / "run_progress.json"

STAGES = ("dat_kim_goc", "rach_nhi", "than_kinh_hoanh")
STAGE_DISPLAY = {
    "dat_kim_goc": "Dat kim goc",
    "rach_nhi": "Rach nhi",
    "than_kinh_hoanh": "Than kinh hoanh",
}

EXTRACT_FPS = 20
SAMPLE_EVERY_N_FRAMES = 20
DRY_RUN = False
OVERWRITE = False
DELETE_INTERMEDIATE_AFTER_MERGE = True
MIN_FREE_GB_BEFORE_CLEANUP = 8
AUTODETECT_DRIVE_PATHS = True
STRICT_AMBIGUOUS_GOC = True
RERUN_SOURCE_NOT_FOUND_ONLY = False
MANUAL_GOC_OVERRIDES_CSV = Path("")
SKIP_FRAME_EXTRACT = False
SKIP_SAMPLE_DOWNLOAD = False
RERUN_AFFECTED_STAGES_ONLY = False
MISSED_ONLY_STAGE_OUTPUTS = False

# Local download config (inside Colab VM and optional browser download)
TRY_LOCAL_DOWNLOAD = True
LOCAL_DOWNLOAD_DIR = Path("/content/stage_sample_download")
LOCAL_MIN_FREE_GB = 5
LOCAL_SAFETY_MULTIPLIER = 1.25
TRIGGER_BROWSER_DOWNLOAD = False
# --------------------------------------------------------------


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_env_overrides() -> None:
    global DOCX_PATH, MERGED_DIR, ROOT_VIDEOS_DIR, CARDIOVIS_RELATED_DIR
    global OUTPUT_ROOT, CLIPS_DIR, FINAL_DIR, FRAMES_DIR, SAMPLES_DIR, REPORTS_DIR, TEMP_DIR
    global EXTRACT_FPS, SAMPLE_EVERY_N_FRAMES, DRY_RUN, OVERWRITE
    global DELETE_INTERMEDIATE_AFTER_MERGE, MIN_FREE_GB_BEFORE_CLEANUP
    global AUTODETECT_DRIVE_PATHS, STRICT_AMBIGUOUS_GOC, RERUN_SOURCE_NOT_FOUND_ONLY
    global MANUAL_GOC_OVERRIDES_CSV
    global SKIP_FRAME_EXTRACT, SKIP_SAMPLE_DOWNLOAD, RERUN_AFFECTED_STAGES_ONLY
    global MISSED_ONLY_STAGE_OUTPUTS
    global TRY_LOCAL_DOWNLOAD, LOCAL_DOWNLOAD_DIR, LOCAL_MIN_FREE_GB, LOCAL_SAFETY_MULTIPLIER
    global TRIGGER_BROWSER_DOWNLOAD

    DOCX_PATH = Path(os.getenv("DOCX_PATH", str(DOCX_PATH)))
    MERGED_DIR = Path(os.getenv("MERGED_DIR", str(MERGED_DIR)))
    ROOT_VIDEOS_DIR = Path(os.getenv("ROOT_VIDEOS_DIR", str(ROOT_VIDEOS_DIR)))
    CARDIOVIS_RELATED_DIR = Path(os.getenv("CARDIOVIS_RELATED_DIR", str(CARDIOVIS_RELATED_DIR)))

    OUTPUT_ROOT = CARDIOVIS_RELATED_DIR / os.getenv("OUTPUT_SUBDIR", "stage_outputs")
    CLIPS_DIR = OUTPUT_ROOT / "clips"
    FINAL_DIR = OUTPUT_ROOT / "final"
    FRAMES_DIR = OUTPUT_ROOT / "frames"
    SAMPLES_DIR = OUTPUT_ROOT / "frame_samples_1_per_20"
    REPORTS_DIR = OUTPUT_ROOT / "reports"
    TEMP_DIR = OUTPUT_ROOT / "tmp"

    EXTRACT_FPS = int(os.getenv("EXTRACT_FPS", str(EXTRACT_FPS)))
    SAMPLE_EVERY_N_FRAMES = int(os.getenv("SAMPLE_EVERY_N_FRAMES", str(SAMPLE_EVERY_N_FRAMES)))
    DRY_RUN = _env_bool("DRY_RUN", DRY_RUN)
    OVERWRITE = _env_bool("OVERWRITE", OVERWRITE)
    DELETE_INTERMEDIATE_AFTER_MERGE = _env_bool(
        "DELETE_INTERMEDIATE_AFTER_MERGE", DELETE_INTERMEDIATE_AFTER_MERGE
    )
    MIN_FREE_GB_BEFORE_CLEANUP = int(
        os.getenv("MIN_FREE_GB_BEFORE_CLEANUP", str(MIN_FREE_GB_BEFORE_CLEANUP))
    )
    AUTODETECT_DRIVE_PATHS = _env_bool("AUTODETECT_DRIVE_PATHS", AUTODETECT_DRIVE_PATHS)
    STRICT_AMBIGUOUS_GOC = _env_bool("STRICT_AMBIGUOUS_GOC", STRICT_AMBIGUOUS_GOC)
    RERUN_SOURCE_NOT_FOUND_ONLY = _env_bool(
        "RERUN_SOURCE_NOT_FOUND_ONLY", RERUN_SOURCE_NOT_FOUND_ONLY
    )
    MANUAL_GOC_OVERRIDES_CSV = Path(
        os.getenv("MANUAL_GOC_OVERRIDES_CSV", str(MANUAL_GOC_OVERRIDES_CSV))
    )
    SKIP_FRAME_EXTRACT = _env_bool("SKIP_FRAME_EXTRACT", SKIP_FRAME_EXTRACT)
    SKIP_SAMPLE_DOWNLOAD = _env_bool("SKIP_SAMPLE_DOWNLOAD", SKIP_SAMPLE_DOWNLOAD)
    RERUN_AFFECTED_STAGES_ONLY = _env_bool(
        "RERUN_AFFECTED_STAGES_ONLY", RERUN_AFFECTED_STAGES_ONLY
    )
    MISSED_ONLY_STAGE_OUTPUTS = _env_bool(
        "MISSED_ONLY_STAGE_OUTPUTS", MISSED_ONLY_STAGE_OUTPUTS
    )

    TRY_LOCAL_DOWNLOAD = _env_bool("TRY_LOCAL_DOWNLOAD", TRY_LOCAL_DOWNLOAD)
    LOCAL_DOWNLOAD_DIR = Path(os.getenv("LOCAL_DOWNLOAD_DIR", str(LOCAL_DOWNLOAD_DIR)))
    LOCAL_MIN_FREE_GB = int(os.getenv("LOCAL_MIN_FREE_GB", str(LOCAL_MIN_FREE_GB)))
    LOCAL_SAFETY_MULTIPLIER = float(
        os.getenv("LOCAL_SAFETY_MULTIPLIER", str(LOCAL_SAFETY_MULTIPLIER))
    )
    TRIGGER_BROWSER_DOWNLOAD = _env_bool(
        "TRIGGER_BROWSER_DOWNLOAD", TRIGGER_BROWSER_DOWNLOAD
    )


def refresh_output_paths() -> None:
    global OUTPUT_ROOT, CLIPS_DIR, FINAL_DIR, FRAMES_DIR, SAMPLES_DIR, REPORTS_DIR, TEMP_DIR, PROGRESS_FILE
    OUTPUT_ROOT = CARDIOVIS_RELATED_DIR / os.getenv("OUTPUT_SUBDIR", "stage_outputs")
    CLIPS_DIR = OUTPUT_ROOT / "clips"
    FINAL_DIR = OUTPUT_ROOT / "final"
    FRAMES_DIR = OUTPUT_ROOT / "frames"
    SAMPLES_DIR = OUTPUT_ROOT / "frame_samples_1_per_20"
    REPORTS_DIR = OUTPUT_ROOT / "reports"
    TEMP_DIR = OUTPUT_ROOT / "tmp"
    PROGRESS_FILE = REPORTS_DIR / "run_progress.json"


def write_progress(payload: Dict[str, object]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload_with_ts = {"timestamp": int(time.time()), **payload}
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload_with_ts, f, ensure_ascii=True, indent=2)


def task_key(task: ClipTask) -> str:
    return (
        f"{task.patient}|{task.stage}|{task.start}|{task.end}|"
        f"{task.video_index or ''}|{task.source_type}"
    )


@dataclass
class ClipTask:
    patient: int
    source_type: str  # merged | goc
    stage: str
    start: str
    end: str
    start_sec: int
    end_sec: int
    video_index: Optional[int]


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    run(["apt-get", "update", "-qq"])
    run(["apt-get", "install", "-y", "-qq", "ffmpeg"])


def ensure_dirs() -> None:
    for base in (CLIPS_DIR, FINAL_DIR, FRAMES_DIR, SAMPLES_DIR, REPORTS_DIR, TEMP_DIR):
        base.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        (CLIPS_DIR / stage).mkdir(parents=True, exist_ok=True)
        (FRAMES_DIR / stage).mkdir(parents=True, exist_ok=True)
        (SAMPLES_DIR / stage).mkdir(parents=True, exist_ok=True)


def parse_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Patient", "\nPatient")
    return text


def normalize_stage_name(raw: str) -> Optional[str]:
    val = raw.strip().lower()
    if "đặt kim gốc" in val or "dat kim goc" in val:
        return "dat_kim_goc"
    if "rạch nhĩ" in val or "rach nhi" in val:
        return "rach_nhi"
    if "thần kinh hoành" in val or "than kinh hoanh" in val:
        return "than_kinh_hoanh"
    return None


def parse_time_to_seconds(token: str) -> Optional[int]:
    token = token.strip()
    token = token.replace(" ", "")
    token = token.replace(",", ".")
    if not token:
        return None
    if token.count(".") == 2:
        parts = token.split(".")
        if not all(p.isdigit() for p in parts):
            return None
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    if token.count(".") == 1:
        a, b = token.split(".")
        if not (a.isdigit() and b.isdigit()):
            return None
        return int(a) * 60 + int(b)
    if token.isdigit():
        return int(token)
    return None


def extract_video_index(text: str) -> Optional[int]:
    m = re.search(r"(?:video|vid)\s*([0-9]{1,3})", text, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def extract_range(text: str) -> Tuple[Optional[str], Optional[str]]:
    m = re.search(r"([0-9]+(?:\.[0-9]+){0,2})\s*-\s*([0-9]+(?:\.[0-9]+){0,2})", text)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def parse_patient_block(block: str) -> Tuple[int, List[ClipTask], List[Dict[str, str]]]:
    skipped: List[Dict[str, str]] = []
    tasks: List[ClipTask] = []

    p = re.search(r"Patient\s+(\d+)", block, flags=re.IGNORECASE)
    if not p:
        return -1, tasks, skipped
    patient = int(p.group(1))
    stage_line_pattern = re.compile(
        r"(Đặt kim gốc|Dat kim goc|Rạch nhĩ|Rach nhi|Thần kinh hoành|Than kinh hoanh)\s*:\s*([^:]+?)(?=(?:Đặt kim gốc|Dat kim goc|Rạch nhĩ|Rach nhi|Thần kinh hoành|Than kinh hoanh|$))",
        flags=re.IGNORECASE,
    )

    for m in stage_line_pattern.finditer(block):
        raw_stage = m.group(1)
        payload = m.group(2).strip()
        stage = normalize_stage_name(raw_stage)
        if not stage:
            continue

        start_tok, end_tok = extract_range(payload)
        if not start_tok or not end_tok:
            skipped.append(
                {
                    "patient": str(patient),
                    "stage": stage,
                    "reason": "missing_or_invalid_range",
                    "raw": payload,
                }
            )
            continue

        start_sec = parse_time_to_seconds(start_tok)
        end_sec = parse_time_to_seconds(end_tok)
        if start_sec is None or end_sec is None or end_sec <= start_sec:
            skipped.append(
                {
                    "patient": str(patient),
                    "stage": stage,
                    "reason": "invalid_time_order_or_format",
                    "raw": payload,
                }
            )
            continue

        payload_norm = payload.lower()
        video_index = extract_video_index(payload_norm)
        source_type = "goc" if video_index is not None else "merged"

        tasks.append(
            ClipTask(
                patient=patient,
                source_type=source_type,
                stage=stage,
                start=start_tok,
                end=end_tok,
                start_sec=start_sec,
                end_sec=end_sec,
                video_index=video_index,
            )
        )

    return patient, tasks, skipped


def parse_doc_manifest(docx_path: Path) -> Tuple[List[ClipTask], List[Dict[str, str]]]:
    text = parse_docx_text(docx_path)
    blocks = [
        b.strip()
        for b in re.findall(r"Patient\s+\d+.*?(?=\nPatient\s+\d+|$)", text, flags=re.IGNORECASE)
        if b.strip()
    ]
    all_tasks: List[ClipTask] = []
    skipped: List[Dict[str, str]] = []
    for b in blocks:
        patient, tasks, patient_skipped = parse_patient_block(b)
        if patient < 0:
            continue
        all_tasks.extend(tasks)
        skipped.extend(patient_skipped)
    return all_tasks, skipped


def stage_filename_slug(task: ClipTask) -> str:
    src = "merged" if task.source_type == "merged" else f"goc_v{task.video_index or 0:03d}"
    return (
        f"patient_{task.patient:02d}_{src}_{task.stage}_"
        f"{task.start.replace('.', '_')}-{task.end.replace('.', '_')}.mp4"
    )


def video_files_recursive(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".m4v"}]


def find_by_name_with_depth_limit(
    roots: List[Path],
    target_name: str,
    is_dir: bool,
    max_depth: int = 7,
) -> Optional[Path]:
    target_name_l = target_name.lower()
    for root in roots:
        if not root.exists():
            continue
        root_depth = len(root.parts)
        for current, dirs, files in os.walk(root):
            current_path = Path(current)
            depth = len(current_path.parts) - root_depth
            if depth > max_depth:
                dirs[:] = []
                continue

            if is_dir:
                if current_path.name.lower() == target_name_l:
                    return current_path
                for d in dirs:
                    if d.lower() == target_name_l:
                        return current_path / d
            else:
                for f in files:
                    if f.lower() == target_name_l:
                        return current_path / f
    return None


def maybe_autodetect_paths() -> None:
    global DOCX_PATH, MERGED_DIR, ROOT_VIDEOS_DIR, CARDIOVIS_RELATED_DIR
    if not AUTODETECT_DRIVE_PATHS:
        return

    search_roots = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]

    if not DOCX_PATH.exists():
        found = find_by_name_with_depth_limit(search_roots, "passio-lab-1.docx", is_dir=False)
        if found:
            DOCX_PATH = found
    if not MERGED_DIR.exists():
        found = find_by_name_with_depth_limit(search_roots, "CardioVis-merged-videos", is_dir=True)
        if found:
            MERGED_DIR = found
    if not ROOT_VIDEOS_DIR.exists():
        found = find_by_name_with_depth_limit(search_roots, "3) Videos", is_dir=True)
        if found:
            ROOT_VIDEOS_DIR = found
    if not CARDIOVIS_RELATED_DIR.exists():
        found = find_by_name_with_depth_limit(search_roots, "Cardiovis-related", is_dir=True)
        if found:
            CARDIOVIS_RELATED_DIR = found


def extract_patient_from_name(name: str) -> Optional[int]:
    m = re.search(r"patient[\s_]*([0-9]{1,3})", name, flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def resolve_merged_file(task: ClipTask, merged_files: List[Path]) -> Optional[Path]:
    candidates = [
        p for p in merged_files if extract_patient_from_name(p.name) == task.patient and "merged" in p.name.lower()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(str(p)))
    return candidates[0]


def extract_video_number_from_filename(path: Path) -> Optional[int]:
    name = path.stem.lower()
    m = re.search(r"vid0*([0-9]{1,4})", name)
    if m:
        return int(m.group(1))
    if re.fullmatch(r"0*([0-9]{1,4})", name):
        return int(name)
    m2 = re.search(r"(?:^|[_\-\s])([0-9]{1,4})(?:$|[_\-\s])", name)
    if m2:
        return int(m2.group(1))
    return None


def build_patient_video_index(root_videos_dir: Path) -> Dict[int, Dict[int, Path]]:
    index: Dict[int, Dict[int, Path]] = {}
    ambiguous_rows: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    patient_dirs = [p for p in root_videos_dir.rglob("*") if p.is_dir() and extract_patient_from_name(p.name) is not None]

    for patient_dir in patient_dirs:
        patient = extract_patient_from_name(patient_dir.name)
        if patient is None:
            continue

        # Traverse exactly as requested: keep descending from Patient folder until folders that contain video lists.
        candidate_video_dirs: List[Path] = []
        for d in patient_dir.rglob("*"):
            if not d.is_dir():
                continue
            direct_videos = [
                f for f in d.iterdir()
                if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".m4v"}
            ]
            if direct_videos:
                candidate_video_dirs.append(d)

        # Also include patient root if it directly holds videos.
        root_direct_videos = [
            f for f in patient_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".m4v"}
        ]
        if root_direct_videos:
            candidate_video_dirs.append(patient_dir)

        pmap = index.setdefault(patient, {})
        vno_candidates: Dict[int, List[Path]] = {}
        for video_dir in sorted(set(candidate_video_dirs), key=lambda p: len(p.parts), reverse=True):
            for f in sorted(video_dir.iterdir()):
                if not (f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".m4v"}):
                    continue
                vno = extract_video_number_from_filename(f)
                if vno is None:
                    continue
                vno_candidates.setdefault(vno, []).append(f)

        for vno, candidates in vno_candidates.items():
            uniq_candidates = sorted(set(candidates), key=lambda p: str(p))
            if len(uniq_candidates) == 1:
                chosen = uniq_candidates[0]
                pmap[vno] = chosen
                selected_rows.append(
                    {
                        "patient": patient,
                        "video_index": vno,
                        "selected_file": str(chosen),
                        "reason": "single_candidate",
                        "candidate_count": 1,
                    }
                )
                continue

            ranked = sorted(
                uniq_candidates,
                key=lambda p: (
                    len(p.parts),
                    p.stat().st_size if p.exists() else 0,
                    p.stat().st_mtime if p.exists() else 0,
                ),
                reverse=True,
            )
            if STRICT_AMBIGUOUS_GOC:
                ambiguous_rows.append(
                    {
                        "patient": patient,
                        "video_index": vno,
                        "candidate_count": len(uniq_candidates),
                        "candidates": " || ".join(str(x) for x in uniq_candidates),
                        "selected_file": "",
                        "reason": "ambiguous_strict_skip",
                    }
                )
                continue

            chosen = ranked[0]
            pmap[vno] = chosen
            ambiguous_rows.append(
                {
                    "patient": patient,
                    "video_index": vno,
                    "candidate_count": len(uniq_candidates),
                    "candidates": " || ".join(str(x) for x in uniq_candidates),
                    "selected_file": str(chosen),
                    "reason": "ambiguous_auto_selected",
                }
            )

    write_csv(
        REPORTS_DIR / "goc_index_selected.csv",
        selected_rows,
        ["patient", "video_index", "selected_file", "reason", "candidate_count"],
    )
    write_csv(
        REPORTS_DIR / "goc_index_ambiguous.csv",
        ambiguous_rows,
        ["patient", "video_index", "candidate_count", "candidates", "selected_file", "reason"],
    )
    return index


def resolve_goc_file(task: ClipTask, patient_index: Dict[int, Dict[int, Path]]) -> Optional[Path]:
    if task.video_index is None:
        return None
    return patient_index.get(task.patient, {}).get(task.video_index)


def load_manual_goc_overrides(path: Path) -> Dict[Tuple[int, int], Path]:
    overrides: Dict[Tuple[int, int], Path] = {}
    if str(path).strip() in {"", "."}:
        return overrides
    if not path.exists():
        return overrides
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                patient = int(str(row.get("patient", "")).strip())
                video_index = int(str(row.get("video_index", "")).strip())
                file_path = Path(str(row.get("file_path", "")).strip())
            except Exception:
                continue
            if file_path.exists():
                overrides[(patient, video_index)] = file_path
    return overrides


def load_missing_task_keys_from_previous_report() -> set[str]:
    keys: set[str] = set()
    report = REPORTS_DIR / "task_results.csv"
    if not report.exists():
        return keys
    with report.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("status", "")).strip() != "skipped":
                continue
            if "source_not_found" not in str(row.get("message", "")):
                continue
            keys.add(
                "|".join(
                    [
                        str(row.get("patient", "")).strip(),
                        str(row.get("stage", "")).strip(),
                        str(row.get("start", "")).strip(),
                        str(row.get("end", "")).strip(),
                        str(row.get("video_index", "")).strip(),
                        str(row.get("source_type", "")).strip(),
                    ]
                )
            )
    return keys


def ffmpeg_trim(input_file: Path, output_file: Path, start_sec: int, end_sec: int) -> Tuple[bool, str]:
    duration = end_sec - start_sec
    output_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-ss",
        str(start_sec),
        "-i",
        str(input_file),
        "-t",
        str(duration),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_file),
    ]
    proc = run(cmd, check=False)
    if proc.returncode == 0:
        return True, "ok"
    return False, proc.stderr[-400:]


def ffmpeg_concat(clips: List[Path], output_file: Path) -> Tuple[bool, str]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if not clips:
        return False, "no_clips"
    list_file = TEMP_DIR / f"concat_{output_file.stem}.txt"
    with list_file.open("w", encoding="utf-8") as f:
        for p in clips:
            escaped = str(p).replace("\\", "\\\\").replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
    cmd = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(output_file),
    ]
    proc = run(cmd, check=False)
    if proc.returncode == 0:
        return True, "ok"
    fallback = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output_file),
    ]
    proc2 = run(fallback, check=False)
    if proc2.returncode == 0:
        return True, "ok_reencoded"
    return False, proc2.stderr[-400:]


def ffmpeg_extract_frames(video_file: Path, stage: str) -> Tuple[bool, str, int]:
    out_pattern = FRAMES_DIR / stage / f"{video_file.stem}_frame_%08d.jpg"
    cmd = [
        "ffmpeg",
        "-y" if OVERWRITE else "-n",
        "-i",
        str(video_file),
        "-vf",
        f"fps={EXTRACT_FPS}",
        "-q:v",
        "2",
        str(out_pattern),
    ]
    proc = run(cmd, check=False)
    if proc.returncode != 0:
        return False, proc.stderr[-400:], 0
    frame_count = len(list((FRAMES_DIR / stage).glob(f"{video_file.stem}_frame_*.jpg")))
    return True, "ok", frame_count


def sample_every_n(stage: str, prefix: str, n: int) -> int:
    src_files = sorted((FRAMES_DIR / stage).glob(f"{prefix}_frame_*.jpg"))
    sampled = src_files[::n]
    target = SAMPLES_DIR / stage
    target.mkdir(parents=True, exist_ok=True)
    for f in sampled:
        dst = target / f.name
        if dst.exists() and not OVERWRITE:
            continue
        shutil.copy2(f, dst)
    return len(sampled)


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def write_csv(path: Path, rows: Iterable[Dict[str, object]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_cleanup_intermediate(stage_clips: Dict[str, List[Path]]) -> None:
    if not DELETE_INTERMEDIATE_AFTER_MERGE:
        return
    if free_gb(CARDIOVIS_RELATED_DIR) >= MIN_FREE_GB_BEFORE_CLEANUP:
        return
    for stage in STAGES:
        for clip in stage_clips.get(stage, []):
            if clip.exists():
                clip.unlink(missing_ok=True)
    for stage in STAGES:
        stage_dir = CLIPS_DIR / stage
        if stage_dir.exists() and not any(stage_dir.iterdir()):
            stage_dir.rmdir()


def maybe_download_samples_locally() -> Dict[str, str]:
    result = {"status": "skipped", "message": ""}
    if not TRY_LOCAL_DOWNLOAD:
        result["message"] = "TRY_LOCAL_DOWNLOAD=False"
        return result

    total_sample_bytes = dir_size_bytes(SAMPLES_DIR)
    required = total_sample_bytes * LOCAL_SAFETY_MULTIPLIER
    LOCAL_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(LOCAL_DOWNLOAD_DIR).free

    if free_bytes < required or free_bytes < LOCAL_MIN_FREE_GB * (1024**3):
        result["status"] = "not_enough_memory"
        result["message"] = (
            f"free_bytes={free_bytes}, required_bytes={int(required)}, "
            f"samples_bytes={total_sample_bytes}"
        )
        return result

    copied = 0
    for stage in STAGES:
        src = SAMPLES_DIR / stage
        dst = LOCAL_DOWNLOAD_DIR / stage
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.jpg"):
            shutil.copy2(f, dst / f.name)
            copied += 1

    result["status"] = "copied_to_local_runtime"
    result["message"] = f"copied_files={copied}, local_dir={LOCAL_DOWNLOAD_DIR}"

    if TRIGGER_BROWSER_DOWNLOAD:
        try:
            from google.colab import files  # type: ignore
        except Exception:
            result["message"] += "; browser_download_unavailable"
            return result
        for stage in STAGES:
            zip_path = LOCAL_DOWNLOAD_DIR / f"{stage}_samples.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in (LOCAL_DOWNLOAD_DIR / stage).glob("*.jpg"):
                    zf.write(f, arcname=f.name)
            files.download(str(zip_path))
        result["status"] = "browser_download_triggered"
    return result


def main() -> None:
    apply_env_overrides()
    maybe_autodetect_paths()
    refresh_output_paths()
    ensure_ffmpeg()
    ensure_dirs()

    if not DOCX_PATH.exists():
        raise FileNotFoundError(f"DOCX not found: {DOCX_PATH}")
    if not MERGED_DIR.exists():
        raise FileNotFoundError(f"Merged folder not found: {MERGED_DIR}")
    if not ROOT_VIDEOS_DIR.exists():
        raise FileNotFoundError(f"3) Videos folder not found: {ROOT_VIDEOS_DIR}")

    tasks, skipped = parse_doc_manifest(DOCX_PATH)
    tasks = [t for t in tasks if t.stage in STAGES]
    tasks.sort(key=lambda x: (x.stage, x.patient, x.start_sec))
    if RERUN_SOURCE_NOT_FOUND_ONLY:
        missing_keys = load_missing_task_keys_from_previous_report()
        if missing_keys:
            tasks = [t for t in tasks if task_key(t) in missing_keys]
            print(
                f"RERUN_SOURCE_NOT_FOUND_ONLY enabled: narrowed to {len(tasks)} tasks.",
                flush=True,
            )
        else:
            print(
                "RERUN_SOURCE_NOT_FOUND_ONLY enabled but no previous missing-source report found.",
                flush=True,
            )
    print(f"Parsed {len(tasks)} valid stage rows from annotation.", flush=True)
    print(f"Skipped rows after parse: {len(skipped)}", flush=True)

    merged_files = video_files_recursive(MERGED_DIR)
    patient_index = build_patient_video_index(ROOT_VIDEOS_DIR)
    manual_overrides = load_manual_goc_overrides(MANUAL_GOC_OVERRIDES_CSV)
    print(f"Indexed merged videos: {len(merged_files)} files", flush=True)
    print(f"Indexed goc patients: {len(patient_index)} patients", flush=True)
    if manual_overrides:
        print(f"Loaded manual goc overrides: {len(manual_overrides)} entries", flush=True)

    manifest_rows = []
    stage_clips: Dict[str, List[Path]] = {s: [] for s in STAGES}
    rerun_stages = set()
    task_results = []
    status_counts: Dict[str, int] = {
        "created": 0,
        "existing": 0,
        "ready": 0,
        "skipped": 0,
        "failed": 0,
    }
    total_tasks = len(tasks)
    write_progress(
        {
            "phase": "trim_segments",
            "total_tasks": total_tasks,
            "processed_tasks": 0,
            "status_counts": status_counts,
            "message": "start",
        }
    )

    for idx, t in enumerate(tasks, start=1):
        clip_name = stage_filename_slug(t)
        out_clip = CLIPS_DIR / t.stage / clip_name
        source_path: Optional[Path]
        if t.source_type == "merged":
            source_path = resolve_merged_file(t, merged_files)
        else:
            source_path = resolve_goc_file(t, patient_index)
            if source_path is None and t.video_index is not None:
                source_path = manual_overrides.get((t.patient, t.video_index))

        row = {
            "patient": t.patient,
            "source_type": t.source_type,
            "stage": t.stage,
            "start": t.start,
            "end": t.end,
            "start_sec": t.start_sec,
            "end_sec": t.end_sec,
            "video_index": t.video_index or "",
            "source_file_path": str(source_path) if source_path else "",
            "clip_file_path": str(out_clip),
            "status": "",
            "message": "",
        }

        if source_path is None:
            row["status"] = "skipped"
            row["message"] = "source_not_found"
            status_counts["skipped"] += 1
            task_results.append(row)
            skipped.append(
                {
                    "patient": str(t.patient),
                    "stage": t.stage,
                    "reason": "source_not_found",
                    "raw": f"source={t.source_type},video_index={t.video_index}",
                }
            )
            print(
                f"[{idx}/{total_tasks}] P{t.patient:02d} {t.stage} -> skipped (source_not_found)",
                flush=True,
            )
            write_progress(
                {
                    "phase": "trim_segments",
                    "total_tasks": total_tasks,
                    "processed_tasks": idx,
                    "status_counts": status_counts,
                    "current_task": {
                        "patient": t.patient,
                        "stage": t.stage,
                        "status": "skipped",
                    },
                }
            )
            continue

        if out_clip.exists() and not OVERWRITE:
            row["status"] = "existing"
            row["message"] = "clip_exists"
            status_counts["existing"] += 1
            task_results.append(row)
            stage_clips[t.stage].append(out_clip)
            if RERUN_SOURCE_NOT_FOUND_ONLY:
                rerun_stages.add(t.stage)
            manifest_rows.append(row)
            print(
                f"[{idx}/{total_tasks}] P{t.patient:02d} {t.stage} -> existing",
                flush=True,
            )
            write_progress(
                {
                    "phase": "trim_segments",
                    "total_tasks": total_tasks,
                    "processed_tasks": idx,
                    "status_counts": status_counts,
                    "current_task": {
                        "patient": t.patient,
                        "stage": t.stage,
                        "status": "existing",
                    },
                }
            )
            continue

        if DRY_RUN:
            row["status"] = "ready"
            row["message"] = "dry_run"
            status_counts["ready"] += 1
            task_results.append(row)
            manifest_rows.append(row)
            print(
                f"[{idx}/{total_tasks}] P{t.patient:02d} {t.stage} -> ready (dry_run)",
                flush=True,
            )
            write_progress(
                {
                    "phase": "trim_segments",
                    "total_tasks": total_tasks,
                    "processed_tasks": idx,
                    "status_counts": status_counts,
                    "current_task": {
                        "patient": t.patient,
                        "stage": t.stage,
                        "status": "ready",
                    },
                }
            )
            continue

        ok, msg = ffmpeg_trim(source_path, out_clip, t.start_sec, t.end_sec)
        row["status"] = "created" if ok else "failed"
        row["message"] = msg
        if ok:
            status_counts["created"] += 1
        else:
            status_counts["failed"] += 1
        task_results.append(row)
        manifest_rows.append(row)
        if ok:
            stage_clips[t.stage].append(out_clip)
            rerun_stages.add(t.stage)
        print(
            f"[{idx}/{total_tasks}] P{t.patient:02d} {t.stage} -> {row['status']}",
            flush=True,
        )
        write_progress(
            {
                "phase": "trim_segments",
                "total_tasks": total_tasks,
                "processed_tasks": idx,
                "status_counts": status_counts,
                "current_task": {
                    "patient": t.patient,
                    "stage": t.stage,
                    "status": row["status"],
                },
            }
        )

    final_rows = []
    merged_outputs: List[Tuple[str, Path]] = []
    stages_to_merge = list(STAGES)
    if RERUN_SOURCE_NOT_FOUND_ONLY and RERUN_AFFECTED_STAGES_ONLY:
        stages_to_merge = sorted(rerun_stages)
        print(f"Rerun mode: merging affected stages only: {stages_to_merge}", flush=True)
    for stage in stages_to_merge:
        if RERUN_SOURCE_NOT_FOUND_ONLY and MISSED_ONLY_STAGE_OUTPUTS:
            clips = sorted(set(stage_clips[stage]), key=lambda p: p.name)
            final_video = FINAL_DIR / f"{stage}_missed_only.mp4"
        else:
            clips = sorted((CLIPS_DIR / stage).glob("*.mp4"), key=lambda p: p.name)
            final_video = FINAL_DIR / f"{stage}_full.mp4"
        print(f"[merge] stage={stage} clip_count={len(clips)}", flush=True)
        if DRY_RUN:
            final_rows.append(
                {"stage": stage, "clip_count": len(clips), "output": str(final_video), "status": "dry_run"}
            )
            merged_outputs.append((stage, final_video))
            continue
        ok, msg = ffmpeg_concat(clips, final_video)
        final_rows.append(
            {"stage": stage, "clip_count": len(clips), "output": str(final_video), "status": "ok" if ok else "failed", "message": msg}
        )
        if ok:
            merged_outputs.append((stage, final_video))
        write_progress(
            {
                "phase": "merge_stage_videos",
                "stage": stage,
                "clip_count": len(clips),
                "status": "ok" if ok else "failed",
            }
        )

    maybe_cleanup_intermediate(stage_clips)

    frame_rows = []
    sample_rows = []
    if not DRY_RUN and not SKIP_FRAME_EXTRACT:
        extract_targets = merged_outputs
        if not extract_targets and not MISSED_ONLY_STAGE_OUTPUTS:
            extract_targets = [(stage, FINAL_DIR / f"{stage}_full.mp4") for stage in STAGES]
        for stage, final_video in extract_targets:
            if not final_video.exists():
                frame_rows.append({"stage": stage, "status": "skipped", "message": "final_video_missing", "frame_count": 0})
                print(f"[extract] stage={stage} skipped (final video missing)", flush=True)
                continue
            print(f"[extract] stage={stage} fps={EXTRACT_FPS}", flush=True)
            ok, msg, frame_count = ffmpeg_extract_frames(final_video, stage)
            frame_rows.append(
                {"stage": stage, "status": "ok" if ok else "failed", "message": msg, "frame_count": frame_count}
            )
            sampled = sample_every_n(stage, final_video.stem, SAMPLE_EVERY_N_FRAMES) if ok else 0
            sample_rows.append({"stage": stage, "sample_rule": "1_per_20", "sample_count": sampled})
            print(
                f"[sample] stage={stage} extracted_frames={frame_count} sampled={sampled}",
                flush=True,
            )
            write_progress(
                {
                    "phase": "extract_and_sample",
                    "stage": stage,
                    "extract_status": "ok" if ok else "failed",
                    "frame_count": frame_count,
                    "sample_count": sampled,
                }
            )
    elif SKIP_FRAME_EXTRACT:
        print("Skipping frame extraction (SKIP_FRAME_EXTRACT=1).", flush=True)
        for stage in STAGES:
            frame_rows.append(
                {
                    "stage": stage,
                    "status": "skipped_by_flag",
                    "message": "SKIP_FRAME_EXTRACT=1",
                    "frame_count": 0,
                }
            )
            sample_rows.append({"stage": stage, "sample_rule": "1_per_20", "sample_count": 0})

    if not DRY_RUN and not SKIP_SAMPLE_DOWNLOAD:
        download_status = maybe_download_samples_locally()
    elif DRY_RUN:
        download_status = {"status": "dry_run", "message": ""}
    else:
        download_status = {"status": "skipped_by_flag", "message": "SKIP_SAMPLE_DOWNLOAD=1"}
    write_progress(
        {
            "phase": "download_samples",
            "status": download_status.get("status", "unknown"),
            "message": download_status.get("message", ""),
        }
    )

    write_csv(
        REPORTS_DIR / "manifest_resolved.csv",
        manifest_rows,
        [
            "patient",
            "source_type",
            "stage",
            "start",
            "end",
            "start_sec",
            "end_sec",
            "video_index",
            "source_file_path",
            "clip_file_path",
            "status",
            "message",
        ],
    )
    write_csv(REPORTS_DIR / "task_results.csv", task_results, list(task_results[0].keys()) if task_results else ["status"])
    write_csv(REPORTS_DIR / "skipped_rows.csv", skipped, ["patient", "stage", "reason", "raw"])
    write_csv(REPORTS_DIR / "final_merge_report.csv", final_rows, ["stage", "clip_count", "output", "status", "message"])
    write_csv(REPORTS_DIR / "frame_extract_report.csv", frame_rows, ["stage", "status", "message", "frame_count"])
    write_csv(REPORTS_DIR / "sample_report.csv", sample_rows, ["stage", "sample_rule", "sample_count"])
    write_csv(
        REPORTS_DIR / "download_status.csv",
        [download_status],
        ["status", "message"],
    )

    print("Pipeline complete.")
    print(f"Reports: {REPORTS_DIR}")
    print(f"Final videos: {FINAL_DIR}")
    print(f"Frames: {FRAMES_DIR}")
    print(f"Samples: {SAMPLES_DIR}")
    print(f"Download status: {download_status}")
    write_progress(
        {
            "phase": "done",
            "status": "completed",
            "reports_dir": str(REPORTS_DIR),
            "final_dir": str(FINAL_DIR),
            "frames_dir": str(FRAMES_DIR),
            "samples_dir": str(SAMPLES_DIR),
        }
    )


if __name__ == "__main__":
    main()
