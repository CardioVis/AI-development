#!/usr/bin/env python3
"""Run Features_timestamps trim locally using Drive API streaming (no full video download).

Writes to local stage_outputs/, then uploads clips / samples / reports / frames to Drive.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Point pipeline outputs at local repo stage_outputs
os.environ.setdefault("CARDIOVIS_RELATED_DIR", str(ROOT))
os.environ.setdefault("FEATURES_TIMESTAMPS_XLSX", str(ROOT / "tmp" / "Features_timestamps.xlsx"))
os.environ.setdefault("DRY_RUN", "0")
os.environ.setdefault("OVERWRITE", "0")

import features_timestamps_trim_pipeline as pipe  # noqa: E402

MCP_CFG = Path.home() / ".config" / "google-docs-mcp"
CARDIOVIS_FOLDER_ID = "1rXt_X2rQkJM-Rae2yhTVgVXZSCWAVmOz"
VIDEOS_ROOT_ID = "1GYGc4JzJotpJ_piuPDQn_a-Oynr4jX2F"
STAGE_OUTPUTS_ID = "1HV6f5YCkJSeAV57HdHgMfomBO-jCN8gn"


def drive_creds() -> Credentials:
    env = {
        k.strip(): v.strip().strip('"')
        for k, v in (
            line.split("=", 1)
            for line in (MCP_CFG / "credentials.env").read_text().splitlines()
            if "=" in line
        )
    }
    token = json.loads((MCP_CFG / "token.json").read_text())
    creds = Credentials(
        None,
        refresh_token=token["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=env["GOOGLE_CLIENT_ID"],
        client_secret=env["GOOGLE_CLIENT_SECRET"],
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    creds.refresh(Request())
    return creds


def list_children(drive, folder_id: str) -> list[dict]:
    files: list[dict] = []
    page = None
    while True:
        res = (
            drive.files()
            .list(
                q=f"'{folder_id}' in parents and trashed=false",
                fields="nextPageToken,files(id,name,mimeType,size)",
                pageSize=200,
                pageToken=page,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        files.extend(res.get("files", []))
        page = res.get("nextPageToken")
        if not page:
            break
    return files


def ensure_child_folder(drive, parent_id: str, name: str) -> str:
    for f in list_children(drive, parent_id):
        if f["name"] == name and f["mimeType"].endswith("folder"):
            return f["id"]
    meta = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    created = drive.files().create(body=meta, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def build_drive_video_index(drive) -> dict[str, list[dict]]:
    """Map lowercase filename -> [{id, name, path_hint}, ...] under 3) Videos."""
    index: dict[str, list[dict]] = {}

    def walk(folder_id: str, path_hint: str, depth: int = 0) -> None:
        if depth > 8:
            return
        for f in list_children(drive, folder_id):
            name = f["name"]
            if f["mimeType"].endswith("folder"):
                walk(f["id"], f"{path_hint}/{name}", depth + 1)
                continue
            if not name.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv")):
                continue
            if name.startswith("._"):
                continue
            index.setdefault(name.lower(), []).append(
                {"id": f["id"], "name": name, "path_hint": f"{path_hint}/{name}"}
            )

    walk(VIDEOS_ROOT_ID, "3) Videos")
    return index


def resolve_drive_video(task: pipe.FeatureTask, index: dict[str, list[dict]]) -> tuple[dict | None, str]:
    candidates = index.get(task.video_name.lower(), [])
    if not candidates:
        return None, "source_not_found"
    if len(candidates) == 1:
        return candidates[0], "unique_filename"
    patient_token = f"patient {task.patient}"
    preferred = [
        c
        for c in candidates
        if patient_token in c["path_hint"].lower()
        or f"patient_{task.patient}" in c["path_hint"].lower()
    ]
    if preferred:
        return preferred[0], "patient_folder_match"
    return candidates[0], "ambiguous_filename_first"


def ffmpeg_trim_drive(creds: Credentials, file_id: str, dst: Path, start_sec: int, end_sec: int) -> tuple[bool, str]:
    duration = end_sec - start_sec
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not pipe.OVERWRITE:
        return True, "existing"
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    cmd = [
        "ffmpeg",
        "-y" if pipe.OVERWRITE else "-n",
        "-ss",
        str(start_sec),
        "-t",
        str(duration),
        "-headers",
        f"Authorization: Bearer {creds.token}\r\n",
        "-i",
        url,
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
    if pipe.DRY_RUN:
        return True, "dry_run"
    # Refresh token if needed before long ffmpeg
    if not creds.valid:
        creds.refresh(Request())
        cmd[cmd.index("-headers") + 1] = f"Authorization: Bearer {creds.token}\r\n"
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
        return True, "created"
    return False, (proc.stderr or "")[-500:]


def upload_file(drive, local_path: Path, parent_id: str, mime: str | None = None) -> str:
    name = local_path.name
    existing = [
        f
        for f in list_children(drive, parent_id)
        if f["name"] == name and not f["mimeType"].endswith("folder")
    ]
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    if existing:
        drive.files().update(fileId=existing[0]["id"], media_body=media, supportsAllDrives=True).execute()
        return existing[0]["id"]
    meta = {"name": name, "parents": [parent_id]}
    created = drive.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def upload_tree(drive, local_dir: Path, parent_id: str, pattern: str = "*") -> int:
    if not local_dir.exists():
        return 0
    n = 0
    for path in sorted(local_dir.glob(pattern)):
        if not path.is_file():
            continue
        upload_file(drive, path, parent_id)
        n += 1
        if n % 20 == 0:
            print(f"  uploaded {n} files into {local_dir.name}...", flush=True)
    return n


def main() -> None:
    pipe.ensure_ffmpeg()
    pipe.ensure_dirs()

    creds = drive_creds()
    drive = build("drive", "v3", credentials=creds)

    tasks = pipe.parse_tasks(pipe.XLSX_PATH)
    print(f"Parsed {len(tasks)} tasks", flush=True)

    print("Indexing Drive videos under 3) Videos...", flush=True)
    video_index = build_drive_video_index(drive)
    print(f"Indexed {sum(len(v) for v in video_index.values())} videos", flush=True)

    next_idx_by_stage = {stage: pipe.max_existing_frame_idx(stage) for stage in pipe.STAGE_SHEETS.values()}
    # Local frames may be empty; force continuation from known batch ends.
    for stage, fallback in pipe.FALLBACK_LAST_FRAME_IDX.items():
        next_idx_by_stage[stage] = max(next_idx_by_stage[stage], fallback)
    print(f"Starting frame indices: {next_idx_by_stage}", flush=True)

    task_rows: list[dict] = []
    frame_map_rows: list[dict] = []

    for i, task in enumerate(tasks, start=1):
        hit, resolve_msg = resolve_drive_video(task, video_index)
        clip_name = pipe.clip_filename(task)
        out_clip = pipe.CLIPS_DIR / task.stage / clip_name
        task.clip_path = out_clip
        task.source_path = Path(hit["path_hint"]) if hit else None

        print(
            f"[{i}/{len(tasks)}] P{task.patient:02d} {task.stage} "
            f"{task.start_sec}-{task.end_sec}s video={task.video_name}",
            flush=True,
        )

        if hit is None:
            task.status = "skipped"
            task.message = resolve_msg
            task_rows.append(pipe.task_to_row(task, resolve_msg))
            continue

        # Refresh token periodically
        if not creds.valid or (creds.expiry and True):
            creds.refresh(Request())

        ok, msg = ffmpeg_trim_drive(creds, hit["id"], out_clip, task.start_sec, task.end_sec)
        if not ok:
            task.status = "failed_trim"
            task.message = msg
            task_rows.append(pipe.task_to_row(task, resolve_msg))
            print(f"  TRIM FAIL: {msg[:200]}", flush=True)
            continue

        start_next = next_idx_by_stage[task.stage]
        ok2, msg2, f_start, f_end, f_count = pipe.extract_and_renumber_frames(task, start_next)
        if not ok2:
            task.status = "failed_extract"
            task.message = msg2
            task_rows.append(pipe.task_to_row(task, resolve_msg))
            print(f"  EXTRACT FAIL: {msg2[:200]}", flush=True)
            continue

        samples = pipe.sample_new_frames(task.stage, f_start, f_end)
        next_idx_by_stage[task.stage] = f_end
        task.status = "ok"
        task.message = f"{resolve_msg}; trim={msg}; extract={msg2}"
        task.frame_start_idx = f_start
        task.frame_end_idx = f_end
        task.frame_count = f_count
        task.sample_count = samples
        task_rows.append(pipe.task_to_row(task, resolve_msg))
        frame_map_rows.append(
            {
                "patient": task.patient,
                "stage": task.stage,
                "clip_name": clip_name,
                "source_path": hit["path_hint"],
                "start_sec": task.start_sec,
                "end_sec": task.end_sec,
                "frame_start_idx": f_start,
                "frame_end_idx": f_end,
                "frame_count": f_count,
                "sample_count": samples,
            }
        )
        print(f"  -> frames={f_start}-{f_end} n={f_count} samples={samples}", flush=True)

    reports = pipe.REPORTS_DIR
    pipe.write_csv(
        reports / "features_timestamps_task_results.csv",
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
    pipe.write_csv(
        reports / "features_frame_index_map.csv",
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
    # Also copy map into tmp for append_features_sample_manifest.py
    shutil.copy2(reports / "features_frame_index_map.csv", ROOT / "tmp" / "features_frame_index_map.csv")

    ok_n = sum(1 for r in task_rows if r["status"] == "ok")
    print(f"Local pipeline done: ok={ok_n}/{len(task_rows)}", flush=True)

    print("Uploading outputs to Drive stage_outputs/...", flush=True)
    clips_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "clips")
    frames_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "frames")
    samples_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "frame_samples_1_per_20")
    reports_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "reports")

    for stage in ("dat_kim_goc", "rach_nhi"):
        stage_clips = ensure_child_folder(drive, clips_id, stage)
        stage_frames = ensure_child_folder(drive, frames_id, stage)
        stage_samples = ensure_child_folder(drive, samples_id, stage)

        # Only upload NEW feature clips / samples / frames (indices after fallback)
        clip_dir = pipe.CLIPS_DIR / stage
        for clip in sorted(clip_dir.glob("patient_3*_*.mp4")):
            # Features patients are 33+
            m = re.search(r"patient_(\d+)", clip.name)
            if not m or int(m.group(1)) < 33:
                continue
            print(f"  upload clip {clip.name}", flush=True)
            upload_file(drive, clip, stage_clips, "video/mp4")

        fallback = pipe.FALLBACK_LAST_FRAME_IDX[stage]
        frame_dir = pipe.FRAMES_DIR / stage
        new_frames = [
            p
            for p in frame_dir.glob(f"{stage}_full_frame_*.jpg")
            if int(re.search(r"_frame_(\d+)\.jpg$", p.name).group(1)) > fallback
        ]
        print(f"  upload {len(new_frames)} frames for {stage}", flush=True)
        for p in sorted(new_frames):
            upload_file(drive, p, stage_frames, "image/jpeg")

        sample_dir = pipe.SAMPLES_DIR / stage
        new_samples = [
            p
            for p in sample_dir.glob(f"{stage}_full_frame_*.jpg")
            if int(re.search(r"_frame_(\d+)\.jpg$", p.name).group(1)) > fallback
        ]
        print(f"  upload {len(new_samples)} samples for {stage}", flush=True)
        for p in sorted(new_samples):
            upload_file(drive, p, stage_samples, "image/jpeg")

    for report_name in (
        "features_timestamps_task_results.csv",
        "features_frame_index_map.csv",
    ):
        upload_file(drive, reports / report_name, reports_id, "text/csv")
        print(f"  uploaded report {report_name}", flush=True)

    print("All uploads complete.", flush=True)
    print(f"Final next indices: {next_idx_by_stage}", flush=True)


if __name__ == "__main__":
    main()
