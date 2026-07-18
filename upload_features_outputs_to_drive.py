#!/usr/bin/env python3
"""Finish uploading Features_timestamps outputs to Drive (clips, samples, reports, frame zips)."""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

ROOT = Path(__file__).resolve().parent
MCP_CFG = Path.home() / ".config" / "google-docs-mcp"
STAGE_OUTPUTS_ID = "1HV6f5YCkJSeAV57HdHgMfomBO-jCN8gn"
FALLBACK = {"dat_kim_goc": 6917, "rach_nhi": 5980}


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
                fields="nextPageToken,files(id,name,mimeType)",
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
    return drive.files().create(body=meta, fields="id", supportsAllDrives=True).execute()["id"]


def upload_file(drive, local_path: Path, parent_id: str, mime: str | None = None) -> str:
    name = local_path.name
    existing = [f for f in list_children(drive, parent_id) if f["name"] == name and not f["mimeType"].endswith("folder")]
    media = MediaFileUpload(str(local_path), mimetype=mime, resumable=True)
    if existing:
        drive.files().update(fileId=existing[0]["id"], media_body=media, supportsAllDrives=True).execute()
        return existing[0]["id"]
    meta = {"name": name, "parents": [parent_id]}
    return drive.files().create(body=meta, media_body=media, fields="id", supportsAllDrives=True).execute()["id"]


def main() -> None:
    creds = drive_creds()
    drive = build("drive", "v3", credentials=creds)

    clips_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "clips")
    frames_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "frames")
    samples_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "frame_samples_1_per_20")
    reports_id = ensure_child_folder(drive, STAGE_OUTPUTS_ID, "reports")

    for stage in ("dat_kim_goc", "rach_nhi"):
        stage_clips = ensure_child_folder(drive, clips_id, stage)
        stage_samples = ensure_child_folder(drive, samples_id, stage)
        stage_frames = ensure_child_folder(drive, frames_id, stage)
        cutoff = FALLBACK[stage]

        clip_dir = ROOT / "stage_outputs" / "clips" / stage
        for clip in sorted(clip_dir.glob("patient_*.mp4")):
            m = re.search(r"patient_(\d+)", clip.name)
            if not m or int(m.group(1)) < 33:
                continue
            print(f"upload clip {clip.name}", flush=True)
            upload_file(drive, clip, stage_clips, "video/mp4")

        sample_dir = ROOT / "stage_outputs" / "frame_samples_1_per_20" / stage
        new_samples = [
            p
            for p in sample_dir.glob(f"{stage}_full_frame_*.jpg")
            if int(re.search(r"_frame_(\d+)\.jpg$", p.name).group(1)) > cutoff
        ]
        print(f"upload {len(new_samples)} samples for {stage}", flush=True)
        for i, p in enumerate(sorted(new_samples), 1):
            upload_file(drive, p, stage_samples, "image/jpeg")
            if i % 25 == 0:
                print(f"  samples {i}/{len(new_samples)}", flush=True)

        # Zip new frames for Drive (faster than 1000+ individual uploads)
        frame_dir = ROOT / "stage_outputs" / "frames" / stage
        new_frames = [
            p
            for p in frame_dir.glob(f"{stage}_full_frame_*.jpg")
            if int(re.search(r"_frame_(\d+)\.jpg$", p.name).group(1)) > cutoff
        ]
        zip_path = ROOT / "stage_outputs" / "tmp" / f"features_frames_{stage}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"zipping {len(new_frames)} frames -> {zip_path.name}", flush=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for p in sorted(new_frames):
                zf.write(p, arcname=p.name)
        print(f"upload zip {zip_path.name} ({zip_path.stat().st_size/1e6:.1f} MB)", flush=True)
        upload_file(drive, zip_path, stage_frames, "application/zip")

    reports = ROOT / "stage_outputs" / "reports"
    for name in ("features_timestamps_task_results.csv", "features_frame_index_map.csv"):
        print(f"upload report {name}", flush=True)
        upload_file(drive, reports / name, reports_id, "text/csv")

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
