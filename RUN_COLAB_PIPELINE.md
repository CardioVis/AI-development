# Run Stage Pipeline In Colab

1. Open `CardioVis_Segment_and_Merge_By_Stage_Colab.ipynb` in Google Colab.
2. Run the mount/setup cells.
3. Ensure `colab_drive_stage_pipeline.py` is available at:
   - `/content/drive/MyDrive/AI-development/colab_drive_stage_pipeline.py`
4. Run the final runner cell.

## Notes for "goc" videos traversal

- The resolver now traverses exactly from `3) Videos` into `Valve ...` folders, then `Patient N`,
  and keeps descending until it finds folders that list video files.
- It maps `video N` / `vidN` from the annotation to filenames like `VID00N.mp4` or `N.mp4`.
- If configured paths are wrong, set `AUTODETECT_DRIVE_PATHS=1` (already enabled in the notebook runner).

## Output locations on Drive

- Final merged videos:
  - `Cardiovis-related/stage_outputs/final/dat_kim_goc_full.mp4`
  - `Cardiovis-related/stage_outputs/final/rach_nhi_full.mp4`
  - `Cardiovis-related/stage_outputs/final/than_kinh_hoanh_full.mp4`
- Extracted frames (FPS=20):
  - `Cardiovis-related/stage_outputs/frames/dat_kim_goc/`
  - `Cardiovis-related/stage_outputs/frames/rach_nhi/`
  - `Cardiovis-related/stage_outputs/frames/than_kinh_hoanh/`
- Sampled frames (1 per 20):
  - `Cardiovis-related/stage_outputs/frame_samples_1_per_20/dat_kim_goc/`
  - `Cardiovis-related/stage_outputs/frame_samples_1_per_20/rach_nhi/`
  - `Cardiovis-related/stage_outputs/frame_samples_1_per_20/than_kinh_hoanh/`
- Reports:
  - `Cardiovis-related/stage_outputs/reports/`

## Memory-safe behavior

- The pipeline can delete intermediate clips after merge if Drive free space is low.
- Sample frame download to local runtime is gated by disk free checks.

## Re-run missing clips and merge back

- Set `RERUN_SOURCE_NOT_FOUND_ONLY=1` to process only rows that were previously `source_not_found`.
- The script now rebuilds final stage videos from all clips in `stage_outputs/clips/*`, so newly recovered clips are merged back automatically.
- Keep `STRICT_AMBIGUOUS_GOC=1` to skip ambiguous `goc` mappings.
- If needed, provide `MANUAL_GOC_OVERRIDES_CSV` with columns:
  - `patient,video_index,file_path`
