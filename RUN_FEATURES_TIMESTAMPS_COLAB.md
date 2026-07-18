# Run: Features_timestamps trim (Colab or local Drive runner)

## Files on `My Drive/Cardiovis-related/`

Already uploaded:

1. [Features_timestamps.xlsx](https://drive.google.com/file/d/1LwgZX2q1SZF4qb-uG020RCeibBsGtOi6/view) (binary Excel)
2. [Features_timestamps](https://docs.google.com/spreadsheets/d/1sxuI17hU4Cf7CdgzBJMc5v6Z_UHthDa7bcK_izO09TQ/edit) (Sheets mirror)
3. [features_timestamps_trim_pipeline.py](https://drive.google.com/file/d/1PFyrvqv5DeklVI7dQxsV9H8kRBoLjwLV/view)
4. [CardioVis_Features_Timestamps_Trim_Colab.ipynb](https://drive.google.com/file/d/1VlRDnxDFI8oFaCpWoPqhPAwuIckjYEXe/view)
5. [RUN Features_timestamps Colab](https://docs.google.com/document/d/1JjlGGViPHDU42To1fja9kuafc2N9WW3tQJBzWLKZitk/edit) (this doc)

## Option A — Colab

1. Open `CardioVis_Features_Timestamps_Trim_Colab.ipynb` in Colab.
2. Mount Drive.
3. Run with `DRY_RUN = True` first → check `stage_outputs/reports/features_timestamps_task_results.csv`.
4. Set `DRY_RUN = False`, run again to trim + extract + sample.
5. Confirm new files under:
   - `stage_outputs/clips/{dat_kim_goc,rach_nhi}/`
   - `stage_outputs/frames/{stage}/` (indices continue after 6917 / 5980)
   - `stage_outputs/frame_samples_1_per_20/{stage}/` (next samples 6921… / 5981…)

## Option B — Local Drive streaming (already used)

From the repo (uses MCP Google OAuth + ffmpeg HTTP stream; no full ~1GB download):

```bash
python3 run_features_timestamps_via_drive.py
```

This writes local `stage_outputs/`, uploads clips/frames/samples/reports to Drive, and copies `features_frame_index_map.csv` into `tmp/`.

## After pipeline finishes (local manifests)

```bash
python3 append_features_sample_manifest.py
# or:
python3 build_sample_patient_manifest.py --include-features-batch
```

## Time parse reminder

Excel shows `HH:MM:SS` but values mean **MM:SS** → `seconds = hour*60 + minute` (each clip ≈ 10s).

## Assumed Label Studio IDs (until real upload)

- `dat_kim_goc` new samples: **5956–6005** (~50; actual LS upload IDs)
- `rach_nhi` new samples: **6006–6095** (~90; actual LS upload IDs)

## Verify on Drive

Under `Cardiovis-related/stage_outputs/`:

| Path | Expect |
|---|---|
| `clips/dat_kim_goc/` | 5 new patient_34/35/39/40/43 clips |
| `clips/rach_nhi/` | 9 new clips (incl. patient_33 `0_05-0_15`) |
| `frames/*/` | ~1000 + ~1800 new jpgs after 6917 / 5980 |
| `frame_samples_1_per_20/*/` | ~50 + ~90 new samples |
| `reports/features_*.csv` | task results + frame index map |
