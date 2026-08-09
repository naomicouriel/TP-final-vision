# Archive

Development notes and one-off scripts kept for traceability. Nothing here is part of the
production path — the maintained code lives in [`src/`](../src) and the entry points are
described in the [top-level README](../README.md).

Paths inside these files are relative to the repository root, so they must be run from
there (`python archive/analyze_ecoglasses_predictions.py`) rather than from this directory.

## Notes

| File | Contents |
|------|----------|
| `REALTIME_DETECTION_GUIDE.md` | Investigation into why real-time ecoglasses detection underperformed static images, and the rejected fine-tuning experiment (58.9% → 46.3%) |
| `ECOGLASSES_PREDICTION_GUIDE.md` | Per-class analysis of the custom ecoglasses set |
| `ECOGLASSES_QUICKSTART.md` | Short runbook for reproducing the ecoglasses evaluation |
| `BUG_FIXES_FINAL.md` | Log of the hardware-integration bugs and their fixes |

## Scripts

| File | Purpose |
|------|---------|
| `analyze_ecoglasses_predictions.py` | Confusion breakdown over the ecoglasses predictions CSV |
| `test_ecoglasses_prediction.py` | Ad-hoc check of the predictor on the custom set |
| `test_finetuned_model.py` | Comparison of the fine-tuned checkpoint against the original |
| `finetune_ecoglasses.py` | Full fine-tuning experiment (rejected — caused catastrophic forgetting) |
| `finetune_ecoglasses_simple.py` | Reduced variant of the same experiment |
| `convert_heic_images.py` | Bulk HEIC → JPEG conversion for the phone-captured dataset |
| `generate_notebooks.py` | Generator used to scaffold the notebooks in `notebooks/` |
| `dump.py` | Concatenates the project source into `project_dump.txt` |

## Data

| File | Contents |
|------|----------|
| `ecoglasses_bandeja_predictions.csv` | Raw prediction output over the 95 custom tray images |
| `project_dump.txt` | Flattened source snapshot produced by `dump.py` |
| `OLD_tp_final_pipeline.ipynb` | Superseded single-notebook version of the pipeline |
