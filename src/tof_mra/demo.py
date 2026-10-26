"""Small, entirely synthetic execution example. It does not run or simulate training."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib

from .common import load_config, write_cases, write_json
from .dataset import prepare_study, inspect_study
from .evaluation import evaluate
from .comparison import compare_reports


def run_demo(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Demo output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    # Also works when installed from a wheel, without repository-relative assets.
    cfg = {
        "manifest": "cases.csv", "work_dir": "work",
        "dataset": {"id": 990, "name": "SyntheticDemo", "channel_name": "CT", "expected_development": 6,
                    "expected_test": 6, "expected_validation": None, "resample_labels": True, "splits_file": None},
        "nnunet": {"raw": "work/nnUNet_raw", "preprocessed": "work/nnUNet_preprocessed", "results": "work/nnUNet_results",
                   "device": "cpu", "threads": 1, "augmentation_processes": 2, "preprocessing_processes": 2, "prediction_processes": 2},
        "baseline": {"dataset": "Dataset666_Vessel", "results": "unused_baseline_weights", "configuration": "2d", "folds": [0], "tta": True},
        "pediatric": {"configuration": "3d_fullres", "fold": 0, "tta": True},
        "evaluation": {"brain_mask_policy": "required", "bahd_normalization": "reference", "expected_pairs": 6,
                       "nonfinite_policy": "error", "font_family": "Arial"},
    }
    write_json(output / "study.json", cfg)
    rows = []
    affine = np.diag([0.45, 0.45, 0.7, 1.0])
    rng = np.random.default_rng(12345)
    for split in ("development", "test"):
        for i in range(6):
            case = ("dev" if split == "development" else "test") + f"{i+1:03d}"
            vessel = np.zeros((24, 24, 24), dtype=np.uint8)
            vessel[10:13, 10:13, 3:21] = 1
            vessel[12:19, 10:13, 13:16] = 1
            image_data = (30 + rng.normal(0, 3, vessel.shape) + vessel.astype(np.float32) * (400+i*15)).astype(np.float32)
            folder = output / "data" / case
            folder.mkdir(parents=True)
            paths = {name: folder / f"{name}.nii.gz" for name in ("tof", "label", "brain_mask")}
            for name, array in (("tof", image_data), ("label", vessel), ("brain_mask", np.ones(vessel.shape, dtype=np.uint8))):
                img = nib.Nifti1Image(array, affine)
                img.header.set_xyzt_units("mm")
                nib.save(img, paths[name])
            rows.append({"case_id": case, "subject_id": case, "split": split, **{k: str(v) for k, v in paths.items()}, "t1": ""})
            if split == "test":
                baseline = vessel.copy()
                baseline[10:13, 10:13, 3:6+i] = 0
                baseline[4:6, 4:6, 4:6] = 1
                pediatric = vessel.copy()
                pediatric[10, 10, 3:4+i] = 0
                for model, array in (("baseline", baseline), ("pediatric", pediatric)):
                    pred_dir = output / "work" / "predictions" / model
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pred = nib.Nifti1Image(array, affine)
                    pred.header.set_xyzt_units("mm")
                    nib.save(pred, pred_dir / f"{case}.nii.gz")
    write_cases(output / "cases.csv", rows)
    config = load_config(output / "study.json")
    inspect_study(config)
    prepare_study(config)
    baseline_csv = evaluate(config, "baseline")
    pediatric_csv = evaluate(config, "pediatric")
    comparison = output / "work" / "comparison"
    compare_reports(baseline_csv, pediatric_csv, comparison, expected_pairs=6,
                    figure_note="Synthetic demonstration - not study results")
    write_json(output / "DEMO.json", {"synthetic": True, "n_test": 6,
                                     "note": "Artificial masks were generated to exercise the evaluator. No training or real model inference was performed."})
    return comparison

