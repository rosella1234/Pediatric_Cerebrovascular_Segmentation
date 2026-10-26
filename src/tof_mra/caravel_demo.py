"""Synthetic regional-summary demo; it does not generate study results or train models."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib

from .common import load_config, write_json, write_cases
from .caravel_data import collect, ROOTS, FEATURES
from .caravel_stats import analyze


def run_demo(output: Path) -> Path:
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty directory for the Caravel demo.")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "manifest": "cases.csv", "work_dir": "work",
        "dataset": {"id": 990, "name": "SyntheticCaravel", "expected_development": 0, "expected_test": 19},
        "evaluation": {"font_family": "Arial"},
        "caravel": {"atlas": "atlas.nii.gz", "demographics": "demographics.csv", "region_labels": list(range(1, 33)),
                    "age_bin_width": 5, "seed": 42, "minimum_age_n": 10, "minimum_sex_n": 5, "minimum_region_n": 5,
                    "figure_note": "Synthetic demonstration - not study results"},
    }
    ages = [.25, 1, 2, 4, 6, 7, 9, 10, 11, 12, 13, 14, 14.5, 15, 16, 17, 18, 19, 19.5]
    sexes = ["M"]*7 + ["M", "F", "M", "F", "M", "F"]*2
    rows, demographics = [], []
    rng = np.random.default_rng(456)
    dummy = output / "synthetic_image.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4)), dummy)
    dummy_label = output / "synthetic_label.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.uint8), np.eye(4)), dummy_label)
    # Unique paths are retained per case to satisfy manifest correspondence checks.
    for i, (age, sex) in enumerate(zip(ages, sexes)):
        case = f"test{i+1:03d}"
        folder = output / "data" / case
        folder.mkdir(parents=True)
        for name in ("tof", "label"):
            import shutil
            shutil.copy2(dummy if name == "tof" else dummy_label, folder / f"{name}.nii.gz")
        rows.append({"case_id": case, "subject_id": case, "split": "test", "tof": str(folder/"tof.nii.gz"), "label": str(folder/"label.nii.gz"), "brain_mask": "", "t1": ""})
        demographics.append({"case_id": case, "age_years": age, "sex": sex})
        for model in ("baseline", "pediatric"):
            records = []
            for region in range(1, 33):
                spatial = 1.5 + np.sin(region/5) + .025*region
                scale = spatial * (1 + .007*age + (.05 if sex == "M" else 0))
                noise = rng.uniform(.85, 1.15)
                multiplier = 1 if model == "baseline" else 1.04
                record = {"region_label": region, "total_length": 100*scale*noise*multiplier,
                          "volume": 35*scale*noise*multiplier, "num_components": max(1, round(5*spatial*noise*(.75 if model == "pediatric" else 1))),
                          "num_bifurcations": round(12*scale*noise), "num_loops": round(2*scale*noise),
                          "num_abnormal_degree_nodes": round(spatial*noise*(.55 if model == "pediatric" else 1.5))}
                for j, root in enumerate(ROOTS):
                    record[root+"mean_curvature"] = (.02 + .0001*age + .001*region)*(1+j*.1)*noise
                    record[root+"mean_square_curvature"] = (.004 + .00001*age + .0001*region)*(1+j*.1)*noise
                records.append(record)
            target = output / "work" / "caravel" / model / case
            target.mkdir(parents=True)
            pd.DataFrame(records).to_csv(target / "region_summary.csv", index=False)
    # An artificial label atlas supports code-path testing only; it is not anatomical.
    atlas = np.zeros((32, 32, 16), dtype=np.int16)
    for x in range(8):
        for y in range(4):
            atlas[x*4:(x+1)*4, y*8:(y+1)*8, 2:14] = x*4+y+1
    nib.save(nib.Nifti1Image(atlas, np.eye(4)), output / "atlas.nii.gz")
    write_cases(output / "cases.csv", rows)
    pd.DataFrame(demographics).to_csv(output / "demographics.csv", index=False)
    write_json(output / "study.json", config)
    cfg = load_config(output / "study.json")
    collect(cfg)
    result = analyze(cfg)
    write_json(output / "DEMO.json", {"synthetic": True, "note": "Artificial regional tables and artificial non-anatomical atlas. No graph extraction, registration, model inference, or training was performed by this demo."})
    return result

