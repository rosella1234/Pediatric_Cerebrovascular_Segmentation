"""Evaluate the same explicitly enumerated independent test cases for each model."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from .common import read_study, write_json, versions, sha256
from .imaging import image, binary_data, require_geometry
from .metrics import overlap_metrics, distance_metrics


def evaluate(cfg: dict, model: str, predictions: Path | None = None, output: Path | None = None) -> Path:
    cases = [r for r in read_study(cfg) if r["split"] == "test"]
    if not cases:
        raise ValueError("No independent test cases in the manifest.")
    predictions = predictions or Path(cfg["work_dir"]) / "predictions" / model
    output = output or Path(cfg["work_dir"]) / "evaluation" / f"{model}.csv"
    policy = cfg.get("evaluation", {}).get("brain_mask_policy", "required")
    if policy not in {"required", "none"}:
        raise ValueError("evaluation.brain_mask_policy must be required or none.")
    records = []
    for row in cases:
        case = row["case_id"]
        source, ref = image(row["tof"]), image(row["label"])
        pred_path = predictions / f"{case}.nii.gz"
        pred = image(pred_path)
        require_geometry(source, ref, context=f"{case}: TOF/reference")
        require_geometry(source, pred, context=f"{case}: TOF/prediction")
        pred_mask, ref_mask = binary_data(pred), binary_data(ref)
        if policy == "required":
            if not row.get("brain_mask"):
                raise ValueError(f"Brain mask missing for {case}. The source study scripts require it; select policy none explicitly for an unmasked analysis.")
            brain = image(row["brain_mask"])
            require_geometry(source, brain, context=f"{case}: TOF/brain mask")
            brain_mask = binary_data(brain)
            if not brain_mask.any():
                raise ValueError(f"Brain mask is empty: {case}")
            pred_mask &= brain_mask
            ref_mask &= brain_mask
        records.append({"case_id": case, "Filename": f"{case}.nii.gz", "model": model,
                        "brain_mask_policy": policy, "n_prediction_voxels": int(pred_mask.sum()),
                        "n_reference_voxels": int(ref_mask.sum()),
                        **overlap_metrics(pred_mask, ref_mask), **distance_metrics(pred_mask, ref_mask, source.affine)})
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    write_json(output.with_suffix(".json"), {
        "model": model, "case_ids": [r["case_id"] for r in cases], "n_cases": len(cases),
        "brain_mask_policy": policy, "manifest_sha256": sha256(Path(cfg["manifest"])),
        "HD95_definition": "max of the two directional 95th percentiles",
        "AHD_definition": "arithmetic mean of the two directional mean surface distances",
        "bAHD_reference": "sum of both directional distances divided by 2*n_reference_surface_points",
        "bAHD_prediction": "sum of both directional distances divided by 2*n_prediction_surface_points; original script variant",
        "surface_definition": "6-connected binary erosion; voxel centers in physical coordinates",
        "empty_mask_policy": "distance metrics infinity if either mask is empty; status and voxel counts retained",
        "versions": versions(),
    })
    return output

