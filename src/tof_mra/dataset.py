"""Explicit, subject-disjoint preparation of the development and test sets."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import tempfile
import nibabel as nib
import numpy as np

from .common import dataset_name, read_study, sha256, write_cases, write_json, versions
from .imaging import copy_nifti, image, prepare_label, same_geometry


def _one_nifti(folder: Path) -> Path:
    found = sorted(p for p in folder.iterdir() if p.is_file() and p.name.endswith((".nii", ".nii.gz")))
    if len(found) != 1:
        raise ValueError(f"Expected exactly one NIfTI in {folder}; found {len(found)}. Select the intended series explicitly.")
    return found[0]


def _matching_file(root: Path, case: str, *, mask: bool = False) -> str:
    if (root / case).is_dir():
        return str(_one_nifti(root / case).resolve())
    stem = case + ("_mask" if mask else "")
    candidates = [root / (stem + ending) for ending in (".nii.gz", ".nii")]
    candidates = [p for p in candidates if p.is_file()]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one matching file for {case} in {root}.")
    return str(candidates[0].resolve())


def make_manifest(dev_images: Path, dev_labels: Path, test_images: Path, test_labels: Path,
                  output: Path, brain_masks: Path | None = None) -> None:
    rows = []
    for split, root, labels in (("development", dev_images, dev_labels), ("test", test_images, test_labels)):
        folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
        if not folders:
            raise ValueError(f"No subject folders in {root}.")
        for folder in folders:
            case = folder.name
            rows.append({"case_id": case, "subject_id": case, "split": split,
                         "tof": str(_one_nifti(folder).resolve()), "label": _matching_file(labels, case),
                         "brain_mask": _matching_file(brain_masks, case, mask=True) if brain_masks and split == "test" else "", "t1": ""})
    # Validate before writing, including duplicate subject/case IDs.
    from .common import read_cases
    with tempfile.TemporaryDirectory() as temporary:
        candidate = Path(temporary) / "cases.csv"
        write_cases(candidate, rows)
        read_cases(candidate)
    if output.exists():
        raise FileExistsError(f"Manifest already exists: {output}")
    write_cases(output.resolve(), rows)


def inspect_study(cfg: dict, output: Path | None = None) -> Path:
    rows = read_study(cfg)
    output = output or Path(cfg["work_dir"]) / "inspection.csv"
    records = []
    for row in rows:
        img, lab = image(row["tof"]), image(row["label"])
        records.append({"case_id": row["case_id"], "split": row["split"],
                        "shape_ijk": "x".join(map(str, img.shape)),
                        "spacing_ijk_mm": ",".join(map(str, nib.affines.voxel_sizes(img.affine))),
                        "orientation": "".join(nib.aff2axcodes(img.affine)),
                        "image_label_geometry_matches": same_geometry(img, lab)})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return output


def validate_splits(splits: list, development_ids: set[str], cfg: dict) -> None:
    fold = int(cfg["pediatric"].get("fold", 0))
    if not isinstance(splits, list) or fold >= len(splits) or fold < 0:
        raise ValueError("splits_final.json does not contain the selected fold.")
    for split in splits:
        train_list, val_list = split.get("train", []), split.get("val", [])
        train, val = set(train_list), set(val_list)
        if len(train_list) != len(train) or len(val_list) != len(val) or not train or not val:
            raise ValueError("Every saved fold must have nonempty, unique train/val case IDs.")
        if train & val or train | val != development_ids:
            raise ValueError("Saved train/val folds must partition the development cases exactly, without test subjects.")
    expected = cfg["dataset"].get("expected_validation")
    if expected is not None and len(splits[fold]["val"]) != expected:
        raise ValueError(f"Selected fold should contain {expected} validation cases.")


def prepare_study(cfg: dict) -> Path:
    rows = read_study(cfg)
    development = [r for r in rows if r["split"] == "development"]
    tests = [r for r in rows if r["split"] == "test"]
    if not development or not tests:
        raise ValueError("Both development and independent test cases are required.")
    raw = Path(cfg["nnunet"]["raw"])
    destination = raw / dataset_name(cfg)
    work = Path(cfg["work_dir"])
    test_destination = work / "test_inputs"
    if destination.exists() or test_destination.exists():
        raise FileExistsError("Prepared dataset/test_inputs already exists. Use a fresh work directory to avoid mixing runs.")
    splits_path = cfg["dataset"].get("splits_file")
    splits = json.loads(Path(splits_path).read_text()) if splits_path else None
    if splits is not None:
        validate_splits(splits, {r["case_id"] for r in development}, cfg)
        existing_splits = Path(cfg["nnunet"]["preprocessed"]) / dataset_name(cfg) / "splits_final.json"
        if existing_splits.exists() and json.loads(existing_splits.read_text()) != splits:
            raise FileExistsError("A different splits_final.json already exists in the preprocessing directory.")
    raw.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    checks = []
    # Build in a temporary directory; a failed geometry check never leaves a half-built dataset.
    with tempfile.TemporaryDirectory(prefix="prepare_", dir=raw) as temporary:
        stage = Path(temporary)
        ds, test_stage = stage / "dataset", stage / "test_inputs"
        (ds / "imagesTr").mkdir(parents=True)
        (ds / "labelsTr").mkdir()
        test_stage.mkdir()
        for row in development:
            case = row["case_id"]
            copy_nifti(Path(row["tof"]), ds / "imagesTr" / f"{case}_0000.nii.gz")
            result = prepare_label(Path(row["label"]), Path(row["tof"]), ds / "labelsTr" / f"{case}.nii.gz",
                                   resample=cfg["dataset"].get("resample_labels", True))
            checks.append({"case_id": case, **result})
        for row in tests:
            image(row["tof"])
            copy_nifti(Path(row["tof"]), test_stage / f"{row['case_id']}_0000.nii.gz")
        write_json(ds / "dataset.json", {
            "channel_names": {"0": cfg["dataset"].get("channel_name", "CT")},
            "labels": {"background": 0, "vessel": 1}, "numTraining": len(development), "file_ending": ".nii.gz",
            "description": "TOF-MRA; CT channel name intentionally preserves the source study's CTNormalization choice.",
        })
        # Only development labels are put in nnUNet_raw. No test labels are staged for nnU-Net.
        shutil.move(str(ds), str(destination))
        shutil.move(str(test_stage), str(test_destination))
    provenance = work / "provenance"
    provenance.mkdir(exist_ok=True)
    write_cases(provenance / "cases_snapshot.csv", rows)
    write_json(provenance / "label_geometry.json", checks)
    write_json(provenance / "preparation.json", {
        "manifest_sha256": sha256(Path(cfg["manifest"])), "config_sha256": sha256(Path(cfg["_file"])),
        "dataset": dataset_name(cfg), "n_development": len(development), "n_test": len(tests),
        "versions": versions(), "saved_splits_supplied": splits is not None,
        "source_hashes": [{"case_id": r["case_id"], "tof": sha256(Path(r["tof"])), "label": sha256(Path(r["label"]))} for r in rows],
    })
    if splits is not None:
        target = Path(cfg["nnunet"]["preprocessed"]) / dataset_name(cfg)
        target.mkdir(parents=True, exist_ok=True)
        existing = target / "splits_final.json"
        write_json(existing, splits)
    return destination


def require_prepared(cfg: dict) -> list[dict]:
    rows = read_study(cfg)
    preparation = Path(cfg["work_dir"]) / "provenance" / "preparation.json"
    if not preparation.is_file():
        raise FileNotFoundError("Run prepare before planning, training, or prediction.")
    saved = json.loads(preparation.read_text())
    if saved["manifest_sha256"] != sha256(Path(cfg["manifest"])):
        raise ValueError("Manifest changed after preparation. Prepare a fresh work directory.")
    if saved["dataset"] != dataset_name(cfg):
        raise ValueError("Dataset identity changed after preparation.")
    for row in rows:
        if row["split"] == "test" and not (Path(cfg["work_dir"]) / "test_inputs" / f"{row['case_id']}_0000.nii.gz").is_file():
            raise FileNotFoundError(f"Missing staged test image: {row['case_id']}")
    return rows

