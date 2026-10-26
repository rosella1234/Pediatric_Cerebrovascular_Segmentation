"""Python orchestration of the official nnU-Net v2.6.2 entry points."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from .common import command_environment, dataset_name, run_command, write_json
from .dataset import require_prepared, validate_splits
from .imaging import image, require_geometry, binary_data


def plan_command(cfg: dict) -> list[str]:
    command = ["nnUNetv2_plan_and_preprocess", "-d", str(cfg["dataset"]["id"]), "--verify_dataset_integrity",
            "-c", cfg["pediatric"]["configuration"], "-np", str(cfg["nnunet"].get("preprocessing_processes", 4)),
            "-npfp", str(cfg["nnunet"].get("preprocessing_processes", 4))]
    if cfg["pediatric"].get("plans", "nnUNetPlans") != "nnUNetPlans":
        command += ["-overwrite_plans_name", cfg["pediatric"]["plans"]]
    return command


def train_command(cfg: dict) -> list[str]:
    model = cfg["pediatric"]
    return ["nnUNetv2_train", dataset_name(cfg), model["configuration"], str(model.get("fold", 0)),
            "-tr", model.get("trainer", "nnUNetTrainer"), "-p", model.get("plans", "nnUNetPlans"),
            "-device", cfg["nnunet"].get("device", "cuda")]


def predict_command(cfg: dict, model_name: str) -> list[str]:
    model = cfg[model_name]
    folds = model.get("folds", [0]) if model_name == "baseline" else [model.get("fold", 0)]
    if folds != [0]:
        raise ValueError("The paper uses fold 0 only; folds must be [0].")
    ds = model["dataset"] if model_name == "baseline" else dataset_name(cfg)
    work = Path(cfg["work_dir"])
    command = ["nnUNetv2_predict", "-i", str(work / "test_inputs"), "-o", str(work / "predictions" / model_name),
               "-d", str(ds), "-c", model["configuration"], "-f", "0", "-tr", model.get("trainer", "nnUNetTrainer"),
               "-p", model.get("plans", "nnUNetPlans"), "-chk", model.get("checkpoint", "checkpoint_final.pth"),
               "-step_size", str(model.get("step_size", 0.5)), "-device", cfg["nnunet"].get("device", "cuda"),
               "-npp", str(cfg["nnunet"].get("prediction_processes", 2)),
               "-nps", str(cfg["nnunet"].get("prediction_processes", 2))]
    if not model.get("tta", True):
        command.append("--disable_tta")
    return command


def _archive_training_metadata(cfg: dict) -> None:
    ds = dataset_name(cfg)
    directory = Path(cfg["nnunet"]["preprocessed"]) / ds
    out = Path(cfg["work_dir"]) / "provenance" / "nnunet"
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("dataset.json", "dataset_fingerprint.json", f"{cfg['pediatric'].get('plans', 'nnUNetPlans')}.json", "splits_final.json"):
        path = directory / filename
        if path.is_file():
            shutil.copy2(path, out / filename)


def plan(cfg: dict, *, dry_run: bool = False) -> None:
    if not dry_run:
        require_prepared(cfg)
    run_command(plan_command(cfg), command_environment(cfg), Path(cfg["work_dir"]) / "logs" / "plan.json", dry_run=dry_run)
    if not dry_run:
        _archive_training_metadata(cfg)


def train(cfg: dict, *, dry_run: bool = False) -> None:
    if int(cfg["pediatric"].get("fold", 0)) != 0:
        raise ValueError("The study trains fold 0 only.")
    if not dry_run:
        rows = require_prepared(cfg)
        preprocessed = Path(cfg["nnunet"]["preprocessed"]) / dataset_name(cfg)
        plans = preprocessed / f"{cfg['pediatric'].get('plans', 'nnUNetPlans')}.json"
        if not plans.exists():
            raise FileNotFoundError("Preprocessed plans missing. Run plan first.")
        splits_file = preprocessed / "splits_final.json"
        if splits_file.exists():
            validate_splits(json.loads(splits_file.read_text()), {r["case_id"] for r in rows if r["split"] == "development"}, cfg)
        fold_dir = Path(cfg["nnunet"]["results"]) / dataset_name(cfg) / f"{cfg['pediatric'].get('trainer', 'nnUNetTrainer')}__{cfg['pediatric'].get('plans', 'nnUNetPlans')}__{cfg['pediatric']['configuration']}" / "fold_0"
        if fold_dir.exists():
            raise FileExistsError("Training output already exists. Use a fresh results directory for training from scratch.")
    # No pretrained-weight argument is accepted: the pediatric model is initialized from scratch.
    run_command(train_command(cfg), command_environment(cfg), Path(cfg["work_dir"]) / "logs" / "train.json", dry_run=dry_run)
    if not dry_run:
        _archive_training_metadata(cfg)
        generated = Path(cfg["nnunet"]["preprocessed"]) / dataset_name(cfg) / "splits_final.json"
        validate_splits(json.loads(generated.read_text()), {r["case_id"] for r in rows if r["split"] == "development"}, cfg)


def predict(cfg: dict, model_name: str, *, dry_run: bool = False) -> None:
    command = predict_command(cfg, model_name)
    if not dry_run:
        rows = require_prepared(cfg)
        output = Path(cfg["work_dir"]) / "predictions" / model_name
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Prediction output is not empty: {output}. Use a fresh output directory.")
    run_command(command, command_environment(cfg, baseline=model_name == "baseline"),
                Path(cfg["work_dir"]) / "logs" / f"predict_{model_name}.json", dry_run=dry_run)
    if not dry_run:
        expected = {row["case_id"] for row in rows if row["split"] == "test"}
        found = {p.name.removesuffix(".nii.gz") for p in output.glob("*.nii.gz")}
        if found != expected:
            raise ValueError(f"Prediction outputs do not match the test cohort. Missing: {sorted(expected-found)}; extra: {sorted(found-expected)}")
        for row in rows:
            if row["split"] == "test":
                prediction = image(output / f"{row['case_id']}.nii.gz")
                require_geometry(image(row["tof"]), prediction, context=f"{row['case_id']} prediction")
                binary_data(prediction)
        write_json(output / "cohort.json", {"case_ids": sorted(expected), "model": model_name,
                                            "configuration": cfg[model_name]["configuration"], "fold": 0})

