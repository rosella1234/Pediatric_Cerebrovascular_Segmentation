"""Configuration, explicit case correspondence, and execution records."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone


def absolute(path: str | Path, base: Path) -> Path:
    path = Path(path).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for package in ("pediatric-tof-mra", "numpy", "scipy", "pandas", "nibabel", "SimpleITK", "matplotlib", "nnunetv2", "torch", "networkx", "scikit-image", "scikit-learn", "nipype", "vedo", "openpyxl"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not installed"
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    path = path.resolve()
    cfg = json.loads(path.read_text())
    cfg["_file"] = str(path)
    cfg["_base"] = str(path.parent)
    for field in ("manifest", "work_dir"):
        if not cfg.get(field):
            raise ValueError(f"Configuration requires {field!r}.")
        cfg[field] = str(absolute(cfg[field], path.parent))
    for section, fields in {
        "nnunet": ("raw", "preprocessed", "results", "executable_dir"),
        "baseline": ("results",),
        "dataset": ("splits_file",),
    }.items():
        for field in fields:
            if cfg.get(section, {}).get(field):
                cfg[section][field] = str(absolute(cfg[section][field], path.parent))
    ds = cfg["dataset"]
    if not isinstance(ds["id"], int) or ds["id"] < 0:
        raise ValueError("dataset.id must be a non-negative integer.")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", ds["name"]):
        raise ValueError("dataset.name must contain only letters and digits, starting with a letter.")
    return cfg


def dataset_name(cfg: dict) -> str:
    return f"Dataset{cfg['dataset']['id']:03d}_{cfg['dataset']['name']}"


def read_cases(path: Path, *, check_files: bool = True) -> list[dict]:
    """Paths in a manifest are relative to that manifest, not the current directory."""
    path = path.resolve()
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"case_id", "split", "tof", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("Manifest requires case_id, split, tof, label columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("Manifest is empty.")
    ids, subjects, source_images, source_labels = set(), set(), set(), set()
    for row in rows:
        for key, value in list(row.items()):
            row[key] = (value or "").strip()
        case = row["case_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case) or case.endswith("_0000"):
            raise ValueError(f"Invalid case_id: {case!r}. Use an ID without the _0000 channel suffix.")
        if case in ids:
            raise ValueError(f"Duplicate case_id: {case}")
        ids.add(case)
        subject = row.get("subject_id") or case
        if subject in subjects:
            raise ValueError(f"Repeated subject_id: {subject}. This study expects one examination per subject.")
        subjects.add(subject)
        row["subject_id"] = subject
        if row["split"] not in {"development", "test"}:
            raise ValueError(f"Invalid split for {case}: use development or test.")
        for field in ("tof", "label", "brain_mask", "t1"):
            value = row.get(field, "")
            row[field] = str(absolute(value, path.parent)) if value else ""
            if field in {"tof", "label"} and not value:
                raise ValueError(f"Missing {field} for {case}.")
            if check_files and value and not Path(row[field]).is_file():
                raise FileNotFoundError(f"{case}: {field} does not exist: {row[field]}")
        # Resolve symlinks before checking for accidental reuse across case IDs.
        for field, seen in (("tof", source_images), ("label", source_labels)):
            if row[field] in seen:
                raise ValueError(f"The same {field} file is assigned to multiple cases: {case}")
            seen.add(row[field])
    return rows


def check_counts(cases: list[dict], cfg: dict) -> None:
    for split, key in (("development", "expected_development"), ("test", "expected_test")):
        expected = cfg["dataset"].get(key)
        actual = sum(row["split"] == split for row in cases)
        if expected is not None and actual != expected:
            raise ValueError(f"Expected {expected} {split} cases; found {actual}. Update the manifest or explicitly change the expected count.")


def read_study(cfg: dict) -> list[dict]:
    rows = read_cases(Path(cfg["manifest"]))
    check_counts(rows, cfg)
    return rows


def write_cases(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "subject_id", "split", "tof", "label", "brain_mask", "t1"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_environment(cfg: dict, *, baseline: bool = False) -> dict[str, str]:
    nn = cfg["nnunet"]
    env = {
        "nnUNet_raw": nn["raw"],
        "nnUNet_preprocessed": nn["preprocessed"],
        "nnUNet_results": cfg["baseline"]["results"] if baseline else nn["results"],
        "nnUNet_compile": "False",
        "nnUNet_n_proc_DA": str(nn.get("augmentation_processes", 2)),
        "nnUNet_n_proc_DA_val": "1",
    }
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(nn.get("threads", 1))
    if nn.get("executable_dir"):
        env["PATH"] = nn["executable_dir"] + os.pathsep + os.environ.get("PATH", "")
    return env


def run_command(argv: list[str], env: dict[str, str], record: Path, *, dry_run: bool = False) -> None:
    """Use argv lists, never a shell. Record only explicitly supplied environment keys."""
    print(shlex.join(argv), flush=True)
    if dry_run:
        print(json.dumps({"command": argv, "environment": env}, indent=2))
        return
    record.parent.mkdir(parents=True, exist_ok=True)
    payload = {"argv": argv, "environment": env, "started_utc": datetime.now(timezone.utc).isoformat(), "versions": versions()}
    write_json(record, payload)
    try:
        with record.with_suffix(".log").open("w") as log:
            process = subprocess.Popen(argv, env={**os.environ, **env}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                returncode = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        payload["returncode"] = returncode
        if returncode:
            raise RuntimeError(f"Command failed with exit code {returncode}; see {record.with_suffix('.log')}")
    except BaseException as exc:
        payload["error"] = str(exc)
        raise
    finally:
        payload["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record, payload)

