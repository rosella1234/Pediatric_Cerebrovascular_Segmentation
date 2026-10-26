"""DICOM-to-NIfTI conversion driven by an explicit series manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .common import absolute, write_json
from .imaging import image


def convert_series(manifest: Path, output: Path, *, dcm2niix: str = "dcm2niix", decompressor: list[str] | None = None) -> list[dict]:
    manifest, output = manifest.resolve(), output.resolve()
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"case_id", "modality", "series_dir"}.issubset(reader.fieldnames or []):
            raise ValueError("Conversion manifest requires case_id, modality, series_dir.")
        rows = list(reader)
    if not rows:
        raise ValueError("Conversion manifest is empty.")
    seen = set()
    for row in rows:
        row.update({k: (v or "").strip() for k, v in row.items()})
        case, modality = row["case_id"], row["modality"].lower()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", case) or modality not in {"tof", "t1"}:
            raise ValueError("Use a pseudonymous case_id and modality tof or t1.")
        row["modality"] = modality
        row["series_dir"] = str(absolute(row["series_dir"], manifest.parent))
        key = case, modality
        if key in seen:
            raise ValueError(f"Multiple series selected for {case}/{modality}. Select one intended series explicitly.")
        seen.add(key)
        if not Path(row["series_dir"]).is_dir():
            raise FileNotFoundError(row["series_dir"])
        if (output / modality / f"{case}.nii.gz").exists() or (output / modality / f"{case}.json").exists():
            raise FileExistsError(f"Conversion output already exists for {case}/{modality}.")
    if decompressor is not None and (not decompressor or not all(isinstance(x, str) and x for x in decompressor)):
        raise ValueError("Decompressor must be a nonempty JSON list of command tokens.")
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in rows:
        case, modality = row["case_id"], row["modality"]
        report = {**row, "attempts": [], "decompression_used": False}
        record = output / "logs" / f"{case}_{modality}.json"
        try:
            with tempfile.TemporaryDirectory(prefix="dcm2niix_") as temporary:
                temporary = Path(temporary)
                converted = temporary / "converted"
                converted.mkdir()

                def attempt(source: Path):
                    argv = [dcm2niix, "-z", "y", "-b", "y", "-ba", "y", "-f", "series_%s", "-o", str(converted), str(source)]
                    result = subprocess.run(argv, capture_output=True, text=True)
                    report["attempts"].append({"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
                    return result.returncode, sorted(converted.glob("*.nii.gz"))

                status, volumes = attempt(Path(row["series_dir"]))
                if status or not volumes:
                    if decompressor is None:
                        raise RuntimeError("dcm2niix failed or produced no NIfTI. Configure a DICOM decompressor if needed; see the conversion log.")
                    for file in converted.iterdir():
                        file.unlink()
                    uncompressed = temporary / "decompressed"
                    sources = sorted(p for p in Path(row["series_dir"]).rglob("*") if p.is_file())
                    if not sources:
                        raise ValueError("DICOM series folder contains no files.")
                    for source in sources:
                        destination = uncompressed / source.relative_to(row["series_dir"])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        result = subprocess.run([*decompressor, str(source), str(destination)], capture_output=True, text=True)
                        if result.returncode:
                            raise RuntimeError(f"DICOM decompression failed: {result.stderr}")
                    report["decompression_used"] = True
                    status, volumes = attempt(uncompressed)
                if status or len(volumes) != 1:
                    raise ValueError(f"Expected exactly one output volume; got {len(volumes)}, exit code {status}. Inspect and select the intended DICOM series.")
                image(volumes[0])
                target = output / modality / f"{case}.nii.gz"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(volumes[0], target)
                sidecar = volumes[0].with_suffix("").with_suffix(".json")
                if sidecar.exists():
                    shutil.copy2(sidecar, target.with_suffix("").with_suffix(".json"))
                report["output"] = str(target)
                report["status"] = "ok"
        except BaseException as exc:
            report["status"] = "failed"
            report["error"] = str(exc)
            raise
        finally:
            write_json(record, report)
        reports.append(report)
        print(f"Converted {case}/{modality}: {report['output']}")
    write_json(output / "conversion_report.json", reports)
    return reports

