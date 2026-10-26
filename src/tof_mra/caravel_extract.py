"""Run the supplied CaravelMetrics implementation on both model outputs."""

from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import json
import pickle
import shutil
import numpy as np
import pandas as pd

from .caravel_data import EXTENSIVE, ROOTS, MODELS, settings
from .common import read_study, write_json, sha256, versions
from .imaging import image, binary_data, require_geometry

SELECTED_METRICS = ["total_length", "num_bifurcations", "volume", "num_loops", "num_abnormal_degree_nodes",
                    "spline_mean_curvature", "spline_mean_square_curvature"]


def read_label_map(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit():
            number = int(fields[0])
            if number in result:
                raise ValueError(f"Duplicate atlas label {number} in label map.")
            result[number] = fields[1]
    if not result:
        raise ValueError("Label map must contain lines of the form: integer_label region_name.")
    return result


def atlas_for_case(cfg: dict, options: dict, row: dict) -> Path:
    tof = image(row["tof"])
    if options.get("registered_atlases"):
        path = Path(options["registered_atlases"]) / f"{row['case_id']}.nii.gz"
    else:
        if not row.get("t1"):
            raise ValueError(f"T1 missing for {row['case_id']}; supply it in the manifest or provide registered_atlases.")
        for program in ("fslroi", "bet", "fslreorient2std", "flirt"):
            if not shutil.which(program):
                raise RuntimeError(f"FSL program {program!r} not found. Load the FSL environment, or supply pre-registered atlases.")
        directory = Path(cfg["work_dir"]) / "caravel" / "registration" / row["case_id"]
        directory.mkdir(parents=True, exist_ok=True)
        t1_name = Path(row["t1"]).name.split(".")[0]
        path = directory / f"{t1_name}_registered_atlas.nii.gz"
        identity = {key: sha256(Path(row[key])) for key in ("tof", "t1")}
        identity["atlas"] = sha256(Path(options["atlas"]))
        marker = directory / "registration.json"
        if path.exists():
            if not marker.exists() or json.loads(marker.read_text()).get("input_hashes") != identity:
                raise ValueError(f"Unverified or stale atlas registration for {row['case_id']}. Use a fresh work directory.")
        else:
            from ._caravel.registration import atlas_registration
            with (directory / "registration.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                # arr_seg in the supplied registration code is not used to estimate the transform.
                # Therefore a single transform is shared between both segmentation strategies.
                atlas_registration(options["atlas"], np.zeros((1, 1, 1), dtype=np.uint8), row["tof"], row["t1"], str(directory))
            write_json(marker, {"input_hashes": identity, "shared_between_models": True})
    atlas = image(path)
    require_geometry(tof, atlas, context=f"{row['case_id']}: registered atlas/TOF")
    labels = np.asanyarray(atlas.dataobj)
    if not np.isfinite(labels).all() or not np.all(labels == np.floor(labels)) or np.any(labels < 0):
        raise ValueError(f"Registered atlas contains invalid discrete labels: {path}")
    if not set(np.unique(labels).astype(int)).issubset({0, *options["region_labels"]}):
        raise ValueError(f"Unexpected label values in {path}")
    return path


def extract(cfg: dict, *, model: str = "both", resume: bool = False) -> None:
    options = settings(cfg)
    rows = [r for r in read_study(cfg) if r["split"] == "test"]
    label_map = Path(options["label_map"])
    if not set(options["region_labels"]).issubset(read_label_map(label_map)):
        raise ValueError("The atlas label map does not define every configured region label.")
    from ._caravel import graph as graph_module
    from ._caravel.metrics import extract_metrics
    overrides = options.get("graph", {})
    allowed = {"SMOOTH_ITERS", "SMOOTH_ALPHA", "INTENSITY_THRESHOLD", "MIN_COMPONENT_SIZE",
               "ORPHAN_DISTANCE_THRESHOLD", "MERGE_DISTANCE", "MIN_ANGLE"}
    if set(overrides) - allowed:
        raise ValueError("Unrecognized Caravel graph settings.")
    defaults = {key: getattr(graph_module, key) for key in allowed}
    graph_settings = {**defaults, **overrides}
    try:
        for key, value in graph_settings.items():
            setattr(graph_module, key, value)
        for row in rows:
            registered = atlas_for_case(cfg, options, row)
            for name in (MODELS if model == "both" else [model]):
                pred_path = Path(cfg["work_dir"]) / "predictions" / name / f"{row['case_id']}.nii.gz"
                pred = image(pred_path)
                require_geometry(image(row["tof"]), pred, context=f"{row['case_id']}: Caravel prediction")
                mask = binary_data(pred)
                # Graph extraction uses the prediction as supplied, without applying the evaluation brain mask.
                out = Path(options["results_dirs"][name]) / row["case_id"]
                marker = out / "extraction.json"
                identity = {"prediction": sha256(pred_path), "registered_atlas": sha256(registered),
                            "label_map": sha256(label_map), "graph_settings": graph_settings}
                if out.exists() and any(out.iterdir()):
                    saved = json.loads(marker.read_text()) if marker.exists() else {}
                    if resume and saved.get("status") == "complete" and saved.get("inputs") == identity and (out / "region_summary.csv").is_file():
                        print(f"Verified existing Caravel result: {name}/{row['case_id']}")
                        continue
                    raise FileExistsError(f"Unverified, incomplete, or stale Caravel output: {out}. Use a fresh results directory.")
                out.mkdir(parents=True, exist_ok=True)
                record = {"status": "running", "inputs": identity, "versions": versions(), "graph_coordinates": "LPS after supplied SLICER_MATRIX; atlas indexing uses SimpleITK LPS"}
                write_json(marker, record)
                print(f"Caravel extraction: {name}/{row['case_id']}", flush=True)
                try:
                    with (out / "extraction.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
                        if mask.any():
                            graph_module.extract_graph(str(pred_path), str(out))
                            # Only the graph just generated by this process is loaded here.
                            with (out / "vessel_data.pkl").open("rb") as stream:
                                graph_data = pickle.load(stream)
                            n_nodes = graph_data["graph"].number_of_nodes()
                        else:
                            n_nodes = 0
                        if n_nodes:
                            extract_metrics(row["case_id"], str(out), str(out / "vessel_data.pkl"), SELECTED_METRICS,
                                            str(label_map), registered_atlas_path=str(registered))
                        if not (out / "region_summary.csv").exists():
                            if n_nodes:
                                raise ValueError("A nonempty graph produced no regional summary. Check atlas overlap, coordinates, and the extraction log.")
                            columns = ["region_label", *EXTENSIVE, *[root+suffix for root in ROOTS for suffix in ("mean_curvature", "mean_square_curvature")]]
                            pd.DataFrame(columns=columns).to_csv(out / "region_summary.csv", index=False)
                        record["n_graph_nodes"] = n_nodes
                        record["status"] = "complete"
                except BaseException as exc:
                    record["status"], record["error"] = "failed", str(exc)
                    raise
                finally:
                    write_json(marker, record)
    finally:
        for key, value in defaults.items():
            setattr(graph_module, key, value)

