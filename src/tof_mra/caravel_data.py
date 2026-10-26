"""Canonical eight-feature tables from the study's CaravelMetrics region summaries."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .common import absolute, read_study, write_json, sha256

EXTENSIVE = ("total_length", "volume", "num_components", "num_bifurcations", "num_loops", "num_abnormal_degree_nodes")
CURVATURE = ("root_mean_curvature_avg", "mean_squared_curvature_avg")
FEATURES = (*EXTENSIVE, *CURVATURE)
ROOTS = ("Largest_endpoint_root_", "2nd_Largest_endpoint_root_", "Largest_bifurcation_root_")
FEATURE_LABELS = {
    "total_length": "Total vessel length", "volume": "Vascular volume",
    "num_components": "Connected components", "num_bifurcations": "Bifurcations",
    "num_loops": "Loops", "num_abnormal_degree_nodes": "Abnormal-degree nodes",
    "root_mean_curvature_avg": "Root-based mean curvature", "mean_squared_curvature_avg": "Mean squared curvature",
}
MODELS = ("baseline", "pediatric")


def settings(cfg: dict) -> dict:
    if "caravel" not in cfg:
        raise ValueError("Add the caravel section from configs/study.example.json to your configuration.")
    options = dict(cfg["caravel"])
    base = Path(cfg["_base"])
    for key in ("atlas", "label_map", "registered_atlases", "demographics", "balanced_cases_file"):
        if options.get(key):
            options[key] = str(absolute(options[key], base))
    options["results_dirs"] = {model: str(absolute(options.get("results_dirs", {}).get(model), base))
                               if options.get("results_dirs", {}).get(model) else str(Path(cfg["work_dir"]) / "caravel" / model)
                               for model in MODELS}
    options.setdefault("region_labels", list(range(1, 33)))
    labels = options["region_labels"]
    if not labels or len(set(labels)) != len(labels) or any(not isinstance(x, int) or x <= 0 for x in labels):
        raise ValueError("caravel.region_labels must be unique positive integer atlas labels.")
    return options


def weighted_mean(values, weights) -> float:
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[valid], weights=weights[valid])) if valid.any() else np.nan


def normalize_summary(frame: pd.DataFrame, *, expected_labels: list[int]) -> pd.DataFrame:
    frame = frame.copy()
    missing = set(("region_label", *EXTENSIVE)) - set(frame)
    if missing:
        raise ValueError(f"Caravel region_summary.csv is missing columns: {sorted(missing)}")
    labels = pd.to_numeric(frame["region_label"], errors="raise")
    if labels.isna().any() or not np.all(labels == np.floor(labels)):
        raise ValueError("Atlas region labels must be integers.")
    frame["region_label"] = labels.astype(int)
    if frame.region_label.duplicated().any():
        raise ValueError("Expected one region_summary row per region, not component-level data.")
    if not set(frame.region_label).issubset(expected_labels):
        raise ValueError("Unexpected atlas labels in region summary.")
    for key in EXTENSIVE:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
        if (~np.isfinite(frame[key])).any() or (frame[key] < 0).any():
            raise ValueError(f"Invalid extensive measurement: {key}. Missing/corrupt measurements are not converted to zero.")
    for target, suffix in zip(CURVATURE, ("mean_curvature", "mean_square_curvature")):
        if target not in frame:
            columns = [root + suffix for root in ROOTS]
            if not set(columns).issubset(frame):
                raise ValueError(f"Curvature requires {target} or all three root-specific columns.")
            frame[target] = frame[columns].apply(pd.to_numeric, errors="raise").mean(axis=1)
        frame[target] = pd.to_numeric(frame[target], errors="raise")
        if np.isinf(frame[target]).any():
            raise ValueError(f"Infinite curvature values: {target}")
    return frame[["region_label", *FEATURES]].sort_values("region_label")


def aggregate_whole_brain(regional: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (model, case), group in regional.groupby(["model", "case_id"], sort=False):
        record = {"model": model, "case_id": case, "n_observed_regions": len(group)}
        record.update({key: float(group[key].sum()) for key in EXTENSIVE})
        record.update({key: weighted_mean(group[key], group.total_length) for key in CURVATURE})
        records.append(record)
    return pd.DataFrame(records)


def collect(cfg: dict) -> tuple[Path, Path]:
    options = settings(cfg)
    cases = [r for r in read_study(cfg) if r["split"] == "test"]
    tables, audit, empty_cases = [], [], []
    for model in MODELS:
        for row in cases:
            case = row["case_id"]
            path = Path(options["results_dirs"][model]) / case / "region_summary.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing {model} result for {case}: {path}")
            frame = normalize_summary(pd.read_csv(path), expected_labels=options["region_labels"])
            frame.insert(0, "case_id", case)
            frame.insert(0, "model", model)
            tables.append(frame)
            if frame.empty:
                empty_cases.append((model, case))
            audit.append({"model": model, "case_id": case, "summary_sha256": sha256(path),
                          "observed_regions": frame.region_label.tolist(),
                          "omitted_regions": sorted(set(options["region_labels"]) - set(frame.region_label))})
    regional = pd.concat(tables, ignore_index=True)
    whole = aggregate_whole_brain(regional)
    if empty_cases:
        empty = [{"model": m, "case_id": c, "n_observed_regions": 0,
                  **{f: 0. for f in EXTENSIVE}, **{f: np.nan for f in CURVATURE}} for m, c in empty_cases]
        whole = pd.concat([whole, pd.DataFrame(empty)], ignore_index=True)
    out = Path(cfg["work_dir"]) / "caravel" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    regional.to_csv(out / "regional.csv", index=False)
    whole.to_csv(out / "whole_brain.csv", index=False)
    write_json(out / "collection_audit.json", {"cases": audit, "missing_region_policy": "omit (source notebook convention)",
               "whole_brain_definition": "sum regional extensive descriptors; vessel-length-weighted curvature; not whole-graph topology"})
    return out / "regional.csv", out / "whole_brain.csv"


def _id_strings(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.replace(r"^(\d+)\.0$", r"\1", regex=True)


def import_demographics(source: Path, manifest: Path, output: Path, *, id_column: str = "case_id", case_map: Path | None = None) -> Path:
    """Import only the three required fields, never distribute the original workbook."""
    from .common import read_cases
    if source.suffix.lower() in {".xlsx", ".xlsm"}:
        frame = pd.read_excel(source, dtype={id_column: str})
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source, dtype={id_column: str})
    else:
        raise ValueError("Demographics source must be CSV or XLSX.")
    if id_column not in frame:
        raise ValueError(f"Demographics has no {id_column!r} column.")
    ids = _id_strings(frame[id_column])
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Demographics IDs must be present and unique.")
    if case_map:
        mapping = pd.read_csv(case_map, dtype=str)
        if not {"source_id", "case_id"}.issubset(mapping):
            raise ValueError("Case mapping needs source_id and case_id columns.")
        mapping["source_id"] = _id_strings(mapping.source_id)
        if mapping.source_id.duplicated().any() or mapping.case_id.duplicated().any():
            raise ValueError("Case mapping must be one-to-one.")
        ids = ids.map(mapping.set_index("source_id").case_id)
    age_column = next((c for c in ("age_years", "AGE") if c in frame), None)
    sex_column = next((c for c in ("sex", "SEX_ID", "SEX_ID (1=m, 2=f)") if c in frame), None)
    if not age_column or not sex_column:
        raise ValueError("Need age_years/AGE and sex/SEX_ID columns.")
    sex = frame[sex_column].astype(str).str.strip().str.upper().replace({"1": "M", "1.0": "M", "2": "F", "2.0": "F", "MALE": "M", "FEMALE": "F"})
    result = pd.DataFrame({"case_id": ids, "age_years": pd.to_numeric(frame[age_column], errors="raise"), "sex": sex})
    tests = {r["case_id"] for r in read_cases(manifest) if r["split"] == "test"}
    if not tests.issubset(set(result.case_id.dropna())):
        raise ValueError("Demographics IDs do not match all test cases. Supply an explicit source_id/case_id mapping; IDs are never guessed.")
    result = result[result.case_id.isin(tests)]
    if result.age_years.isna().any() or not np.isfinite(result.age_years).all() or (result.age_years < 0).any() or not result.sex.isin(["M", "F"]).all():
        raise ValueError("Each test subject needs a finite nonnegative age in years and sex M/F (source codes 1=M, 2=F).")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output

