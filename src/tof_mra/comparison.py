"""Paired statistics by case ID; never assume that CSV row orders correspond."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import sha256, write_json, versions
from .metrics import OVERLAP


def read_report(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, dtype={"case_id": str, "Filename": str})
    if "case_id" not in data:
        if "Filename" not in data:
            raise ValueError(f"CSV needs case_id or Filename: {path}")
        data["case_id"] = data["Filename"].str.replace(r"\.nii(\.gz)?$", "", regex=True)
    if data["case_id"].isna().any() or (data["case_id"].str.strip() == "").any() or data["case_id"].duplicated().any():
        raise ValueError(f"CSV has missing or duplicate case IDs: {path}")
    data["case_id"] = data["case_id"].str.strip()
    if data["case_id"].duplicated().any():
        raise ValueError(f"CSV has duplicate case IDs after stripping spaces: {path}")
    # The supplied historical distance script uses the prediction as denominator.
    if "Balanced_AHD_mm" in data and "Balanced_AHD_prediction_mm" not in data:
        data["Balanced_AHD_prediction_mm"] = data["Balanced_AHD_mm"]
    return data


def merge_reports(overlap: Path, distance: Path, output: Path) -> Path:
    left, right = read_report(overlap), read_report(distance)
    if set(left.case_id) != set(right.case_id):
        raise ValueError("Overlap and distance reports contain different case IDs.")
    # Common metadata such as Filename is retained from the overlap report.
    extra = [c for c in right if c not in left or c == "case_id"]
    result = left.merge(right[extra], on="case_id", validate="one_to_one")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.sort_values("case_id").to_csv(output, index=False)
    return output


def compare_reports(baseline: Path, trained: Path, output: Path, *, bahd_normalization: str = "reference",
                    expected_pairs: int | None = None, allow_unpaired: bool = False,
                    nonfinite_policy: str = "error", make_plots: bool = True,
                    font_family: str = "Arial", figure_note: str | None = None) -> pd.DataFrame:
    if bahd_normalization not in {"reference", "prediction"}:
        raise ValueError("bAHD normalization must be reference or prediction.")
    if nonfinite_policy not in {"error", "drop"}:
        raise ValueError("nonfinite_policy must be error or drop.")
    left, right = read_report(baseline), read_report(trained)
    ids_left, ids_right = set(left.case_id), set(right.case_id)
    if ids_left != ids_right and not allow_unpaired:
        raise ValueError(f"Paired comparison requires identical case sets. Baseline only: {sorted(ids_left-ids_right)}; trained only: {sorted(ids_right-ids_left)}")
    if "brain_mask_policy" in left and "brain_mask_policy" in right:
        policies = left[["case_id", "brain_mask_policy"]].merge(right[["case_id", "brain_mask_policy"]], on="case_id", suffixes=("_baseline", "_trained"), validate="one_to_one")
        if not (policies.brain_mask_policy_baseline == policies.brain_mask_policy_trained).all():
            raise ValueError("Baseline and trained reports use different brain-mask policies.")
    selected = [*OVERLAP, "HD95_mm", "AHD_mm", f"Balanced_AHD_{bahd_normalization}_mm"]
    for name, frame in (("baseline", left), ("trained", right)):
        missing = set(selected) - set(frame)
        if missing:
            raise ValueError(f"{name} report is missing {sorted(missing)}. Historical Balanced_AHD_mm is prediction-normalized; select --bahd-normalization prediction or recompute from masks.")
        for metric in selected:
            frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    joined = left[["case_id", *selected]].merge(right[["case_id", *selected]], on="case_id", suffixes=("_baseline", "_trained"), validate="one_to_one").sort_values("case_id")
    if joined.empty:
        raise ValueError("No paired cases.")
    if expected_pairs is not None and len(joined) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} paired cases; found {len(joined)}. No full-cohort p-values will be reused for a smaller cohort.")
    output.mkdir(parents=True, exist_ok=True)
    audit = {"n_joined_pairs": len(joined), "baseline_only": sorted(ids_left-ids_right), "trained_only": sorted(ids_right-ids_left),
             "excluded_nonfinite_by_metric": {}, "paired_case_ids": joined.case_id.tolist()}
    summaries = []
    plot_data = {}
    for metric in selected:
        base = joined[metric + "_baseline"].to_numpy(dtype=float)
        pediatric = joined[metric + "_trained"].to_numpy(dtype=float)
        valid = np.isfinite(base) & np.isfinite(pediatric)
        audit["excluded_nonfinite_by_metric"][metric] = joined.loc[~valid, "case_id"].tolist()
        if not valid.all() and nonfinite_policy == "error":
            write_json(output / "pairing_audit.json", audit)
            raise ValueError(f"Non-finite values for {metric}. See pairing_audit.json; exclusion requires --nonfinite-policy drop explicitly.")
        base, pediatric = base[valid], pediatric[valid]
        if not len(base):
            raise ValueError(f"No finite paired values remain for {metric}.")
        differences = base - pediatric
        if np.all(differences == 0):
            statistic, pvalue = 0.0, 1.0
        else:
            test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
            statistic, pvalue = float(test.statistic), float(test.pvalue)
        bq1, bmed, bq3 = np.quantile(base, [0.25, 0.5, 0.75])
        tq1, tmed, tq3 = np.quantile(pediatric, [0.25, 0.5, 0.75])
        summaries.append({"metric": metric, "n_pairs": len(base), "n_excluded_nonfinite": int((~valid).sum()),
                          "baseline_median": bmed, "baseline_q1": bq1, "baseline_q3": bq3,
                          "trained_median": tmed, "trained_q1": tq1, "trained_q3": tq3,
                          "median_paired_change_trained_minus_baseline": float(np.median(pediatric-base)),
                          "wilcoxon_statistic": statistic, "p_value": pvalue})
        plot_data[metric] = (base, pediatric)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "traditional_metrics_statistics.csv", index=False)
    joined.to_csv(output / "paired_metrics.csv", index=False)
    write_json(output / "pairing_audit.json", audit)
    metadata = {"baseline_csv_sha256": sha256(baseline), "trained_csv_sha256": sha256(trained),
                "bahd_normalization": bahd_normalization, "expected_pairs": expected_pairs,
                "nonfinite_policy": nonfinite_policy, "test": "paired two-sided Wilcoxon signed-rank",
                "zero_method": "wilcox", "method": "auto", "continuity_correction": False,
                "all_zero_difference_policy": "statistic=0, p=1", "multiple_comparison_correction": "none (paper's nominal tests)",
                "historical_bahd_column_assumption": "Balanced_AHD_mm from the supplied script is prediction-normalized",
                "figure_note": figure_note, "versions": versions()}
    if make_plots:
        from .plotting import plot_comparison
        metadata["font_used"] = plot_comparison(summary, plot_data, output, font_family=font_family, note=figure_note)
    write_json(output / "comparison_metadata.json", metadata)
    return summary

