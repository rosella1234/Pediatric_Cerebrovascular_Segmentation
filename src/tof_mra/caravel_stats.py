"""The paper's exploratory age, balanced-sex, and regional analyses."""

from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind, f_oneway, kruskal

from .caravel_data import FEATURES, MODELS, settings
from .common import read_study, sha256, write_json, versions


def fdr_bh(pvalues) -> np.ndarray:
    """Keep the eight-descriptor family size even when a descriptor is untestable."""
    values = np.asarray(pvalues, dtype=float)
    finite = np.isfinite(values)
    if ((values[finite] < 0) | (values[finite] > 1)).any():
        raise ValueError("p-values must be between zero and one.")
    safe = np.where(finite, values, 1.)
    order = np.argsort(safe)
    adjusted = np.minimum.accumulate((safe[order] * len(values) / np.arange(1, len(values)+1))[::-1])[::-1]
    result = np.empty(len(values))
    result[order] = np.minimum(1, adjusted)
    result[~finite] = np.nan
    return result


def age_correlations(frame: pd.DataFrame, *, minimum: int = 10) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        valid = frame[["age_years", feature]].replace([np.inf, -np.inf], np.nan).dropna()
        row = {"feature": feature, "n": len(valid), "pearson_r": np.nan, "pearson_p": np.nan,
               "spearman_r": np.nan, "spearman_p": np.nan, "status": "insufficient_observations"}
        if len(valid) >= minimum and valid.age_years.nunique() > 1 and valid[feature].nunique() > 1:
            pearson = pearsonr(valid.age_years, valid[feature])
            spearman = spearmanr(valid.age_years, valid[feature])
            row.update(pearson_r=float(pearson.statistic), pearson_p=float(pearson.pvalue),
                       spearman_r=float(spearman.statistic), spearman_p=float(spearman.pvalue), status="ok")
        elif len(valid) >= minimum:
            row["status"] = "constant_age_or_feature"
        rows.append(row)
    result = pd.DataFrame(rows)
    for method in ("pearson", "spearman"):
        result[method+"_p_fdr"] = fdr_bh(result[method+"_p"])
    return result


def balance_sexes(demographics: pd.DataFrame, *, width: float = 5, seed: int = 42,
                  fixed_ids: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Age bin width must be positive.")
    frame = demographics.sort_values("case_id").copy()
    start = np.floor(frame.age_years.min())
    stop = np.floor(frame.age_years.max()) + width
    bins = np.arange(start, stop + np.finfo(float).eps*max(stop, 1), width)
    if bins[-1] <= frame.age_years.max():
        bins = np.append(bins, bins[-1]+width)
    frame["age_bin"] = pd.cut(frame.age_years, bins=bins, right=False, include_lowest=True)
    if frame.age_bin.isna().any():
        raise ValueError("An age fell outside the balancing bins.")
    if fixed_ids is not None:
        if len(fixed_ids) != len(set(fixed_ids)) or not set(fixed_ids).issubset(set(frame.case_id)):
            raise ValueError("Saved balanced IDs must be unique test case IDs.")
    selected, report = [], []
    for age_bin, group in frame.groupby("age_bin", observed=False, sort=True):
        males, females = group[group.sex == "M"], group[group.sex == "F"]
        number = min(len(males), len(females))
        if fixed_ids is None:
            chosen = pd.concat([males.sample(n=number, random_state=seed), females.sample(n=number, random_state=seed)])
        else:
            chosen = group[group.case_id.isin(fixed_ids)]
            if sum(chosen.sex == "M") != sum(chosen.sex == "F"):
                raise ValueError("Saved balanced IDs are not sex-balanced within each age bin.")
        selected.append(chosen)
        report.append({"age_bin": str(age_bin), "original_male": len(males), "original_female": len(females),
                       "balanced_male": int(sum(chosen.sex == "M")), "balanced_female": int(sum(chosen.sex == "F"))})
    balanced = pd.concat(selected, ignore_index=True)
    if balanced.empty:
        raise ValueError("No age bin contains both sexes; no balanced subgroup can be formed.")
    return balanced.sort_values(["age_years", "case_id"]).drop(columns="age_bin"), pd.DataFrame(report)


def sex_comparisons(frame: pd.DataFrame, *, minimum: int = 5) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        male = frame.loc[frame.sex == "M", feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        female = frame.loc[frame.sex == "F", feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        row = {"feature": feature, "n_male": len(male), "n_female": len(female), "t_statistic": np.nan,
               "t_pvalue": np.nan, "cohens_d": np.nan, "male_mean": np.mean(male) if len(male) else np.nan,
               "female_mean": np.mean(female) if len(female) else np.nan, "status": "insufficient_observations"}
        if min(len(male), len(female)) >= minimum:
            pooled_variance = ((len(male)-1)*male.var(ddof=1)+(len(female)-1)*female.var(ddof=1))/(len(male)+len(female)-2)
            if pooled_variance > 0:
                test = ttest_ind(male, female, equal_var=True, alternative="two-sided")
                row.update(t_statistic=float(test.statistic), t_pvalue=float(test.pvalue),
                           cohens_d=float((male.mean()-female.mean())/np.sqrt(pooled_variance)), status="ok")
            elif male.mean() == female.mean():
                row.update(t_statistic=0., t_pvalue=1., cohens_d=0., status="equal_constants")
            else:
                row.update(t_pvalue=0., status="zero_pooled_variance_effect_size_undefined")
        rows.append(row)
    result = pd.DataFrame(rows)
    result["t_pvalue_fdr"] = fdr_bh(result.t_pvalue)
    return result


def regional_statistics(frame: pd.DataFrame, *, minimum: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, tests = [], []
    for feature in FEATURES:
        groups = []
        for label, group in frame.groupby("region_label", sort=True):
            data = group[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
            summaries.append({"feature": feature, "region_label": int(label), "n": len(data),
                              "mean": float(data.mean()) if len(data) else np.nan,
                              "std": float(data.std(ddof=1)) if len(data) > 1 else np.nan,
                              "median": float(np.median(data)) if len(data) else np.nan})
            if len(data) >= minimum:
                groups.append(data)
        row = {"feature": feature, "n_regions": len(groups), "n_observations": sum(map(len, groups)),
               "f_statistic": np.nan, "anova_pvalue": np.nan, "kruskal_pvalue": np.nan,
               "eta_squared": np.nan, "status": "insufficient_regions"}
        if len(groups) >= 2:
            all_values = np.concatenate(groups)
            grand = all_values.mean()
            total = float(np.square(all_values-grand).sum())
            between = float(sum(len(g)*(g.mean()-grand)**2 for g in groups))
            row["eta_squared"] = between/total if total > 0 else 0.
            if total == 0:
                row.update(f_statistic=0., anova_pvalue=1., kruskal_pvalue=1., status="equal_constants")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    test = f_oneway(*groups)
                row.update(f_statistic=float(test.statistic), anova_pvalue=float(test.pvalue),
                           kruskal_pvalue=float(kruskal(*groups).pvalue), status="ok")
        tests.append(row)
    result = pd.DataFrame(tests)
    result["anova_pvalue_fdr"] = fdr_bh(result.anova_pvalue)
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        def standardize(values):
            std = values.std(ddof=1)
            return (values-values.mean())/std if std > 0 else values-values.mean()
        summary["z_score"] = summary.groupby("feature")["mean"].transform(standardize)
    return result, summary


def analyze(cfg: dict, *, make_plots: bool = True) -> Path:
    options = settings(cfg)
    root = Path(cfg["work_dir"]) / "caravel"
    regional = pd.read_csv(root / "tables" / "regional.csv", dtype={"case_id": str})
    whole = pd.read_csv(root / "tables" / "whole_brain.csv", dtype={"case_id": str})
    demographics = pd.read_csv(options["demographics"], dtype={"case_id": str})
    if not {"case_id", "age_years", "sex"}.issubset(demographics):
        raise ValueError("Import demographics first: canonical columns are case_id, age_years, sex.")
    expected = {r["case_id"] for r in read_study(cfg) if r["split"] == "test"}
    if set(whole.model) != set(MODELS) or not set(regional.model).issubset(MODELS):
        raise ValueError("Caravel tables must contain only baseline and pediatric model labels.")
    if not set(regional.case_id).issubset(expected):
        raise ValueError("Regional tables contain subjects outside the independent test cohort.")
    if demographics.case_id.duplicated().any() or set(demographics.case_id) != expected:
        raise ValueError("Canonical demographics must contain exactly one row for every independent test case.")
    demographics["age_years"] = pd.to_numeric(demographics.age_years, errors="raise")
    if not np.isfinite(demographics.age_years).all() or (demographics.age_years < 0).any() or not demographics.sex.isin(["M", "F"]).all():
        raise ValueError("Invalid canonical ages or sex labels.")
    for model in MODELS:
        if set(whole.loc[whole.model == model, "case_id"]) != expected:
            raise ValueError(f"Whole-brain {model} measurements do not match the independent test cohort.")
    if whole.duplicated(["model", "case_id"]).any() or regional.duplicated(["model", "case_id", "region_label"]).any():
        raise ValueError("Duplicate Caravel observations.")
    fixed_ids = None
    if options.get("balanced_cases_file"):
        fixed_ids = pd.read_csv(options["balanced_cases_file"], dtype=str).case_id.tolist()
    balanced, report = balance_sexes(demographics, width=options.get("age_bin_width", 5), seed=options.get("seed", 42), fixed_ids=fixed_ids)
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(out / "balanced_cases.csv", index=False)
    report.to_csv(out / "age_bin_balance.csv", index=False)
    joined = whole.merge(demographics, on="case_id", validate="many_to_one")
    all_age, all_sex, all_tests, all_means = [], [], [], []
    for model in MODELS:
        local = joined[joined.model == model]
        age = age_correlations(local, minimum=options.get("minimum_age_n", 10))
        sex = sex_comparisons(local[local.case_id.isin(balanced.case_id)], minimum=options.get("minimum_sex_n", 5))
        tests, means = regional_statistics(regional[regional.model == model], minimum=options.get("minimum_region_n", 5))
        for frame, collection in ((age, all_age), (sex, all_sex), (tests, all_tests), (means, all_means)):
            frame.insert(0, "model", model)
            collection.append(frame)
    frames = {"age_correlations": pd.concat(all_age), "sex_comparisons": pd.concat(all_sex),
              "regional_tests": pd.concat(all_tests), "regional_profiles": pd.concat(all_means)}
    for name, frame in frames.items():
        frame.to_csv(out / (name+".csv"), index=False)
    pairs = whole.pivot(index="case_id", columns="model", values=list(FEATURES))
    summaries = []
    for feature in FEATURES:
        b, p = pairs[feature]["baseline"].to_numpy(), pairs[feature]["pediatric"].to_numpy()
        valid = np.isfinite(b) & np.isfinite(p)
        summaries.append({"feature": feature, "n_pairs": int(valid.sum()),
                          "baseline_median": float(np.median(b[valid])) if valid.any() else np.nan,
                          "trained_median": float(np.median(p[valid])) if valid.any() else np.nan,
                          "median_paired_change": float(np.median(p[valid]-b[valid])) if valid.any() else np.nan})
    pd.DataFrame(summaries).to_csv(out / "paired_descriptor_summary.csv", index=False)
    metadata = {"demographics_sha256": sha256(Path(options["demographics"])), "n_test": len(expected),
                "n_balanced": len(balanced), "balanced_case_ids": balanced.case_id.tolist(), "seed": options.get("seed", 42),
                "sampling_order": "stable case_id sort; pandas sample(random_state=seed) separately within sex/bin",
                "age_bins": "left-closed, right-open, start=floor(min_age); final bin always includes max_age",
                "fdr_families": "eight descriptors, separately by model and test method", "regional_z_score_ddof": 1,
                "regional_test_limitation": "ANOVA/Kruskal-Wallis do not model within-subject dependence; exploratory",
                "versions": versions()}
    if make_plots:
        from .caravel_plots import figures
        metadata["font_used"] = figures(cfg, frames, joined, demographics, balanced, out)
    write_json(out / "analysis_metadata.json", metadata)
    return out

