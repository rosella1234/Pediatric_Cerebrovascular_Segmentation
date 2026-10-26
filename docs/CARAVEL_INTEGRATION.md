# CaravelMetrics setup and execution

Version 0.2 integrates the uploaded study extraction modules and translates the paper-relevant notebook analyses into Python scripts. The upstream framework is credited in THIRD_PARTY_NOTICES.md. No MATLAB or notebook execution is required.

## Dependencies and atlas inputs

Install the extraction extras with `python -m pip install -e ".[caravel]"`. Supply the original arterial atlas NIfTI and numeric-label text file in the `caravel` configuration section. Neither atlas file was included in the uploaded ZIP. The paper's default is labels 1–32: 30 arterial territories and two lateral ventricles. Use the original label map, with one `integer_label region_name` entry per line.

Predictions must exist as `work/predictions/baseline/<case_id>.nii.gz` and `work/predictions/pediatric/<case_id>.nii.gz`, matching the TOF geometry and manifest IDs. Caravel extracts graphs from these predictions directly; the evaluation brain mask is not applied again.

Provide each subject's T1 path in the manifest and an FSL environment exposing fslroi, bet, fslreorient2std, and flirt. Registration preserves the supplied y-axis crop starting at voxel 50, BET fraction 0.30, q-form-based T1 resampling, and two-step 12-DOF atlas alignment with nearest-neighbor labels and final ±30-degree searches. The q-form resampling assumes physically consistent T1/TOF headers; it does not estimate a new T1-to-TOF registration. Review alignment locally.

Alternatively, set `caravel.registered_atlases` to a directory containing `<case_id>.nii.gz` atlases already on each subject's TOF grid. This bypasses FSL. Geometry and discrete labels are checked. One registered atlas is shared between both models per subject. Local registration outputs are saved under `work/caravel/registration/<case_id>` with source hashes.

## Import demographics

The canonical demographics CSV contains `case_id,age_years,sex`, with sex M/F. The uploaded workbook format uses IXI_ID, AGE and SEX_ID (1=M, 2=F):

```console
tof-study --config configs/study.json caravel-demographics --source /path/IXI_METADATA.xlsx --id-column IXI_ID --output data/demographics.csv
```

If metadata IDs differ from the manifest, add `--case-map data/case_mapping.csv`. That CSV must contain one-to-one `source_id,case_id` correspondence. IDs are never guessed from row order, prefix stripping, or assumed numeric equivalence. Only the three required fields for independent test subjects are exported. The clinical workbook is not distributed.

## Extract, collect, and analyze

```console
tof-study --config configs/study.json caravel-extract
tof-study --config configs/study.json caravel-collect
tof-study --config configs/study.json caravel-analyze
```

Extraction defaults to both models. Use `--model baseline` or `--model pediatric` for one. `--resume` reuses only completed outputs with matching prediction/atlas/label-map hashes and graph settings. Incomplete or unverified directories cause an error. A nonempty graph without regional measurements causes an error rather than a silent zero result.

The graph calculation retains 26-neighbor construction, orphan handling/pruning, removal of components smaller than seven nodes, two smoothing iterations at alpha 0.8, physical edge weights, the original mean-spacing radius approximation, and three-root path curvature sampled at 500 positions. Both models use identical settings. Graph coordinates are converted from RAS to LPS by the supplied SLICER_MATRIX; SimpleITK atlas indexing already expects LPS, so the wrapper does not flip them again.

Existing study results can be analyzed directly: point `caravel.results_dirs.baseline` and `.pediatric` to roots containing `<case_id>/region_summary.csv`, then run collection and analysis. Case directories must match the manifest. Copy/link legacy names explicitly if needed. Component-level `all_components_by_region.csv` is not accepted in place of a region summary.

## Features and aggregation

| Features | Whole-brain aggregation |
|---|---|
| total_length, volume | Sum over observed regions |
| num_components, num_bifurcations, num_loops, num_abnormal_degree_nodes | Sum over regional subgraphs |
| root_mean_curvature_avg, mean_squared_curvature_avg | Vessel-length-weighted mean |

Each curvature descriptor first averages the three root-specific regional columns. Undefined curvature values and their weights are excluded from a weighted average. Missing regional rows remain omitted and are audited. Regional topology sums retain atlas-boundary fragmentation and are not direct whole-graph topology. Bifurcations include degree ≥3; abnormal-degree nodes include degree >3; loops are independent cycles, preserving the source implementation.

## Statistics and subgroup reproducibility

Age associations use Pearson and Spearman correlations, with at least ten finite observations by default. Sex analyses use equal male/female counts in five-year bins with seed 42, pooled-variance two-sided t-tests, and male-minus-female Cohen's d, with at least five observations per sex. Regional tests use raw observations with at least five subjects per region, ANOVA, complementary Kruskal–Wallis, and eta-squared. Regional tests retain the paper's exploratory limitation because within-subject dependence is not modeled.

FDR families are eight descriptors, separately by model and test method. Untestable descriptors retain status/missing results; they are treated as p=1 during adjustment to preserve family size, while their own adjusted result stays missing. Paired model summaries report descriptive medians and changes without introducing additional inferential claims.

The subgroup is selected once and shared between models. IDs and per-bin counts are saved. Case IDs are sorted before seeded sampling to remove dependence on directory loading order, and the maximum-age endpoint is included. Consequently, an old notebook's subgroup may differ if its input order was not saved. Set `caravel.balanced_cases_file` to a CSV containing the original `case_id` list for exact historical selection; within-bin sex balance is checked. No 10–20-year restriction is forced to obtain N=12: subgroup size/ages are consequences of the current data.

## Tables and figures

| Output under work/caravel | Contents |
|---|---|
| `<model>/<case_id>/` | Graph pickle/VTP, component/regional CSVs, extraction log/provenance |
| `tables/regional.csv` | Subject/model/region values for eight descriptors |
| `tables/whole_brain.csv` | Subject-level regional aggregates |
| `tables/collection_audit.json` | Observed/omitted regions and source hashes |
| `analysis/balanced_cases.csv`, `age_bin_balance.csv` | Selected subgroup and original/selected counts |
| `analysis/age_correlations.csv` | Coefficients, nominal/FDR p-values, sample sizes/status |
| `analysis/sex_comparisons.csv` | Pooled t-tests, Cohen's d, nominal/FDR p-values and group counts |
| `analysis/regional_tests.csv` | ANOVA/Kruskal–Wallis, eta-squared and adjusted ANOVA p-values |
| `analysis/regional_profiles.csv` | Raw summaries and standardized regional means |
| `analysis/paired_descriptor_summary.csv` | Descriptive baseline/trained changes |
| `analysis/analysis_metadata.json` | Subgroup IDs, settings, hashes and versions |

Editable SVG, PNG and PDF figures include age correlations, curvature scatter panels, sex effect sizes, complete/balanced age distributions, regional profiles, and main/supplementary atlas maps. Baseline is left and trained is right. Heatmaps/maps share color scales between models. Sex-effect stars indicate nominal p<0.05; adjusted results remain in the tables.

Regional Z-scores use sample SD (ddof=1), separately within each feature/model, matching the notebook. Tests use unstandardized observations. A relative Z-score change does not establish an absolute reduction. Maps use canonical RAS orientation and three axial slices selected at quartiles of occupied atlas slices. Supply `caravel.atlas` for maps, set `caravel.plot_atlas=false` explicitly for non-spatial plots, or use `caravel-analyze --no-plots` for tables only.

## Synthetic demonstration and limits

```console
tof-study caravel-demo --output caravel-demo-output
```

This creates 19 artificial subjects, artificial regional summaries, a non-anatomical label atlas, and a 12-subject balanced subgroup. Every figure is marked synthetic. This demo exercises collection/statistics/plots; it does not perform registration, graph extraction, inference or training. The separate extraction test runs the real graph/metric modules on a synthetic vessel with translated/anisotropic physical geometry and validates RAS-to-LPS atlas assignment. FSL registration and the clinical dataset were unavailable for local execution. Actual study findings still require the original data, atlas, model exports and exact subject/subgroup allocations.
