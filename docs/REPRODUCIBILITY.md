# Reproducibility notes

## Implemented scope

This release incorporates the supplied conversion, nnU-Net, traditional metrics, and CaravelMetrics modules/notebook. All processing is exposed through Python scripts. The paper-relevant age, sex, regional statistics, and figures have been translated from the notebook; unrelated adult-cohort analyses are outside the manuscript workflow.

## Original artifacts still needed for exact study reconstruction

- The original case-to-development/test allocation, with one consistent pseudonymous ID scheme.
- The original `splits_final.json` or an equivalent list identifying the 18 training and five validation subjects. If IDs are pseudonymized, update every saved split entry consistently.
- The original pediatric `dataset.json`, `nnUNetPlans.json`, and `dataset_fingerprint.json`. Plans should be checked for CTNormalization, the recorded foreground statistics, target spacing, batch size, and patch size.
- The adult baseline model export and its metadata, checkpoint, preprocessing plans, and provenance. The dataset-folder name alone does not identify a model release.
- Manual test labels and the HD-BET brain masks actually used for metric evaluation. The provided scripts do not generate those brain masks.
- Subject-level original CSVs to verify the reported numerical results, especially the bAHD variant and the distinction between 12 retained reports and the full 19-subject tests.

Neither clinical data nor weights are available in this package. A successful synthetic run verifies software behavior rather than reproducing the numerical findings of the manuscript.

## Normalization

The source training script sets `channel_names={0: "CT"}` for TOF-MRA. This intentionally remains the example configuration's channel label, selecting CTNormalization in nnU-Net. It must not be casually changed to MR. The study's clipped intensity range, mean, and standard deviation are properties of the development data and saved plans, not constants to impose on a new cohort.

The dataset ID is 66833 because that is the executable value in the supplied training script; a nearby comment names 668. Dataset IDs are configurable and do not determine the scientific method.

## Geometry

Development labels may be resampled with the identity transform and nearest-neighbor interpolation, matching the supplied script. This reconciles voxel grids using existing physical coordinates; it is not image registration and cannot repair an incorrect anatomical header.

Evaluation checks prediction/reference/TOF/brain-mask geometry and stops if it differs. It does not resample predictions or test labels automatically. NIfTI spatial units must represent millimeters. Integer masks are interpreted as foreground above zero; fractional probabilities and non-finite/negative labels are rejected.

## Distances and bAHD

Distances use the 6-connected erosion boundary, as in the original Python script. Surface voxels are represented by their centers. HD95 is the maximum of directional 95th percentiles, not the percentile of concatenated distances. AHD weights the two directional means equally.

The historical script computes the balanced distance denominator from the **prediction** surface size. The manuscript defines it from the **reference** surface size. Both are exported independently. The primary comparison defaults to the manuscript's reference definition. Use `--bahd-normalization prediction` to compare the source implementation. A manuscript/table decision should be made after checking the original results against both values.

## Statistics and sample sizes

The seven reported metrics are compared with paired two-sided Wilcoxon signed-rank tests. F1 is retained only as a duplicate of Dice in the raw evaluation outputs. SciPy's `method="auto"`, `zero_method="wilcox"`, and no continuity correction are used; the SciPy version is recorded. Ties, zeros, and exact-versus-asymptotic selection can produce small differences from MATLAB defaults. All-zero paired differences are assigned p=1.

There is no multiple-comparison correction for these seven tests, following the paper's traditional-metric section. CaravelMetrics analyses use separate Benjamini-Hochberg families of eight descriptors for each model and test method; these corrections are implemented in caravel_stats.py.

Missing pairs are errors by default. Non-finite measurements are also errors by default. Explicit intersection/exclusion options retain an audit and the actual test-specific sample sizes. Tests are never populated with p-values copied from the paper.

## Validation

The release's validation report records the actual local test count, synthetic execution, versions, and checks against the two supplied Python metric functions. DICOM decoding and nnU-Net GPU execution are not claimed as locally validated; the converter fallback is tested with a mock external process and nnU-Net command construction is checked against the official v2.6.2 interfaces.


## Caravel-specific conventions

The supplied Graph_extractor, Metric_extractor, and Atlas_Registration code is retained under tof_mra/_caravel with attribution. Graph and metric calculations are unchanged. Atlas registration now specifies the crop output explicitly so temporary files stay in the subject’s registration directory. No segmentation or reference annotation is used to estimate the atlas transform; the original implementation’s unused segmentation-array argument receives an inert dummy array. A single registration is shared across both model outputs.

Graph positions are transformed from NIfTI RAS to LPS by the supplied SLICER_MATRIX. SimpleITK atlas indexing expects LPS; no additional flip is applied. The radius estimate remains the source distance transform in voxel units multiplied by mean voxel spacing, rather than an alternative anisotropic physical-distance estimate.

The source notebook’s extensive sums, three-root curvature average, and length-weighted aggregation are preserved. Missing regional rows remain omitted and are audited; they are not silently converted to new observations with value zero. A nonempty graph that produces no regional summary causes an error. A wholly empty/pruned graph produces an explicit zero-count whole-brain summary with undefined curvature.

The balancing code uses left-closed/right-open five-year bins beginning at floor(minimum age), a fixed seed of 42, and a stable case-ID order. It ensures the maximum age is included, fixing the source notebook’s possible upper-bound exclusion. Stable ordering can select different tied candidates than an old notebook whose file-loading order was not saved. Exported balanced_cases.csv freezes the new selection; set caravel.balanced_cases_file to the original selected case list to reproduce the historical subgroup exactly.

Regional Z-scores are computed separately within each descriptor and model using sample SD (ddof=1). They describe relative regional profiles and do not establish absolute reductions between models. Statistical tests use raw values. Region-level repeated observations are not modeled as independent evidence of biological effects; regional ANOVA/Kruskal-Wallis results retain the manuscript’s exploratory limitation.

No clinical metadata workbook or notebook output cells are bundled. Caravel source provenance includes file hashes and the checked upstream revision. Actual age/sex statistics are never populated from the manuscript’s reported numbers.
