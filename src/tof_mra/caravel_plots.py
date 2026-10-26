"""Paper-aligned CaravelMetrics figures from explicit statistical output tables."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm

from .caravel_data import FEATURES, FEATURE_LABELS, MODELS, settings


def _save(fig, out: Path, name: str, note: str | None):
    if note:
        fig.suptitle(note, fontsize=11, fontweight="bold")
    for extension in ("svg", "png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=220)
    plt.close(fig)


def figures(cfg: dict, tables: dict, joined, demographics, balanced, out: Path) -> str:
    options = settings(cfg)
    requested = cfg.get("evaluation", {}).get("font_family", "Arial")
    try:
        font_manager.findfont(font_manager.FontProperties(family=requested), fallback_to_default=False)
        font = requested
    except ValueError:
        font = "DejaVu Sans"
    note = options.get("figure_note")
    style = {"font.family": font, "font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42,
             "axes.spines.right": False, "axes.spines.top": False}
    labels = [FEATURE_LABELS[f] for f in FEATURES]
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
        for i, method in enumerate(("pearson", "spearman")):
            for j, model in enumerate(MODELS):
                ax = axes[i, j]
                local = tables["age_correlations"].query("model == @model").set_index("feature").reindex(FEATURES)
                values = local[method+"_r"].to_numpy(dtype=float)
                img = ax.imshow(values[:, None], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
                ax.set_yticks(range(8), labels if j == 0 else [])
                ax.set_xticks([0], [method.capitalize()])
                ax.set_title("Baseline" if model == "baseline" else "Trained", fontweight="bold")
                for k, value in enumerate(values):
                    ax.text(0, k, f"{value:.2f}" if np.isfinite(value) else "NA", ha="center", va="center",
                            color="white" if abs(value) > .65 else "black")
        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=.7, label="Age correlation coefficient")
        _save(fig, out, "age_correlations", note)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
        for ax, model in zip(axes, MODELS):
            local = tables["sex_comparisons"].query("model == @model").set_index("feature").reindex(FEATURES)
            effects = local.cohens_d.to_numpy(dtype=float)
            ax.barh(np.arange(8), effects, color="#4779aa" if model == "baseline" else "#db8650")
            for i, (effect, pvalue) in enumerate(zip(effects, local.t_pvalue)):
                if np.isfinite(effect) and pvalue < .05:
                    ax.annotate("*", (effect, i), xytext=(5 if effect >= 0 else -8, 0), textcoords="offset points", va="center")
            ax.set_yticks(range(8), labels)
            ax.set_title("Baseline" if model == "baseline" else "Trained", fontweight="bold")
            ax.axvline(0, color="#555555", lw=.7)
            ax.set_xlabel("Cohen's d (male - female); * nominal p < 0.05")
        axes[0].invert_yaxis()
        _save(fig, out, "sex_effect_sizes", note)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
        bins = np.arange(0, demographics.age_years.max()+5.01, 5)
        for ax, frame, title in zip(axes, (demographics, balanced), ("Complete test cohort", "Age-balanced subgroup")):
            ax.hist([frame.loc[frame.sex == "M", "age_years"], frame.loc[frame.sex == "F", "age_years"]],
                    bins=bins, color=["#4779aa", "#db8650"], label=["Male", "Female"], edgecolor="white")
            ax.set_title(f"{title} (N = {len(frame)})", fontweight="bold")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Subjects")
            ax.legend(frameon=False)
        _save(fig, out, "age_sex_distribution", note)

        profiles = tables["regional_profiles"]
        region_labels = options["region_labels"]
        matrices = [profiles.query("model == @model").pivot(index="feature", columns="region_label", values="z_score").reindex(index=FEATURES, columns=region_labels).to_numpy(dtype=float) for model in MODELS]
        finite = np.concatenate([m.ravel() for m in matrices])
        finite = finite[np.isfinite(finite)]
        limit = max(1., float(np.abs(finite).max())) if len(finite) else 1.
        fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True, layout="constrained")
        for ax, matrix, model in zip(axes, matrices, MODELS):
            img = ax.imshow(np.ma.masked_invalid(matrix), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
            ax.set_title("Baseline" if model == "baseline" else "Trained", fontweight="bold")
            ax.set_xticks(range(len(region_labels)), region_labels, fontsize=8)
            ax.set_yticks(range(8), labels)
            ax.set_xlabel("Atlas region")
        fig.colorbar(img, ax=axes.tolist(), shrink=.8, label="Regional Z-score")
        _save(fig, out, "regional_profiles", note)

        # Age scatter panels corresponding to the two reported curvature descriptors.
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
        for i, feature in enumerate(FEATURES[-2:]):
            for j, model in enumerate(MODELS):
                local = joined.query("model == @model")
                ax = axes[i, j]
                ax.scatter(local.age_years, local[feature], s=24, color="#4779aa" if j == 0 else "#db8650")
                valid = local[["age_years", feature]].dropna()
                if len(valid) >= 2 and valid.age_years.nunique() > 1:
                    slope, intercept = np.polyfit(valid.age_years, valid[feature], 1)
                    xs = np.array([valid.age_years.min(), valid.age_years.max()])
                    ax.plot(xs, slope*xs+intercept, color="#555555", lw=1)
                ax.set_xlabel("Age (years)")
                ax.set_ylabel(FEATURE_LABELS[feature])
                ax.set_title("Baseline" if j == 0 else "Trained", fontweight="bold")
        _save(fig, out, "age_scatter_curvature", note)
        if options.get("plot_atlas", True):
            atlas_path = Path(options["atlas"])
            if not atlas_path.is_file():
                raise FileNotFoundError("Atlas maps require caravel.atlas. Set caravel.plot_atlas=false explicitly for plots without spatial maps.")
            _atlas_maps(atlas_path, profiles, out, note)
    return font


def _atlas_maps(path: Path, profiles: pd.DataFrame, out: Path, note: str | None):
    atlas = nib.as_closest_canonical(nib.load(path))
    data = np.asanyarray(atlas.dataobj)
    if data.ndim != 3 or not np.isfinite(data).all() or not np.all(data == np.floor(data)):
        raise ValueError("Atlas visualization requires a 3D, finite, discrete label atlas.")
    occupied = np.flatnonzero(np.any(data > 0, axis=(0, 1)))
    if not len(occupied):
        raise ValueError("Atlas has no labeled voxels.")
    slices = np.unique(np.quantile(occupied, [.25, .5, .75]).astype(int))
    label_data = data.astype(int)
    limit = max(1., float(profiles.z_score.abs().max()))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    groups = {"regional_maps_main": ("total_length", "num_bifurcations", "num_components"),
              "regional_maps_supplementary": ("num_abnormal_degree_nodes", "volume", "num_loops", *FEATURES[-2:])}
    for name, features in groups.items():
        fig, axes = plt.subplots(len(features), 2, figsize=(13, 2.1*len(features)), squeeze=False, layout="constrained")
        for i, feature in enumerate(features):
            for j, model in enumerate(MODELS):
                frame = profiles.query("model == @model and feature == @feature")
                lookup = dict(zip(frame.region_label, frame.z_score))
                panels = []
                for z in slices:
                    labels = label_data[:, :, z].T
                    panel = np.full(labels.shape, np.nan)
                    for region, value in lookup.items():
                        panel[labels == region] = value
                    panels.append(panel)
                img = axes[i, j].imshow(np.ma.masked_invalid(np.concatenate(panels, axis=1)), cmap=cmap, origin="lower", vmin=-limit, vmax=limit)
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])
                if i == 0:
                    axes[i, j].set_title("Baseline" if j == 0 else "Trained", fontweight="bold")
                if j == 0:
                    axes[i, j].set_ylabel(FEATURE_LABELS[feature])
                axes[i, j].text(.01, .03, "L", transform=axes[i, j].transAxes, fontsize=8)
                axes[i, j].text(.99, .03, "R", ha="right", transform=axes[i, j].transAxes, fontsize=8)
        fig.colorbar(img, ax=axes.ravel().tolist(), shrink=.7, label="Regional Z-score")
        _save(fig, out, name, note)

