"""Editable SVG and static figures with paired observations, medians, and IQRs."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def plot_comparison(summary, data: dict, output: Path, *, font_family: str = "Arial", note: str | None = None) -> str:
    try:
        font_manager.findfont(font_manager.FontProperties(family=font_family), fallback_to_default=False)
        font = font_family
    except ValueError:
        font = "DejaVu Sans"
    style = {"font.family": font, "font.size": 11, "svg.fonttype": "none", "pdf.fonttype": 42,
             "axes.spines.top": False, "axes.spines.right": False}
    rng = np.random.default_rng(42)
    colors = ("#4779aa", "#db8650")
    labels = {"Dice": "Dice", "Recall": "Recall", "Precision": "Precision", "IoU": "IoU",
              "HD95_mm": "HD95", "AHD_mm": "AHD", "Balanced_AHD_reference_mm": "bAHD (reference)",
              "Balanced_AHD_prediction_mm": "bAHD (prediction)"}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(13, 7.2), layout="constrained")
        grid = fig.add_gridspec(2, 12)
        for i, row in summary.iterrows():
            metric = row["metric"]
            ax = fig.add_subplot(grid[0, i*3:(i+1)*3] if i < 4 else grid[1, (i-4)*4:(i-3)*4])
            base, trained = data[metric]
            xs = []
            for j, values in enumerate((base, trained), start=1):
                if len(values) > 1 and np.ptp(values) > 0:
                    violin = ax.violinplot([values], positions=[j], widths=0.65, showextrema=False)
                    violin["bodies"][0].set_facecolor(colors[j-1])
                    violin["bodies"][0].set_alpha(0.35)
                    violin["bodies"][0].set_edgecolor(colors[j-1])
                jitter = j + rng.uniform(-0.065, 0.065, len(values))
                xs.append(jitter)
                ax.scatter(jitter, values, s=21, color=colors[j-1], alpha=0.7, zorder=3)
                q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.plot([j, j], [q1, q3], color="black", lw=3, zorder=4)
                ax.scatter(j, med, marker="D", s=32, facecolor="white", edgecolor="black", zorder=5)
            for x1, x2, b, t in zip(xs[0], xs[1], base, trained):
                ax.plot([x1, x2], [b, t], color="#888888", alpha=0.25, lw=0.6, zorder=1)
            ax.set_xticks([1, 2], ["Baseline", "Trained"])
            ax.set_xlim(0.45, 2.55)
            ax.set_title(labels[metric], fontweight="bold", pad=9)
            ax.set_ylabel("Score" if i < 4 else "Distance (mm)")
            p = row["p_value"]
            ptext = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
            ax.text(0.5, 0.97, f"{ptext}; n = {int(row['n_pairs'])}", ha="center", va="top", transform=ax.transAxes, fontsize=10)
            low, high = min(base.min(), trained.min()), max(base.max(), trained.max())
            span = max(high-low, 0.01 if i < 4 else max(abs(high)*0.1, 0.001))
            bottom, top = low-0.12*span, high+0.38*span
            if i < 4:
                bottom, top = max(-0.025, bottom), min(1.035, top)
            ax.set_ylim(bottom, top)
            ax.grid(axis="y", alpha=0.2, linestyle=":")
        if note:
            fig.suptitle(note, fontsize=12, fontweight="bold")
        for extension in ("svg", "png", "pdf"):
            fig.savefig(output / f"traditional_metrics.{extension}", dpi=250)
        plt.close(fig)
    return font

