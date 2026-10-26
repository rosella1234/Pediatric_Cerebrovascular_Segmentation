"""The study's overlap metrics and explicitly named bAHD variants."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, generate_binary_structure
from scipy.spatial import cKDTree

OVERLAP = ("Dice", "Recall", "Precision", "IoU")
DISTANCE = ("HD95_mm", "AHD_mm", "Balanced_AHD_reference_mm", "Balanced_AHD_prediction_mm")


def overlap_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    pred, ref = np.asarray(prediction) > 0, np.asarray(reference) > 0
    if pred.shape != ref.shape:
        raise ValueError("Prediction and reference shapes differ.")
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    if tp + fp + fn == 0:
        return {key: 1.0 for key in (*OVERLAP, "F1_Score")}
    dice = 2 * tp / (2 * tp + fp + fn)
    return {"Dice": dice, "Recall": tp / (tp + fn) if tp + fn else 0.0,
            "Precision": tp / (tp + fp) if tp + fp else 0.0,
            "IoU": tp / (tp + fp + fn), "F1_Score": dice}


def surface_points(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Voxel-center surface points: 6-connected erosion, then full physical affine."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Distance metrics require 3D masks.")
    border = mask ^ binary_erosion(mask, structure=generate_binary_structure(3, 1), border_value=0)
    voxels = np.argwhere(border)
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("A finite 4x4 physical affine is required.")
    return voxels @ affine[:3, :3].T + affine[:3, 3]


def distance_metrics(prediction: np.ndarray, reference: np.ndarray, affine: np.ndarray) -> dict:
    pred, ref = np.asarray(prediction) > 0, np.asarray(reference) > 0
    if pred.shape != ref.shape:
        raise ValueError("Prediction and reference shapes differ.")
    points_pred, points_ref = surface_points(pred, affine), surface_points(ref, affine)
    n_pred, n_ref = len(points_pred), len(points_ref)
    output = {"n_surface_prediction": n_pred, "n_surface_reference": n_ref}
    if not n_pred or not n_ref:
        output.update({key: float("inf") for key in DISTANCE})
        output["distance_status"] = "both_empty" if not n_pred and not n_ref else "one_empty"
        return output
    to_ref = cKDTree(points_ref).query(points_pred)[0]
    to_pred = cKDTree(points_pred).query(points_ref)[0]
    summed = float(to_ref.sum() + to_pred.sum())
    output.update({
        "HD95_mm": float(max(np.percentile(to_ref, 95), np.percentile(to_pred, 95))),
        "AHD_mm": float((to_ref.mean() + to_pred.mean()) / 2),
        "Balanced_AHD_reference_mm": summed / (2 * n_ref),
        "Balanced_AHD_prediction_mm": summed / (2 * n_pred),
        "distance_status": "ok",
    })
    return output

