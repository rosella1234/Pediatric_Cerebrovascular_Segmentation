"""NIfTI geometry and binary label handling."""

from __future__ import annotations

from pathlib import Path
import shutil
import nibabel as nib
import numpy as np


def image(path: str | Path):
    img = nib.load(str(path))
    if len(img.shape) != 3:
        raise ValueError(f"Expected a 3D NIfTI, found {img.shape}: {path}")
    if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine[:3, :3])) < 1e-12:
        raise ValueError(f"Invalid spatial affine: {path}")
    if img.header.get_xyzt_units()[0] not in {"mm", "unknown"}:
        raise ValueError(f"Spatial units must be millimeters (unknown assumes mm): {path}")
    return img


def same_geometry(first, second, *, atol: float = 1e-5) -> bool:
    return first.shape == second.shape and np.allclose(first.affine, second.affine, rtol=0, atol=atol)


def require_geometry(reference, candidate, *, context: str) -> None:
    if not same_geometry(reference, candidate):
        raise ValueError(f"Geometry mismatch ({context}): shape, spacing, origin, and direction must agree. Evaluation does not resample masks.")


def binary_data(img) -> np.ndarray:
    data = np.asanyarray(img.dataobj)
    if not np.isfinite(data).all() or np.any(data < 0) or not np.all(data == np.floor(data)):
        raise ValueError("A binary/integer label mask is required; probabilities, NaNs, and negative labels are not accepted.")
    return data > 0


def copy_nifti(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.name.endswith(".nii.gz"):
        shutil.copy2(source, destination)
    else:
        nib.save(image(source), str(destination))


def prepare_label(source: Path, reference: Path, destination: Path, *, resample: bool) -> dict:
    img, lab = image(reference), image(source)
    source_mask = binary_data(lab)
    changed = not same_geometry(img, lab)
    if changed and not resample:
        require_geometry(img, lab, context=str(source))
    if changed:
        import SimpleITK as sitk

        lab_sitk = sitk.ReadImage(str(source))
        aligned = sitk.Resample(lab_sitk, sitk.ReadImage(str(reference)), sitk.Transform(), sitk.sitkNearestNeighbor, 0, lab_sitk.GetPixelID())
        sitk.WriteImage(aligned, str(destination))
        result = image(destination)
        require_geometry(img, result, context=str(destination))
        mask = binary_data(result)
    else:
        mask = source_mask
    if source_mask.any() and not mask.any():
        raise ValueError(f"Resampling removed every labeled voxel: {source}. Check physical alignment.")
    if not mask.any():
        raise ValueError(f"Development annotation contains no vessel voxels: {source}")
    # Save discrete 0/1 labels on exactly the image geometry, including its spatial forms.
    header = img.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine, header), str(destination))
    return {"resampled": changed, "source_foreground_voxels": int(source_mask.sum()), "prepared_foreground_voxels": int(mask.sum())}

