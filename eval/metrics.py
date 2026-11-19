""" metrics.py

Segmentation agreement metrics.

Both masks must lie on the same voxel grid, as the ones `ASAHSegmenter.segment`
returns do.

Exported Types:
* MaskSource: anything a label map can be read from.

Exported Functions:
* dice_coefficient: Dice coefficient of one label in two masks.
* absolute_volume_difference: absolute difference of the volume of one label
  in two masks, in millilitres.

"""

from os import PathLike

import nibabel as nib
import numpy as np
import numpy.typing as npt
import torch


type MaskSource = (str | PathLike | nib.Nifti1Image | nib.Nifti2Image |
                   torch.Tensor | npt.ArrayLike)


def dice_coefficient(prediction: MaskSource,
                     reference: MaskSource,
                     label: int | None = None,
                     empty_score: float = 1.0,
                     atol: float = 1e-4) -> float:
    """
    Computes the Dice coefficient of a label in two masks.

    DICE = 2 |P ∩ R| / (|P| + |R|), where P and R are the voxels of
    `prediction` and `reference` holding `label`.

    Parameters
    ----------
    prediction : str, PathLike, Nifti1Image, Nifti2Image, Tensor or ArrayLike
        The predicted label map (a path is read with NiBabel)

    reference : str, PathLike, Nifti1Image, Nifti2Image, Tensor or ArrayLike
        The label map to compare against (e.g. the legacy TensorFlow mask or
        a manual annotation)

    label : int, optional
        The label to score; if None, every non-zero voxel is foreground

    empty_score : float
        Returned when `label` appears in neither mask, where the coefficient
        is undefined; 1.0 counts two empty masks as a perfect agreement

    atol : float
        Absolute tolerance on the affine matrices when both masks are NIfTI
        images

    Returns
    -------
    float
        The coefficient, in [0, 1]

    Raises
    ------
    ValueError
        If the masks have different shapes or, when both are NIfTI images,
        different affines
    """
    pred_fg, ref_fg, _ = _foregrounds(prediction, reference, label, atol)

    total = int(np.count_nonzero(pred_fg)) + int(np.count_nonzero(ref_fg))
    if total == 0:
        return float(empty_score)

    intersection = int(np.count_nonzero(pred_fg & ref_fg))
    return 2.0 * intersection / total


def absolute_volume_difference(prediction: MaskSource,
                               reference: MaskSource,
                               label: int | None = None,
                               voxel_volume: float | None = None,
                               atol: float = 1e-4) -> float:
    """
    Computes the absolute volume difference of a label in two masks.

    AVD = | |P| - |R| | * v, where P and R are the voxels of `prediction` and
    `reference` holding `label` and v is the volume of one voxel.

    Parameters
    ----------
    prediction : str, PathLike, Nifti1Image, Nifti2Image, Tensor or ArrayLike
        The predicted label map (a path is read with NiBabel)

    reference : str, PathLike, Nifti1Image, Nifti2Image, Tensor or ArrayLike
        The label map to compare against

    label : int, optional
        The label to measure; if None, every non-zero voxel is foreground

    voxel_volume : float, optional
        The volume of one voxel in mm³; if None, it is read from the affine of
        whichever mask is a NIfTI image (spatial units assumed to be mm)

    atol : float
        Absolute tolerance on the affine matrices when both masks are NIfTI
        images

    Returns
    -------
    float
        The difference in millilitres, >= 0

    Raises
    ------
    ValueError
        If the masks have different shapes or, when both are NIfTI images,
        different affines; or if `voxel_volume` is None and neither mask is a
        NIfTI image
    """
    pred_fg, ref_fg, affine = _foregrounds(prediction, reference, label, atol)
    if voxel_volume is None:
        if affine is None:
            raise ValueError("voxel_volume is required when neither mask is "
                             "a NIfTI image.")
        voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))

    voxels = abs(int(np.count_nonzero(pred_fg)) -
                 int(np.count_nonzero(ref_fg)))
    return voxels * voxel_volume / 1000.0  # mm³ -> mL


def _foregrounds(
        prediction: MaskSource,
        reference: MaskSource,
        label: int | None,
        atol: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Boolean foreground of `label` in both masks, checked to share a grid,
    and the affine of the first one that is a NIfTI image (None if neither).
    """
    pred_img = _as_image(prediction)
    ref_img = _as_image(reference)
    pred = _mask_data(pred_img if pred_img is not None else prediction)
    ref = _mask_data(ref_img if ref_img is not None else reference)

    pred = pred.squeeze()
    ref = ref.squeeze()

    if pred.shape != ref.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} and {ref.shape}.")
    if (pred_img is not None and ref_img is not None and
        not np.allclose(pred_img.affine, ref_img.affine, atol=atol)):
        raise ValueError("Mask affines differ: the masks are not on the same "
                         "voxel grid.")

    if label is None:
        pred_fg, ref_fg = pred != 0, ref != 0
    else:
        pred_fg, ref_fg = pred == label, ref == label

    img = pred_img if pred_img is not None else ref_img
    return pred_fg, ref_fg, None if img is None else img.affine


def _as_image(
        src: MaskSource
    ) -> nib.Nifti1Image | nib.Nifti2Image | None:
    """Loads paths, passes NIfTI images through; None for in-memory data."""
    if isinstance(src, str | PathLike):
        img = nib.load(src)
        if not isinstance(img, nib.Nifti1Image):
            raise TypeError("Expected a valid NIfTI-1 or NIfTI-2 image.")
        return img
    if isinstance(src, nib.Nifti1Image):  # Nifti2Image subclasses it
        return src
    return None


def _mask_data(src: MaskSource) -> np.ndarray:
    """Label map as an integer array (float labels are rounded)."""
    if isinstance(src, nib.Nifti1Image):
        data = np.asanyarray(src.dataobj)
    elif isinstance(src, torch.Tensor):
        data = src.detach().cpu().numpy()
    else:
        data = np.asarray(src)

    if np.issubdtype(data.dtype, np.floating):
        data = np.rint(data)
    return data.astype(np.int64, copy=False)
