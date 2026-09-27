""" preprocessing.py

Utility functions to preprocess NIfTI and DICOM files.

Exported Functions:
* preprocess_nifti: preprocesses a NIfTI volume.
* preprocess_dicom: wrapper for DICOM to NIfTI conversion and NIfTI
preprocessing.

"""

import numpy as np
import numpy.typing as npt
import nilearn.image
import nibabel as nib
import dicom2nifti


def preprocess_nifti(img: nib.Nifti1Image,
                     downsample: int = 2,
                     size: int = 256,
                     clip: tuple[int, int] = (0, 150),
                     dtype: npt.DTypeLike=np.int16) -> nib.Nifti1Image:
    """
    Preprocesses a NIfTI volume.
    Target configuration:
        

    Parameters
    ----------
    img : Nifti1Image
        Source image

    downsample : int
        Downsample rate

    size : int
        Target volume in-plane resolution (square)

    clip : tuple[int, int]
        Clamping values for intensity units

    dtype : dtypelike
        The data type to be used
    
    Returns
    -------
    Nifti1Image
        The preprocessed image
    
    Raises
    ------
    ValueError
        If the provided image does not contain a valid affine matrix
    """

    if img.affine is None:
        raise ValueError(f"{img!r} image is not a valid NIfTI image")

    new_affine = img.affine.copy()
    new_affine[:3, :3] = np.matmul(img.affine[:3, :3],
                                   np.diag((downsample, downsample, 1)))
    min_value = img.get_fdata().min()
    tmp_img = nilearn.image.resample_img(img, target_affine=new_affine,
        target_shape=(size, size, img.shape[2]), fill_value=min_value)
    data = tmp_img.get_fdata()
    if clip:
        data = data.clip(min=clip[0], max=clip[1])
    return nib.Nifti1Image(data.astype(dtype), tmp_img.affine)


def preprocess_dicom(path: str, *args, **kwargs) -> nib.Nifti1Image:
    """
    Preprocess a DICOM volume.
    Actually converts DICOM to NIfTI and runs preprocess_nifti().

    Parameters
    ----------
    img : str
        The path to the source DICOM series
    *args : any, optional
        Variable length argument list
    **kwargs : any, optional
        Arbitrary keyword arguments
    
    Additional arguments are passed to the preprocessing function on the
    converted DICOM image.
    See preprocess_nifti for further reference.
    
    Returns
    -------
    Nifti1Image
        The preprocessed image
    """

    img = dicom2nifti.dicom_series_to_nifti(path,
                                            output_file=None,
                                            reorient_nifti=False)["NII"]
    return preprocess_nifti(img, *args, **kwargs)
