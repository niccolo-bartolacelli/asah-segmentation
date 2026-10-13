""" io.py

Utility functions to load NIfTI volumes.

Both NIfTI-1 and NIfTI-2 sources are read as they are stored and then
converted to the requested version, in either direction: the format a volume
is handled in is always the caller's choice, never inferred from the source.

Exported Types:
* NiftiVersion: supported NIfTI versions (`1` or `2`).

Exported Functions:
* load_volume: loads a NIfTI volume from a local path or from a file object.
* convert_volume: converts a NIfTI image to the requested version.
* nifti_version_of: returns the NIfTI version of an image or header.
* nifti_image_type: returns the image class of a NIfTI version.

"""

from contextlib import nullcontext
import gzip
from io import IOBase
from os import PathLike
from typing import Literal

import nibabel as nib
from nibabel.spatialimages import HeaderDataError
import numpy as np

type NiftiVersion = Literal[1, 2]

_NIFTI1_MAX_DIM = np.iinfo(np.int16).max


def nifti_image_type(
        nifti_version: NiftiVersion
    ) -> type[nib.Nifti1Image | nib.Nifti2Image]:
    """
    Returns the image class of a NIfTI version.

    Raises
    ------
    ValueError
        If `nifti_version` is neither 1 nor 2
    """
    if nifti_version == 1:
        return nib.Nifti1Image
    if nifti_version == 2:
        return nib.Nifti2Image
    raise ValueError(f"nifti_version must be 1 or 2, got {nifti_version!r}.")


def nifti_version_of(
        obj: nib.Nifti1Image | nib.Nifti1Header |
             nib.Nifti2Image | nib.Nifti2Header
    ) -> NiftiVersion:
    """
    Returns the NIfTI version of an image or header.

    Notes
    -----
    `Nifti2Image` and `Nifti2Header` subclass their NIfTI-1 counterparts, so
    the NIfTI-2 check has to come first.
    """
    header = obj.header if isinstance(obj, nib.Nifti1Image) else obj
    if isinstance(header, nib.Nifti2Header):
        return 2
    if isinstance(header, nib.Nifti1Header):
        return 1
    raise TypeError(f"Expected a NIfTI image or header, got "
                    f"{type(obj).__name__}.")


def convert_volume(img: nib.Nifti1Image | nib.Nifti2Image,
                   nifti_version: NiftiVersion
                   ) -> nib.Nifti1Image | nib.Nifti2Image:
    """
    Converts a NIfTI image to the requested version.

    Images already of the requested version are returned unchanged. Data,
    affine and header fields (qform, sform, zooms, ...) are carried over.

    Parameters
    ----------
    img : Nifti1Image or Nifti2Image
        The source image

    nifti_version : {1, 2}
        The target NIfTI version

    Returns
    -------
    Nifti1Image or Nifti2Image

    Raises
    ------
    ValueError
        If `nifti_version` is invalid
    HeaderDataError
        If the image cannot be represented in the target version (only
        possible when converting NIfTI-2 to NIfTI-1, e.g. dimensions above
        the NIfTI-1 limit of 32767)
    """
    image_type = nifti_image_type(nifti_version)
    if type(img) is image_type:
        return img
    # nibabel would otherwise store an (N, 1, 1) shape beyond the limit with
    # the non-standard FreeSurfer `dim[1] = -1` hack instead of failing.
    if nifti_version == 1 and max(img.shape) > _NIFTI1_MAX_DIM:
        raise HeaderDataError(
            f"Shape {img.shape} exceeds the NIfTI-1 dimension limit "
            f"({_NIFTI1_MAX_DIM})."
        )
    return image_type.from_image(img)


def load_volume(src: str | PathLike | IOBase,
                nifti_version: NiftiVersion = 1
                ) -> nib.Nifti1Image | nib.Nifti2Image:
    """
    Loads a volume from a NIfTI file or archive.

    Parameters
    ----------
    src : str or PathLike or FileObject
        The path to the volume to load or the already loaded volume

    nifti_version : {1, 2}
        NIfTI version of the returned image (defaults to 1 for backward
        compatibility); the source is converted if stored in the other one

    Returns
    -------
    Nifti1Image or Nifti2Image
        A NiBabel NIfTI Volume

    Raises
    ------
    TypeError
        If the source is not a NIfTI-1 or NIfTI-2 image
    HeaderDataError
        If the source cannot be represented in `nifti_version` (see
        `convert_volume`)
    """
    nifti_image_type(nifti_version)  # validates before any I/O

    if isinstance(src, str | PathLike):
        file_img = nib.load(src)

        if not isinstance(file_img, nib.Nifti1Image):
            raise TypeError("Expected a valid NIfTI-1 or NIfTI-2 image.")

        return convert_volume(file_img, nifti_version)

    src.seek(0)
    is_compressed = src.read(2) == b"\x1f\x8b"
    src.seek(0)

    ctx = (gzip.GzipFile(fileobj=src, mode="rb")
           if is_compressed else
           nullcontext(src))

    with ctx as f:
        source_type = _sniff_image_type(f.read(nib.Nifti2Header.sizeof_hdr))
        f.seek(0)
        img = source_type.from_file_map({"image": nib.FileHolder(fileobj=f)})

        if img.affine is None:
            raise TypeError("Expected a valid NIfTI-1 or NIfTI-2 image.")

        ret_header = img.header.copy()
        ret_header.set_data_dtype(np.float32)

        ret_image = type(img)(
            img.get_fdata(dtype=np.float32).copy(),
            img.affine.copy(),
            header=ret_header,
            extra=img.extra.copy(),
        )

    return convert_volume(ret_image, nifti_version)


def _sniff_image_type(head: bytes) -> type[nib.Nifti1Image]:
    """Identifies the NIfTI version of a single-file header block."""
    if nib.Nifti2Header.may_contain_header(head):
        return nib.Nifti2Image
    if nib.Nifti1Header.may_contain_header(head):
        return nib.Nifti1Image
    raise TypeError("Expected a valid NIfTI-1 or NIfTI-2 image.")
