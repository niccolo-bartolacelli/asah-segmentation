""" io.py

Utility functions to load NIfTI volumes.

Exported Functions:
* load_volume: loads a NIfTI volume from a local path or from a file object.

"""

from contextlib import nullcontext
import gzip
from io import IOBase
from os import PathLike

import nibabel as nib
import numpy as np


def load_volume(src: str | PathLike | IOBase,
                v2: bool = False) -> nib.Nifti1Image | nib.Nifti2Image:
    """
    Loads a volume from a NIfTI file or archive.

    Parameters
    ----------
    src : str or PathLike or FileObject
        The path to the volume to load or the already loaded volume
    
    v2 : bool
        Whether to load data as NIfTI2 volume (defaults to `False`
        for backward compatibility)

    Returns
    -------
    Nifti1Image or Nifti2Image
        A NiBabel NIfTI Volume
    """
    image_type = nib.Nifti2Image if v2 else nib.Nifti1Image

    if isinstance(src, str | PathLike):
        file_img = nib.load(src)

        if not isinstance(file_img, image_type):
            raise TypeError(f"Expected a valid NIfTI{2 if v2 else 1} image.")
        
        img = image_type.from_image(file_img)
        return img

    src.seek(0)
    is_compressed = src.read(2) == b"\x1f\x8b"
    src.seek(0)
    
    ctx = (gzip.GzipFile(fileobj=src, mode="rb")
           if is_compressed else
           nullcontext(src))

    with ctx as f:
        img = image_type.from_file_map({"image": nib.FileHolder(fileobj=f)})

        if img.affine is None:
            raise TypeError(f"Expected a valid NIfTI{2 if v2 else 1} image.")      

        ret_header = img.header.copy()
        ret_header.set_data_dtype(np.float32)

        ret_image = type(img)(
            img.get_fdata(dtype=np.float32).copy(),
            img.affine.copy(),
            header=ret_header,
            extra=img.extra.copy(),
        )

        return ret_image
