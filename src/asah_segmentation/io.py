""" io.py

Utility functions to load volumes from files to usable `Nifti1Image` image
objects.

Exported Functions:
* load_volume: loads a volume from a local path or from an already loaded file
object.

"""

from contextlib import nullcontext
import gzip
from io import IOBase

import nibabel as nib
import numpy as np

def load_volume(src: str | IOBase,
                v2: bool = False) -> tuple[nib.Nifti1Image | nib.Nifti2Image,
                                          np.ndarray]:
    """
    Loads a volume from a NIfTI file or archive.
    If you need to load a DICOM volume, consider running preprocess_dicom
    instead.

    Parameters
    ----------
    param : str or FileObject
        The path to the volume to load or the already loaded volume
    
    v2 : bool
        Whether to load data as NIfTI2 volume (defaults to `False`
        for backward compatibility)

    Returns
    -------
    tuple[Nifti1Image or Nifti2Image, ndarray]
        A NiBabel NIfTI Volume
    """
    image_type = nib.Nifti2Image if v2 else nib.Nifti1Image

    if isinstance(src, str):
        file_img = nib.load(src)

        if not isinstance(file_img, image_type):
            raise TypeError(f"Expected a valid NIfTI{2 if v2 else 1} image.")
        
        img = image_type.from_image(file_img)
        return img, img.get_fdata(dtype=np.float32).copy()

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

        data = img.get_fdata(dtype=np.float32).copy()

        ret_header = img.header.copy()
        ret_header.set_data_dtype(np.float32)

        ret_image = type(img)(
            data,
            img.affine.copy(),
            header=ret_header,
            extra=img.extra.copy(),
        )

        return ret_image, data
