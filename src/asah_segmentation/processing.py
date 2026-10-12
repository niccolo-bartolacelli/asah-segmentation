""" processing.py

Pre and post processing interface.

Exported Classes:
* PreprocessingConfiguration: preprocessing configuration options definition.
* Processor: context manager and methods for pre and post processing tasks.

"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import nilearn.image
import nibabel as nib
import torch


class ImageContext:
    def __init__(self,
                 header: nib.Nifti1Header | nib.Nifti2Header | None,
                 original_shape: tuple[int, ...] | None,
                 original_affine: np.ndarray,
                 processed_shape: tuple[int, ...] | None = None,
                 processed_affine: np.ndarray | None = None) -> None:
        
        self.original_header = header.copy() if header is not None else None
        self.original_shape = original_shape
        self.original_affine = (original_affine.copy()
                                if original_affine is not None
                                else None)
        self.processed_shape = processed_shape
        self.processed_affine = (processed_affine.copy()
                                 if processed_affine is not None
                                 else None)


    @staticmethod
    def is_valid(
        context: ImageContext | None,
        fields: list[str] | None = None,
        mode: Literal["include", "exclude", "all_required"] = "all_required"
    ) -> tuple[bool, str]:
        """
        Checks if the provided context is valid.

        Parameters
        ----------
        context : ImageContext
            The relevant image context

        fields : list[str], optional
            Relevant object fields.

        mode : str
            Available options:

            * `"all_required"`: each entry in object `__dict__` is required not
            to be None

            * `"include"`: each entry in object `__dict__` matching a valid
            `"fields"` entry is required not to be None
            
            * `"exclude"`: each entry in object `__dict__` NOT matching a valid
            `"fields"` entry is required not to be None
        
        Returns
        -------
        None

        Raises
        ------
        ValueError
            If any of the requested constraints is not met
        """

        if mode in ("include", "exclude") and fields is None:
            raise ValueError("fields parameter must not be null if mode is"
                                "either 'include' or 'exclude'")
        
        if context is None:
            raise ValueError("Context is not initialized")

        for k, v in context.__dict__.items():
            if ((mode == "all_required" or
                (mode == "include" and k in fields) or
                (mode == "exclude" and k not in fields))
                and v is None):
                return False, f"Context is not valid: {k} is mandatory"

        return True, ""


    @classmethod
    def from_image(cls, img: nib.Nifti1Image | nib.Nifti2Image):
        """
        Constructs an ImageContext from NIfTI image.

        Parameters
        ----------
        img : Nifti1Image or Nifti2Image
            The relevant image
        
        Returns
        -------
        ImageContext

        Notes
        -----
        N.B. This function internally calls `ImageContext.assert_valid`,
        requiring input image header, shape and affine not to be None.

        See the function documentation for further reference
        """

        if img.affine is None:
            raise ValueError(f"image must have a valid affine matrix.")
        
        inst = cls(img.header, img.shape, img.affine)

        valid, msg = cls.is_valid(inst,
                                  fields=["processed_shape",
                                          "processed_affine"],
                                  mode="exclude")
        if not valid:
            raise ValueError(msg)
        
        return inst


@dataclass(frozen=True)
class PreprocessingConfiguration:
    """
    Processing configuration, applied in-plane to each slice of the processed
    volume.
    """
    downsample: tuple[int, int] = (2, 2)
    shape: tuple[int, int] = (256, 256)
    clip: tuple[int, int] = (0, 150)
    dtype: npt.DTypeLike=np.int16


class Processor:
    """
    Pre and post-inference processing interface.

    This class handles both preprocessing and postprocessing while maintaining
    context on the data being processed across the segmentation pipeline.

    Content-agnostic configuration parameters are passed to the initialization
    function, while content-relative configuration (like image headers and
    affine matrix) are managed and validated internally by the Processor.
    """
    def __init__(self,
                 config: PreprocessingConfiguration | None = None) -> None:
        """
        Parameters
        ----------
        config : PreprocessingConfiguration
            The default (in-plane) configuration used for image preprocessing.
        """

        self._context = None
        self.config = config or PreprocessingConfiguration()


    @contextmanager
    def slices_from(self, img: nib.Nifti1Image):
        if self._context != None:
            raise ValueError("Context is already initialized")
        
        try:
            self._context = ImageContext.from_image(img)
            p_img = self.preprocess_volume(img)
            
            self._context.processed_shape = p_img.shape
            self._context.processed_affine = p_img.affine

            yield self.volume_to_slices(p_img)
        finally:
            self._context = None


    def preprocess_volume(self, img: nib.Nifti1Image) -> nib.Nifti1Image:
        """
        Preprocesses a NIfTI volume with the object configuration setup.
        
        For default preprocessing values, see `PreprocessingConfiguration`.

        Parameters
        ----------
        img : Nifti1Image
            Source image
        
        Returns
        -------
        Nifti1Image
            The preprocessed image
        
        Raises
        ------
        ValueError
            If the provided image does not contain a valid affine matrix
        """

        new_affine = img.affine.copy()
        new_affine[:3, :3] = np.matmul(img.affine[:3, :3],
                                       np.diag((*self.config.downsample, 1)))
        min_value = img.get_fdata().min()
        tmp_img = nilearn.image.resample_img(img,
                                             target_affine=new_affine,
                                             target_shape=(*self.config.shape,
                                                           img.shape[2]),
                                             fill_value=min_value)
        data = tmp_img.get_fdata()
        if self.config.clip:
            data = data.clip(min=self.config.clip[0], max=self.config.clip[1])

        return nib.Nifti1Image(data.astype(self.config.dtype), tmp_img.affine)


    def volume_to_slices(
            self,
            volume: nib.Nifti1Image | nib.Nifti2Image
        ) -> torch.Tensor:
        data = volume.get_fdata(dtype=np.float32).copy()

        if data.ndim != 3:
            raise ValueError("Expected a 3D volume.")

        return torch.from_numpy(
            data.transpose((2, 0, 1)).copy()
        ).unsqueeze(1)


    def predictions_to_nifti(
            self,
            logits: torch.Tensor,
            v2: bool = False
        ) -> nib.Nifti1Image | nib.Nifti2Image:
        valid, msg = ImageContext.is_valid(self._context)
        if not valid:
            raise ValueError(msg)

        image_type = nib.Nifti2Image if v2 else nib.Nifti1Image

        labels = logits.argmax(dim=1)
        data = (labels.permute(1, 2, 0)
                .to("cpu").clone().numpy().astype(np.uint8))

        img = image_type(
            data,
            affine=self._context.processed_affine,
            header=self._context.original_header
        )

        r_img = nilearn.image.resample_img(
            img,
            target_affine=self._context.original_affine,
            target_shape=self._context.original_shape,
            interpolation="nearest"
        )

        data = np.asarray(r_img.dataobj)
        if data.shape != self._context.original_shape:
            raise RuntimeError(
                f"Prediction shape {data.shape} does not match input "
                f"{self._context.original_shape}."
            )

        labels = np.rint(data).astype(np.uint8)
        header = self._context.original_header.copy()
        out = image_type(
            labels,
            affine=self._context.original_affine,
            header=header
        )
        out.set_data_dtype(np.uint8)
        out.header.set_slope_inter(1, 0)
        out.header["cal_min"] = 0
        out.header["cal_max"] = int(labels.max(initial=0))
        out.header["intent_code"] = 1002  # NIFTI_INTENT_LABEL
        out.header["descrip"] = b"segmentation label map"

        qform, qcode = header.get_qform(coded=True)
        sform, scode = header.get_sform(coded=True)
        out.set_qform(
            qform if qform is not None else self._context.original_affine,
            int(qcode or 1)
        )
        out.set_sform(
            sform if sform is not None else self._context.original_affine,
            int(scode or 1)
        )
        return out
