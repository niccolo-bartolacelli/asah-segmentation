""" api.py

Public, stable interface to ASAH segmentation.

Everything a consumer needs (loading weights, choosing a device, reading a
volume, predicting a mask, saving it) goes through `ASAHSegmenter`. The
modules it wraps (`model`, `processing`, `inference`, `io`, `serialization`)
are implementation details and may change between minor releases; the names
exported here (see `__all__`) keep their signatures and behaviour.

Exported Classes:
* ASAHSegmenter: loads a model once and segments NIfTI volumes with it.
* CheckpointError: a checkpoint is missing, unreadable or does not fit the
model.
* VolumeError: an input is not a usable 3D NIfTI volume.

Exported Constants:
* BACKGROUND_LABEL, SAH_LABEL: label values of the predicted mask.

Examples
--------
>>> from asah_segmentation import ASAHSegmenter
>>> segmenter = ASAHSegmenter.from_checkpoint("asah_nonewnet.pt")
>>> mask = segmenter.segment("scan.nii.gz")          # Nifti1Image, uint8
>>> segmenter.segment_file("scan.nii.gz", "mask.nii.gz")

"""

from __future__ import annotations

import hashlib
from io import IOBase
from os import PathLike
from pathlib import Path
from typing import Any

import nibabel as nib
import torch
import torch.nn as nn

from asah_segmentation.inference import predict_volume
from asah_segmentation.io import (NiftiVersion, convert_volume, load_volume,
                                  nifti_image_type)
from asah_segmentation.model import ASAHSegmentationNet
from asah_segmentation.processing import PreprocessingConfiguration

__all__ = [
    "ASAHSegmenter",
    "CheckpointError",
    "VolumeError",
    "PreprocessingConfiguration",
    "BACKGROUND_LABEL",
    "SAH_LABEL",
]

BACKGROUND_LABEL = 0
SAH_LABEL = 1

type VolumeSource = str | PathLike | IOBase | nib.Nifti1Image | nib.Nifti2Image
type DeviceLike = str | torch.device


class CheckpointError(RuntimeError):
    """A checkpoint is missing, unreadable or does not fit the model."""


class VolumeError(ValueError):
    """An input is not a usable 3D NIfTI volume."""


class ASAHSegmenter:
    """
    Aneurysmal SAH segmentation of non-contrast head CT volumes.

    Holds a loaded model on a device; `segment` turns a NIfTI volume into a
    `uint8` label map (`BACKGROUND_LABEL` / `SAH_LABEL`) with the same shape,
    affine, qform and sform as the input, so it overlays the source scan.

    Instances are cheap to reuse: load once, segment many volumes.
    """

    def __init__(self,
                 model: nn.Module,
                 *,
                 device: DeviceLike = "auto",
                 batch_size: int = 16,
                 preprocessing: PreprocessingConfiguration | None = None,
                 nifti_version: NiftiVersion = 1,
                 checkpoint: str | PathLike | None = None) -> None:
        """
        Parameters
        ----------
        model : Module
            A model with loaded weights (normally built by `from_checkpoint`
            or `from_legacy_checkpoint`)

        device : str or device
            `"auto"` (CUDA if available, otherwise CPU), or any torch device
            specification such as `"cpu"`, `"cuda"`, `"cuda:1"`

        batch_size : int
            Slices submitted to the model per forward pass

        preprocessing : PreprocessingConfiguration, optional
            Preprocessing parameters; the defaults match the published model

        nifti_version : {1, 2}
            NIfTI format inputs are read as and masks are returned in. The
            published model was validated on NIfTI-1 only (as was the original
            TensorFlow implementation), hence the default

        checkpoint : str or PathLike, optional
            Path the weights were loaded from (provenance only)
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        nifti_image_type(nifti_version)  # raises ValueError if invalid

        self._device = self.resolve_device(device)
        self._model = model.to(self._device).eval()
        self._batch_size = int(batch_size)
        self._preprocessing = preprocessing or PreprocessingConfiguration()
        self._nifti_version: NiftiVersion = nifti_version
        self._checkpoint = Path(checkpoint) if checkpoint is not None else None
        self._checkpoint_sha256: str | None = None


    # -- construction -------------------------------------------------------

    @classmethod
    def from_checkpoint(cls,
        path: str | PathLike,
        *,
        device: DeviceLike = "auto",
        batch_size: int = 16,
        preprocessing: PreprocessingConfiguration | None = None,
        nifti_version: NiftiVersion = 1
    ) -> ASAHSegmenter:
        """
        Builds a segmenter from a PyTorch checkpoint.

        Parameters
        ----------
        path : str or PathLike
            A file holding the model `state_dict` (as written by
            `save_checkpoint`); a `{"state_dict": ...}` wrapper is accepted

        device, batch_size, preprocessing, nifti_version
            See `ASAHSegmenter.__init__`

        Returns
        -------
        ASAHSegmenter

        Raises
        ------
        CheckpointError
            If the file is missing, unreadable, or does not match the model
        """
        path = Path(path)
        if not path.is_file():
            raise CheckpointError(f"Checkpoint not found: '{path}'.")

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise CheckpointError(
                f"Could not read checkpoint '{path}': {exc}"
            ) from exc

        state_dict = cls._extract_state_dict(payload)
        if state_dict is None:
            raise CheckpointError(
                f"'{path}' does not contain a model state_dict."
            )

        model = ASAHSegmentationNet()
        try:
            # Materialises the LazyConv2d parameters from the stored shapes.
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            raise CheckpointError(
                f"Checkpoint '{path}' does not match the model: {exc}"
            ) from exc

        return cls(model, device=device, batch_size=batch_size,
                   preprocessing=preprocessing, nifti_version=nifti_version,
                   checkpoint=path)


    @classmethod
    def from_legacy_checkpoint(
        cls,
        path: str | PathLike,
        *,
        device: DeviceLike = "auto",
        batch_size: int = 16,
        preprocessing: PreprocessingConfiguration | None = None,
        nifti_version: NiftiVersion = 1
    ) -> ASAHSegmenter:
        """
        Builds a segmenter from the original TensorFlow checkpoint.

        Requires the `legacy` optional dependencies (TensorFlow). Use it once
        and persist the result with `save_checkpoint`; production should load
        the PyTorch checkpoint with `from_checkpoint`.

        Raises
        ------
        CheckpointError
            If TensorFlow is not installed or the checkpoint cannot be read
        """
        from asah_segmentation import serialization

        if not serialization.TF_AVAILABLE:
            raise CheckpointError(
                "TensorFlow is required to read legacy checkpoints: install "
                "asah-segmentation with the `legacy` extra."
            )
        try:
            model = serialization.load_legacy_model(str(path))
        except Exception as exc:
            raise CheckpointError(
                f"Could not convert legacy checkpoint '{path}': {exc}"
            ) from exc

        return cls(model, device=device, batch_size=batch_size,
                   preprocessing=preprocessing, nifti_version=nifti_version,
                   checkpoint=path)


    # -- inference ----------------------------------------------------------

    def segment(self,
                volume: VolumeSource) -> nib.Nifti1Image | nib.Nifti2Image:
        """
        Predicts the SAH mask of a volume.

        Parameters
        ----------
        volume : str, PathLike, file object, Nifti1Image or Nifti2Image
            The non-contrast CT volume (3D; a singleton 4th axis is accepted),
            read as the segmenter's `nifti_version`

        Returns
        -------
        Nifti1Image or Nifti2Image
            `uint8` label map on the input's voxel grid, in the segmenter's
            `nifti_version`

        Raises
        ------
        VolumeError
            If the input is not a readable 3D volume of the configured NIfTI
            version
        """
        img = self.read_volume(volume, self._nifti_version)
        return predict_volume(self._model, img, self._device,
                              self._batch_size, self._preprocessing,
                              self._nifti_version)


    def segment_file(self,
                     src: str | PathLike | IOBase,
                     dst: str | PathLike) -> Path:
        """
        Segments a NIfTI file and writes the mask to `dst`.

        Returns
        -------
        Path
            The written file (`.nii` or `.nii.gz`, from `dst`'s suffix)
        """
        mask = self.segment(src)
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(mask, dst)
        return dst


    # -- persistence --------------------------------------------------------

    def save_checkpoint(self, path: str | PathLike) -> Path:
        """
        Writes the model `state_dict` in the format `from_checkpoint` reads.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.detach().cpu()
                      for k, v in self._model.state_dict().items()}
        tmp = path.with_name(path.name + ".tmp")
        torch.save(state_dict, tmp)
        tmp.replace(path)
        return path


    # -- properties ---------------------------------------------------------

    @property
    def device(self) -> torch.device:
        """The device inference runs on."""
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def preprocessing(self) -> PreprocessingConfiguration:
        return self._preprocessing

    @property
    def nifti_version(self) -> NiftiVersion:
        """NIfTI format inputs are read as and masks are returned in."""
        return self._nifti_version

    @property
    def checkpoint(self) -> Path | None:
        """Path the weights were loaded from, if any."""
        return self._checkpoint

    @property
    def checkpoint_sha256(self) -> str | None:
        """
        SHA-256 of the checkpoint file (computed lazily), if loaded from one.
        """
        if (self._checkpoint_sha256 is None and
            self._checkpoint is not None and
            self._checkpoint.is_file()):
            h = hashlib.sha256()
            with open(self._checkpoint, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._checkpoint_sha256 = h.hexdigest()
        return self._checkpoint_sha256

    @property
    def model(self) -> nn.Module:
        """The underlying module (not covered by the stability guarantee)."""
        return self._model

    def info(self) -> dict[str, Any]:
        """JSON-serialisable provenance of this segmenter."""
        from asah_segmentation import __version__

        return {
            "package_version": __version__,
            "checkpoint": self._checkpoint.name if self._checkpoint else None,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self._device),
            "batch_size": self._batch_size,
            "nifti_version": self._nifti_version,
            "torch": torch.__version__,
        }

    def __repr__(self) -> str:
        ckpt = self._checkpoint.name if self._checkpoint else None
        return (f"{type(self).__name__}(device={str(self._device)!r}, "
                f"batch_size={self._batch_size}, checkpoint={ckpt!r})")


    # -- helpers ------------------------------------------------------------

    @staticmethod
    def resolve_device(device: DeviceLike = "auto") -> torch.device:
        """`"auto"` picks CUDA if available, otherwise CPU."""
        if isinstance(device, torch.device):
            return device
        if device != "auto":
            resolved = torch.device(device)
            if resolved.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
            return resolved
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


    @staticmethod
    def read_volume(volume: VolumeSource,
                    nifti_version: NiftiVersion = 1
                    ) -> nib.Nifti1Image | nib.Nifti2Image:
        """
        Loads a volume as NIfTI-`nifti_version` and checks it is 3D.

        NIfTI-1 and NIfTI-2 inputs are both accepted and converted to
        `nifti_version` when stored in the other one, in either direction.
        Already loaded images of the requested version are returned as they
        are (after the same checks). A singleton 4th axis is dropped.

        Raises
        ------
        ValueError
            If `nifti_version` is neither 1 nor 2
        VolumeError
            If the input cannot be read, cannot be represented in
            `nifti_version` (NIfTI-2 images exceeding NIfTI-1 limits), or is
            not 3D
        """
        nifti_image_type(nifti_version)  # raises ValueError if invalid

        try:
            # Nifti2Image subclasses Nifti1Image.
            if isinstance(volume, nib.Nifti1Image):
                img = convert_volume(volume, nifti_version)
            else:
                img = load_volume(volume, nifti_version)
        except Exception as exc:
            raise VolumeError(
                f"Could not read volume as NIfTI-{nifti_version}: {exc}"
            ) from exc

        if img.ndim == 4 and img.shape[3] == 1:
            img = img.slicer[..., 0]
        if img.ndim != 3:
            raise VolumeError(
                f"Expected a 3D volume, got shape {tuple(img.shape)}."
            )
        if img.affine is None:
            raise VolumeError("Volume has no affine.")
        return img


    @staticmethod
    def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
        if (isinstance(payload, dict) and
            isinstance(payload.get("state_dict"), dict)):
            payload = payload["state_dict"]
        if (isinstance(payload, dict) and
            payload and
            all(isinstance(v, torch.Tensor) for v in payload.values())):
            return payload
        return None
