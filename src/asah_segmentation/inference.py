""" inference.py

Volume inference interface.

Exported Functions:
* predict_volume: predicts NIfTI volume segmentation mask.

"""

import nibabel as nib
import torch
import torch.nn as nn

from asah_segmentation.io import NiftiVersion
from asah_segmentation.processing import PreprocessingConfiguration, Processor


@torch.inference_mode()
def predict_volume(
        model: nn.Module,
        volume: nib.Nifti1Image | nib.Nifti2Image,
        device: torch.device,
        batch_size: int = 16,
        config: PreprocessingConfiguration | None = None,
        nifti_version: NiftiVersion | None = None
    ) -> nib.Nifti1Image | nib.Nifti2Image:
    """
    Predicts NIfTI volume segmentation mask.

    Parameters
    ----------
    model : Module
        The already loaded module
    
    volume : Nifti1Image or Nifti2Image
        The volume to run inference on
    
    device : Device
        The device to run inference on
    
    batch_size : int
        Number of slices effectively submitted to the model during inference
        step
    
    config : PreprocessingConfiguration
        Preprocessing requirements (see
        `asah_segmentation.processing.PreprocessingConfiguration` for further
        reference)

    nifti_version : {1, 2}, optional
        NIfTI version of the returned mask; defaults to the version of
        `volume`

    Returns
    -------
    Nifti1Image or Nifti2Image
        The predicted segmentation mask as a NIfTI volume
    
    Notes
    -----
    For raw logits, consider using batch prediction, but keep in mind that
    while this function expects (C, H, W, D) format, model forward step
    expects (B, C, H, W) format and returns a Tensor in the format (N, 2, H, W)
    """

    model.eval()
    processor = Processor(config)

    with processor.slices_from(volume) as slices:
        logits_shape = list(slices.shape)
        logits_shape[1] = 2
        logits = torch.empty(logits_shape,
                             dtype=torch.float32,
                             device=slices.device)

        for start in range(0, slices.shape[0], batch_size):
            batch = slices[start:start + batch_size].to(device)
            logits[start:start + batch_size] = model(batch).to(
                dtype=torch.float32).cpu()

        return processor.predictions_to_nifti(logits, nifti_version)
