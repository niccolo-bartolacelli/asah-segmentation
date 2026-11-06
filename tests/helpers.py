"""Shared fixtures for the test suite: synthetic volumes and random weights.

Everything here is synthetic. Tests check interfaces and geometry, never
segmentation quality.
"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from asah_segmentation.model import ASAHSegmentationNet


def make_ct(shape=(96, 80, 3),
            zooms=(0.5, 0.5, 5.0),
            seed=0) -> nib.Nifti1Image:
    """A small CT-like int16 volume with coded qform/sform."""
    rng = np.random.default_rng(seed)
    data = rng.integers(-1000, 120, size=shape).astype(np.int16)
    affine = np.diag((*zooms, 1.0))
    affine[:3, 3] = (-20.0, -18.0, -10.0)
    img = nib.Nifti1Image(data, affine)
    img.set_qform(affine, 1)
    img.set_sform(affine, 1)
    img.header.set_xyzt_units("mm", "sec")
    return img


def materialized_model(seed=0) -> ASAHSegmentationNet:
    """ASAHSegmentationNet with its LazyConv2d layers initialised."""
    torch.manual_seed(seed)
    model = ASAHSegmentationNet().eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 32, 32))
    return model


class ThresholdModel(nn.Module):
    """Deterministic stand-in for the network: class 1 where input > threshold.

    Returns (N, 2, H, W) logits, like ASAHSegmentationNet, and records the
    devices and batch sizes it was called with.
    """

    def __init__(self, threshold: float = 50.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.anchor = nn.Parameter(torch.zeros(1))  # gives the module a device
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(x.shape[0])
        positive = (x > self.threshold).float()
        return torch.cat((1 - positive, positive), dim=1) + 0 * self.anchor


class TempDirTestCase(unittest.TestCase):
    """Gives each test a fresh temporary directory as ``self.tmp``."""

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
