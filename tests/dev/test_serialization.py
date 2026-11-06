"""Tests for the legacy checkpoint conversion script,
``scripts/convert_legacy_checkpoint.py``.

TensorFlow is never needed: ``tf`` is replaced by a fake checkpoint reader
built from the weights of a randomly initialised model.
"""

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from asah_segmentation import ASAHSegmenter
from scripts import convert_legacy_checkpoint
from tests.helpers import TempDirTestCase, materialized_model

REPO_ROOT = Path(__file__).resolve().parents[2]

# The legacy NiftyNet naming, restated independently of the script:
# (TF scope, torch module, (first layer, second layer)) for every block.
BLOCKS = [
    ("L1", "down_blocks.0", ("30_in_prelu", "30_in_prelu_1")),
    ("L2", "down_blocks.1", ("60_in_prelu", "60_in_prelu_1")),
    ("L3", "down_blocks.2", ("120_in_prelu", "120_in_prelu_1")),
    ("L4", "down_blocks.3", ("240_in_prelu", "240_in_prelu_1")),
    ("bottom", "bottleneck", ("480_in_prelu", "240_in_prelu")),
    ("R4", "up_blocks.0", ("240_in_prelu", "120_in_prelu")),
    ("R3", "up_blocks.1", ("120_in_prelu", "60_in_prelu")),
    ("R2", "up_blocks.2", ("60_in_prelu", "30_in_prelu")),
    ("R1_FC", "head", ("30_in_prelu", "30_in_prelu_1")),
]


def tf_checkpoint_from(model):
    """What a TF checkpoint holding ``model``'s weights would contain."""
    sd = model.state_dict()
    tensors = {}
    for scope, module, layers in BLOCKS:
        for i, layer in enumerate(layers):
            prefix = f"NoNewNet/{scope}/{layer}"
            # conv kernels: torch (out, in, kh, kw) <- TF (kh, kw, in, out)
            tensors[f"{prefix}/conv_/w"] = \
                sd[f"{module}.blocks.{4 * i}.weight"].permute(
                    2, 3, 1, 0).numpy()
            tensors[f"{prefix}/in_/InstanceNorm/gamma"] = \
                sd[f"{module}.blocks.{4 * i + 1}.weight"].numpy()
            tensors[f"{prefix}/in_/InstanceNorm/beta"] = \
                sd[f"{module}.blocks.{4 * i + 1}.bias"].numpy()
            tensors[f"{prefix}/prelu_acti_/alpha"] = \
                sd[f"{module}.blocks.{4 * i + 2}.weight"].numpy()
    tensors["NoNewNet/R1_FC/2/conv_/w"] = \
        sd["head.blocks.8.weight"].permute(2, 3, 1, 0).numpy()
    return tensors


def fake_tf(tensors, requested):
    def load_checkpoint(path):
        def get_tensor(name):
            requested.append(name)
            return tensors[name]
        return SimpleNamespace(get_tensor=get_tensor)
    return SimpleNamespace(
        train=SimpleNamespace(load_checkpoint=load_checkpoint)
    )


class ConvertLegacyCheckpointTest(TempDirTestCase):

    def convert(self, tensors, requested=None):
        """Runs the conversion against a fake TF; returns the output path."""
        dest = self.tmp / "converted.pt"
        with mock.patch.object(convert_legacy_checkpoint, "TF_AVAILABLE",
                               True), \
             mock.patch.object(convert_legacy_checkpoint, "tf",
                               fake_tf(tensors, [] if requested is None
                                       else requested), create=True):
            convert_legacy_checkpoint.convert_legacy_checkpoint("ckpt",
                                                                str(dest))
        return dest

    def test_requires_tensorflow(self):
        dest = self.tmp / "converted.pt"
        with mock.patch.object(convert_legacy_checkpoint, "TF_AVAILABLE",
                               False):
            with self.assertRaises(ImportError):
                convert_legacy_checkpoint.convert_legacy_checkpoint(
                    "ckpt", str(dest))
        self.assertFalse(dest.exists())

    def test_weights_are_mapped_and_transposed(self):
        reference = materialized_model(seed=1)
        tensors = tf_checkpoint_from(reference)
        requested = []
        dest = self.convert(tensors, requested)

        # every variable, once
        self.assertEqual(sorted(requested), sorted(tensors))
        # the output is what production loads (strict key/shape matching)
        model = ASAHSegmenter.from_checkpoint(dest, device="cpu").model
        ref_sd = reference.state_dict()
        self.assertEqual(model.state_dict().keys(), ref_sd.keys())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, ref_sd[key], msg=key)
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(model(x), reference.eval()(x))

    def test_missing_variable_propagates(self):
        tensors = tf_checkpoint_from(materialized_model())
        tensors.pop("NoNewNet/L1/30_in_prelu/conv_/w")
        with self.assertRaises(KeyError):
            self.convert(tensors)
        self.assertFalse((self.tmp / "converted.pt").exists())

    def test_command_line_help(self):
        proc = subprocess.run(
            [sys.executable, "scripts/convert_legacy_checkpoint.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("src", proc.stdout)
        self.assertIn("dest", proc.stdout)


if __name__ == "__main__":
    unittest.main()
