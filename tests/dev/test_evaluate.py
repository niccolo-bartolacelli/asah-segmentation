"""Tests for the evaluation CLI, ``eval/evaluate.py``.

Synthetic scans only: the model is the deterministic ``ThresholdModel``
stand-in (so Dice values can be asserted) or random weights saved to a
temporary checkpoint.
"""

import contextlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from asah_segmentation import ASAHSegmenter
from eval import evaluate
from eval.metrics import dice_coefficient
from tests.helpers import (TempDirTestCase, ThresholdModel, make_ct,
                           materialized_model)

REPO_ROOT = Path(__file__).resolve().parents[2]


class EvaluateTest(TempDirTestCase):

    @classmethod
    def setUpClass(cls):
        cls.segmenter = ASAHSegmenter(ThresholdModel(50.0), device="cpu")
        base = make_ct(shape=(96, 80, 2))
        data = np.zeros(base.shape, np.int16)
        data[:48] = 100  # the ThresholdModel segments the upper half
        cls.scan = nib.Nifti1Image(data, base.affine, base.header)
        cls.mask = cls.segmenter.segment(cls.scan)

    def setUp(self):
        super().setUp()
        self.source, self.target = self.tmp / "images", self.tmp / "labels"
        self.source.mkdir()
        self.target.mkdir()

    def add_case(self, name, label=None, affine=None):
        nib.save(self.scan, self.source / name)
        if label is not None:
            nib.save(nib.Nifti1Image(label.astype(np.uint8),
                                     self.mask.affine if affine is None
                                     else affine), self.target / name)

    def run_main(self, *argv):
        """Runs ``main`` with the stand-in model; returns (status, out, err)."""
        out, err = io.StringIO(), io.StringIO()
        with (mock.patch.object(evaluate.ASAHSegmenter, "from_checkpoint",
                                return_value=self.segmenter),
              contextlib.redirect_stdout(out),
              contextlib.redirect_stderr(err)):
            status = evaluate.main([str(self.source), str(self.target),
                                    *map(str, argv)])
        return status, out.getvalue(), err.getvalue()

    def test_scores_and_json(self):
        perfect = np.asarray(self.mask.dataobj)
        half = perfect.copy()
        half[:24] = 0
        expected = dice_coefficient(perfect, half)
        self.add_case("a.nii.gz", perfect)
        self.add_case("b.nii", half)
        (self.source / "notes.txt").write_text("not a scan")
        report_path = self.tmp / "dice.json"

        status, out, err = self.run_main("--json", report_path)

        self.assertEqual(status, 0, err)
        self.assertIn("a.nii.gz  1.0000", out)
        self.assertIn(f"b.nii  {expected:.4f}", out)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["cases"], 2)
        self.assertAlmostEqual(report["dice"]["b.nii"], expected)
        self.assertAlmostEqual(report["mean"], (1 + expected) / 2)
        self.assertEqual(report["skipped_cases"], {})

    def test_label_option(self):
        self.add_case("a.nii.gz", np.asarray(self.mask.dataobj) * 2)
        _, out, _ = self.run_main("--label", 2)
        self.assertIn("a.nii.gz  0.0000", out)  # the prediction holds 1s only

    def test_bad_cases_are_skipped(self):
        good = np.asarray(self.mask.dataobj)
        self.add_case("good.nii.gz", good)
        self.add_case("missing.nii.gz")
        self.add_case("shifted.nii.gz", good, affine=np.eye(4) * 2)
        (self.source / "broken.nii.gz").write_bytes(b"not a NIfTI file")
        nib.save(self.mask, self.target / "broken.nii.gz")

        status, out, err = self.run_main()

        self.assertEqual(status, 0)
        self.assertIn("good.nii.gz  1.0000", out)
        self.assertIn("cases 1  skipped 3", out)
        for name in ("missing", "shifted", "broken"):
            self.assertIn(f"skipping {name}.nii.gz", err)

    def test_nothing_scored(self):
        self.add_case("a.nii.gz")
        status, out, _ = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("cases 0  skipped 1", out)

    def test_command_line(self):
        self.add_case("a.nii.gz", np.asarray(self.mask.dataobj))
        checkpoint = self.tmp / "weights.pt"
        torch.save(materialized_model().state_dict(), checkpoint)
        proc = subprocess.run(
            [sys.executable, "eval/evaluate.py", str(self.source),
             str(self.target), "--checkpoint", str(checkpoint),
             "--device", "cpu", "--batch-size", "2"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("cases 1  skipped 0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
