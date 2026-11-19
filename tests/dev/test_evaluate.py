"""Tests for the evaluation CLI, ``eval/evaluate.py``.

Synthetic masks only: the "port" masks go in SOURCE, the "legacy" ones in
TARGET, on a grid of 0.5 x 0.5 x 5 mm voxels (1.25 mm³, 0.00125 mL).
"""

import contextlib
import csv
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from eval import evaluate
from eval.metrics import dice_coefficient
from tests.helpers import TempDirTestCase, make_ct

REPO_ROOT = Path(__file__).resolve().parents[2]
VOXEL_ML = 0.5 * 0.5 * 5.0 / 1000


class EvaluateTest(TempDirTestCase):

    @classmethod
    def setUpClass(cls):
        cls.affine = make_ct(shape=(16, 16, 2)).affine
        cls.full = np.zeros((16, 16, 2), np.uint8)
        cls.full[:8] = 1                      # 256 voxels
        cls.half = cls.full.copy()
        cls.half[:4] = 0                      # 128 voxels

    def setUp(self):
        super().setUp()
        self.source, self.target = self.tmp / "port", self.tmp / "legacy"
        self.source.mkdir()
        self.target.mkdir()

    def save(self, folder, name, data, affine=None):
        nib.save(nib.Nifti1Image(data, self.affine if affine is None
                                 else affine), folder / name)

    def add_case(self, name, port=None, legacy=None):
        if port is not None:
            self.save(self.source, name, port)
        if legacy is not None:
            self.save(self.target, name, legacy)

    def run_main(self, *argv):
        """Runs ``main``; returns (status, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        with (contextlib.redirect_stdout(out),
              contextlib.redirect_stderr(err)):
            status = evaluate.main([str(self.source), str(self.target),
                                    *map(str, argv)])
        return status, out.getvalue(), err.getvalue()

    def test_per_case_results_and_summary(self):
        self.add_case("a.nii.gz", self.full, self.full)
        self.add_case("b.nii", self.half, self.full)
        (self.source / "notes.txt").write_text("not a mask")
        dice_b = dice_coefficient(self.half, self.full)
        avd_b = 128 * VOXEL_ML

        status, out, err = self.run_main()

        self.assertEqual(status, 0, err)
        lines = out.splitlines()
        self.assertEqual(lines[1].split(), ["a.nii.gz", "1.0000", "0.0000"])
        self.assertEqual(lines[2].split(),
                         ["b.nii", f"{dice_b:.4f}", f"{avd_b:.4f}"])
        self.assertIn("cases 2  skipped 0", out)
        dice_row = next(l for l in lines if l.startswith("Dice "))
        mean, std, median, q1, q3, lo, hi = dice_row.split()[1:]
        self.assertEqual(mean, f"{(1 + dice_b) / 2:.4f}")
        self.assertEqual((lo, hi), (f"{dice_b:.4f}", "1.0000"))
        self.assertIn(f"lowest Dice:  b.nii ({dice_b:.4f})", out)
        self.assertIn(f"largest AVD:  b.nii ({avd_b:.4f} mL)", out)

    def test_json_output(self):
        self.add_case("a.nii.gz", self.full, self.full)
        self.add_case("b.nii.gz", self.half, self.full)
        self.add_case("orphan.nii.gz", port=self.full)
        path = self.tmp / "out" / "results.json"

        status, _, err = self.run_main("--output", path)

        self.assertEqual(status, 0, err)
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual((report["cases"], report["skipped"]), (2, 1))
        self.assertAlmostEqual(report["results"]["b.nii.gz"]["avd_ml"],
                               128 * VOXEL_ML)
        self.assertAlmostEqual(report["summary"]["avd_ml"]["mean"],
                               64 * VOXEL_ML)
        self.assertAlmostEqual(report["summary"]["dice"]["max"], 1.0)
        self.assertEqual(report["worst"], {"dice": "b.nii.gz",
                                           "avd_ml": "b.nii.gz"})
        self.assertIn("orphan.nii.gz", report["skipped_cases"])

    def test_csv_output(self):
        self.add_case("a.nii.gz", self.full, self.half)
        self.add_case("orphan.nii.gz", legacy=self.full)
        path = self.tmp / "results.csv"

        self.run_main("-o", path)

        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([r["case"] for r in rows],
                         ["a.nii.gz", "orphan.nii.gz"])
        self.assertAlmostEqual(float(rows[0]["avd_ml"]), 128 * VOXEL_ML)
        self.assertEqual(rows[0]["skipped"], "")
        self.assertIn("no source mask", rows[1]["skipped"])

    def test_single_case_has_no_std(self):
        self.add_case("a.nii.gz", self.full, self.full)
        _, out, _ = self.run_main()
        dice_row = next(l for l in out.splitlines() if l.startswith("Dice "))
        self.assertEqual(dice_row.split()[2], "n/a")

    def test_label_option(self):
        self.add_case("a.nii.gz", self.full, self.full * 2)
        _, out, _ = self.run_main("--label", 2)
        # The port mask holds 1s only: no overlap, all 256 legacy voxels.
        self.assertEqual(out.splitlines()[1].split(),
                         ["a.nii.gz", "0.0000", f"{256 * VOXEL_ML:.4f}"])

    def test_bad_cases_are_skipped(self):
        self.add_case("good.nii.gz", self.full, self.full)
        self.add_case("no_legacy.nii.gz", port=self.full)
        self.add_case("no_port.nii.gz", legacy=self.full)
        self.add_case("shifted.nii.gz", port=self.full)
        self.save(self.target, "shifted.nii.gz", self.full, np.eye(4) * 2)
        (self.source / "broken.nii.gz").write_bytes(b"not a NIfTI file")
        self.save(self.target, "broken.nii.gz", self.full)

        status, out, err = self.run_main()

        self.assertEqual(status, 0)
        self.assertIn("good.nii.gz", out)
        self.assertIn("cases 1  skipped 4", out)
        for name in ("no_legacy", "no_port", "shifted", "broken"):
            self.assertIn(f"skipping {name}.nii.gz", err)
        self.assertIn("no target mask", err)

    def test_nothing_scored(self):
        self.add_case("a.nii.gz", port=self.full)
        status, out, _ = self.run_main()
        self.assertEqual(status, 1)
        self.assertIn("cases 0  skipped 1", out)
        self.assertNotIn("Dice ", out.split("cases")[1])

    def test_bad_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                evaluate.main([str(self.source), str(self.tmp / "missing")])
            with self.assertRaises(SystemExit):
                evaluate.main([str(self.source), str(self.target),
                               "--output", str(self.tmp / "results.txt")])

    def test_command_line(self):
        self.add_case("a.nii.gz", self.full, self.full)
        proc = subprocess.run(
            [sys.executable, "eval/evaluate.py", str(self.source),
             str(self.target)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("cases 1  skipped 0", proc.stdout)


if __name__ == "__main__":
    unittest.main()
