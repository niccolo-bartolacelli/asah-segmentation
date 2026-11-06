import unittest

import nibabel as nib
import numpy as np
import torch

from eval.metrics import dice_coefficient
from tests.helpers import TempDirTestCase, make_ct


def _mask(shape=(8, 8, 2), voxels=()) -> np.ndarray:
    data = np.zeros(shape, dtype=np.uint8)
    for v in voxels:
        data[v] = 1
    return data


class DiceCoefficientTest(TempDirTestCase):

    def test_identical_masks(self):
        m = _mask(voxels=[(0, 0, 0), (1, 1, 1)])
        self.assertEqual(dice_coefficient(m, m), 1.0)

    def test_disjoint_masks(self):
        a = _mask(voxels=[(0, 0, 0)])
        b = _mask(voxels=[(1, 1, 1)])
        self.assertEqual(dice_coefficient(a, b), 0.0)

    def test_partial_overlap(self):
        a = _mask(voxels=[(0, 0, 0), (0, 1, 0)])
        b = _mask(voxels=[(0, 0, 0), (2, 2, 1), (3, 3, 1)])
        # 2 * 1 / (2 + 3)
        self.assertAlmostEqual(dice_coefficient(a, b), 0.4)

    def test_is_symmetric(self):
        a = _mask(voxels=[(0, 0, 0), (0, 1, 0)])
        b = _mask(voxels=[(0, 0, 0), (2, 2, 1), (3, 3, 1)])
        self.assertEqual(dice_coefficient(a, b), dice_coefficient(b, a))

    def test_both_empty_gives_empty_score(self):
        m = _mask()
        self.assertEqual(dice_coefficient(m, m), 1.0)
        self.assertEqual(dice_coefficient(m, m, empty_score=0.0), 0.0)

    def test_one_empty_gives_zero(self):
        self.assertEqual(dice_coefficient(_mask(), _mask(voxels=[(0, 0, 0)])),
                         0.0)

    def test_label_selection(self):
        a = np.array([0, 1, 2, 2])
        b = np.array([0, 1, 1, 2])
        self.assertAlmostEqual(dice_coefficient(a, b, label=1), 2 / 3)
        self.assertAlmostEqual(dice_coefficient(a, b, label=2), 2 / 3)
        self.assertEqual(dice_coefficient(a, b, label=None), 1.0)

    def test_float_and_bool_inputs(self):
        a = np.array([0.0, 0.9999, 1.0, 0.0])
        b = np.array([False, True, True, False])
        self.assertEqual(dice_coefficient(a, b), 1.0)

    def test_tensor_input(self):
        m = _mask(voxels=[(0, 0, 0)])
        self.assertEqual(dice_coefficient(torch.from_numpy(m), m), 1.0)

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            dice_coefficient(_mask((8, 8, 2)), _mask((8, 8, 3)))

    def test_nifti_images_and_paths(self):
        ct = make_ct(shape=(8, 8, 2))
        data = _mask(voxels=[(0, 0, 0), (4, 4, 1)])
        img = nib.Nifti1Image(data, ct.affine)
        img2 = nib.Nifti2Image(data, ct.affine)
        path = self.tmp / "mask.nii.gz"
        nib.save(img, path)

        self.assertEqual(dice_coefficient(img, img2), 1.0)
        self.assertEqual(dice_coefficient(path, img), 1.0)
        self.assertEqual(dice_coefficient(str(path), data), 1.0)

    def test_affine_mismatch_raises(self):
        data = _mask(voxels=[(0, 0, 0)])
        a = nib.Nifti1Image(data, np.eye(4))
        b = nib.Nifti1Image(data, np.diag((2.0, 2.0, 2.0, 1.0)))
        with self.assertRaises(ValueError):
            dice_coefficient(a, b)


if __name__ == "__main__":
    unittest.main()
