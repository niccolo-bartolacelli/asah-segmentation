import unittest

import nibabel as nib
import numpy as np
import torch

from eval.metrics import absolute_volume_difference, dice_coefficient
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


class AbsoluteVolumeDifferenceTest(unittest.TestCase):

    def test_identical_masks(self):
        m = _mask(voxels=[(0, 0, 0), (1, 1, 1)])
        self.assertEqual(absolute_volume_difference(m, m, voxel_volume=1.0),
                         0.0)

    def test_counts_voxels_not_overlap(self):
        a = _mask(voxels=[(0, 0, 0), (0, 1, 0)])
        b = _mask(voxels=[(5, 5, 1), (6, 6, 1), (7, 7, 1)])
        # | 2 - 3 | voxels of 1000 mm³ = 1 mL
        self.assertAlmostEqual(
            absolute_volume_difference(a, b, voxel_volume=1000.0), 1.0)

    def test_is_symmetric(self):
        a = _mask(voxels=[(0, 0, 0)])
        b = _mask(voxels=[(0, 0, 0), (1, 1, 1), (2, 2, 1)])
        self.assertEqual(absolute_volume_difference(a, b, voxel_volume=2.0),
                         absolute_volume_difference(b, a, voxel_volume=2.0))

    def test_voxel_volume_from_affine(self):
        a = _mask(voxels=[(0, 0, 0)])
        b = _mask(voxels=[(0, 0, 0), (1, 1, 1), (2, 2, 1)])
        affine = np.diag((0.5, -0.5, 5.0, 1.0))  # 1.25 mm³, flipped axis
        expected = 2 * 1.25 / 1000
        img_a, img_b = nib.Nifti1Image(a, affine), nib.Nifti1Image(b, affine)
        self.assertAlmostEqual(absolute_volume_difference(img_a, img_b),
                               expected)
        self.assertAlmostEqual(absolute_volume_difference(img_a, b), expected)
        self.assertAlmostEqual(absolute_volume_difference(a, img_b), expected)
        # An explicit voxel volume wins over the affine.
        self.assertAlmostEqual(
            absolute_volume_difference(img_a, img_b, voxel_volume=1000.0), 2.0)

    def test_label_selection(self):
        a = np.array([0, 1, 2, 2])
        b = np.array([0, 1, 1, 2])
        self.assertEqual(absolute_volume_difference(a, b, label=1,
                                                    voxel_volume=1000.0), 1.0)
        self.assertEqual(absolute_volume_difference(a, b, label=None,
                                                    voxel_volume=1000.0), 0.0)

    def test_arrays_need_voxel_volume(self):
        m = _mask(voxels=[(0, 0, 0)])
        with self.assertRaises(ValueError):
            absolute_volume_difference(m, m)

    def test_grid_mismatch_raises(self):
        with self.assertRaises(ValueError):
            absolute_volume_difference(_mask((8, 8, 2)), _mask((8, 8, 3)),
                                       voxel_volume=1.0)
        data = _mask(voxels=[(0, 0, 0)])
        with self.assertRaises(ValueError):
            absolute_volume_difference(
                nib.Nifti1Image(data, np.eye(4)),
                nib.Nifti1Image(data, np.diag((2.0, 2.0, 2.0, 1.0))))


if __name__ == "__main__":
    unittest.main()
