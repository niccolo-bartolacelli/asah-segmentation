import dataclasses
import unittest

import nibabel as nib
import numpy as np
import torch

from asah_segmentation.processing import (ImageContext,
                                          PreprocessingConfiguration,
                                          Processor)
from tests.helpers import make_ct


class PreprocessingConfigurationTest(unittest.TestCase):

    def test_defaults_match_published_model(self):
        cfg = PreprocessingConfiguration()
        self.assertEqual(cfg.downsample, (2, 2))
        self.assertEqual(cfg.shape, (256, 256))
        self.assertEqual(cfg.clip, (0, 150))
        self.assertEqual(np.dtype(cfg.dtype), np.int16)

    def test_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            PreprocessingConfiguration().shape = (128, 128)


class ImageContextTest(unittest.TestCase):

    def setUp(self):
        self.img = make_ct()

    def test_from_image(self):
        ctx = ImageContext.from_image(self.img)
        self.assertEqual(ctx.original_shape, self.img.shape)
        np.testing.assert_allclose(ctx.original_affine, self.img.affine)
        self.assertIsNone(ctx.processed_shape)
        self.assertIsNone(ctx.processed_affine)

    def test_from_image_copies(self):
        ctx = ImageContext.from_image(self.img)
        ctx.original_affine[0, 0] = 99
        self.assertNotEqual(self.img.affine[0, 0], 99)

    def test_is_valid_modes(self):
        ctx = ImageContext.from_image(self.img)
        ok, msg = ImageContext.is_valid(ctx)
        self.assertFalse(ok)
        self.assertIn("processed_shape", msg)
        self.assertTrue(ImageContext.is_valid(
            ctx,
            fields=["processed_shape", "processed_affine"],
            mode="exclude"
        )[0])
        self.assertTrue(ImageContext.is_valid(
            ctx, fields=["original_shape"], mode="include")[0])
        self.assertFalse(ImageContext.is_valid(
            ctx, fields=["processed_affine"], mode="include")[0])

        ctx.processed_shape = (256, 256, 3)
        ctx.processed_affine = np.eye(4)
        self.assertEqual(ImageContext.is_valid(ctx), (True, ""))

    def test_is_valid_errors(self):
        with self.assertRaises(ValueError):
            ImageContext.is_valid(None)
        ctx = ImageContext.from_image(self.img)
        for mode in ("include", "exclude"):
            with self.assertRaises(ValueError):
                ImageContext.is_valid(ctx, mode=mode)


class PreprocessVolumeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.img = make_ct()
        cls.out = Processor().preprocess_volume(cls.img)

    def test_shape_keeps_slice_count(self):
        self.assertEqual(self.out.shape, (256, 256, self.img.shape[2]))

    def test_affine_scaled_in_plane_only(self):
        expected = self.img.affine.copy()
        expected[:3, :3] = self.img.affine[:3, :3] @ np.diag((2, 2, 1))
        np.testing.assert_allclose(self.out.affine, expected)

    def test_clipped_and_cast(self):
        data = np.asarray(self.out.dataobj)
        self.assertEqual(data.dtype, np.int16)
        self.assertGreaterEqual(data.min(), 0)
        self.assertLessEqual(data.max(), 150)

    def test_custom_configuration(self):
        cfg = PreprocessingConfiguration(downsample=(1, 1), shape=(64, 32),
                                         clip=(-10, 10), dtype=np.float32)
        out = Processor(cfg).preprocess_volume(self.img)
        self.assertEqual(out.shape, (64, 32, 3))
        data = np.asarray(out.dataobj)
        self.assertEqual(data.dtype, np.float32)
        self.assertTrue(((data >= -10) & (data <= 10)).all())

    def test_outside_field_of_view_is_filled_with_minimum(self):
        # The 48 x 40 mm scan sits inside a 256 x 256 mm grid: the padding
        # takes the volume minimum, which the clip then raises to 0.
        data = np.asarray(self.out.dataobj)
        self.assertEqual(data[-1, -1, 0], 0)


class VolumeToSlicesTest(unittest.TestCase):

    def test_layout(self):
        data = np.arange(4 * 6 * 3, dtype=np.int16).reshape(4, 6, 3)
        slices = Processor().volume_to_slices(nib.Nifti1Image(data, np.eye(4)))
        self.assertEqual(tuple(slices.shape), (3, 1, 4, 6))
        self.assertEqual(slices.dtype, torch.float32)
        for k in range(3):
            np.testing.assert_array_equal(slices[k, 0].numpy(), data[:, :, k])

    def test_rejects_non_3d(self):
        with self.assertRaises(ValueError):
            Processor().volume_to_slices(
                nib.Nifti1Image(np.zeros((4, 4, 2, 2), np.int16), np.eye(4)))


class SlicesContextTest(unittest.TestCase):

    def test_yields_preprocessed_slices(self):
        img = make_ct()
        with Processor().slices_from(img) as slices:
            self.assertEqual(tuple(slices.shape), (3, 1, 256, 256))

    def test_not_reentrant(self):
        processor = Processor()
        with processor.slices_from(make_ct()):
            with self.assertRaises(ValueError):
                with processor.slices_from(make_ct()):
                    pass

    def test_context_is_reset_even_on_error(self):
        processor = Processor()
        with self.assertRaises(RuntimeError):
            with processor.slices_from(make_ct()):
                raise RuntimeError("boom")
        self.assertIsNone(processor._context)
        with processor.slices_from(make_ct()):
            pass

    def test_postprocessing_requires_context(self):
        with self.assertRaises(ValueError):
            Processor().predictions_to_nifti(torch.zeros(3, 2, 256, 256))


class PredictionsToNiftiTest(unittest.TestCase):

    def setUp(self):
        self.img = make_ct()
        self.processor = Processor()

    def predict(self, fill_class: int):
        with self.processor.slices_from(self.img) as slices:
            logits = torch.zeros(slices.shape[0], 2, *slices.shape[2:])
            logits[:, fill_class] = 1.0
            return self.processor.predictions_to_nifti(logits)

    def test_geometry_matches_source(self):
        mask = self.predict(1)
        self.assertEqual(mask.shape, self.img.shape)
        np.testing.assert_allclose(mask.affine, self.img.affine)
        self.assertEqual(int(mask.header["qform_code"]), 1)
        self.assertEqual(int(mask.header["sform_code"]), 1)
        np.testing.assert_allclose(mask.header.get_zooms(),
                                   self.img.header.get_zooms())

    def test_label_map_header(self):
        mask = self.predict(1)
        self.assertEqual(mask.get_data_dtype(), np.uint8)
        self.assertEqual(int(mask.header["intent_code"]), 1002)
        slope, inter = mask.header.get_slope_inter()
        self.assertIn(slope, (1.0, None))
        self.assertIn(inter, (0.0, None))

    def test_argmax_labels(self):
        np.testing.assert_array_equal(np.unique(np.asarray(
            self.predict(1).dataobj
        )), [1])
        np.testing.assert_array_equal(np.unique(np.asarray(
            self.predict(0).dataobj
        )), [0])
        self.assertEqual(int(self.predict(0).header["cal_max"]), 0)

    def test_output_version(self):
        v2 = nib.Nifti2Image(np.asarray(self.img.dataobj), self.img.affine,
                             self.img.header)
        cases = ((self.img, None, nib.Nifti1Image),
                 (v2, None, nib.Nifti2Image),
                 (self.img, 2, nib.Nifti2Image),
                 (v2, 1, nib.Nifti1Image))
        for src, version, expected in cases:
            with self.subTest(src=type(src).__name__, version=version):
                with self.processor.slices_from(src) as slices:
                    logits = torch.zeros(slices.shape[0], 2, *slices.shape[2:])
                    logits[:, 1] = 1.0
                    out = self.processor.predictions_to_nifti(logits, version)
                self.assertIs(type(out), expected)
                self.assertEqual(out.shape, self.img.shape)
                np.testing.assert_allclose(out.affine, self.img.affine)
                self.assertEqual(int(out.header["intent_code"]), 1002)

    def test_preprocessing_keeps_version(self):
        v2 = nib.Nifti2Image(np.asarray(self.img.dataobj), self.img.affine)
        self.assertIs(type(self.processor.preprocess_volume(v2)),
                      nib.Nifti2Image)
        self.assertIs(type(self.processor.preprocess_volume(self.img)),
                      nib.Nifti1Image)

    def test_source_header_is_not_mutated(self):
        before = self.img.header.get_data_dtype()
        self.predict(1)
        self.assertEqual(self.img.header.get_data_dtype(), before)


if __name__ == "__main__":
    unittest.main()
