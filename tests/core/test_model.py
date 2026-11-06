import unittest

import numpy as np
import torch

from asah_segmentation.model import (ASAHSegmentationNet, ConvBlock,
                                     LinearInterpolationLayer)
from tests.helpers import materialized_model


def reference_resize(x: np.ndarray, out_shape) -> np.ndarray:
    """
    Legacy TF resize_bilinear (align_corners=False, half_pixel_centers=False),
    written independently as separable np.interp over the spatial axes.
    """
    res = x.astype(np.float64)
    for axis, n_out in enumerate(out_shape, start=2):
        n_in = res.shape[axis]
        src = np.arange(n_out) * (n_in / n_out)
        res = np.apply_along_axis(lambda v: np.interp(src, np.arange(n_in), v),
                                  axis, res)
    return res


class ASAHSegmentationNetTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = materialized_model()

    def test_output_shape(self):
        with torch.no_grad():
            out = self.model(torch.randn(3, 1, 32, 48))
        self.assertEqual(tuple(out.shape), (3, 2, 32, 48))

    def test_default_architecture(self):
        self.assertEqual(self.model.num_classes, 2)
        self.assertEqual(self.model.n_features, (30, 60, 120, 240, 480))
        self.assertEqual(len(self.model.down_blocks), 4)
        self.assertEqual(len(self.model.up_blocks), 3)

    def test_rejects_non_4d_input(self):
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 32, 32))

    def test_rejects_spatial_size_not_multiple_of_pooling(self):
        # 4 pooling levels -> each spatial dim must be a multiple of 16
        with self.assertRaises(ValueError):
            self.model(torch.zeros(1, 1, 40, 32))

    def test_requires_three_levels(self):
        with self.assertRaises(ValueError):
            ASAHSegmentationNet(n_features=(8, 16))

    def test_custom_architecture(self):
        model = ASAHSegmentationNet(num_classes=3,
                                    n_features=(4, 8, 16)).eval()
        with torch.no_grad():
            out = model(torch.randn(2, 1, 8, 12))
        self.assertEqual(tuple(out.shape), (2, 3, 8, 12))

    def test_dropout_inactive_in_eval(self):
        torch.manual_seed(0)
        model = ASAHSegmentationNet(n_features=(4, 8, 16),
                                    dropout_rate=0.5).eval()
        x = torch.randn(1, 1, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), model(x)))

    def test_deterministic_and_batch_independent(self):
        x = torch.randn(4, 1, 32, 32)
        with torch.no_grad():
            full = self.model(x)
            parts = torch.cat([self.model(x[:1]), self.model(x[1:])])
        torch.testing.assert_close(full, parts, rtol=1e-4, atol=1e-5)

    def test_state_dict_loads_into_fresh_lazy_model(self):
        fresh = ASAHSegmentationNet()
        fresh.load_state_dict(self.model.state_dict(), strict=True)
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(fresh.eval()(x), self.model(x))


class ConvBlockTest(unittest.TestCase):

    def test_downsample_and_partial(self):
        block = ConvBlock((6, 6), (3, 3), action="downsample")
        out, partial = block(torch.randn(1, 2, 8, 8), with_partial=True)
        self.assertEqual(tuple(out.shape), (1, 6, 4, 4))
        self.assertEqual(tuple(partial.shape), (1, 6, 8, 8))

    def test_partial_is_none_by_default(self):
        block = ConvBlock((4,), (3,))
        out, partial = block(torch.randn(1, 1, 8, 8))
        self.assertIsNone(partial)
        self.assertEqual(tuple(out.shape), (1, 4, 8, 8))

    def test_upsample(self):
        block = ConvBlock((4,), (3,), action="upsample")
        out, _ = block(torch.randn(1, 1, 4, 6))
        self.assertEqual(tuple(out.shape), (1, 4, 8, 12))

    def test_skip_is_concatenated_on_channels(self):
        block = ConvBlock((4,), (3,))
        block(torch.randn(1, 3, 8, 8), skip=torch.randn(1, 5, 8, 8))
        self.assertEqual(block.blocks[0].weight.shape[1], 8)

    def test_pointwise_layer_has_no_norm_or_activation(self):
        block = ConvBlock((4, 2), (3, 1))
        # 3x3 conv + norm + prelu + dropout, then a bare 1x1 conv
        self.assertEqual(len(block.blocks), 5)
        self.assertIsInstance(block.blocks[4], torch.nn.LazyConv2d)


class LinearInterpolationLayerTest(unittest.TestCase):

    def test_matches_legacy_tensorflow_resize(self):
        x = torch.arange(2 * 3 * 4,
                         dtype=torch.float32).reshape(1, 2, 3, 4) ** 1.5
        out = LinearInterpolationLayer(scale=2.0)(x)
        np.testing.assert_allclose(out.numpy(),
                                   reference_resize(x.numpy(), (6, 8)),
                                   rtol=1e-5, atol=1e-4)

    def test_new_shape(self):
        x = torch.randn(1, 1, 4, 4)
        out = LinearInterpolationLayer(new_shape=(6, 10))(x)
        self.assertEqual(tuple(out.shape), (1, 1, 6, 10))
        np.testing.assert_allclose(out.numpy(),
                                   reference_resize(x.numpy(), (6, 10)),
                                   rtol=1e-5, atol=1e-5)

    def test_preserves_constants_and_corner(self):
        x = torch.full((1, 1, 3, 3), 7.0)
        torch.testing.assert_close(LinearInterpolationLayer(2.0)(x),
                                   torch.full((1, 1, 6, 6), 7.0))
        y = torch.randn(1, 1, 3, 3)
        self.assertEqual(LinearInterpolationLayer(2.0)(y)[0, 0, 0, 0],
                         y[0, 0, 0, 0])

    def test_casts_to_float32(self):
        out = LinearInterpolationLayer(2.0)(torch.ones(1, 1, 2, 2,
                                                       dtype=torch.float64))
        self.assertEqual(out.dtype, torch.float32)

    def test_argument_validation(self):
        with self.assertRaises(ValueError):
            LinearInterpolationLayer()
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(scale=2.0, new_shape=(2, 2))
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(scale=0)
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(new_shape=())
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(new_shape=(2, 0))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(2.0)(torch.ones(4, 4))
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(new_shape=(4,))(torch.ones(1, 1, 2, 2))
        with self.assertRaises(ValueError):
            LinearInterpolationLayer(scale=0.1)(torch.ones(1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
