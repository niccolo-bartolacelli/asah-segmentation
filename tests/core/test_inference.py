import unittest

import nibabel as nib
import numpy as np
import torch

from asah_segmentation.inference import predict_volume
from asah_segmentation.processing import PreprocessingConfiguration
from tests.helpers import ThresholdModel, make_ct


def half_bright_volume(z=5):
    """
    Left half 100 HU, right half 0 HU: the threshold model labels the left.
    """
    base = make_ct(shape=(96, 80, z))
    data = np.zeros(base.shape, np.int16)
    data[:48] = 100
    return nib.Nifti1Image(data, base.affine, base.header)


class PredictVolumeTest(unittest.TestCase):

    def setUp(self):
        self.volume = half_bright_volume()
        self.model = ThresholdModel(threshold=50.0)

    def test_mask_follows_model_output(self):
        mask = np.asarray(predict_volume(self.model, self.volume,
                                         torch.device("cpu")).dataobj)
        self.assertEqual(mask.shape, self.volume.shape)
        # Away from the step, where in-plane resampling blurs it, and from the
        # last in-plane row/column: after the 2x downsampling the outermost
        # voxels fall half a voxel outside the processed grid's support.
        self.assertTrue((mask[:40, :-1] == 1).all())
        self.assertTrue((mask[56:] == 0).all())

    def test_batching_does_not_change_the_result(self):
        cpu = torch.device("cpu")
        one = predict_volume(ThresholdModel(), self.volume, cpu, batch_size=1)
        model = ThresholdModel()
        two = predict_volume(model, self.volume, cpu, batch_size=2)
        self.assertEqual(model.batch_sizes, [2, 2, 1])
        np.testing.assert_array_equal(np.asarray(one.dataobj),
                                      np.asarray(two.dataobj))

    def test_model_put_in_eval_mode(self):
        self.model.train()
        predict_volume(self.model, self.volume, torch.device("cpu"))
        self.assertFalse(self.model.training)

    def test_no_autograd_graph(self):
        seen = []

        class Spy(ThresholdModel):
            def forward(self, x):
                seen.append(torch.is_grad_enabled())
                return super().forward(x)

        predict_volume(Spy(), self.volume, torch.device("cpu"))
        self.assertTrue(seen)
        self.assertFalse(any(seen))

    def test_batches_are_sent_to_the_device(self):
        devices = []

        class Spy(ThresholdModel):
            def forward(self, x):
                devices.append(x.device)
                return super().forward(x)

        predict_volume(Spy(), self.volume, torch.device("cpu"))
        self.assertTrue(all(d.type == "cpu" for d in devices))

    def test_custom_preprocessing_is_used(self):
        seen_shapes = []

        class Spy(ThresholdModel):
            def forward(self, x):
                seen_shapes.append(tuple(x.shape[2:]))
                return super().forward(x)

        cfg = PreprocessingConfiguration(shape=(64, 64))
        mask = predict_volume(Spy(),
                              self.volume,
                              torch.device("cpu"),
                              config=cfg)
        self.assertEqual(set(seen_shapes), {(64, 64)})
        self.assertEqual(mask.shape, self.volume.shape)

    def test_works_with_the_real_network(self):
        from tests.helpers import materialized_model
        mask = predict_volume(materialized_model(), make_ct(shape=(96, 80, 2)),
                              torch.device("cpu"), batch_size=2)
        self.assertEqual(mask.shape, (96, 80, 2))
        self.assertEqual(mask.get_data_dtype(), np.uint8)


if __name__ == "__main__":
    unittest.main()
