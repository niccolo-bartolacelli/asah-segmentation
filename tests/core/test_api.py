"""Contract tests for the public API, ``asah_segmentation.ASAHSegmenter``."""

import hashlib
import io
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import torch

from asah_segmentation import (SAH_LABEL, ASAHSegmenter, CheckpointError,
                               PreprocessingConfiguration, VolumeError, api)
from tests.helpers import (TempDirTestCase, ThresholdModel, make_ct,
                           materialized_model)


class CheckpointTestCase(TempDirTestCase):
    """Writes a random-weights checkpoint to ``self.checkpoint``."""

    @classmethod
    def setUpClass(cls):
        cls.state_dict = materialized_model().state_dict()

    def setUp(self):
        super().setUp()
        self.checkpoint = self.tmp / "weights.pt"
        torch.save(self.state_dict, self.checkpoint)


class FromCheckpointTest(CheckpointTestCase):

    def test_loads_plain_state_dict(self):
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu")
        for key, value in seg.model.state_dict().items():
            self.assertTrue(torch.equal(value, self.state_dict[key]), key)
        self.assertFalse(seg.model.training)
        self.assertEqual(seg.checkpoint, self.checkpoint)

    def test_accepts_str_path_and_wrapper(self):
        wrapped = self.tmp / "wrapped.pt"
        torch.save({"state_dict": self.state_dict, "epoch": 3}, wrapped)
        ASAHSegmenter.from_checkpoint(str(wrapped), device="cpu")

    def test_options_are_forwarded(self):
        cfg = PreprocessingConfiguration(shape=(128, 128))
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu",
                                            batch_size=3, preprocessing=cfg,
                                            nifti_version=2)
        self.assertEqual(seg.batch_size, 3)
        self.assertIs(seg.preprocessing, cfg)
        self.assertEqual(seg.nifti_version, 2)

    def test_missing_file(self):
        with self.assertRaisesRegex(CheckpointError, "not found"):
            ASAHSegmenter.from_checkpoint(self.tmp / "missing.pt")

    def test_unreadable_file(self):
        junk = self.tmp / "junk.pt"
        junk.write_bytes(b"definitely not a torch file")
        with self.assertRaisesRegex(CheckpointError, "Could not read"):
            ASAHSegmenter.from_checkpoint(junk)

    def test_not_a_state_dict(self):
        for payload in ({},
                        {"a": 1},
                        [torch.zeros(1)], {"state_dict": {"a": "b"}}):
            path = self.tmp / "bad.pt"
            torch.save(payload, path)
            with self.subTest(payload=type(payload).__name__), \
                    self.assertRaisesRegex(CheckpointError, "state_dict"):
                ASAHSegmenter.from_checkpoint(path)

    def test_mismatched_state_dict(self):
        partial = dict(self.state_dict)
        partial.pop(next(iter(partial)))
        path = self.tmp / "partial.pt"
        torch.save(partial, path)
        with self.assertRaisesRegex(CheckpointError, "does not match"):
            ASAHSegmenter.from_checkpoint(path)

    def test_refuses_arbitrary_pickles(self):
        path = self.tmp / "pickle.pt"
        torch.save({"weights": object()}, path)
        with self.assertRaises(CheckpointError):
            ASAHSegmenter.from_checkpoint(path)


class ConstructorTest(unittest.TestCase):

    def test_defaults(self):
        seg = ASAHSegmenter(ThresholdModel(), device="cpu")
        self.assertEqual(seg.batch_size, 16)
        self.assertEqual(seg.preprocessing, PreprocessingConfiguration())
        self.assertEqual(seg.nifti_version, 1)
        self.assertIsNone(seg.checkpoint)
        self.assertIsNone(seg.checkpoint_sha256)

    def test_invalid_nifti_version(self):
        with self.assertRaises(ValueError):
            ASAHSegmenter(ThresholdModel(), device="cpu", nifti_version=3)

    def test_model_moved_to_device_and_eval(self):
        model = ThresholdModel().train()
        seg = ASAHSegmenter(model, device=torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertEqual(seg.device, torch.device("cpu"))
        self.assertEqual(model.anchor.device.type, "cpu")

    def test_invalid_batch_size(self):
        for bad in (0, -1):
            with self.subTest(batch_size=bad), self.assertRaises(ValueError):
                ASAHSegmenter(ThresholdModel(), batch_size=bad)


class ResolveDeviceTest(unittest.TestCase):

    def test_explicit(self):
        self.assertEqual(ASAHSegmenter.resolve_device("cpu"),
                         torch.device("cpu"))
        dev = torch.device("cpu")
        self.assertIs(ASAHSegmenter.resolve_device(dev), dev)

    def test_cuda_requested_but_missing(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                ASAHSegmenter.resolve_device("cuda")

    def test_auto_prefers_cuda_then_cpu(self):
        with mock.patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(ASAHSegmenter.resolve_device(),
                             torch.device("cuda"))
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(ASAHSegmenter.resolve_device("auto"),
                             torch.device("cpu"))


class ReadVolumeTest(TempDirTestCase):

    def setUp(self):
        super().setUp()
        self.src = make_ct()
        self.path = self.tmp / "ct.nii.gz"
        nib.save(self.src, self.path)

    def test_sources(self):
        for source in (self.path, str(self.path),
                       io.BytesIO(self.path.read_bytes()),
                       self.src):
            with self.subTest(source=type(source).__name__):
                img = ASAHSegmenter.read_volume(source)
                self.assertEqual(img.shape, self.src.shape)
                np.testing.assert_allclose(img.affine, self.src.affine)

    def test_loaded_image_passes_through(self):
        self.assertIs(ASAHSegmenter.read_volume(self.src), self.src)

    def test_both_versions_converted_either_way(self):
        v2 = nib.Nifti2Image(np.asarray(self.src.dataobj), self.src.affine)
        path2 = self.tmp / "ct2.nii.gz"
        nib.save(v2, path2)
        sources = {
            "nifti1": lambda: (self.path,
                               io.BytesIO(self.src.to_bytes()),
                               self.src),
            "nifti2": lambda: (path2,
                               io.BytesIO(v2.to_bytes()),
                               v2),
        }
        for stored, make_sources in sources.items():
            for version, expected in ((1, nib.Nifti1Image),
                                      (2, nib.Nifti2Image)):
                for source in make_sources():
                    with self.subTest(stored=stored, version=version,
                                      source=type(source).__name__):
                        img = ASAHSegmenter.read_volume(source, version)
                        self.assertIs(type(img), expected)
                        self.assertEqual(img.shape, self.src.shape)
                        np.testing.assert_allclose(img.affine, self.src.affine)

    def test_downgrade_beyond_nifti1_limits(self):
        big = nib.Nifti2Image(np.zeros((40000, 1, 1), np.uint8), np.eye(4))
        self.assertEqual(ASAHSegmenter.read_volume(big, 2).shape,
                         (40000, 1, 1))
        with self.assertRaisesRegex(VolumeError, "NIfTI-1"):
            ASAHSegmenter.read_volume(big, 1)

    def test_invalid_version(self):
        with self.assertRaises(ValueError):
            ASAHSegmenter.read_volume(self.src, nifti_version=3)

    def test_singleton_fourth_axis_is_dropped(self):
        data = np.asarray(self.src.dataobj)[..., None]
        img = ASAHSegmenter.read_volume(nib.Nifti1Image(data, self.src.affine))
        self.assertEqual(img.shape, self.src.shape)

    def test_non_3d_refused(self):
        data = np.asarray(self.src.dataobj)
        for bad in (np.stack([data, data], axis=-1), data[..., 0]):
            with self.subTest(shape=bad.shape), self.assertRaises(VolumeError):
                ASAHSegmenter.read_volume(nib.Nifti1Image(bad,
                                                          self.src.affine))

    def test_unreadable(self):
        bad = self.tmp / "bad.nii.gz"
        bad.write_bytes(b"not a nifti")
        for source in (bad, self.tmp / "missing.nii.gz",
                       io.BytesIO(b"x" * 400)):
            with (self.subTest(source=str(source)),
                  self.assertRaises(VolumeError)):
                ASAHSegmenter.read_volume(source)


class SegmentTest(TempDirTestCase):
    """Uses a deterministic stand-in model, so results can be asserted."""

    def setUp(self):
        super().setUp()
        base = make_ct(shape=(96, 80, 4))
        data = np.zeros(base.shape, np.int16)
        data[:48] = 100
        self.volume = nib.Nifti1Image(data, base.affine, base.header)
        self.model = ThresholdModel(threshold=50.0)
        self.seg = ASAHSegmenter(self.model, device="cpu", batch_size=3)

    def test_label_map_on_the_source_grid(self):
        mask = self.seg.segment(self.volume)
        data = np.asarray(mask.dataobj)
        self.assertEqual(mask.shape, self.volume.shape)
        np.testing.assert_allclose(mask.affine, self.volume.affine)
        self.assertEqual(mask.get_data_dtype(), np.uint8)
        self.assertTrue((data[:40, :-1] == SAH_LABEL).all())
        self.assertTrue((data[56:] == 0).all())

    def test_batch_size_is_used(self):
        self.seg.segment(self.volume)
        self.assertEqual(self.model.batch_sizes, [3, 1])

    def test_preprocessing_is_used(self):
        shapes = []

        class Spy(ThresholdModel):
            def forward(self, x):
                shapes.append(tuple(x.shape[2:]))
                return super().forward(x)

        ASAHSegmenter(Spy(), device="cpu",
                      preprocessing=PreprocessingConfiguration(shape=(64, 64))
                      ).segment(self.volume)
        self.assertEqual(set(shapes), {(64, 64)})

    def test_same_result_for_every_source_type(self):
        path = self.tmp / "ct.nii.gz"
        nib.save(self.volume, path)
        expected = np.asarray(self.seg.segment(self.volume).dataobj)
        for source in (path, str(path), io.BytesIO(path.read_bytes())):
            with self.subTest(source=type(source).__name__):
                np.testing.assert_array_equal(
                    np.asarray(self.seg.segment(source).dataobj), expected)

    def test_bad_input(self):
        with self.assertRaises(VolumeError):
            self.seg.segment(self.tmp / "missing.nii.gz")

    def test_output_follows_nifti_version(self):
        # The segmenter's version wins, whatever the input's version.
        expected = np.asarray(self.seg.segment(self.volume).dataobj)
        volume2 = nib.Nifti2Image(np.asarray(self.volume.dataobj),
                                  self.volume.affine, self.volume.header)
        for version, mask_type in ((1, nib.Nifti1Image), (2, nib.Nifti2Image)):
            seg = ASAHSegmenter(self.model,
                                device="cpu",
                                nifti_version=version)
            for volume in (self.volume, volume2):
                with self.subTest(version=version,
                                  input=type(volume).__name__):
                    mask = seg.segment(volume)
                    self.assertIs(type(mask), mask_type)
                    np.testing.assert_array_equal(np.asarray(mask.dataobj),
                                                  expected)

    def test_segment_file(self):
        src = self.tmp / "ct.nii.gz"
        nib.save(self.volume, src)
        for name in ("sub/mask.nii.gz", "mask.nii"):
            with self.subTest(dst=name):
                out = self.seg.segment_file(src, self.tmp / name)
                self.assertEqual(out, self.tmp / name)
                written = nib.load(out)
                self.assertEqual(written.shape, self.volume.shape)
                self.assertEqual(written.get_data_dtype(), np.uint8)

    def test_real_network_end_to_end(self):
        seg = ASAHSegmenter(materialized_model(), device="cpu", batch_size=2)
        mask = seg.segment(make_ct(shape=(96, 80, 2)))
        self.assertEqual(mask.shape, (96, 80, 2))
        self.assertTrue(set(np.unique(np.asarray(mask.dataobj))) <= {0, 1})


class PersistenceAndIntrospectionTest(CheckpointTestCase):

    def test_save_checkpoint_roundtrip(self):
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu")
        out = seg.save_checkpoint(self.tmp / "nested" / "copy.pt")
        self.assertTrue(out.is_file())
        self.assertFalse(out.with_name("copy.pt.tmp").exists())
        again = ASAHSegmenter.from_checkpoint(out, device="cpu")
        x = torch.randn(1, 1, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(seg.model(x), again.model(x))
        self.assertIsInstance(torch.load(out, weights_only=True), dict)

    def test_checkpoint_sha256(self):
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu")
        expected = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.assertEqual(seg.checkpoint_sha256, expected)

    def test_info_is_json_ready(self):
        import json
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu",
                                            batch_size=4)
        info = seg.info()
        json.dumps(info)
        self.assertEqual(info["checkpoint"], "weights.pt")
        self.assertEqual(info["device"], "cpu")
        self.assertEqual(info["batch_size"], 4)
        self.assertEqual(len(info["checkpoint_sha256"]), 64)
        self.assertEqual(set(info), {"package_version", "checkpoint",
                                     "checkpoint_sha256", "device",
                                     "batch_size", "nifti_version", "torch"})
        self.assertEqual(info["nifti_version"], 1)

    def test_repr(self):
        seg = ASAHSegmenter.from_checkpoint(self.checkpoint, device="cpu")
        self.assertEqual(repr(seg),
                         "ASAHSegmenter(device='cpu', batch_size=16, "
                         "checkpoint='weights.pt')")

    def test_read_only_properties(self):
        seg = ASAHSegmenter(ThresholdModel(), device="cpu")
        for name in ("device", "batch_size", "preprocessing", "nifti_version",
                     "checkpoint", "checkpoint_sha256", "model"):
            with self.subTest(name=name), self.assertRaises(AttributeError):
                setattr(seg, name, None)


class ExportsTest(unittest.TestCase):

    def test_module_all(self):
        for name in api.__all__:
            self.assertTrue(hasattr(api, name), name)


if __name__ == "__main__":
    unittest.main()
