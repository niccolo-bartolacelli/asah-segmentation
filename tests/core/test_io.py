import gzip
import io
import unittest

import nibabel as nib
import numpy as np
from nibabel.spatialimages import HeaderDataError

from asah_segmentation.io import (convert_volume, load_volume,
                                  nifti_image_type, nifti_version_of)
from tests.helpers import TempDirTestCase, make_ct


class LoadVolumeFromPathTest(TempDirTestCase):

    def test_nifti1_gz(self):
        src = make_ct()
        path = self.tmp / "ct.nii.gz"
        nib.save(src, path)
        img = load_volume(path)
        self.assertIsInstance(img, nib.Nifti1Image)
        self.assertEqual(img.shape, src.shape)
        np.testing.assert_allclose(img.affine, src.affine)
        np.testing.assert_array_equal(np.asarray(img.dataobj),
                                      np.asarray(src.dataobj))

    def test_str_path_and_plain_nii(self):
        path = self.tmp / "ct.nii"
        nib.save(make_ct(), path)
        self.assertEqual(load_volume(str(path)).shape, (96, 80, 3))

    def test_both_versions_converted_either_way(self):
        src = make_ct()
        data = np.asarray(src.dataobj)
        for stored in (nib.Nifti1Image, nib.Nifti2Image):
            path = self.tmp / f"{stored.__name__}.nii.gz"
            nib.save(stored(data, src.affine, src.header), path)
            for version, expected in ((1, nib.Nifti1Image),
                                      (2, nib.Nifti2Image)):
                with self.subTest(stored=stored.__name__, version=version):
                    img = load_volume(path, nifti_version=version)
                    self.assertIs(type(img), expected)
                    np.testing.assert_allclose(img.affine, src.affine)
                    np.testing.assert_allclose(img.header.get_zooms(),
                                               src.header.get_zooms())
                    np.testing.assert_array_equal(np.asarray(img.dataobj),
                                                  data)

    def test_non_nifti_file(self):
        path = self.tmp / "ct.mgz"
        nib.save(nib.MGHImage(np.zeros((4, 4, 4), np.float32), np.eye(4)),
                 path)
        with self.assertRaises(TypeError):
            load_volume(path)

    def test_invalid_version(self):
        with self.assertRaises(ValueError):
            load_volume(self.tmp / "unused.nii.gz", nifti_version=3)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_volume(self.tmp / "missing.nii.gz")


class LoadVolumeFromFileObjectTest(unittest.TestCase):

    def setUp(self):
        self.src = make_ct()
        self.raw = self.src.to_bytes()

    def check(self, img):
        self.assertIsInstance(img, nib.Nifti1Image)
        self.assertEqual(img.shape, self.src.shape)
        np.testing.assert_allclose(img.affine, self.src.affine)
        self.assertEqual(img.get_data_dtype(), np.float32)
        np.testing.assert_array_equal(
            np.asarray(img.dataobj),
            np.asarray(self.src.dataobj).astype(np.float32)
        )

    def test_uncompressed(self):
        self.check(load_volume(io.BytesIO(self.raw)))

    def test_gzip_compressed(self):
        self.check(load_volume(io.BytesIO(gzip.compress(self.raw))))

    def test_stream_position_is_ignored(self):
        buf = io.BytesIO(self.raw)
        buf.seek(100)
        self.check(load_volume(buf))

    def test_data_survives_closing_the_source(self):
        buf = io.BytesIO(gzip.compress(self.raw))
        img = load_volume(buf)
        buf.close()
        self.assertEqual(np.asarray(img.dataobj).shape, self.src.shape)

    def test_both_versions_converted_either_way(self):
        v2_raw = nib.Nifti2Image(np.asarray(self.src.dataobj), self.src.affine,
                                 self.src.header).to_bytes()
        for name, raw in (("nifti1", self.raw), ("nifti2", v2_raw)):
            for version, expected in ((1, nib.Nifti1Image),
                                      (2, nib.Nifti2Image)):
                for wrap in (bytes, gzip.compress):
                    with self.subTest(stored=name, version=version,
                                      wrap=wrap.__name__):
                        img = load_volume(io.BytesIO(wrap(raw)),
                                          nifti_version=version)
                        self.assertIs(type(img), expected)
                        self.check(img)

    def test_garbage(self):
        with self.assertRaises(TypeError):
            load_volume(io.BytesIO(b"not a nifti" * 100))


class ConvertVolumeTest(unittest.TestCase):

    def test_same_version_is_returned_unchanged(self):
        v1 = make_ct()
        v2 = nib.Nifti2Image(np.asarray(v1.dataobj), v1.affine)
        self.assertIs(convert_volume(v1, 1), v1)
        self.assertIs(convert_volume(v2, 2), v2)

    def test_roundtrip_keeps_geometry(self):
        src = make_ct()
        back = convert_volume(convert_volume(src, 2), 1)
        self.assertIs(type(back), nib.Nifti1Image)
        np.testing.assert_allclose(back.affine, src.affine)
        np.testing.assert_allclose(back.header.get_qform(),
                                   src.header.get_qform())
        np.testing.assert_allclose(back.header.get_sform(),
                                   src.header.get_sform())

    def test_downgrade_beyond_nifti1_limits(self):
        big = nib.Nifti2Image(np.zeros((40000, 1, 1), np.uint8), np.eye(4))
        self.assertIs(type(convert_volume(big, 2)), nib.Nifti2Image)
        with self.assertRaises(HeaderDataError):
            convert_volume(big, 1)

    def test_version_helpers(self):
        v1 = make_ct()
        v2 = convert_volume(v1, 2)
        self.assertEqual(nifti_version_of(v1), 1)
        self.assertEqual(nifti_version_of(v2), 2)
        self.assertEqual(nifti_version_of(v2.header), 2)
        self.assertIs(nifti_image_type(2), nib.Nifti2Image)
        with self.assertRaises(ValueError):
            nifti_image_type(3)


if __name__ == "__main__":
    unittest.main()
