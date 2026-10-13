# ASAH Segmentation

PyTorch pipeline to perform Aneurysmal Subarachnoid Hemorrhage segmentation tasks on Non-Contrast CT Scans.


## Acknowledgements

Parts of the source code and the general architecture are inspired or derived from [segment-ncct-sah](https://github.com/athanell/segment-ncct-sah) by [@athanell](https://github.com/athanell), distributed under GPL-3.0 license.


## Installation

```bash
uv sync                    # inference: torch, nibabel, nilearn, numpy
uv sync --extra legacy     # + TensorFlow, only to convert the original checkpoint
```


## Usage

`ASAHSegmenter` is the public, stable interface; everything else in the package
is an implementation detail that may change between minor releases.

```python
from asah_segmentation import ASAHSegmenter

segmenter = ASAHSegmenter.from_checkpoint("asah_nonewnet.pt", device="auto")
mask = segmenter.segment("scan.nii.gz")            # nib.Nifti1Image, uint8
segmenter.segment_file("scan.nii.gz", "mask.nii.gz")
print(segmenter.info())                            # provenance (checkpoint SHA-256, device, ...)
```

The mask is a `uint8` label map (`BACKGROUND_LABEL = 0`, `SAH_LABEL = 1`) on the
input's voxel grid: same shape, affine, qform and sform. Inputs can be a path, a
file object or a loaded NIfTI image; a singleton 4th axis is dropped.

The NIfTI format is an explicit choice, `nifti_version=1` (default) or `2`, set
when building the segmenter: NIfTI-1 and NIfTI-2 inputs are both accepted and
converted to that version (in either direction), and the mask is returned in
it. Only a NIfTI-2 volume exceeding NIfTI-1 limits (a dimension above 32767)
cannot be read as NIfTI-1. The published model, like the original TensorFlow
implementation, was validated on NIfTI-1 only.

Errors are reported as `VolumeError` (bad input) and `CheckpointError` (bad
weights).

### Converting the original TensorFlow checkpoint

Once, with the `legacy` extra installed:

```python
from asah_segmentation import ASAHSegmenter

ASAHSegmenter.from_legacy_checkpoint("path/to/tf_checkpoint", device="cpu") \
    .save_checkpoint("asah_nonewnet.pt")
```

Production code then only needs `from_checkpoint`.


## Development

Tests use the standard library `unittest` (no extra dependencies), with random
weights and synthetic volumes:

```bash
uv run python -m unittest
```
