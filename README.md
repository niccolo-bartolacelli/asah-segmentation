# ASAH Segmentation

PyTorch pipeline to perform Aneurysmal Subarachnoid Hemorrhage segmentation tasks on Non-Contrast CT Scans.


## Acknowledgements

Parts of the source code and the general architecture are inspired or derived from [segment-ncct-sah](https://github.com/athanell/segment-ncct-sah) by [@athanell](https://github.com/athanell), distributed under GPL-3.0 license.


## Installation

Requires Python 3.14 (see `requires-python` and `.python-version`).

The repository holds one package, `asah-segmentation` (import name
`asah_segmentation`), with production inference only: `ASAHSegmenter`, the
model, pre/post-processing and NIfTI I/O. Its dependencies are torch,
nibabel, nilearn and numpy.

Development tooling lives next to it, outside the package, and is never
shipped or imported by it (the test suite enforces this):

| Path | Contents |
|---|---|
| `eval/` | evaluation CLI (`evaluate.py`) and segmentation metrics (`metrics.py`) |
| `scripts/` | one-off legacy TensorFlow checkpoint conversion |
| `tests/` | unit tests (`tests/core`: the package, `tests/dev`: the tooling) |
| `weights/` | local model weights, git-ignored |

```bash
# Production: the package alone
pip install asah-segmentation        # or, from this repository:
uv sync --no-dev                     # (add --frozen in deployments)

# Development
uv sync                              # the package; eval/ and tests/ need nothing more
uv sync --group legacy               # + TensorFlow, only to convert the original checkpoint
```

The `legacy` dependency group is local to this repository: dependency groups
are not part of the published package metadata, so a production install never
pulls TensorFlow in. Building the project (`uv build`) produces a wheel with
`asah_segmentation` only.


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

This is a one-off development task. Once, after `uv sync --group legacy`, from
the repository root:

```bash
uv run --group legacy python scripts/convert_legacy_checkpoint.py  path/to/tf_checkpoint weights/asah_nonewnet.pt
```

The script writes a PyTorch `state_dict`; production code then only needs
`ASAHSegmenter.from_checkpoint("weights/asah_nonewnet.pt")`.


## Results

The original authors validated the TensorFlow model on external sets (their README): patient level, sensitivity 100%
(Swiss, 1100 scans), 97.3% (Indian, 279) and 100% (Finnish, 519 consecutive emergency scans), specificity 57.7%, 86.0% and 87.3%;
slice level, sensitivity 87.4% (Swiss) and 75.3% (Finnish), specificity 95.3% and 98.8%.

**Parity with the original**: running both the original implementation and the PyTorch port on a subset of 41 images of the original Indian validation dataset gave the following results, which provide strong evidence of parity:

|         | mean   | std    | median | q1      | q3     | min    | max    |
|---------|--------|--------|--------|---------|--------|--------|--------|
|Dice     | 0.9995 | 0.0018 | 0.9999 |  0.9997 | 1.0000 | 0.9886 | 1.0000 |
|AVD (mL) | 0.0011 | 0.0019 | 0.0006 |  0.0000 | 0.0011 | 0.0000 | 0.0087 |


## Development

New code goes in `src/asah_segmentation` only if production inference needs
it; everything else (training, datasets, augmentation, evaluation) stays out
of the package, in `eval/`, `scripts/` or a similar top-level directory, and
is imported from the repository root (e.g. `from eval.metrics import
dice_coefficient`).

Tests use the standard library `unittest` (no extra dependencies), with random
weights and synthetic volumes. `tests/core` covers the production package and
also runs on a production-only environment; `tests/dev` covers `eval/` and
`scripts/` (TensorFlow is faked, so it is not needed):

```bash
uv run python -m unittest                                          # everything
uv run --no-dev python -m unittest discover -s tests/core -t .     # core, production-only environment
```

### Evaluation

`eval/evaluate.py` compares two folders of masks (`*.nii`, `*.nii.gz`),
matched by file name: the masks to evaluate in SOURCE (typically this
package's segmentations) and the reference masks in TARGET, which can be
manual annotations, the legacy TensorFlow model's output, or another model's.
Each case is scored with the Dice coefficient and the absolute volume
difference (AVD, in mL, from the voxel size in the affine); the per-case
results are followed by a summary of each metric (mean, standard deviation,
median, quartiles, min, max) and the worst case for each. Masks are never
committed.

```bash
uv run python eval/evaluate.py SOURCE TARGET
uv run python eval/evaluate.py SOURCE TARGET --output results.json   # full report
uv run python eval/evaluate.py SOURCE TARGET --output results.csv    # one row per case
```

Options: `--label` (default: every non-zero voxel) and `-o/--output PATH`
(also write the results: a JSON report with per-case results, summary and
skipped cases, or per-case rows if PATH ends in `.csv`). Masks present in only
one folder, that fail to load, or whose counterpart is on a different voxel
grid are skipped with a warning. The exit status is 0 if at least one case was
scored, 1 otherwise.

The metrics are also usable on their own:

```python
from eval.metrics import absolute_volume_difference, dice_coefficient

dice_coefficient("mask_torch.nii.gz", "mask_tf.nii.gz")   # every non-zero voxel
dice_coefficient(mask, reference, label=1)                # SAH label only
absolute_volume_difference("mask_torch.nii.gz", "mask_tf.nii.gz")   # mL
absolute_volume_difference(mask, reference, voxel_volume=1.25)      # arrays: mm³ per voxel
```


## License

Copyright (C) 2026 Niccolò Bartolacelli

Derived from [segment-ncct-sah](https://github.com/athanell/segment-ncct-sah) by [@athanell](https://github.com/athanell), and distributed under the same terms.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.

This program is experimental software, without any warranty. See the [LICENSE](LICENSE) file for details.
