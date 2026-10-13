""" asah_segmentation

Aneurysmal Subarachnoid Hemorrhage segmentation on Non-Contrast CT scans.

The public, stable interface is `ASAHSegmenter` (see `asah_segmentation.api`);
the other modules are implementation details.

"""

from importlib.metadata import PackageNotFoundError, version

from asah_segmentation.api import (
    BACKGROUND_LABEL,
    SAH_LABEL,
    ASAHSegmenter,
    CheckpointError,
    PreprocessingConfiguration,
    VolumeError,
)

try:
    __version__ = version("asah-segmentation")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0+unknown"

__all__ = [
    "ASAHSegmenter",
    "CheckpointError",
    "VolumeError",
    "PreprocessingConfiguration",
    "BACKGROUND_LABEL",
    "SAH_LABEL",
    "__version__",
]
