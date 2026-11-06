import ast
import pkgutil
import subprocess
import sys
import unittest
from pathlib import Path

import asah_segmentation
from asah_segmentation import api

# Top-level modules the production package must never import: installing it
# alone (without the `dev` dependency group) has to be enough to run it.
DEV_ONLY_MODULES = {"asah_segmentation_dev", "tensorflow"}


class PublicSurfaceTest(unittest.TestCase):

    def test_all_names_are_exported(self):
        for name in asah_segmentation.__all__:
            self.assertTrue(hasattr(asah_segmentation, name), name)

    def test_root_reexports_the_api(self):
        for name in api.__all__:
            self.assertIs(getattr(asah_segmentation, name), getattr(api, name))

    def test_stable_names(self):
        # Removing or renaming any of these is a breaking change.
        self.assertTrue({"ASAHSegmenter", "CheckpointError", "VolumeError",
                         "PreprocessingConfiguration", "BACKGROUND_LABEL",
                         "SAH_LABEL"} <= set(api.__all__))
        self.assertEqual((api.BACKGROUND_LABEL, api.SAH_LABEL), (0, 1))

    def test_version(self):
        self.assertIsInstance(asah_segmentation.__version__, str)
        self.assertTrue(asah_segmentation.__version__)

    def test_error_hierarchy(self):
        self.assertTrue(issubclass(api.CheckpointError, RuntimeError))
        self.assertTrue(issubclass(api.VolumeError, ValueError))


class ProductionIsolationTest(unittest.TestCase):
    """The core package stands alone: no development code, no TensorFlow."""

    def test_no_dev_imports_in_source(self):
        # Static, so lazy imports inside functions are caught as well.
        root = Path(asah_segmentation.__file__).parent
        for path in sorted(root.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    with self.subTest(file=path.name, line=node.lineno):
                        self.assertNotIn(name.split(".")[0], DEV_ONLY_MODULES)

    def test_importing_every_module_loads_no_dev_code(self):
        modules = [m.name for m in pkgutil.walk_packages(
            asah_segmentation.__path__, "asah_segmentation.")]
        code = ("import sys\n"
                f"for m in {modules!r}: __import__(m)\n"
                "print(sorted({n.split('.')[0] for n in sys.modules}))")
        out = subprocess.run([sys.executable, "-c", code], check=True,
                             capture_output=True, text=True).stdout
        self.assertFalse(DEV_ONLY_MODULES & set(ast.literal_eval(out)))


if __name__ == "__main__":
    unittest.main()
