# ASAH Segmentation

PyTorch pipeline to perform Aneurysmal Subarachnoid Hemorrage segmentation tasks on Non-Contrast CT Scans.


## Acknowledgements

Parts of the source code and the general architecture are inspired or derived from [segment-ncct-sah](https://github.com/athanell/segment-ncct-sah) by [@athanell](https://github.com/athanell), distributed under GPL-3.0 license.


## Configuration
This project was built with torch:xpu extension (for Intel embedded GPU).
If you wish to run it on different hardware, please edit accordingly in `pyproject.toml` the following lines
```toml
[build-system]
requires = ["uv_build>=0.12.14,<0.13.0"]
build-backend = "uv_build"

[tool.uv.sources]
torch = [
    { index = "pytorch-xpu" },
]
torchvision = [
    { index = "pytorch-xpu" },
]
# torch-xpu also requires triton–xpu to be resolved explicitly
triton-xpu = [
    { index = "pytorch-xpu" },
]

[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true
```
