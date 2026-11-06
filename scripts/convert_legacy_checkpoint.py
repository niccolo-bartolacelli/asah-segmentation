""" convert_legacy_checkpoint.py

Convert legacy TensorFlow checkpoint to PyTorch state dict.
Requires TF as a dependency (install with `uv sync --group legacy`).

Conversion is a one-off development task: it is run once on the legacy
checkpoint, which is thence loaded in production with
`ASAHSegmenter.from_checkpoint`, which needs neither TensorFlow nor this
script.

"""


try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import torch


TF_TO_TORCH_BLOCKS = [
    ("L1", "down_blocks.0", ("30_in_prelu", "30_in_prelu_1")),
    ("L2", "down_blocks.1", ("60_in_prelu", "60_in_prelu_1")),
    ("L3", "down_blocks.2", ("120_in_prelu", "120_in_prelu_1")),
    ("L4", "down_blocks.3", ("240_in_prelu", "240_in_prelu_1")),
    ("bottom", "bottleneck", ("480_in_prelu", "240_in_prelu")),
    ("R4", "up_blocks.0", ("240_in_prelu", "120_in_prelu")),
    ("R3", "up_blocks.1", ("120_in_prelu", "60_in_prelu")),
    ("R2", "up_blocks.2", ("60_in_prelu", "30_in_prelu")),
    ("R1_FC", "head", ("30_in_prelu", "30_in_prelu_1")),
]

TF_TO_TORCH_PARAMS = [
    ("conv_/w", 0, "weight"),
    ("in_/InstanceNorm/gamma", 1, "weight"),
    ("in_/InstanceNorm/beta", 1, "bias"),
    ("prelu_acti_/alpha", 2, "weight"),
]

def convert_legacy_checkpoint(src: str, dest: str) -> None:

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to perform legacy model "
                          "weights conversion (`uv sync --group legacy`).")

    reader = tf.train.load_checkpoint(src)
    
    mapping = {}
    for tf_block, torch_block, layers in TF_TO_TORCH_BLOCKS:
        for i, l in enumerate(layers):
            for tf_param, offset, torch_param in TF_TO_TORCH_PARAMS:
                source = f"NoNewNet/{tf_block}/{l}/{tf_param}"
                target = f"{torch_block}.blocks.{4 * i + offset}.{torch_param}"
                mapping[source] = target
    mapping["NoNewNet/R1_FC/2/conv_/w"] = "head.blocks.8.weight"

    state_dict = {}
    for source, target in mapping.items():
        array = reader.get_tensor(source)

        if source.endswith("/conv_/w"):
            array = array.transpose(3, 2, 0, 1)

        tensor = torch.from_numpy(array.copy()).to(torch.float32)

        state_dict[target] = tensor

    torch.save(state_dict, dest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert legacy TensorFlow checkpoint to PyTorch state dict."
    )
    parser.add_argument("src", help="path or prefix of the legacy TF checkpoint")
    parser.add_argument("dest", help="output path of the PyTorch state dict")
    args = parser.parse_args()

    convert_legacy_checkpoint(args.src, args.dest)
