""" serialization.py

Deserialize legacy TensorFlow checkpoints.
Requires TF as a dependency (consider installing with `legacy` optional
dependencies).

Exported Functions:
* load_legacy_model: load model checkpoint from legacy TensorFlow
raw data.

"""

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

import torch

from asah_segmentation.model import ASAHSegmentationNet


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

def load_legacy_model(path: str) -> ASAHSegmentationNet:
    """
    Loads legacy TensorFlow weigths to PyTorch model.

    Parameters
    ----------
    path : str
        The path to the TF checkpoint folder
    
    Returns
    -------
    ASAHSegmentationNet
        The loaded model
    """

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow is required to perform legacy model "
                          "weights conversion (install the `legacy` extra).")

    model = ASAHSegmentationNet()
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 1, 32, 32))    
    state_dict = model.state_dict()

    reader = tf.train.load_checkpoint(path)
    
    mapping = {}
    for tf_block, torch_block, layers in TF_TO_TORCH_BLOCKS:
        for i, l in enumerate(layers):
            for tf_param, offset, torch_param in TF_TO_TORCH_PARAMS:
                source = f"NoNewNet/{tf_block}/{l}/{tf_param}"
                target = f"{torch_block}.blocks.{4 * i + offset}.{torch_param}"
                mapping[source] = target
    mapping["NoNewNet/R1_FC/2/conv_/w"] = "head.blocks.8.weight"

    assert len(mapping) == len(state_dict)

    for source, target in mapping.items():
        array = reader.get_tensor(source)

        if source.endswith("/conv_/w"):
            array = array.transpose(3, 2, 0, 1)

        tensor = torch.from_numpy(array.copy()).to(torch.float32)

        state_dict[target] = tensor

    model.load_state_dict(state_dict, strict=True)

    return model
