""" model.py

Core segmentation interface.

Exported Classes:
* ASAHSegmentationNet: a Torch module which implements the NoNewNet
architecture, used for Aneurysmal Subarachnoid Hemorrhage automatic
segmentation on Non-Contrast CT Scans.

"""

from itertools import chain
from typing import Literal

from asah_segmentation.utils import safe_tuple as IF

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ASAHSegmentationNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 n_features: tuple[int, ...] = (30, 60, 120, 240, 480),
                 dropout_rate: float = 0.0) -> None:
        """
        Parameters
        ----------
        num_classes : int
            Number of output classes predicted by the model
        
        n_features : tuple[int, ...]
            The number of features produced by each layer
        
        dropout_rate : float
            The dropout rate to be used during training (has no effect during
            validation and testing)
        """

        super().__init__()

        if len(n_features) < 3:
            raise ValueError(f"{repr(self)} Parameter n_features must" \
            "contain at least 3 elements.")

        self.num_classes = num_classes
        self.n_features = n_features

        # TODO: add weights and bias regularizers (initializers not needed
        #       if using pretrained model?)

        self.down_blocks = nn.ModuleList([
            ConvBlock((f, f),
                      (3, 3),
                      action="downsample",
                      dr=dropout_rate)
            for f in self.n_features[:-1]
        ])

        self.bottleneck = ConvBlock((self.n_features[-1], self.n_features[-2]),
                                    (3, 3),
                                    action="upsample",
                                    dr=dropout_rate)

        self.up_blocks = nn.ModuleList([
            ConvBlock((f1, f2),
                      (3, 3),
                      action="upsample",
                      dr=dropout_rate)
            for f1, f2 in zip(self.n_features[-2::-1], self.n_features[-3::-1])
        ])

        self.head = ConvBlock((self.n_features[0],
                               self.n_features[0],
                               self.num_classes),
                              (3, 3, 1),
                              dr=dropout_rate)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Safety checks
        if X.ndim != 4:
            raise ValueError("Expected input shape [N, C, H, W].")

        # MaxPool3d(2,2) is applied at each level
        if any(dim % (2 ** (len(self.n_features) - 1)) != 0 for dim in X.shape[2:]):
            raise ValueError(f"Each dimension of input tensor size must be"
                             f"multiple of {2 ** (len(self.n_features) - 1)}.")

        # Iterate through down blocks and append result in the form
        # (out_tensor, partial_result), where partial_result is the convolved
        # tensor and out_tensor is the downsampled tensor.
        d_res = [(X, None)]
        for d_block in self.down_blocks:
            d_res.append(d_block(d_res[-1][0], with_partial=True))

        u, _ = self.bottleneck(d_res[-1][0])

        for u_block, skip_conn in zip(self.up_blocks, reversed(d_res)):
            u, _ = u_block(u, skip=skip_conn[1])

        logits, _ = self.head(u, skip= d_res[1][1])
        return logits


class ConvBlock(nn.Module):
    def __init__(self,
                 out_ch: tuple[int, ...],
                 k_size: tuple[int, ...],
                 *,
                 action:  Literal["downsample", "upsample", "none"] = "none",
                 dr: float = 0.0) -> None:
        super().__init__()

        # TODO: maybe use Conv2d instead LazyConv2d
        self.blocks = nn.Sequential(
            *chain(*((
                nn.LazyConv2d(o, k, bias=False, padding=1 if k != 1 else 0),
                *IF(k != 1, nn.InstanceNorm2d(o, eps=1e-5, affine=True)),
                *IF(k != 1, nn.PReLU(o, init=0.0)),
                *IF(k != 1, nn.Dropout(dr)))
            for o, k in zip(out_ch, k_size)))
        )

        self.action = \
            nn.MaxPool2d(2, 2) if action == "downsample" else \
            LinearInterpolationLayer(2.0) if action == "upsample" else \
            nn.Identity()

    def forward(self,
                X: torch.Tensor,
                with_partial: bool = False,
                skip: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Performs Convolutional Block forward step.

        Parameters
        ----------
        X : Tensor
            The input tensor
        
        with_partial : bool
            If `True`, the second value returned is the raw result of the
            convolution. If `False`, it is `None`

        skip : Tensor
            Tensor to be used to perform skip connection
        
        Returns
        -------
        output : Tensor
            The result of the forward step of the convolution block
        
        partial : Tensor or None
            The raw result of convolution (used for skip connections),
            `None` if with_partial is `False`
        """
        raw_conv = self.blocks(X if skip is None else
                               torch.cat((skip, X), dim=1))
        
        return self.action(raw_conv), raw_conv if with_partial else None


class LinearInterpolationLayer(nn.Module):
    """
    Implements legacy TenforFlow (multi)linear interpolation behavior,
    which defaults to
        
    * `align_corners=False`
    * `half_pixel_centers=False`

    See `tf.compat.v1.image.resize_bilinear` for further reference.
    """
    def __init__(self,
                 scale: float | None = None,
                 new_shape: torch.Size | None = None) -> None:
        super().__init__()

        if not (scale is None) ^ (new_shape is None):
            raise ValueError("Exactly one out of scale and new_shape must be "
                             "specified")

        if scale is not None and scale <= 0:
            raise ValueError("scale must be a positive float")

        if new_shape is not None:
            if len(new_shape) == 0:
                raise ValueError("new_shape cannot be empty")
            if any(d <= 0 for d in new_shape):
                raise ValueError("new_shape dimensions must be positive")
        
        self.scale = scale
        self.new_shape = new_shape


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Performs Linear Interpolation forward step.

        Parameters
        ----------
        X : Tensor
            The input tensor, which must be in the format
            (N, C, *spatial_dims)

        Returns
        -------
        Tensor
            The linearly interpolated tensor
        """
        if X.ndim < 3:
            raise ValueError("X must be a 3D+ tensor in the format "
                             "(N, C, *spatial_dims)")
        
        spatial_dims = list(X.shape[2:])
        if self.new_shape is not None:
            new_spatial_dims = list(self.new_shape)

            if len(spatial_dims) != len(new_spatial_dims):
                raise ValueError(f"Input tensor ({len(spatial_dims)}D) is not "
                                 "compatible with the requested shape "
                                 f"({len(new_spatial_dims)}D)")
        else:
            assert self.scale is not None

            new_spatial_dims = [int(self.scale * s) for s in spatial_dims]

            if any(d <= 0 for d in new_spatial_dims):
                raise ValueError(
                    "Input tensor is not compatible with the requested scale "
                    "(scale * d >= 1 must be true for each dimension d of the)"
                    "input tensor"
                )

        ratio = [in_d / out_d
                 for in_d, out_d in zip(spatial_dims, new_spatial_dims)]

        if X.dtype != torch.float32:
            X = X.to(torch.float32)
        
        res = X
        for dim in range(len(new_spatial_dims)):
            x = torch.arange(new_spatial_dims[dim],
                             device=X.device,
                             dtype=torch.float32) * ratio[dim]
            x_a = torch.floor(x).long()
            x_b = torch.clamp(x_a + 1, max=spatial_dims[dim] - 1)

            y_a = torch.index_select(res, 2 + dim, x_a)
            y_b = torch.index_select(res, 2 + dim, x_b)

            t_broadcast_shape = [1] * X.ndim
            t_broadcast_shape[2 + dim] = new_spatial_dims[dim]
            t = (x - x_a).reshape(t_broadcast_shape).to(res.dtype)

            res = (y_b - y_a) * t + y_a
        
        return res
