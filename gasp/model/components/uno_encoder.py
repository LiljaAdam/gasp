from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.profiler import record_function
from torchvision.ops import SqueezeExcitation

from gasp.model.components.checkpointing import checkpoint
from gasp.model.components.dynamic_conv import DynamicConv
from gasp.model.components.py_deformable_attention.deformable_attention import MSDeformSelfAttnBlockAllToAll


class UnoResBlock(nn.Module):
    def __init__(self, feature_dim: int, stride: int = 1, resblock_convs: Literal["dynamic", "standard"] = "dynamic"):
        super().__init__()
        if resblock_convs == "dynamic":
            self.conv1 = DynamicConv(feature_dim, feature_dim, 3, stride=stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(feature_dim, feature_dim, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_dim)
        self.relu = nn.ReLU(inplace=True)
        if resblock_convs == "dynamic":
            self.conv2 = DynamicConv(feature_dim, feature_dim, 3, stride=1, padding=1, bias=False)
        else:
            self.conv2 = nn.Conv2d(feature_dim, feature_dim, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        self.squeeze_and_excite = SqueezeExcitation(feature_dim, 1)
        self.downsample = (
            nn.Sequential(
                DynamicConv(feature_dim, feature_dim, 1, stride=stride, bias=False)
                if resblock_convs == "dynamic"
                else nn.Conv2d(feature_dim, feature_dim, 1, stride=stride, bias=False),
                nn.BatchNorm2d(feature_dim),
            )
            if stride != 1
            else None
        )
        self.stride = stride
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.squeeze_and_excite(out)

        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        out = self.relu(out)

        return out


class UnoBackbone(nn.Module):
    def __init__(
        self,
        feature_dim: int = 64,
        attention: Literal["none", "deformable_py"] = "deformable_py",
        gradient_checkpointing: bool = False,
        resblock_convs: Literal["dynamic", "standard"] = "dynamic",
        attention_layers: int = 6,
        normalize_before: bool = False,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.stage1 = nn.Sequential(
            UnoResBlock(feature_dim, stride=2, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
        )

        self.stage2 = nn.Sequential(
            UnoResBlock(feature_dim, stride=2, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
        )

        self.stage3 = nn.Sequential(
            UnoResBlock(feature_dim, stride=2, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
            UnoResBlock(feature_dim, stride=1, resblock_convs=resblock_convs),
        )

        if attention == "deformable_py":
            self.attention_py = nn.ModuleList(
                [
                    MSDeformSelfAttnBlockAllToAll(
                        dim=feature_dim,
                        num_points=4,
                        num_heads=8,
                        num_scales=3,
                    )
                    for _ in range(attention_layers)
                ]
            )
            self.attention = None
        else:
            self.attention = None
            self.attention_py = None

    @record_function("UnoBackbone.forward")
    def forward(self, bev_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x1 = self.stage1(bev_features)
        x2 = self.stage2(x1)

        if self.gradient_checkpointing:
            x3 = checkpoint(self.stage3, x2, use_reentrant=False)
        else:
            x3 = self.stage3(x2)

        if self.attention is not None:
            x1, x2, x3 = self.attention(x1, x2, x3)
        elif self.attention_py is not None:
            out = [x1, x2, x3]
            for layer in self.attention_py:
                out = layer(out)
            x1, x2, x3 = out

        return (x1, x2, x3)


class UnoFPN(nn.Module):
    def __init__(self, feature_dim: int = 64):
        super().__init__()
        self.feature_dim = feature_dim
        self.lateral_conv1 = nn.Conv2d(feature_dim, feature_dim, 1)
        self.lateral_conv2 = nn.Conv2d(feature_dim, feature_dim, 1)
        self.lateral_conv3 = nn.Conv2d(feature_dim, feature_dim, 1)

        self.fpn_conv1 = nn.Conv2d(feature_dim, feature_dim, 3, padding=1)

    @record_function("UnoFPN.forward")
    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3 = self.lateral_conv3(x3)
        p2 = self.lateral_conv2(x2) + F.interpolate(p3, size=x2.shape[-2:], mode="nearest")
        p1 = self.lateral_conv1(x1) + F.interpolate(p2, size=x1.shape[-2:], mode="nearest")

        return self.fpn_conv1(p1)


class UnoBEVEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        attention: Literal["none", "deformable_py"] = "deformable_py",
        resblock_convs: Literal["dynamic", "standard"] = "dynamic",
        attention_layers: int = 6,
        swap_batch_norm_to_instance_norm: bool = False,
        gradient_checkpointing: bool = False,
        normalize_before: bool = False,
        use_positional_encoding: bool = True,
    ) -> None:
        """Initialize the UnoEncoder.
        feature_dim: The dimension of the feature vector.
        use_deformable_attention: Whether to use deformable attention.
        """
        super().__init__()
        self.backbone = UnoBackbone(
            feature_dim=feature_dim,
            attention=attention,
            resblock_convs=resblock_convs,
            attention_layers=attention_layers,
            gradient_checkpointing=gradient_checkpointing,
            normalize_before=normalize_before,
            use_positional_encoding=use_positional_encoding,
        )
        self.fpn = UnoFPN(feature_dim)

        if swap_batch_norm_to_instance_norm:
            for module in [self.backbone, self.fpn]:
                for layer in module.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer = nn.InstanceNorm2d(layer.num_features, affine=True)
                        layer.weight.data = torch.ones_like(layer.weight.data)
                        layer.bias.data = torch.zeros_like(layer.bias.data)

    @record_function("UnoBEVEncoder.forward")
    def forward(self, bev_features: torch.Tensor) -> torch.Tensor:
        ms_bev_features = self.backbone(bev_features)
        return self.fpn(*ms_bev_features)
