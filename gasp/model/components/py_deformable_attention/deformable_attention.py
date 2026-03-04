import math
from typing import List

import einops
import torch
from torch import nn
from torch.nn import functional as F

from gasp.model.components.py_deformable_attention.channels_first_layernorm import MLP, CFLayerNorm


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_data: int,
        num_points: int,
        num_heads: int,
        num_scales: int,
        init_head_directional: bool = False,
        align_corners: bool = False,
        prenorm_spatial_data: bool = False,
    ):
        assert (
            dim % num_heads == 0
        ), f"The model latent dimensionality (={dim}) has to be divisible by the number of attention heads (={num_heads})."
        super().__init__()

        self._n_heads = num_heads
        self._n_points = num_points
        self._n_scales = num_scales

        if prenorm_spatial_data:
            self._value_proj = nn.Sequential(CFLayerNorm(dim_data), nn.Conv2d(dim_data, dim, kernel_size=1))
        else:
            self._value_proj = nn.Conv2d(dim_data, dim, kernel_size=1)

        self._output_proj = nn.Linear(dim, dim)
        self._offset_attn_proj = nn.Linear(dim, num_heads * num_scales * num_points * 3)
        self._offset_split_index = num_heads * num_scales * num_points * 2

        self._use_align_corners = align_corners
        self.reset_parameters(init_head_directional=init_head_directional)

    def _init_offset_projection(self, init_head_directional: bool) -> None:
        nn.init.constant_(self._offset_attn_proj.weight.data, 0.0)
        num = self._n_heads if init_head_directional else self._n_points
        angles = (2 * math.pi) * torch.arange(
            num, dtype=next(self._offset_attn_proj.parameters()).dtype
        ) / num
        init_sample_loc = torch.stack([angles.cos(), angles.sin()], dim=-1)
        init_sample_loc /= init_sample_loc.abs().max(-1, keepdim=True)[0]
        if init_head_directional:
            init_sample_loc = init_sample_loc.view(self._n_heads, 1, 1, 2) * torch.arange(
                1,
                self._n_points + 1,
            ).view(1, 1, self._n_points, 1).repeat(1, self._n_scales, 1, 1)
        else:
            init_sample_loc = init_sample_loc.view(1, 1, self._n_points, 2) * torch.arange(
                1, self._n_heads + 1
            ).view(self._n_heads, 1, 1, 1).repeat(1, self._n_scales, 1, 1)
        self._offset_attn_proj.bias.data[: self._offset_split_index] = init_sample_loc.view(-1)
        self._offset_attn_proj.bias.data[self._offset_split_index :] = 0

    def reset_parameters(self, init_head_directional: bool = True) -> None:
        self._init_offset_projection(init_head_directional=init_head_directional)
        value_conv = self._value_proj[1] if isinstance(self._value_proj, nn.Sequential) else self._value_proj
        nn.init.xavier_uniform_(value_conv.weight.data)
        nn.init.xavier_uniform_(self._output_proj.weight.data)

    def _prepare_ref_points_scale(self, ref_points: torch.Tensor) -> torch.Tensor:
        if len(ref_points.shape) == 3:
            return ref_points.unsqueeze(2).repeat(self._n_heads, 1, self._n_points, 1)
        if len(ref_points.shape) == 4:
            return ref_points.repeat(self._n_heads, 1, 1, 1)
        raise NotImplementedError

    def forward(
        self, latent: torch.Tensor, ref_points: torch.Tensor, spatial_data_scales: torch.Tensor
    ) -> torch.Tensor:
        offset_and_attn = self._offset_attn_proj(latent)
        sampling_offsets = einops.rearrange(
            offset_and_attn[..., : self._offset_split_index],
            "b n (ah s p t) -> s (b ah) n p t",
            ah=self._n_heads,
            s=self._n_scales,
            t=2,
        )

        B, N, S, _ = ref_points.shape if len(ref_points.shape) == 4 else ref_points.shape[:-1]
        C = self._offset_attn_proj.in_features

        return self._output_proj(
            (
                einops.rearrange(
                    torch.stack(
                        [
                            F.grid_sample(
                                input=self._value_proj(sds).reshape(
                                    B * self._n_heads,
                                    C // self._n_heads,
                                    sds.shape[2],
                                    sds.shape[3],
                                ),
                                grid=(
                                    self._prepare_ref_points_scale(ref_points[:, :, i, :])
                                    + 2
                                    * sampling_offsets[i]
                                    / torch.tensor(
                                        (sds.shape[3], sds.shape[2]),
                                        dtype=sampling_offsets.dtype,
                                        device=sampling_offsets.device,
                                    ).view(1, 1, 1, 2)
                                ),
                                mode="bilinear",
                                padding_mode="zeros",
                                align_corners=self._use_align_corners,
                            )
                            for i, sds in enumerate(spatial_data_scales)
                        ],
                        dim=0,
                    ),
                    "s (b ah) hc n p -> b n (s p) ah hc",
                    ah=self._n_heads,
                )
                * F.softmax(
                    offset_and_attn[..., self._offset_split_index :].view(
                        B, N, self._n_scales * self._n_points, self._n_heads
                    ),
                    dim=2,
                ).unsqueeze(-1)
            )
            .sum(dim=2)
            .view(B, N, C)
        )


class MSDeformCrossAttnBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_data: int,
        num_points: int,
        num_heads: int,
        num_scales: int,
        init_head_directional: bool = False,
        align_corners: bool = False,
        ff_mult: int = 4,
        activation=nn.ReLU(inplace=True),
        prenorm_spatial_data: bool = False,
    ):
        super().__init__()

        self._q_norm = nn.LayerNorm(dim)
        self._ffn_norm = nn.LayerNorm(dim)
        self._deform_attn = MSDeformAttn(
            dim=dim,
            dim_data=dim_data,
            num_points=num_points,
            num_heads=num_heads,
            num_scales=num_scales,
            init_head_directional=init_head_directional,
            align_corners=align_corners,
            prenorm_spatial_data=prenorm_spatial_data,
        )
        self._ffn = MLP(
            input_dim=dim,
            hidden_dim=ff_mult * dim,
            output_dim=dim,
            num_layers=2,
            activation=activation,
        )

    def forward(
        self,
        latent: torch.Tensor,
        ref_points: torch.Tensor,
        spatial_data_scales: List[torch.Tensor],
    ) -> torch.Tensor:
        latent = latent + self._deform_attn(self._q_norm(latent), ref_points, spatial_data_scales)
        return latent + self._ffn(self._ffn_norm(latent))


def _grid_ref_points(spatial_data: torch.Tensor, num_scales: int = 1) -> torch.Tensor:
    B, _, H, W = spatial_data.shape
    ref = torch.stack(
        torch.meshgrid(
            (torch.arange(H, dtype=torch.float32) + 0.5) / H,
            (torch.arange(W, dtype=torch.float32) + 0.5) / W,
            indexing="ij",
        ),
        dim=-1,
    ).to(device=spatial_data.device)
    ref = ref.flip(-1)
    return (ref * 2 - 1).view(1, H * W, 1, 2).expand(B, H * W, num_scales, 2)


class MSDeformSelfAttnBlockOneToAll(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self._attn_block = MSDeformCrossAttnBlock(dim=dim, dim_data=dim, align_corners=False, **kwargs)

    def forward(self, spatial_data_scales: List[torch.Tensor], self_index: int) -> List[torch.Tensor]:
        spatial_latent = spatial_data_scales[self_index]
        ref_points = _grid_ref_points(spatial_latent, num_scales=len(spatial_data_scales))
        _, _, H, W = spatial_latent.shape
        spatial_latent = einops.rearrange(spatial_latent, "b c h w -> b (h w) c")
        spatial_latent = self._attn_block(
            latent=spatial_latent, ref_points=ref_points, spatial_data_scales=spatial_data_scales
        )
        spatial_data_scales[self_index] = einops.rearrange(spatial_latent, "b (h w) c -> b c h w", h=H, w=W)
        return spatial_data_scales


class MSDeformSelfAttnBlockAllToAll(MSDeformSelfAttnBlockOneToAll):
    def forward(self, spatial_data_scales: List[torch.Tensor]) -> List[torch.Tensor]:
        hws = [(sds.shape[2], sds.shape[3]) for sds in spatial_data_scales]
        sizes = [h * w for h, w in hws]
        split_indices = [0] + [sum(sizes[: i + 1]) for i in range(len(sizes))]
        split_indices = list(zip(split_indices[:-1], split_indices[1:]))
        spatial_latent = torch.cat(
            [einops.rearrange(sds, "b c h w -> b (h w) c") for sds in spatial_data_scales], dim=1
        )
        ref_points = torch.cat(
            [_grid_ref_points(sds, num_scales=len(spatial_data_scales)) for sds in spatial_data_scales],
            dim=1,
        )
        spatial_latent = self._attn_block(
            latent=spatial_latent, ref_points=ref_points, spatial_data_scales=spatial_data_scales
        )
        return [
            einops.rearrange(spatial_latent[:, start:end, :], "b (h w) c -> b c h w", w=w, h=h)
            for (start, end), (h, w) in zip(split_indices, hws)
        ]


