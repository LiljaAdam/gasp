import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.profiler import record_function

from gasp.common.types import PointCloudRange
from gasp.model.components.checkpointing import checkpoint

MIN_TIME, MAX_TIME = -3.0, 3.0  # current max future time is 3 seconds


def interpolate_grid_features(
    feats: torch.Tensor, query_xy_norm: torch.Tensor, fast_interpolation: bool = False
) -> torch.Tensor:
    """Interpolate a feature map based on the query points.

    Args:
        feats: The feature map to interpolate (bs x c x h x w)
        query_xt: The xy coords to interpolate using (bs, n, 2)
    """
    if fast_interpolation:
        feats = feats.to(memory_format=torch.channels_last).contiguous()

    # flip x-y to y-x for grid_sample
    query_xy_norm = query_xy_norm.flip(-1).unsqueeze(-2)
    interp_feats = nn.functional.grid_sample(
        feats, query_xy_norm, mode="bilinear", padding_mode="border", align_corners=False
    ).squeeze(-1)

    return interp_feats


class UnoResLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CoordinateEncoder(nn.Module):
    def __init__(
        self,
        coordinate_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_freqs: int,
        min_freq: float = 1.0,
        max_freq: float = 8.0,
    ):
        super(CoordinateEncoder, self).__init__()

        self._coordinate_dim = coordinate_dim
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        self._num_freqs = num_freqs

        _freqs_log = torch.linspace(math.log(min_freq), math.log(max_freq), num_freqs // 2)
        _freqs = torch.exp(_freqs_log)

        self.register_buffer(
            "_freqs",
            _freqs,
        )

        self.mlp = nn.Sequential(
            nn.Linear(coordinate_dim * num_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @record_function("CoordinateEncoder.forward")
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        encoded = self.sinusoidal_embedding(coordinates)  # B, N, C
        B, N, _ = encoded.shape
        # Reshape, apply MLP, then restore dimensions
        encoded = encoded.view(-1, encoded.size(-1))  # (B*N), C
        encoded = self.mlp(encoded)  # (B*N), out_dim
        return encoded.view(B, N, -1)  # B, N, out_dim

    def sinusoidal_embedding(self, coordinates: torch.Tensor) -> torch.Tensor:
        # coordinates: B, N, C
        # B, N, C
        B, N, _ = coordinates.size()
        # B, N, C, sinusoide_dim // 2
        sin = torch.einsum("bnc,d->bncd", coordinates, self._freqs)
        # B, N, C, sinusoide_dim
        sin = torch.cat([sin.sin(), sin.cos()], dim=-1)
        # B, N, C * sinusoide_dim
        return sin.view(B, N, -1)

    def fast_sinusoidal_embedding(self, coordinates: torch.Tensor) -> torch.Tensor:
        frequencies = self._freqs[None, None, None, :]  # (1, 1, 1, F)
        coordinates = coordinates[..., None]  # (B, N, C, 1)
        sinusoid = coordinates * frequencies  # (B, N, C, F)
        sinusoid = torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)  # (B, N, C, 2F)
        sinusoid = sinusoid.flatten(-2)  # (B, N, C * 2F)
        return sinusoid


class UnoOffsetPrediction(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        offset_dim: int,
        pc_dim: int,
        num_offsets: int = 1,
        tanh_offsets: bool = False,
        tanh_offsets_scale: float = 1.0,
    ):
        super().__init__()
        self.offset_dim = offset_dim
        self.feature_dim = feature_dim
        self.pc_dim = pc_dim
        self.num_offsets = num_offsets
        self.coords_linear = nn.Linear(self.feature_dim, offset_dim)
        self.feats_linear = nn.Linear(self.feature_dim, offset_dim)
        self.offset_head = nn.Linear(offset_dim, 2 * self.num_offsets)  # xy
        self.residual_layer = UnoResLayer(offset_dim)
        self.relu = nn.ReLU(inplace=True)

        if tanh_offsets:
            self.tanh = nn.Tanh()
        else:
            self.tanh = None

        self.tanh_offsets_scale = tanh_offsets_scale

        # custom init on the offset_head as per the paper:
        nn.init.normal_(self.offset_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.offset_head.bias, 0.0)

    def forward(self, query_feats: torch.Tensor, encoded_query_points: torch.Tensor):
        bs, num_queries, _ = encoded_query_points.shape

        lp_feats = self.feats_linear(query_feats.transpose(1, 2))
        lp_coords = self.coords_linear(encoded_query_points)

        x = lp_feats + lp_coords
        x = self.residual_layer(x)
        x = self.relu(x)
        offsets = self.offset_head(x)
        offsets = offsets.reshape(bs, num_queries, self.num_offsets, 2)

        if self.tanh is not None:
            offsets = self.tanh(offsets) * self.tanh_offsets_scale

        return offsets


class UnoDecoder(nn.Module):
    def __init__(
        self,
        point_cloud_range: PointCloudRange,
        in_feature_dim: int = 128,
        hidden_dim: int = 16,
        offset_dim: int = 16,
        pc_dim: int = 4,
        n_occ_resblocks: int = 3,
        num_offsets: int = 1,
        output_dim: int = 1,
        fast_interpolation: bool = False,
        gradient_checkpointing: bool = False,
        tanh_offsets: bool = False,
        tanh_offsets_scale: float = 1.0,
        coordinate_encoder_max_freq: float = 8.0,
        chunking_by_pre_allocate: bool = True,
    ):
        super().__init__()
        self.offset_dim = offset_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = in_feature_dim
        self.pc_dim = pc_dim
        self.output_dim = output_dim
        self.n_occ_resblocks = n_occ_resblocks
        self.num_offsets = num_offsets
        self.fast_interpolation = fast_interpolation
        self.gradient_checkpointing = gradient_checkpointing
        self.chunking_by_pre_allocate = chunking_by_pre_allocate

        self.point_cloud_range = torch.nn.Parameter(
            torch.tensor(
                [*point_cloud_range[:3], MIN_TIME, *point_cloud_range[3:], MAX_TIME],
                dtype=torch.float32,
            ),
            requires_grad=False,
        )

        self.coords_encoder = CoordinateEncoder(
            pc_dim, in_feature_dim, in_feature_dim, num_freqs=16, max_freq=coordinate_encoder_max_freq
        )

        if num_offsets > 0:
            self.offset_predictor = UnoOffsetPrediction(
                in_feature_dim, offset_dim, pc_dim, num_offsets, tanh_offsets, tanh_offsets_scale
            )

        # occupancy prediction
        self.occ_point_lin = nn.Linear(in_feature_dim, hidden_dim)
        self.occ_feat_lins = nn.ModuleList(
            [nn.Linear((1 + self.num_offsets) * self.feature_dim, hidden_dim) for _ in range(n_occ_resblocks)]
        )
        self.occ_pred_head = nn.Linear(self.hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

        self.build_block(n_occ_resblocks)

    def build_block(self, n_blocks: int):
        self.occ_resblocks = nn.ModuleList([UnoResLayer(self.hidden_dim) for _ in range(n_blocks)])

    def _normalize_queries(self, query_points: torch.Tensor) -> torch.Tensor:
        # normalize the query points (x,y,z,t) to [-1, 1]
        lower, upper = self.point_cloud_range[: self.pc_dim], self.point_cloud_range[4 : 4 + self.pc_dim]
        query_points = 2 * (query_points - lower) / (upper - lower) - 1
        return query_points

    def _fetch_features(self, bev_feats: torch.Tensor, query_points: torch.Tensor):
        bs, num_queries, _ = query_points.shape

        # Encode query points
        encoded_query_points = self.coords_encoder(query_points)
        # Get query features
        query_xy = query_points[..., :2].contiguous()
        del query_points

        query_feats = interpolate_grid_features(bev_feats, query_xy, self.fast_interpolation)

        combined_query_feats_parts = [query_feats]

        if self.num_offsets > 0:
            new_query_xy = query_xy.unsqueeze(2).expand(-1, -1, self.num_offsets, -1)
            del query_xy
            # Get offset features
            offsets_coords = self.offset_predictor(query_feats, encoded_query_points)  # bs, N, num_offsets, 2
            new_query_xy = new_query_xy + offsets_coords
            del query_feats

            new_query_xy = new_query_xy.reshape(bs, num_queries * self.num_offsets, 2)
            offset_feats = interpolate_grid_features(bev_feats, new_query_xy, self.fast_interpolation)
            del new_query_xy
            offset_feats = offset_feats.reshape(bs, self.feature_dim, num_queries, self.num_offsets)
            offset_feats = offset_feats.transpose(2, 3)  # bs, c, num_offsets, N
            offset_feats = offset_feats.reshape(bs, self.feature_dim * self.num_offsets, num_queries)

            combined_query_feats_parts.append(offset_feats)
            del offset_feats
        else:
            offsets_coords = None

        combined_query_feats = torch.cat(combined_query_feats_parts, dim=1).transpose(1, 2)

        return combined_query_feats, offsets_coords, encoded_query_points

    def _resblocks_forward(self, x: torch.Tensor, combined_query_feats: torch.Tensor) -> torch.Tensor:
        occ_feats_all = [lin(combined_query_feats) for lin in self.occ_feat_lins]
        for i in range(self.n_occ_resblocks):
            x = x + occ_feats_all[i]
            x = self.occ_resblocks[i](x)
        return x

    def forward(
        self, bev_feats: torch.Tensor, query_points: torch.Tensor, max_queries: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        query_points = self._normalize_queries(query_points)

        # get features from the BEV using deformable attention
        if self.gradient_checkpointing:
            combined_query_feats, offsets_coords, encoded_query_points = checkpoint(
                self._fetch_features, bev_feats, query_points, use_reentrant=False
            )
        else:
            combined_query_feats, offsets_coords, encoded_query_points = self._fetch_features(bev_feats, query_points)
        del query_points

        # process combined features
        x = self.occ_point_lin(encoded_query_points)
        del encoded_query_points
        if self.gradient_checkpointing:
            x = checkpoint(self._resblocks_forward, x, combined_query_feats, use_reentrant=False)
        else:
            x = self._resblocks_forward(x, combined_query_feats)

        # predict occupancy
        occupancy = self.occ_pred_head(self.relu(x))

        return {
            "prediction": occupancy,
            "offsets_coords": offsets_coords,
        }

    def forward_chunks(
        self, bev_feats: torch.Tensor, query_points: torch.Tensor, max_queries: int, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for chunks of queries."""
        # In training (grad-enabled), avoid pre-allocation with in-place writes, which
        # severs the autograd graph. Fall back to list+cat accumulation.
        use_prealloc = self.chunking_by_pre_allocate and (not torch.is_grad_enabled())
        if use_prealloc:
            outs = {
                "prediction": torch.zeros(
                    (query_points.shape[0], query_points.shape[1], self.output_dim),
                    device=bev_feats.device,
                    dtype=bev_feats.dtype,
                ),
                "offsets_coords": torch.zeros(
                    (query_points.shape[0], query_points.shape[1], self.num_offsets, 2),
                    device=bev_feats.device,
                    dtype=bev_feats.dtype,
                ),
            }
            for i in range(0, query_points.shape[1], max_queries):
                end_idx = min(i + max_queries, query_points.shape[1])
                query_points_chunk = query_points[:, i:end_idx, :]
                out = self.forward(bev_feats, query_points_chunk)
                for k, v in out.items():
                    if v is None:
                        continue
                    outs[k][:, i:end_idx, :] = v

        else:
            chunked_results = {
                "prediction": [],
                "offsets_coords": [],
            }

            for i in range(0, query_points.shape[1], max_queries):
                end_idx = min(i + max_queries, query_points.shape[1])
                query_points_chunk = query_points[:, i:end_idx, :]

                out = self.forward(bev_feats, query_points_chunk)

                for k, v in out.items():
                    if k in chunked_results:
                        chunked_results[k].append(v)

            outs = {}
            for k, v_list in chunked_results.items():
                if any(v is None for v in v_list):
                    outs[k] = None
                    continue
                outs[k] = torch.cat(v_list, dim=1)
                if torch.isnan(outs[k]).any() or torch.isinf(outs[k]).any():
                    print("NaNs in prediction!")

        return outs


if __name__ == "__main__":
    # Test UnoDecoder
    decoder = UnoDecoder(
        (-50.0, -50.0, -5.0, 50.0, 50.0, 5.0),
        in_feature_dim=128,
        hidden_dim=16,
        pc_dim=4,
        n_occ_resblocks=3,
        output_dim=1,
    )
    print(sum(p.numel() for p in decoder.parameters() if p.requires_grad))
