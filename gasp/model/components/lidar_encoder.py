from typing import List, Literal, Optional, Tuple, Union

import torch
from torch import nn
from torch.profiler import record_function

from gasp.common.types import PointCloudRange, VoxelSize
from gasp.config.logging_configs import LoggingConfig
from gasp.model.components.checkpointing import checkpoint
from gasp.model.components.decoders.uno_decoder import CoordinateEncoder
from gasp.model.components.pointpillars_encoder import PillarLayer
from gasp.model.components.positional_encoder import _frequencies

tcnn = None


class MLP(nn.Module):
    """A simple MLP network with linear layers and batchnorm and relu activation."""

    def __init__(self, in_dim: int, out_dim: int, n_layers: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class TinyCudaPointEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feature_dim: int,
        num_freqs: int,
    ):
        super().__init__()

        self.encoding = tcnn.Encoding(
            n_input_dims=in_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": num_freqs // 2,
            },
        )

        self.network = tcnn.Network(
            n_input_dims=in_dim * num_freqs,
            n_output_dims=out_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": feature_dim,
                "n_hidden_layers": 3,
            },
        )

    @record_function("TinyCudaPointEncoder.forward")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        dtype = x.dtype
        x = x.view(-1, x.shape[-1])  # Flatten the input
        x = self.encoding(x)  # Apply encoding
        x = self.network(x)  # Apply the MLP
        x = x.view(batch_size, num_points, -1)  # Reshape back to original shape
        return x.to(dtype=dtype)  # Ensure the output has the same dtype as the input


class UnoVoxelizer(nn.Module):
    def __init__(
        self,
        voxel_size: VoxelSize,
        point_cloud_range: PointCloudRange,
        feature_dim: int = 64,
        input_dim: int = 5,  # x,y,z,intensity,timestamp
        points_positional_encoding: Literal["none", "sinusoidal", "tcnn"] = "sinusoidal",
        initial_features: Literal["zeros", "sinusoidal"] = "zeros",
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # we only consider 2D BEV voxelization
        self.voxel_size = torch.nn.Parameter(torch.tensor(voxel_size[:2], dtype=torch.float32), requires_grad=False)
        self.point_cloud_lb = torch.nn.Parameter(
            torch.tensor(point_cloud_range[:2], dtype=torch.float32), requires_grad=False
        )
        self.point_cloud_ub = torch.nn.Parameter(
            torch.tensor(point_cloud_range[3:5], dtype=torch.float32), requires_grad=False
        )
        self.voxel_grid_size = torch.nn.Parameter(
            ((self.point_cloud_ub - self.point_cloud_lb) / self.voxel_size).long(), requires_grad=False
        )  # 2

        if points_positional_encoding == "sinusoidal":
            self.mlp = CoordinateEncoder(input_dim, feature_dim, feature_dim, num_freqs=16)
        elif points_positional_encoding == "tcnn":
            self.mlp = TinyCudaPointEncoder(input_dim, feature_dim, feature_dim, num_freqs=16)
        else:
            self.mlp = nn.Linear(input_dim, feature_dim)

        self._initialize_features(initial_features)

    def _batched_forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        points: B x N x C
        return B x F x H x W
        """
        bs, N, C = points.shape
        feats = self.mlp(points)  # .float()  # NxC
        # feats = feats.view(bs, N, self.feature_dim)
        # voxelize time. we discretize the points in bev space into voxels,
        # and then aggregate the points in each voxel into a single feature vector by summing them up.
        xy = points[..., :2].contiguous()  # Nx2
        # find the voxel index for each point
        voxel_idx = ((xy - self.point_cloud_lb) / self.voxel_size).long()  # BxNx2
        # clamp the voxel index to the valid range,
        voxel_idx = torch.clamp(voxel_idx, torch.zeros_like(self.voxel_grid_size), self.voxel_grid_size - 1)  # BxNx2
        # calculate the voxel index in the flattened voxel grid
        voxel_idx = voxel_idx[..., 0] * self.voxel_grid_size[1] + voxel_idx[..., 1]  # N
        # aggregate the features in the same voxel
        features = self._initial_features(bs, feats.dtype)

        voxel_idx = voxel_idx.unsqueeze(-1).repeat(1, 1, self.feature_dim)

        features = features.scatter_add_(1, voxel_idx, feats)
        features = features.view(bs, self.voxel_grid_size[0], self.voxel_grid_size[1], -1)  # HxWxC
        features = features.permute(0, 3, 1, 2).contiguous()  # BSxCxHxW

        return features

    def _initialize_features(self, initial_features: str):
        if initial_features == "sinusoidal":
            embed_dim = self.feature_dim
            axis_sizes: List = list([self.voxel_grid_size[0], self.voxel_grid_size[1]])
            max_freq = 10_000
            perceiver = False

            dim_offset = embed_dim % (2 * len(axis_sizes))
            fqs = _frequencies(embed_dim + dim_offset, len(axis_sizes), max_freq, perceiver).unsqueeze(
                0
            )  # (1, embed_dim/2*pos_dim)
            # Make positional encodings along each dimension.
            axis_pes = []
            for ax in axis_sizes:
                pos = torch.arange(ax, dtype=torch.float).unsqueeze(1)  # (pos, 1)
                pe = pos * fqs
                axis_pes.append(torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1))
            # Expand all the 1D positional encodings so they can be stacked along all dimensions.
            for i, _ in enumerate(axis_pes):
                repeats = axis_sizes.copy() + [1]
                repeats[i] = 1
                view_shape = [1 for _ in axis_sizes] + [axis_pes[i].shape[-1]]
                view_shape[i] = axis_sizes[i]
                axis_pes[i] = axis_pes[i].view(*view_shape).repeat(*repeats)
            # Stack and flatten instead of concatenate to interleave the dimensions. This allows a
            # clean dimensionality cutoff that does not favor any particular dimension.
            pe_full = torch.stack(axis_pes, dim=-1)  # (X1, X2, .., XN, embed_dim/N, N)
            pe_full = pe_full.flatten(start_dim=-2, end_dim=-1)
            # Now make a cutoff if needed.
            init_features = pe_full[..., :embed_dim].view(-1, self.feature_dim)

        elif initial_features == "zeros":
            init_features = torch.zeros(int(self.voxel_grid_size[0] * self.voxel_grid_size[1]), self.feature_dim)
        else:
            raise ValueError(f"Unknown initial features: {initial_features}")

        self.register_buffer("initial_features", init_features)

    def _initial_features(self, bs, dtype) -> torch.Tensor:
        if self.initial_features.dtype != dtype:
            self.initial_features = self.initial_features.to(dtype)
        return self.initial_features.unsqueeze(0).repeat(bs, 1, 1)

    def _looped_forward(self, batched_points: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for points in batched_points:
            feats = self._batched_forward(points.unsqueeze(0)).squeeze(0)
            out.append(feats)

        return torch.stack(out, dim=0)

    def _forward(self, batched_points: List[torch.Tensor]) -> torch.Tensor:
        out = []
        for points in batched_points:
            feats = self.mlp(points.unsqueeze(0)).squeeze(0)  # NxC
            # voxelize time. we discretize the points in bev space into voxels,
            # and then aggregate the points in each voxel into a single feature vector by summing them up.
            xy = points[:, :2]  # Nx2
            # find the voxel index for each point
            voxel_idx = ((xy - self.point_cloud_lb) / self.voxel_size).long()  # Nx2
            # clamp the voxel index to the valid range,
            voxel_idx = torch.clamp(voxel_idx, torch.zeros_like(self.voxel_grid_size), self.voxel_grid_size - 1)  # Nx2
            # calculate the voxel index in the flattened voxel grid
            voxel_idx = voxel_idx[:, 0] * self.voxel_grid_size[1] + voxel_idx[:, 1]  # N
            # aggregate the features in the same voxel
            features = torch.zeros(self.voxel_grid_size[0] * self.voxel_grid_size[1], feats.shape[1]).to(
                device=feats.device
            )
            features = features.index_add(0, voxel_idx, feats)
            features = features.view(self.voxel_grid_size[0], self.voxel_grid_size[1], -1)  # HxWxC
            features = features.permute(2, 0, 1).contiguous()  # CxHxW
            out.append(features)
        return torch.stack(out, dim=0)

    def forward(self, points: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(points, list):
            return self._looped_forward(points)
        return self._batched_forward(points)


class UnoVoxelizerCustomKernel(nn.Module):
    def __init__(
        self,
        voxel_size: VoxelSize,
        point_cloud_range: PointCloudRange,
        feature_dim: int = 64,
        input_dim: int = 5,  # x,y,z,intensity,timestamp
        points_positional_encoding: Literal["none", "sinusoidal", "tcnn"] = "sinusoidal",
        initial_features: Literal["zeros", "sinusoidal"] = "zeros",
        max_num_voxels: Tuple[int, int] = (16000, 40000),
        max_num_points: int = 16,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        assert initial_features == "zeros", "Only zeros are supported for now"

        self.pillar_layer = PillarLayer(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            max_num_points=max_num_points,
            max_voxels=max_num_voxels,
        )
        if points_positional_encoding == "sinusoidal":
            self.mlp = CoordinateEncoder(input_dim, feature_dim, feature_dim, num_freqs=16)
        elif points_positional_encoding == "tcnn":
            self.mlp = TinyCudaPointEncoder(input_dim, feature_dim, feature_dim, num_freqs=16)
        else:
            self.mlp = nn.Linear(input_dim, feature_dim)

        self.x_l = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
        self.y_l = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])

    def forward(self, points: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(points)

        # drop the coordinate information'
        # features = self.mlp(pillars.view(-1, pillars.shape[-1])).view(pillars.shape[0], pillars.shape[1], -1)
        features = self.mlp(pillars)
        # features = pillars[..., 3:]  # (p1 + p2 + ... + pb, num_points, c)

        # 4. find mask for (0, 0, 0) and update the encoded features
        # a very beautiful implementation
        voxel_ids = torch.arange(0, pillars.size(1), device=pillars.device)  # (num_points, )
        mask = voxel_ids[:, None] < npoints_per_pillar[None, :]  # (num_points, p1 + p2 + ... + pb)
        mask = mask.permute(1, 0).contiguous()  # (p1 + p2 + ... + pb, num_points)
        features *= mask[:, :, None]

        # 5. pooling features
        pooling_features = torch.sum(features, dim=1)  # (p1 + p2 + ... + pb, out_channels)

        # 6. pillar scatter
        batched_canvas = []
        bs = coors_batch[-1, 0] + 1
        for i in range(bs):
            cur_coors_idx = coors_batch[:, 0] == i
            cur_coors = coors_batch[cur_coors_idx, :]
            cur_features = pooling_features[cur_coors_idx]

            canvas = torch.zeros(
                (self.x_l, self.y_l, self.feature_dim),
                dtype=torch.float32,
                device=pillars.device,
            )
            canvas[cur_coors[:, 1], cur_coors[:, 2]] = cur_features
            canvas = canvas.permute(2, 0, 1).contiguous()
            batched_canvas.append(canvas)
        batched_canvas = torch.stack(batched_canvas, dim=0)  # (bs, c, x, y)
        return batched_canvas


class UnoLidarEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        point_cloud_range: PointCloudRange,
        voxel_size: VoxelSize,
        use_custom_cuda: bool = False,
        points_positional_encoding: Literal["none", "sinusoidal", "tcnn"] = "sinusoidal",
        voxelization_initial_features: Literal["zeros", "sinusoidal"] = "zeros",
        gradient_checkpointing: bool = False,
        swap_batch_norm_to_instance_norm: bool = False,
        logging_config: Optional[LoggingConfig] = None,
    ) -> None:
        """Initialize the UnoEncoder.
        feature_dim: The dimension of the feature vector.
        point_cloud_range: The range of the BEV grid. (x_min, y_min, zmin, x_max, y_max, zmax). TODO: Rename this?
        voxel_size: The size of the voxel. (x, y, z)
        """
        super().__init__()

        self.gradient_checkpointing = gradient_checkpointing

        voxelizer_cls = UnoVoxelizerCustomKernel if use_custom_cuda else UnoVoxelizer
        self.voxelizer = voxelizer_cls(
            feature_dim=feature_dim,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            points_positional_encoding=points_positional_encoding,
            initial_features=voxelization_initial_features,
        )
        self.feature_dim = feature_dim

        if swap_batch_norm_to_instance_norm:
            Norm2d = nn.InstanceNorm2d
        else:
            Norm2d = nn.BatchNorm2d

        self.init_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 2, padding=1),
            Norm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, padding=1),
            Norm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, padding=1),
            Norm2d(feature_dim),
            nn.ReLU(),
        )

    def forward(self, points: List[torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        if self.gradient_checkpointing:
            bev_features = checkpoint(
                self.voxelizer,
                points,
                use_reentrant=False,
            )
        else:
            bev_features = self.voxelizer(points)

        if self.gradient_checkpointing:
            bev_features = checkpoint(
                self.init_block,
                bev_features,
                use_reentrant=False,
            )
        else:
            bev_features = self.init_block(bev_features)

        return bev_features, {}
