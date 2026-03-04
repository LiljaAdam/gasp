from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor

from gasp.common.types import PointCloudRange, VoxelSize
from gasp.config.base_config import InstantiateConfig
from gasp.config.logging_configs import LoggingConfig
from gasp.model.components.decoders.uno_decoder import UnoDecoder
from gasp.model.components.lidar_encoder import UnoLidarEncoder
from gasp.model.components.uno_encoder import UnoBEVEncoder

PROBE_RANGE = (-2.0, -10, -0.5, 60, 10, 0.5)

MAX_VFM_WIDTH = 273  # pixels after stride
MAX_VFM_HEIGHT = 109


@dataclass
class GaspConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: Gasp)
    """The target class for instantiation."""

    compile: bool = False
    """Whether to compile the model with torch.compile()."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing."""
    swap_batch_norm_to_instance_norm: bool = False
    """Whether to swap batch norm to instance norm."""

    hidden_dim: int = 128

    # Encoders
    lidar_encoder: bool = True
    """ The lidar encoder to use """

    lidar_points_positional_encoding: Literal["none", "sinusoidal", "tcnn"] = "none"
    """ The type of positional encoding to use for the lidar points """
    voxelization_initial_features: Literal["zeros", "sinusoidal"] = "zeros"
    """ The type of initial features to use for the voxelization """

    lifter_canvas_size: Tuple[int, int, int] = field(init=False)  # will be set by the parent model
    """ The canvas size for the lifter """
    bev_features_canvas_size: Tuple[int, int] = field(init=False)  # will be set by the parent model
    """ The canvas size for the encoded BEV features """

    # BEV Processing
    bev_attention: Literal["none", "deformable_py"] = "deformable_py"
    """ Whether to use deformable attention in the encoder model """
    resblock_convs: Literal["dynamic", "standard"] = "dynamic"
    """ Whether to use dynamic or standard resblock convolutions in the encoder model """
    attention_layers: int = 6
    """ The number of attention layers in the encoder model """
    deformable_attention_normalize_before: bool = True
    """Whether to normalize before the deformable attention."""
    deformable_attention_use_positional_encoding: bool = True
    """Whether to use positional encoding in the deformable attention."""

    # Decoding
    fast_bev_feature_interpolation: bool = False
    """ Whether to use fast BEV feature interpolation """

    # Occupancy
    f_predict_occupancy: bool = True
    """ Whether to predict occupancy in the decoder model """
    force_init_occupancy_decoder: bool = False
    """ Whether to force the occupancy decoder to be initialized """

    self_attention_pre_resblocks: bool = False
    """ Whether to use self-attention before the resblocks in the occupancy decoder """
    self_attention_post_resblocks: bool = False
    """ Whether to use self-attention after the resblocks in the occupancy decoder """

    # Ego Path
    f_predict_ego_path: bool = False
    """ Whether to predict ego path in the decoder model """
    force_init_ego_path_decoder: bool = False
    """ Whether to force the ego path decoder to be initialized """
    ego_path_dimensions: int = 2
    """ The number of dimensions for the ego path """

    # VFM Features
    f_predict_vfm_features: bool = False
    """ Whether to predict vfm features in the decoder model """
    learnable_pe_features: bool = False
    """Wheter to use learnable Positional Encoding features"""
    vfm_feature_dimensions: int = 768
    """ The number of dimensions for the vfm features """
    vfm_feature_layers: Tuple[int, ...] = (12,)
    """ The layers to use for the vfm features """
    force_init_vfm_features_decoder: bool = False
    """ Whether to force the vfm features decoder to be initialized """

    @property
    def f_use_occupancy_decoder(self) -> bool:
        return self.f_predict_occupancy or self.f_predict_vfm_features

    @property
    def f_use_occupancy_sampler(self) -> bool:
        return self.f_use_occupancy_decoder

    @property
    def f_use_vfm_features_decoder(self) -> bool:
        return self.f_predict_vfm_features

    @property
    def f_use_vfm_features_sampler(self) -> bool:
        return self.f_use_vfm_features_decoder

    @property
    def f_use_ego_path_sampler(self) -> bool:
        return self.f_predict_ego_path 

    @property
    def relevant_feature_dim_and_layers(self) -> Tuple[int, Tuple[int, ...]]:
        if self.f_use_vfm_features_decoder:
            relevant_feature_dim = self.vfm_feature_dimensions
            relevant_feature_layers = self.vfm_feature_layers
        else:
            relevant_feature_dim = self.one_head.vfm_feature_dim
            relevant_feature_layers = self.one_head.vfm_feature_layers
        return relevant_feature_dim, relevant_feature_layers

class Gasp(torch.nn.Module):
    def __init__(
        self,
        config: GaspConfig,
        point_cloud_range: PointCloudRange,
        voxel_size: VoxelSize,
        logging_config: LoggingConfig,
        min_query_time: float,
        max_query_time: float,
    ):
        super().__init__()
        self.config = config
        self.logging_config = logging_config
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        if config.lidar_encoder:
            self.lidar_encoder = UnoLidarEncoder(
                feature_dim=config.hidden_dim,
                point_cloud_range=point_cloud_range,
                voxel_size=voxel_size,
                use_custom_cuda=False,
                points_positional_encoding=config.lidar_points_positional_encoding,
                voxelization_initial_features=config.voxelization_initial_features,
                gradient_checkpointing=config.gradient_checkpointing,
                swap_batch_norm_to_instance_norm=config.swap_batch_norm_to_instance_norm,
                logging_config=logging_config,
            )

        else:
            self.lidar_encoder = None

        self.bev_encoder = UnoBEVEncoder(
            feature_dim=config.hidden_dim,
            attention=config.bev_attention,
            resblock_convs=config.resblock_convs,
            attention_layers=config.attention_layers,
            gradient_checkpointing=config.gradient_checkpointing,
            swap_batch_norm_to_instance_norm=config.swap_batch_norm_to_instance_norm,
            normalize_before=config.deformable_attention_normalize_before,
            use_positional_encoding=config.deformable_attention_use_positional_encoding,
        )

        if config.f_use_occupancy_decoder or config.force_init_occupancy_decoder:
            self.occupancy_decoder = UnoDecoder(
                in_feature_dim=config.hidden_dim,
                point_cloud_range=point_cloud_range,
                hidden_dim=16,
                offset_dim=16,
                pc_dim=4,
                n_occ_resblocks=3,
                output_dim=1,
                fast_interpolation=config.fast_bev_feature_interpolation,
                gradient_checkpointing=config.gradient_checkpointing,
            )

        if config.f_predict_ego_path or config.force_init_ego_path_decoder:
            self.ego_path_decoder = UnoDecoder(
                in_feature_dim=config.hidden_dim,
                point_cloud_range=point_cloud_range,
                hidden_dim=16,
                offset_dim=16,
                pc_dim=self.config.ego_path_dimensions,
                n_occ_resblocks=3,
                fast_interpolation=config.fast_bev_feature_interpolation,
                gradient_checkpointing=config.gradient_checkpointing,
            )
        else:
            self.ego_path_decoder = None

        self.vfm_features_decoder = None
        self.pe_head = None
        self.pe_embeddings = None
        if config.f_use_vfm_features_decoder or config.force_init_vfm_features_decoder:
            self.vfm_features_decoder = UnoDecoder(
                in_feature_dim=config.hidden_dim,
                point_cloud_range=point_cloud_range,
                hidden_dim=16,
                offset_dim=16,
                pc_dim=4,
                n_occ_resblocks=3,
                output_dim=config.vfm_feature_dimensions,
                fast_interpolation=config.fast_bev_feature_interpolation,
                gradient_checkpointing=config.gradient_checkpointing,
            )
            if config.learnable_pe_features:
                # shamelessly copied from emernerf
                self.pe_embeddings = torch.nn.Parameter(
                    0.05 * torch.randn(1, config.vfm_feature_dimensions // 2, MAX_VFM_HEIGHT, MAX_VFM_WIDTH),
                    requires_grad=True,
                )
                self.pe_head = torch.nn.Sequential(
                    torch.nn.Linear(config.vfm_feature_dimensions // 2, config.vfm_feature_dimensions),
                )

            if config.compile:
                self.vfm_features_decoder = torch.compile(self.vfm_features_decoder)

        self.fusion = None

    @property
    def can_decode_occupancy(self) -> bool:
        return self.config.f_predict_occupancy

    def get_encoders(self) -> List[torch.nn.Module]:
        return [e for e in [self.lidar_encoder, self.bev_encoder] if e is not None]

    def get_decoders(self) -> List[torch.nn.Module]:
        return [
            d
            for d in [
                self.occupancy_decoder,
                self.ego_path_decoder,
                self.vfm_features_decoder,
                self.pe_head,
                self.pe_embeddings,
            ]
            if d is not None
        ]

    def forward(self, lidar: dict, camera: dict) -> Tuple[Tensor, dict]:
        return self.encode(lidar, camera)

    def encode(self, lidar: dict) -> Tuple[Tensor, dict]:
        # Encode sensors (lidar only)
        if self.lidar_encoder:
            encoding, info = self.lidar_encoder(format_point_cloud(lidar["pointcloud"]))
        else:
            raise ValueError("No encoder specified")

        # Process joint BE representation
        encoding, info = self._bev_encode(encoding, info)

        return encoding, info

    def _bev_encode(self, bev_features: Tensor, info: dict) -> Tuple[Tensor, dict]:
        return self.bev_encoder(bev_features), info

    def decode(
        self, encoding: Tensor, queries: Tensor, to_decode: list[str], max_queries: Optional[int] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        out = {}

        if "occupancy" in to_decode:
            out["occupancy"] = self.decode_occupancy(encoding, queries, **kwargs)["prediction"]
        if "vfm_features" in to_decode:
            out["vfm_features"] = self.decode_vfm_features(encoding, queries, **kwargs)["prediction"]
        if "ego_path" in to_decode:
            out["ego_path"] = self.decode_ego_path(encoding, queries)["prediction"]

        return out

    def decode_occupancy(self, encoding: Tensor, queries: Tensor) -> Dict[str, torch.Tensor]:
        assert self.occupancy_decoder is not None
        out = self.occupancy_decoder(encoding, queries)
        return out

    def decode_ego_path(self, encoding: Tensor, queries: Tensor) -> Tensor:
        assert self.ego_path_decoder is not None
        return self.ego_path_decoder(encoding, queries)["prediction"]

    def decode_vfm_features(
        self,
        encoding: Tensor,
        queries: Tensor,
        # note: these are only used if learnable_pe_features is True and during supervision
        pixel_xy: Optional[Tensor] = None,
        vfm_h: Optional[float] = None,
        vfm_w: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        assert self.vfm_features_decoder is not None
        vfm_feats = self.vfm_features_decoder(encoding, queries)
        if not self.config.learnable_pe_features or pixel_xy is None:
            return vfm_feats
        else:
            return self.add_positional_encoding(vfm_feats, pixel_xy, vfm_h, vfm_w)

    def add_positional_encoding(
        self, vfm_feats: Dict[str, Tensor], pixel_xy: Tensor, vfm_h: Optional[float], vfm_w: Optional[float]
    ) -> Dict[str, Tensor]:
        assert vfm_h is not None and vfm_w is not None
        offset_x = MAX_VFM_WIDTH / 2 - vfm_w / 2
        offset_y = MAX_VFM_HEIGHT / 2 - vfm_h / 2
        pixels_ = pixel_xy + torch.tensor([offset_x, offset_y], device=pixel_xy.device)
        norm_pixels = 2 * pixels_ / torch.tensor([MAX_VFM_WIDTH, MAX_VFM_HEIGHT], device=pixel_xy.device) - 1
        # flip as grid sample expects y, x
        norm_pixels = norm_pixels.flip(-1)
        pe = torch.nn.functional.grid_sample(
            self.pe_embeddings, norm_pixels.unsqueeze(1), align_corners=True, padding_mode="border"
        )
        pe = pe.squeeze(2)  # b c//2 n
        pe = self.pe_head(pe.permute(0, 2, 1))  # b n c
        vfm_feats["prediction"] = vfm_feats["prediction"] + pe
        return vfm_feats


def format_point_cloud(x):
    """Format point cloud to be used in the model"""
    # TODO: (willj) add support for multiple lidar scans; either by aggregation or by processing each scan separately
    if x[0].shape[0] > 1:
        raise NotImplementedError
    if isinstance(x, list):
        x = [pc[-1, :, :] for pc in x]
    else:  # tensor
        x = x[:, -1, :, :]
    return x
