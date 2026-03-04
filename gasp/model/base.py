from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

import lightning as L
import torch
from torch.optim.lr_scheduler import OneCycleLR

from gasp.common.types import PointCloudRange, VoxelSize
from gasp.config.base_config import InstantiateConfig
from gasp.config.logging_configs import LoggingConfig
from gasp.data.datasets.zod_base import ZODBaseDatasetConfig
from gasp.model.components.gasp import Gasp, GaspConfig
from gasp.optimizer.cAdam import CautiousAdamW


def warmup_dataloader(dl, num_batches=5):
    it = iter(dl)
    for _ in range(num_batches):
        try:
            next(it)
        except StopIteration:
            break


DEFAULT_PCD_RANGE_UNO = (-100.04, -100, -2, 150.04, 100, 6)
DEFAULT_PCD_RANGE_SURROUND = (-69.12, -69.12, -2, 69.12, 69.12, 6)
DEFAULT_TASK_RANGE_UNO = (-40.0, -40.0, 0.0, 40.0, 40.0, 3.0)


def pcd_range_defaults(defaults: str):
    if defaults == "uno":
        return DEFAULT_PCD_RANGE_UNO
    elif defaults == "surround":
        return DEFAULT_PCD_RANGE_SURROUND
    else:
        raise ValueError(f"Invalid pcd range defaults: {defaults}")


def lifter_canvas_size(pc_range, voxel_size) -> Tuple[int, int, int]:
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    canvas_x = int(x_range / (voxel_size[0] * 2))
    canvas_y = int(y_range / (voxel_size[1] * 2))
    canvas_z = int(6)
    return (canvas_x, canvas_y, canvas_z)


def bev_features_canvas_size(pc_range, voxel_size) -> Tuple[int, int]:
    x_range = pc_range[3] - pc_range[0]
    y_range = pc_range[4] - pc_range[1]
    canvas_x = int(x_range / (voxel_size[0] * 2) / 2)
    canvas_y = int(y_range / (voxel_size[1] * 2) / 2)
    return (canvas_x, canvas_y)


def get_task_range_defaults(defaults: str):
    if defaults == "uno":
        return DEFAULT_TASK_RANGE_UNO
    else:
        raise ValueError(f"Invalid task range defaults: {defaults}")


def get_task_canvas_size(tasks_region_of_interest, tasks_cell_size):
    num_x = int((tasks_region_of_interest[3] - tasks_region_of_interest[0]) / tasks_cell_size)
    num_y = int((tasks_region_of_interest[4] - tasks_region_of_interest[1]) / tasks_cell_size)
    return (num_x, num_y)


@dataclass
class ModelConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: BaseModel)

    point_cloud_range: PointCloudRange = DEFAULT_PCD_RANGE_SURROUND
    """The range of the point cloud (x_min, y_min, z_min, x_max, y_max, z_max)."""
    voxel_size: VoxelSize = (0.16, 0.16, 8.0)
    """The size of the voxels in the point cloud (x, y, z)."""
    roi: str = "surround"
    """The default ROI to use for the model."""

    gasp_model: GaspConfig = field(default_factory=lambda: GaspConfig())
    """The GASP model configuration."""

    # tasks settings
    tasks_region_of_interest: PointCloudRange = field(init=False)  # will be set in post_init
    """ The region of interest for the tasks. If applicable. """
    tasks_cell_size: float = 0.3
    """ The cell size for the tasks. If applicable. """
    tasks_roi: str = "uno"
    """ The region of interest for the tasks. If applicable. """
    recall_at_precision: float = 0.7
    """ The precision to use for recall at precision """

    # optimizer settings
    max_lr: float = 5e-5
    pct_start: float = 0.02
    div_factor: float = 10
    final_div_factor: float = 1e4
    optimizer_cls: Literal["adam", "adamw", "cadam"] = "adamw"
    weight_decay: float = 1e-4
    fused_optimizer: bool = False
    image_backbone_reduce_lr_factor: float = 100.0

    # logging
    compute_train_metrics: bool = False
    """ Whether to compute train metrics """
    logging: LoggingConfig = field(default_factory=lambda: LoggingConfig())

    _min_query_time: float = -3.0
    """ The minimum query time """
    _max_query_time: float = 3.0
    """ The maximum query time """

    def __post_init__(self) -> None:
        self.point_cloud_range = pcd_range_defaults(self.roi)
        self.gasp_model.lifter_canvas_size = lifter_canvas_size(self.point_cloud_range, self.voxel_size)
        self.gasp_model.bev_features_canvas_size = bev_features_canvas_size(self.point_cloud_range, self.voxel_size)
        self.tasks_region_of_interest = get_task_range_defaults(self.tasks_roi)
        self.tasks_canvas_size = get_task_canvas_size(self.tasks_region_of_interest, self.tasks_cell_size)

    def configure_dataset(self, dataset: ZODBaseDatasetConfig):  # pragma: no cover
        dataset.point_cloud_range = self.point_cloud_range
        if self.gasp_model.lidar_encoder:
            dataset.load_lidar = True
        dataset.tasks_region_of_interest = self.tasks_region_of_interest
        dataset.tasks_canvas_size = self.tasks_canvas_size


class BaseModel(L.LightningModule):
    config: ModelConfig

    def __init__(self, config: ModelConfig, visualization_mode: bool = False):
        super().__init__()
        self.config = config
        self.visualization_mode = visualization_mode
        self.gasp_model: Gasp = config.gasp_model.setup(
            point_cloud_range=config.point_cloud_range,
            voxel_size=config.voxel_size[:2],
            logging_config=config.logging,
            min_query_time=config._min_query_time,
            max_query_time=config._max_query_time,
        )

    @property
    def _compute_metrics(self):
        if self.config.compute_train_metrics and self.training:
            return True
        elif not self.training:
            return True
        else:
            return False

    def on_train_start(self) -> None:
        # Only rank 0 does the warmup to avoid duplicate I/O
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print("\n🧊 Warming up dataloaders.\n")
            warmup_dataloader(self.trainer.train_dataloader)
            print("🔥 Warmed up dataloaders.\n")

        # Synchronize all ranks before training starts
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def configure_optimizers(self):
        if self.config.optimizer_cls not in ("adam", "cadam", "adamw"):
            raise ValueError(f"Invalid optimizer class: {self.config.optimizer_cls}")

        if self.config.optimizer_cls == "cadam":
            optimizer_cls = CautiousAdamW
        elif self.config.optimizer_cls == "adam":
            optimizer_cls = torch.optim.Adam  # type: ignore
        elif self.config.optimizer_cls == "adamw":
            optimizer_cls = torch.optim.AdamW  # type: ignore
        else:
            raise ValueError(f"Invalid optimizer class: {self.config.optimizer_cls}")

        main_lr = self.config.max_lr

        param_groups = [{"params": list(self.parameters()), "lr": main_lr}]
        print(f"Using main LR ({main_lr:.2e}) for all parameters.")

        if self.config.optimizer_cls == "cadam":
            optimizer = optimizer_cls(param_groups, weight_decay=self.config.weight_decay)
        else:
            optimizer = optimizer_cls(
                param_groups, weight_decay=self.config.weight_decay, fused=self.config.fused_optimizer
            )

        scheduler = OneCycleLR(
            optimizer,
            max_lr=main_lr,
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=self.config.pct_start,
            div_factor=self.config.div_factor,
            anneal_strategy="cos",
            final_div_factor=self.config.final_div_factor,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        if hasattr(self.trainer.datamodule, "set_epoch_for_sampler"):  # type: ignore
            self.trainer.datamodule.set_epoch_for_sampler(self.current_epoch)  # type: ignore

    def _get_lidar_data(self, batch: dict) -> Optional[Dict[str, torch.Tensor]]:
        if "lidars" in batch:
            lidar_data = {
                "pointcloud": batch["lidars"],
            }
            return lidar_data
        else:
            return {}

    def _get_probe_queries(
        self, delta_x, delta_y, delta_z, ranges, dims=3, time=0.0
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        x_min, y_min, z_min, x_max, y_max, z_max = ranges

        if dims == 4:
            x = torch.arange(x_min, x_max + delta_x, delta_x, device=self.device)
            y = torch.arange(y_min, y_max + delta_y, delta_y, device=self.device)
            z = torch.arange(z_min, z_max + delta_z, delta_z, device=self.device)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
            ts = torch.ones(xx.numel(), device=self.device) * time
            queries = torch.stack((xx.flatten(), yy.flatten(), zz.flatten(), ts), dim=-1).unsqueeze(0)
            num_samples_per_dim = (len(x), len(y), len(z))
        elif dims == 3:
            x = torch.arange(x_min, x_max + delta_x, delta_x, device=self.device)
            y = torch.arange(y_min, y_max + delta_y, delta_y, device=self.device)
            z = torch.arange(z_min, z_max + delta_z, delta_z, device=self.device)
            xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
            queries = torch.stack((xx.flatten(), yy.flatten(), zz.flatten()), dim=-1).unsqueeze(0)
            num_samples_per_dim = (len(x), len(y), len(z))
        elif dims == 2:
            x = torch.arange(x_min, x_max + delta_x, delta_x, device=self.device)
            y = torch.arange(y_min, y_max + delta_y, delta_y, device=self.device)
            xx, yy = torch.meshgrid(x, y, indexing="ij")
            queries = torch.stack((xx.flatten(), yy.flatten()), dim=-1).unsqueeze(0)
            num_samples_per_dim = (len(x), len(y))
        else:
            raise ValueError(f"Invalid dimensions: {dims}")

        return queries, num_samples_per_dim
