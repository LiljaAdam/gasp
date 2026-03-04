import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Accuracy, F1Score, Precision, Recall

from gasp.common.types import PointCloudRange
from gasp.data.components.occupancy_sampler_gpu import LidarGPUOccupancySamplerConfig
from gasp.data.components.vfm_sampler_gpu import VFMFeaturesProjectionGPUSamplerConfig
from gasp.data.datasets.zod_base import ZODBaseDatasetConfig
from gasp.model.base import BaseModel, ModelConfig

# Suppress the specific deprecation warning about torch.cpu.amp.autocast
warnings.filterwarnings("ignore", message=r".*torch\.cpu\.amp\.autocast.*is deprecated.*")
DEFAULT_OCCUPANCY_PROBE_RANGE: PointCloudRange = (-69.12, -69.12, -2, 69.12, 69.12, 6)
DEFAULT_EGO_PATH_PROBE_RANGE: PointCloudRange = (-2.0, -50, -0.5, 69, 50, 0.5)
DEFAULT_DENSE_EGO_PATH_PROBE_RANGE: PointCloudRange = (-0.55, -4.55, -0.55, 20.5, 4.55, 0.55)
PROBE_OBJECTS_SHRINK = 0.2  # How much to shrink the object bounding boxes for the probe

EGO_VEHICLE_WIDTH = 1.3
EGO_VEHICLE_LENGTH = 4.0
EGO_VEHICLE_HEIGHT = 1.5


@dataclass
class GaspPreTrainConfig(ModelConfig):
    _target: type = field(default_factory=lambda: GaspPreTrain)

    freeze_encoder: bool = False
    """ Whether to freeze the encoder or not """

    occupancy_loss: Literal["bce", "focal"] = "bce"
    """ The loss function for the occupancy. (bce, l1, l2) """
    focal_loss_alpha: Union[int, float] = -1
    """ The alpha parameter for the focal loss """

    vfm_feature_loss: Literal["l1", "l2", "smoothl1", "cos"] = "l1"
    """ The loss function for the vfm features. """
    vfm_feature_smoothl1_beta: float = 0.1
    """ The beta parameter for the smooth l1 loss """

    weight_occupancy_loss: float = 1.0
    """ Weight of the occupancy loss """
    weight_ego_path_loss: float = 0.1
    """ Weight of the ego path loss """
    weight_vfm_features_loss: float = 0.5
    """ Weight of the vfm features loss """

    occupancy_sampler: LidarGPUOccupancySamplerConfig = field(default_factory=lambda: LidarGPUOccupancySamplerConfig())
    """ The occupancy sampler configuration """

    vfm_sampler: VFMFeaturesProjectionGPUSamplerConfig = field(
        default_factory=lambda: VFMFeaturesProjectionGPUSamplerConfig()
    )
    """ The vfm sampler configuration """

    rank_zero_only_logging: bool = True
    """ Whether to log only for rank zero or not """

    def __post_init__(self) -> None:
        super().__post_init__()

    def configure_dataset(self, dataset: ZODBaseDatasetConfig):  # pragma: no cover
        super().configure_dataset(dataset)
        dataset.load_offset_pointclouds = True
        load_cached_vfm_features = self.gasp_model.f_use_vfm_features_sampler and self.vfm_sampler.use_cached_features
        use_vfm_features_sampler = (
            self.gasp_model.f_use_vfm_features_sampler and not self.vfm_sampler.use_cached_features
        )
        dataset.load_cached_vfm_features = load_cached_vfm_features
        if use_vfm_features_sampler:
            dataset.load_vfm_stuff = True
        dataset.load_ego_path_samples = self.gasp_model.f_use_ego_path_sampler
        dataset.eps_cfg.query_dim = self.gasp_model.ego_path_dimensions

        relevant_feature_dim, relevant_feature_layers = self.gasp_model.relevant_feature_dim_and_layers

        dataset.cached_vfm_cfg._load_feature_dim = relevant_feature_dim

        self._min_query_time = -dataset.max_past_frame_s if dataset.num_past_frames_to_sample > 0 else -0.1
        self._max_query_time = dataset.max_future_frame_s if dataset.num_future_frames_to_sample > 0 else 0.1


class GaspPreTrain(BaseModel):
    config: GaspPreTrainConfig

    def __init__(self, config: GaspPreTrainConfig, visualization_mode: bool = False):
        super().__init__(config, visualization_mode=visualization_mode)

        if self.gasp_model.config.f_use_occupancy_sampler:
            self.occupancy_sampler = self.config.occupancy_sampler.setup(
                point_cloud_range=self.config.point_cloud_range
            )

        if self.gasp_model.config.f_use_vfm_features_sampler:
            relevant_feature_dim, relevant_feature_layers = self.gasp_model.config.relevant_feature_dim_and_layers
            self.vfm_sampler = self.config.vfm_sampler.setup(
                point_cloud_range=self.config.point_cloud_range,
                feature_dim=relevant_feature_dim,
                feature_layers=relevant_feature_layers,
            )

        if config.freeze_encoder:
            self.freeze_encoder()

        self._initialize_metrics()

    def freeze_encoder(self) -> None:
        for enc in self.gasp_model.get_encoders():
            for p in enc.parameters():
                p.requires_grad = False

    def _initialize_metrics(self):
        # Initialize occupancy metrics
        if self.gasp_model.config.f_use_occupancy_decoder:
            metrics = ["acc", "prec", "rec", "f1"]
            metric_classes = [Accuracy, Precision, Recall, F1Score]

            for split in ["train", "val"]:
                for metric, cls in zip(metrics, metric_classes):
                    setattr(self, f"{split}_occupancy_{metric}", cls(task="binary"))

        # Initialize ego path metrics
        if self.config.gasp_model.f_predict_ego_path:
            metrics = ["acc", "prec", "rec", "f1"]
            metric_classes = [Accuracy, Precision, Recall, F1Score]

            for split in ["train", "val"]:
                for metric, cls in zip(metrics, metric_classes):
                    setattr(self, f"{split}_ego_path_{metric}", cls(task="binary"))

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: GaspPreTrainConfig,
        strict=True,
        is_from_pretrain: bool = True,
        visualization_mode: bool = False,
    ) -> "GaspPreTrain":
        """Load the model from a checkpoint."""
        # This code assumes the checkpoint is from a Pretrained model, if we want to load from a BEVSegmentor model,
        # we should use the resume training flag
        del is_from_pretrain  # we dont use this here
        model: GaspPreTrain = config.setup(visualization_mode=visualization_mode)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = checkpoint["state_dict"]

        sdrep = {k.replace("gasp_model.", ""): v for k, v in sd.items() if k.startswith("gasp_model.")}
        sdrep = {k.replace("one_head_decoder.decoder", "one_head_decoder._decoder"): v for k, v in sdrep.items()}
        sdrep = {k: v for k, v in sdrep.items() if not k.startswith("one_head_decoder.rendering.decoder")}

        okay_to_miss = ["probe_vfm"]

        if strict:
            info = model.gasp_model.load_state_dict(sdrep, strict=False)
            missing_keys = [mk for mk in info.missing_keys if not any(ok in mk for ok in okay_to_miss)]
            unexpected_keys = info.unexpected_keys
            if len(missing_keys) > 0 or len(unexpected_keys) > 0:
                raise ValueError(f"Missing keys: {missing_keys}, Unexpected keys: {unexpected_keys}")

        else:
            model.gasp_model.load_state_dict(sdrep, strict=False)

        return model

    def _get_occupancy(
        self, occ_queries: Tensor, occ_labels: Tensor, bev_feats: Tensor
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        occupancy_out = self.gasp_model.decode_occupancy(bev_feats, occ_queries)
        occupancy_y = occupancy_out["prediction"]
        occupancy_loss = self._get_occupancy_loss(occupancy_y, occ_labels)
        return occupancy_out, occupancy_loss

    def _get_ego_path(self, batch: dict, bev_feats: Tensor) -> Tuple[Tensor, Tensor]:
        ego_path_queries = batch["ego_path_queries"]
        ego_path_labels = batch["ego_path_label"]
        ego_path_valid = batch["ego_path_valid"]

        y_ego_path = self.gasp_model.decode_ego_path(bev_feats, ego_path_queries)
        bce_loss_ego_path = F.binary_cross_entropy_with_logits(y_ego_path, ego_path_labels.float(), reduction="none")
        bce_loss_ego_path = (bce_loss_ego_path * ego_path_valid).sum() / (ego_path_valid.sum() + 1e-6)

        return y_ego_path, bce_loss_ego_path

    def _get_occupancy_loss(self, occupancy_y: Tensor, occ_labels: Tensor) -> Tensor:
        if self.config.occupancy_loss == "bce":
            occupancy_loss = F.binary_cross_entropy_with_logits(occupancy_y, occ_labels.float(), reduction="mean")
        else:
            raise (ValueError(f"Invalid occupancy loss: {self.config.occupancy_loss}"))
        return occupancy_loss

    def _get_vfm_loss(self, vfm_gt: Tensor, vfm_pred: Tensor, valid_batch: Optional[Tensor] = None) -> Tensor:
        if valid_batch is not None:
            vfm_gt = vfm_gt * valid_batch
            vfm_pred = vfm_pred * valid_batch

        if self.config.vfm_feature_loss == "l2":
            loss = F.mse_loss(vfm_pred, vfm_gt, reduction="mean")
        elif self.config.vfm_feature_loss == "l1":
            loss = F.l1_loss(vfm_pred, vfm_gt, reduction="mean")
        elif self.config.vfm_feature_loss == "cos":
            feat_dim = vfm_gt.shape[-1]
            vfm_pred = vfm_pred.view(-1, feat_dim)
            vfm_gt = vfm_gt.view(-1, feat_dim)
            loss = F.cosine_embedding_loss(
                vfm_pred,
                vfm_gt,
                torch.ones(vfm_gt.shape[0], device=vfm_gt.device),
                reduction="mean",
            )
        elif self.config.vfm_feature_loss == "smoothl1":
            loss = F.smooth_l1_loss(vfm_pred, vfm_gt, reduction="mean", beta=self.config.vfm_feature_smoothl1_beta)
        else:
            raise ValueError(f"Invalid vfm feature loss: {self.config.vfm_feature_loss}")

        return loss

    def _valid_batch_loss(self, loss: Tensor, valid_batch: Tensor) -> Tensor:
        loss = loss * valid_batch
        n_valid_batches = valid_batch.sum()
        batchwise_loss = loss.mean((-2, -1))  # B
        # note that loss we take mean here as it will do the same as generating a new tensor if we dont have any valid batches
        loss = batchwise_loss.sum() / n_valid_batches if n_valid_batches > 0 else torch.tensor(0.0, device=loss.device)
        return loss

    def _get_vfm_features(
        self,
        vfm_queries: Tensor,
        vfm_targets: Tensor,
        vfm_batch_valid: Tensor,
        vfm_xy: Tensor,
        vfm_h: float,
        vfm_w: float,
        bev_feats: Tensor,
        vfm_min: Optional[Tensor] = None,
        vfm_max: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        vfm_out = self.gasp_model.decode_vfm_features(bev_feats, vfm_queries, vfm_xy, vfm_h, vfm_w)
        vfm_pred = vfm_out["prediction"]

        if vfm_min is not None:
            vfm_targets = (vfm_targets - vfm_min) / (vfm_max - vfm_min)

        loss = self._get_vfm_loss(vfm_targets, vfm_pred, vfm_batch_valid)

        if vfm_min is not None:
            vfm_pred = vfm_pred * (vfm_max - vfm_min) + vfm_min

        return vfm_pred, loss

    def _encode(self, batch: dict) -> Tuple[Tensor, dict]:
        lidar = self._get_lidar_data(batch)
        return self.gasp_model.encode(lidar)  # type: ignore

    def step(self, batch: dict, batch_idx: int) -> dict:
        bs = len(batch["id"])

        if self.config.freeze_encoder:
            with torch.no_grad():
                bev_feats, info = self._encode(batch)
        else:
            bev_feats, info = self._encode(batch)

        res = {
            "batch_size": bs,
            "loss": torch.zeros(1, device=self.device, dtype=bev_feats.dtype),
            "info": info,
        }

        if self.config.logging.bev_features:
            res["info"]["bev_features"] = bev_feats

        if self.config.gasp_model.f_predict_occupancy:
            res = self._step_occupancy(batch, bev_feats, res)

        if self.config.gasp_model.f_predict_ego_path:
            res = self._step_ego_path(batch, bev_feats, res)

        if self.config.gasp_model.f_predict_vfm_features:
            res = self._step_vfm_features(batch, bev_feats, res)

        res = detach_dict(res, skip_keys=["loss"])

        return res

    def _step_occupancy(self, batch: dict, bev_feats: Tensor, res: dict) -> dict:
        occ_queries, occ_labels, aux_labels = self.occupancy_sampler(
            positive_points=batch["offset_pointclouds"],
            ego_poses=batch["ego_poses"],
            ego_times=batch["ego_pose_times"],
            extrinsics=batch["lidar_extrinsics"],
            negative_rays=batch.get("missing_rays", None),
            future_points_ratio_inside=batch["future_points_ratio_inside"],
            positive_point_labels=batch.get("offset_pointclouds_labels", {}),
        )
        occupancy_out, occupancy_loss = self._get_occupancy(occ_queries, occ_labels, bev_feats)
        occupancy_y = occupancy_out["prediction"]

        res["loss"] += self.config.weight_occupancy_loss * occupancy_loss
        res["occupancy_loss"] = occupancy_loss.detach()
        res["occupancy_logits"] = occupancy_y.detach()
        res["occupancy_probs"] = occupancy_y.detach().sigmoid()
        res["occupancy_predictions"] = (occupancy_y > 0).detach().float()
        res["occupancy_queries"] = occ_queries
        res["occupancy_label"] = occ_labels
        res["occupancy_aux_labels"] = aux_labels

        if self.config.logging.offsets and "offsets_coords" in occupancy_out:
            res["occupancy_decoder_offsets"] = occupancy_out["offsets_coords"].detach()

        return res

    def _step_ego_path(self, batch: dict, bev_feats: Tensor, res: dict) -> dict:
        ego_path_y, ego_path_loss = self._get_ego_path(batch, bev_feats)
        res["loss"] += self.config.weight_ego_path_loss * ego_path_loss
        res["ego_path_loss"] = ego_path_loss.detach()
        res["ego_path_logits"] = ego_path_y.detach()
        res["ego_path_probs"] = ego_path_y.detach().sigmoid()
        res["ego_path_predictions"] = (ego_path_y > 0).detach().float()

        return res

    def _step_vfm_features(self, batch: dict, bev_feats: Tensor, res: dict) -> dict:

        (
            vfm_queries,
            vfm_targets,
            vfm_pixel_xy,
            vfm_target_occ,
            vfm_batch_valid,
            vfm_img_features,
        ) = self.vfm_sampler(
            vfm_features=batch.get("vfm_features", None),
            vfm_images=batch.get("vfm_images", None),
            vfm_timestamps=batch["vfm_timestamps"],
            ego_poses=batch["ego_poses"],
            ego_pose_times=batch["ego_pose_times"],
            lidar_extrinsics=batch["lidar_extrinsics"],
            camera_extrinsics_inverse=batch["vfm_extrinsics_inverse"],
            camera_intrinsics=batch["vfm_intrinsics"],
            camera_distortion_params=batch["vfm_distortion"],
            positive_points=batch["offset_pointclouds"],
            augmentation_angle=batch["augmentation_angle"],
        )

        vfm_h = batch["vfm_intrinsics"][0, 1, 2] * 2
        vfm_w = batch["vfm_intrinsics"][0, 0, 2] * 2

        vfm_min = batch.get("vfm_compression_min", None)
        vfm_max = batch.get("vfm_compression_max", None)

        vfm_features, vfm_loss = self._get_vfm_features(
            vfm_queries,
            vfm_targets,
            vfm_batch_valid,
            vfm_pixel_xy,
            vfm_h,
            vfm_w,
            bev_feats,
            vfm_min,
            vfm_max,
        )

        res["vfm_3dquery_targets"] = vfm_targets.detach()
        res["vfm_3dquery_queries"] = vfm_queries[..., :3].detach()
        res["vfm_3dquery_timestamps"] = vfm_queries[..., 3].detach()
        res["vfm_3dquery_occupancy"] = vfm_target_occ.detach()
        res["vfm_3dquery_preds"] = vfm_features.detach()
        res["vfm_3dquery_valid"] = vfm_batch_valid.detach()
        if vfm_img_features is not None:
            res["vfm_features"] = vfm_img_features

        # Accumulate the (possibly zero) loss
        weighted_loss = self.config.weight_vfm_features_loss * vfm_loss
        res["loss"] = res["loss"] + weighted_loss
        res["vfm_features_loss"] = vfm_loss.detach()  # Logging only

        return res

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        res = self.step(batch, batch_idx)
        self._log_train(batch, res)
        return res

    def _log_train(self, batch: dict, res: dict):
        # log losses
        self.log(
            "train/total_loss", res["loss"].detach(), on_step=True, rank_zero_only=self.config.rank_zero_only_logging
        )
        for k, v in res.items():
            if k.endswith("_loss"):
                self.log(
                    f"train/loss/{k[:-5]}", v.detach(), on_step=True, rank_zero_only=self.config.rank_zero_only_logging
                )

        if self.config.gasp_model.f_predict_occupancy:
            if "occupancy_label" in batch:
                occupancy_label = batch["occupancy_label"]
            elif "occupancy_label" in res:
                occupancy_label = res["occupancy_label"]
            else:
                occupancy_label = None

            self.train_occupancy_acc(res["occupancy_predictions"], occupancy_label)
            self.train_occupancy_prec(res["occupancy_predictions"], occupancy_label)
            self.train_occupancy_rec(res["occupancy_predictions"], occupancy_label)
            self.train_occupancy_f1(res["occupancy_predictions"], occupancy_label)
            self.log(
                "train/occupancy/accuracy",
                self.train_occupancy_acc,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/occupancy/precision",
                self.train_occupancy_prec,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/occupancy/recall",
                self.train_occupancy_rec,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/occupancy/f1",
                self.train_occupancy_f1,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )

        if self.config.gasp_model.f_predict_ego_path and "ego_path_queries" in batch:
            self.train_ego_path_acc(res["ego_path_predictions"], batch["ego_path_label"])
            self.train_ego_path_prec(res["ego_path_predictions"], batch["ego_path_label"])
            self.train_ego_path_rec(res["ego_path_predictions"], batch["ego_path_label"])
            self.train_ego_path_f1(res["ego_path_predictions"], batch["ego_path_label"])
            self.log(
                "train/ego_path/accuracy",
                self.train_ego_path_acc,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/ego_path/precision",
                self.train_ego_path_prec,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/ego_path/recall",
                self.train_ego_path_rec,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "train/ego_path/f1",
                self.train_ego_path_f1,
                on_step=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        with torch.no_grad():
            res = self.step(batch, batch_idx)

        self._log_val(batch, res)
        return res

    def _log_stuff_epoch(self, split: str) -> None:
        pass

    def on_validation_epoch_end(self) -> None:
        self._log_stuff_epoch("val")

    def on_test_epoch_end(self) -> None:
        self._log_stuff_epoch("test")

    def _log_val(self, batch: dict, res: dict):
        # log losses
        self.log(
            "val/total_loss",
            res["loss"].detach(),
            on_step=False,
            on_epoch=True,
            batch_size=res["batch_size"],
            sync_dist=True,
            rank_zero_only=self.config.rank_zero_only_logging,
        )
        for k, v in res.items():
            if k.endswith("_loss"):
                self.log(
                    f"val/loss/{k[:-5]}",
                    v.detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

        # Log task metrics
        if "occupancy_predictions" in res:
            if "occupancy_label" in batch:
                occupancy_label = batch["occupancy_label"]
            elif "occupancy_label" in res:
                occupancy_label = res["occupancy_label"]
            else:
                occupancy_label = None

            if hasattr(self, "val_occupancy_acc"):
                self.val_occupancy_acc(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/accuracy",
                    self.val_occupancy_acc,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_prec"):
                self.val_occupancy_prec(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/precision",
                    self.val_occupancy_prec,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_rec"):
                self.val_occupancy_rec(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/recall",
                    self.val_occupancy_rec,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_f1"):
                self.val_occupancy_f1(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/f1",
                    self.val_occupancy_f1,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

        if self.config.gasp_model.f_predict_ego_path and "ego_path_queries" in batch:
            self.val_ego_path_acc(res["ego_path_predictions"], batch["ego_path_label"])
            self.val_ego_path_prec(res["ego_path_predictions"], batch["ego_path_label"])
            self.val_ego_path_rec(res["ego_path_predictions"], batch["ego_path_label"])
            self.val_ego_path_f1(res["ego_path_predictions"], batch["ego_path_label"])
            self.log(
                "val/ego_path/accuracy",
                self.val_ego_path_acc,
                batch_size=res["batch_size"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "val/ego_path/precision",
                self.val_ego_path_prec,
                batch_size=res["batch_size"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "val/ego_path/recall",
                self.val_ego_path_rec,
                batch_size=res["batch_size"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )
            self.log(
                "val/ego_path/f1",
                self.val_ego_path_f1,
                batch_size=res["batch_size"],
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=self.config.rank_zero_only_logging,
            )

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        if self.visualization_mode:
            return {}

        with torch.no_grad():
            res = self.step(batch, batch_idx)

        return res

    @torch.no_grad()
    def probe_occupancy(
        self,
        batch: dict,
        sample_idx: int = 0,
        probe_range: PointCloudRange = DEFAULT_OCCUPANCY_PROBE_RANGE,
        time: float = 0.0,
        delta_x=0.1,
        delta_y=0.1,
        delta_z=0.1,
        max_queries: Optional[int] = None,
    ) -> dict:
        bev_feats, _ = self._encode(batch)

        queries, num_samples_per_dim = self._get_probe_queries(
            delta_x, delta_y, delta_z, probe_range, dims=4, time=time
        )

        if max_queries is not None:
            total_queries = queries.shape[1]
            occupancy = torch.zeros((queries.shape[0], total_queries, 1), device=queries.device)

            for i in range(0, total_queries, max_queries):
                end_idx = min(i + max_queries, total_queries)
                queries_chunk = queries[:, i:end_idx]
                occupancy[:, i:end_idx] = self.gasp_model.decode(bev_feats, queries_chunk, ["occupancy"])["occupancy"]
        else:
            occupancy = self.gasp_model.decode(bev_feats, queries, ["occupancy"])["occupancy"]

        res = {
            "occupancy_num_samples_per_dim": num_samples_per_dim,
            "occupancy_queries": queries,
            "occupancy_logits": occupancy.detach().squeeze(-1),
            "occupancy_probs": occupancy.sigmoid().detach().squeeze(-1),
            "occupancy_predictions": (occupancy > 0).float().detach().squeeze(-1),
        }

        return res

    @torch.no_grad()
    def probe_ego_path(
        self,
        batch: dict,
        sample_idx: int = 0,
        probe_range: PointCloudRange = DEFAULT_EGO_PATH_PROBE_RANGE,
        delta_x=0.1,
        delta_y=0.1,
        delta_z=0.1,
        add_dense=False,
    ) -> dict:
        bev_feats, _ = self._encode(batch)

        queries, num_samples_per_dim = self._get_probe_queries(
            delta_x, delta_y, delta_z, probe_range, dims=self.config.gasp_model.ego_path_dimensions
        )

        y = self.gasp_model.decode(bev_feats, queries, ["ego_path"])["ego_path"]

        res = {
            "ego_path_num_samples_per_dim": num_samples_per_dim,
            "ego_path_queries": queries,
            "ego_path_logits": y.detach().squeeze(-1),
            "ego_path_probs": y.sigmoid().detach().squeeze(-1),
            "ego_path_predictions": (y > 0).float().detach().squeeze(-1),
        }

        return res

    @torch.no_grad()
    def probe_vfm_features(
        self,
        batch: dict,
        sample_idx: int = 0,
        probe_range: PointCloudRange = DEFAULT_OCCUPANCY_PROBE_RANGE,
        time: float = 0.0,
        delta_x=0.1,
        delta_y=0.1,
        delta_z=0.1,
        max_queries: Optional[int] = None,
    ) -> dict:
        bev_feats, _ = self._encode(batch)

        queries, num_samples_per_dim = self._get_probe_queries(
            delta_x, delta_y, delta_z, probe_range, dims=4, time=time
        )

        was_training = self.gasp_model.training
        self.gasp_model.eval()

        if max_queries is not None:
            total_queries = queries.shape[1]
            occupancy = torch.zeros((queries.shape[0], total_queries, 1), device=queries.device)
            features = torch.zeros((queries.shape[0], total_queries, 1), device=queries.device)

            for i in range(0, total_queries, max_queries):
                end_idx = min(i + max_queries, total_queries)
                queries_chunk = queries[:, i:end_idx]
                decode_out = self.gasp_model.decode(bev_feats, queries_chunk, ["vfm_features", "occupancy"])
                occupancy[:, i:end_idx] = decode_out["occupancy"]
                features[:, i:end_idx] = decode_out["vfm_features"]
        else:
            decode_out = self.gasp_model.decode(bev_feats, queries, ["vfm_features", "occupancy"])
            features = decode_out["vfm_features"]
            occupancy = decode_out["occupancy"]

        self.gasp_model.train(was_training)

        if self.config.gasp_model.normalize_vfm_features:
            vfm_min = batch["vfm_compression_min"][sample_idx]
            vfm_max = batch["vfm_compression_max"][sample_idx]
            features = features * (vfm_max - vfm_min) + vfm_min

        res = {
            "vfm_num_samples_per_dim": num_samples_per_dim,
            "vfm_feature_queries": queries,
            "vfm_features": features,
            "vfm_feature_occupancy_logits": occupancy,
            "vfm_feature_occupancy_probs": occupancy.sigmoid(),
            "vfm_feature_occupancy_predictions": (occupancy > 0).float(),
        }

        return res

    def optimizer_step(self, *args, **kwargs):
        try:
            super().optimizer_step(*args, **kwargs)
        except Exception as e:
            for n, p in self.gasp_model.named_parameters():
                if p.grad is None and p.requires_grad:
                    print(f"{n} has no grad but requires grad")

            raise e


def t_to_str(t: float) -> str:
    return str(t).replace(".", "p")


def detach_dict(d, skip_keys=["loss"]):
    out = {}
    for k, v in d.items():
        if k in skip_keys:
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach()
        elif isinstance(v, dict):
            out[k] = detach_dict(v)  # recursive
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    model = GaspPreTrainConfig().setup()
