"""Extremely inspired by nerfstudio, credit to them for this pattern."""

import copy
from pathlib import Path
from typing import Dict, OrderedDict

import tyro

from gasp.config.base_config import CONSOLE
from gasp.config.trainer import TrainerConfig, TrainingConfig
from gasp.data.components.cached_vfm_reader import CachedVFMFeaturesReaderConfig
from gasp.data.components.vfm_sampler_gpu import VFMFeaturesProjectionGPUSamplerConfig
from gasp.data.datasets.zod_base import ZODBaseDatasetConfig
from gasp.data.joint_datamodule import JointDataModuleConfig
from gasp.model.components.gasp import GaspConfig
from gasp.model.gasp_pre_train import GaspPreTrainConfig

method_configs: Dict[str, TrainingConfig] = {}
descriptions = {
    "gasp-av2": "GASP pre-training with AV2 data (lidar-only, occupancy + DINO features + ego path).",
    "gasp-av2-online-vfm": "GASP pre-training with AV2 data (lidar-only, occupancy + DINO features + ego path + online VFM features).",
    "uno-av2": "UNO baseline pre-training with AV2 data (lidar-only, occupancy only).",
}

av2_weight_ego_path_loss = 0.1
av2_weight_vfm_features_loss = 0.5

method_configs["gasp-av2"] = TrainingConfig(
    trainer=TrainerConfig(
        max_steps=800_000,
        val_check_interval=0.5,
        precision="bf16-mixed",
    ),
    monitor_key="val/total_loss",
    monitor_mode="max",
    model=GaspPreTrainConfig(
        gasp_model=GaspConfig(
            lidar_encoder=True,
            f_predict_occupancy=True,
            f_predict_ego_path=True,
            f_predict_vfm_features=True,
            vfm_feature_dimensions=16,
            bev_attention="deformable_py",
            swap_batch_norm_to_instance_norm=False,
            gradient_checkpointing=True,
        ),
        vfm_sampler=VFMFeaturesProjectionGPUSamplerConfig(
            rough_camera_fov_deg=-1.0,
            use_cached_features=True,
        ),
        weight_ego_path_loss=av2_weight_ego_path_loss,
        weight_vfm_features_loss=av2_weight_vfm_features_loss,
    ),
    datamodule=JointDataModuleConfig(
        datasets_to_include=("sensor",),
        val_on_sensor_only=True,
        batch_size=1,
        splits="original",
        dataset=ZODBaseDatasetConfig(
            add_missing_points=True,
            n_prev_lidar=3,
            prev_lidar_max_time=3.0,
            num_future_frames_to_sample=3,
            max_future_frame_s=3.0,
            num_past_frames_to_sample=0,
            max_past_frame_s=0.0,
            rotation_augmentation=True,
            translation_augmentation=False,
            cached_vfm_cfg=CachedVFMFeaturesReaderConfig(
                extracted_features_dim=32,
                extracted_features_path=Path("/datasets/av2_dino_features"),
            ),
        ),
    ),
)

method_configs["gasp-av2-online-vfm"] = copy.deepcopy(method_configs["gasp-av2"])
method_configs["gasp-av2-online-vfm"].model.vfm_sampler.use_cached_features = False

method_configs["uno-av2"] = copy.deepcopy(method_configs["gasp-av2"])
method_configs["uno-av2"].model.gasp_model.f_predict_ego_path = False
method_configs["uno-av2"].model.gasp_model.f_predict_vfm_features = False
method_configs["uno-av2"].datamodule.dataset.add_missing_points = False
method_configs["uno-av2"].datamodule.dataset.rotation_augmentation = False


def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting."""
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()


def sort_methods(methods, method_descriptions):
    """Sort methods and descriptions by method name."""
    methods = OrderedDict(sorted(methods.items(), key=lambda x: x[0]))
    method_descriptions = OrderedDict(sorted(method_descriptions.items(), key=lambda x: x[0]))
    return methods, method_descriptions


method_configs, descriptions = sort_methods(method_configs, descriptions)

AnnotatedBaseConfigUnion = tyro.conf.SuppressFixed[
    tyro.extras.subcommand_type_from_defaults(defaults=method_configs, descriptions=descriptions)
]
"""Union[] type over config types, annotated with default instances for use with
tyro.cli(). Allows the user to pick between one of several base configurations, and
then override values in it."""
