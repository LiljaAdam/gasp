import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Type

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

import gasp.config.base_config as cfg
from gasp.data.components.cached_vfm_reader import AV2CachedVFMFeaturesReaderConfig
from gasp.data.datasets.argoverse2 import AV2DatasetConfig
from gasp.data.datasets.zod_base import ZODBaseDatasetConfig, new_base_collate_fn
from gasp.data.utils.distributed_weighted_sampler import is_ddp_active

logger = logging.getLogger(__name__)


@dataclass
class JointDataModuleConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: JointDataModule)

    av2_data: Path = Path("/datasets/av2")
    """The path to av2 dataset."""

    version: Literal["full", "mini"] = "mini"
    """What version of the dataset to use."""
    splits: Literal["original", "geo_near"] = "original"
    """ The splits to use. Only applicable for Argoverse2 and nuScenes datasets."""

    debug: bool = False
    """If to run overfitting on one sample"""

    # dataloader parameters
    batch_size: int = 1
    """The batch size."""
    num_workers: int = 16
    """The number of workers."""
    pin_memory: bool = True
    """Whether to pin memory."""
    persistent_workers: bool = False
    """Whether to use persistent workers."""
    prefetch_factor: int = 2
    """The prefetch factor."""
    shuffle_test: bool = False
    """Whether to shuffle the test dataloader."""
    balanced_dataset_sampling: bool = False
    """Whether to balance the dataset sampling."""
    num_samples_per_rank: Optional[int] = None
    """The number of samples per rank per epoch when using multiple datasets."""

    dataset: ZODBaseDatasetConfig = field(default_factory=lambda: ZODBaseDatasetConfig())
    """The dataset configuration."""
    datasets_to_include: Tuple[str, ...] = ("sensor",)  # "lidar" is also supported
    """The datasets to include."""

    train_dataset_usage_ratio: float = 1.0  # should > 0 and <= 1
    """The ratio of the training dataset to use."""

    train_dataset_usage_num: Optional[int] = None
    """The number of samples to use from the training dataset. -1 means all samples."""

    run_test_on_enitre_dataset: bool = False
    """If to run test on the entire dataset."""

    val_on_sensor_only: bool = False
    """If to validate only on sensor."""

    min_dataset_length: int = 1000

    def __post_init__(self) -> None:
        self.dataset.version = self.version
        assert self.train_dataset_usage_ratio > 0 and self.train_dataset_usage_ratio <= 1, (
            "train_dataset_usage_ratio must be > 0 and <= 1"
        )
        if self.train_dataset_usage_ratio < 1.0:
            assert self.train_dataset_usage_num is None, (
                "train_dataset_usage_num must be None when train_dataset_usage_ratio is set"
            )
        if self.train_dataset_usage_num is not None:
            # should not be used at the same time
            assert self.train_dataset_usage_ratio == 1.0, (
                "train_dataset_usage_ratio must be 1.0 when train_dataset_usage_num is set"
            )


class JointDataModule(LightningDataModule):
    """`LightningDataModule` for the ZOD dataset."""

    config: JointDataModuleConfig

    def __init__(self, config: JointDataModuleConfig) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.config = config
        self.dataset_name = "autonomous_driving"

        self.data_train = None
        self.data_val = None
        self.data_test = None

        self.batch_size_per_device = self.config.batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.config.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.config.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.config.batch_size // self.trainer.world_size

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            train_datasets = []
            val_datasets = []

            for dataset_name in self.config.datasets_to_include:
                if dataset_name == "sensor":
                    dataset_cfg = self.config.dataset.__dict__.copy()
                    dataset_cfg["data"] = self.config.av2_data
                    dataset_cfg["subset"] = "sensor"
                    dataset_cfg["cached_vfm_cfg"] = AV2CachedVFMFeaturesReaderConfig.from_dino_feature_reader_config(
                        dataset_cfg["cached_vfm_cfg"]
                    )
                    dataset_train = AV2DatasetConfig(**dataset_cfg).setup(
                        train_or_val="train", split=f"sensor_{self.config.splits}"
                    )
                    dataset_val = AV2DatasetConfig(**dataset_cfg).setup(
                        train_or_val="val", split=f"sensor_{self.config.splits}"
                    )
                    train_datasets.append(dataset_train)
                    val_datasets.append(dataset_val)
                elif dataset_name == "lidar":
                    dataset_cfg = self.config.dataset.__dict__.copy()
                    dataset_cfg["data"] = self.config.av2_data
                    dataset_cfg["subset"] = "lidar"
                    dataset_cfg["cached_vfm_cfg"] = AV2CachedVFMFeaturesReaderConfig.from_dino_feature_reader_config(
                        dataset_cfg["cached_vfm_cfg"]
                    )
                    dataset_cfg["load_cached_vfm_features"] = False
                    dataset_train = AV2DatasetConfig(**dataset_cfg).setup(
                        train_or_val="train", split=f"lidar_{self.config.splits}"
                    )
                    train_datasets.append(dataset_train)

                    if "sensor" not in self.config.datasets_to_include:
                        dataset_val_cfg = self.config.dataset.__dict__.copy()
                        dataset_val_cfg["data"] = self.config.av2_data
                        dataset_val_cfg["subset"] = "sensor"
                        dataset_val_cfg["cached_vfm_cfg"] = (
                            AV2CachedVFMFeaturesReaderConfig.from_dino_feature_reader_config(
                                dataset_val_cfg["cached_vfm_cfg"]
                            )
                        )
                        dataset_val = AV2DatasetConfig(**dataset_val_cfg).setup(
                            train_or_val="val", split=f"sensor_{self.config.splits}"
                        )
                        val_datasets.append(dataset_val)

                else:
                    raise ValueError(f"Dataset {dataset_name} not supported")

            if self.config.debug:
                # # debug, only train on one sample for faster iteration, remove later
                self.data_train = Subset(dataset_train, [0] * 100)
                self.data_val = Subset(dataset_train, [0, 0])
                self.data_test = Subset(dataset_train, [0, 0])
                return

            if len(train_datasets) > 1:
                print(f"Concatenating {len(train_datasets)} train datasets")
                for ds, name in zip(train_datasets, self.config.datasets_to_include):
                    print(f"Dataset {name} has {len(ds)} samples")
                if self.config.balanced_dataset_sampling:
                    print("Balancing dataset sampling is active")

            self.data_train = ConcatDataset(train_datasets)

            if self.config.train_dataset_usage_ratio < 1.0:
                train_size = int(len(self.data_train) * self.config.train_dataset_usage_ratio)
                self.data_train = Subset(self.data_train, list(range(train_size)))

            if self.config.train_dataset_usage_num is not None:
                train_size = self.config.train_dataset_usage_num
                if train_size == -1 or train_size > len(self.data_train):
                    idx_ = list(range(len(self.data_train)))
                else:
                    idx_ = np.random.choice(len(self.data_train), train_size, replace=False).tolist()
                    # make replicas so we get roughly 10_000 samples
                    if train_size < self.config.min_dataset_length:
                        idx_ = idx_ * (self.config.min_dataset_length // train_size + 1)
                        idx_ = idx_[: self.config.min_dataset_length]
                self.data_train = Subset(self.data_train, idx_)

            if self.config.val_on_sensor_only and "sensor" in self.config.datasets_to_include:
                idx = self.config.datasets_to_include.index("sensor")
                self.data_val = val_datasets[idx]

            else:
                self.data_val = ConcatDataset(val_datasets)

            self.data_test = self.data_val

            if self.config.run_test_on_enitre_dataset:
                self.data_test = ConcatDataset(val_datasets + train_datasets)

    def set_epoch_for_sampler(self, epoch: int):
        if hasattr(self, "train_sampler") and hasattr(self.train_sampler, "set_epoch"):
            self.train_sampler.set_epoch(epoch)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        assert self.data_train is not None, "Data not loaded. Run `setup()` first."

        if is_ddp_active():
            print("DDP Distribution Sampler is used.")
            sampler = DistributedSampler(self.data_train, shuffle=True, drop_last=False)  # drop_last=True
            shuffle = False
        else:
            print("Default sampler is used.")
            sampler = None
            shuffle = True

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=new_base_collate_fn,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        assert self.data_val is not None, "Data not loaded. Run `setup()` first."
        if is_ddp_active():
            sampler_val = DistributedSampler(self.data_val, shuffle=False, drop_last=False)
        else:
            sampler_val = None

        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=False,
            sampler=sampler_val,
            collate_fn=new_base_collate_fn,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        assert self.data_test is not None, "Data not loaded. Run `setup()` first."
        if is_ddp_active():
            sampler_test = DistributedSampler(self.data_test, shuffle=False, drop_last=False)
        else:
            sampler_test = None

        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            shuffle=self.config.shuffle_test,
            sampler=sampler_test,
            collate_fn=new_base_collate_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
