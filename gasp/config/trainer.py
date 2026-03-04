from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union

import yaml

from gasp.data.joint_datamodule import JointDataModuleConfig
from gasp.model.base import ModelConfig


@dataclass
class TrainerConfig:
    accumulate_grad_batches: int = 1
    """The number of batches to accumulate before performing a backward pass."""

    val_check_interval: Union[int, float] = 0.5
    """The validation check interval."""

    limit_val_batches: Union[int, float] = 160
    """The number of validation batches."""

    limit_test_batches: Union[int, float] = 1.0
    """The number of test batches."""

    check_val_every_n_epoch: int = 1
    """The number of training epochs between validation."""

    num_sanity_val_steps: int = 2
    """Number of sanity validation steps to run before training. Set to 0 to disable."""

    max_steps: int = 1_000_000
    """The maximum number of steps."""

    strategy: str = "ddp"  
    """The strategy for distributed training."""

    precision: Optional[str] = "bf16-mixed"
    """The precision to use (None, 16-mixed)."""

    devices: int = 1
    """The number of devices to use."""

    num_nodes: int = 1
    """The number of nodes."""

    sync_batchnorm: bool = True
    """Sync batchnorm."""

    overfit_batches: int = 0
    """ The number of batches to overfit on."""

    gradient_clip_val: float = 2.5
    """Gradient clipping value."""

    gradient_clip_algorithm: str = "norm"
    """Gradient clipping."""

    profiler: Union[str, None] = None
    """The profiler to use (simple, advanced, pytorch, xla)."""

    detect_anomaly: bool = False
    """Detect anomalies."""

    deterministic: bool = False
    """Deterministic training."""

    use_distributed_sampler: bool = True
    """Replace the sampler with a distributed sampler."""

    enable_model_summary: bool = False
    """Enable model summary."""


@dataclass
class TrainingConfig:
    out_dir: Path = Path("/outputs")
    """The output directory."""
    exp_name: str = "unnamed"
    """The experiment name."""
    debug: bool = False
    """Debug mode."""
    seed: int = 1337
    """The one seed to rule them all."""
    train: bool = True
    """Train the model."""
    test: bool = True
    """Test the model."""
    test_after_fit_ckpt_path: Literal["best", "last"] = "last"
    """The checkpoint to test after training."""

    monitor_key: str = "val/total_loss"
    """The key to monitor for early stopping."""
    monitor_patience: int = -1
    """The patience for early stopping. Set to -1 for no early stopping."""
    monitor_mode: str = "min"
    """The mode for early stopping."""
    monitor_strict: bool = True
    """If we want to strictly monitor the metric or not."""

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())
    """The trainer config"""
    datamodule: JointDataModuleConfig = field(default_factory=lambda: JointDataModuleConfig())
    """The dataset configuration."""
    model: ModelConfig = field(default_factory=lambda: ModelConfig())
    """The model configuration."""
    load_ckpt_from: Optional[Path] = None
    """Weights to resume training from."""
    load_ckpt_from_best: bool = False
    """If we want to load the best checkpoint if load_ckpt_from is a directory. Else load last."""
    resume_training: bool = False
    """If we want to resume training from the checkpoint. I.e., load optimizers etc. from the checkpoint. Has to be coupled with `load_ckpt_from`."""
    save_weights_only: bool = False
    """If we want to save only the weights of the model in a .pth file."""

    # logging
    use_logging: bool = True
    """Use logging."""
    relative_checkpoint_dir: Path = Path("checkpoints")
    """The relative path to the checkpoint directory."""
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    """The timestamp for the experiment."""

    validate_config: bool = True
    """Validate the config when loading from checkpoint."""

    cache_bev_feats_to: Optional[Path] = None
    """Cache BEV features."""

    disable_gradient_checkpointing_on_large_node: bool = True
    """Disable gradient checkpointing if on large node."""

    def get_base_dir(self) -> Path:
        """Get the base directory."""
        path = self.out_dir / "experiments" / self.exp_name / self.timestamp
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory."""
        return self.get_base_dir() / self.relative_checkpoint_dir

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        config_yaml_path.write_text(yaml.dump(self), "utf8")

    @classmethod
    def load_config_from_yaml(cls, config_yaml_path: Path) -> "TrainingConfig":
        """Load config from yaml file."""
        return yaml.load(config_yaml_path.read_text(), Loader=yaml.Loader)
