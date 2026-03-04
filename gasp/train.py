"""Train an end-to-end driving model with gasp.
For pre-training, use [bright_yellow]gasp-av2[/bright_yellow] (full GASP) or [bright_yellow]uno-av2[/bright_yellow] (UNO baseline).

We allow for customizing your training and eval configs from the CLI in a powerful way, but there are some
things to understand.

The most demonstrative and helpful example of the CLI structure is the difference in output between the following
commands:

    train.py -h
    train.py uno -h

In each of these examples, the -h applies to the previous subcommand (train.py and uno).

In the first example, we get the help menu for the train script.
In the second example, we get the help menu for the uno model.

With our scripts, your arguments will apply to the preceding subcommand in your command, and thus where you put your
arguments matters! Any optional arguments you discover from running

    train.py -h uno

need to come directly after the train.py command, since these optional arguments only belong to the top-level train command:

    train.py (trainer args) uno (model args)
"""

import functools
import glob
from copy import deepcopy
from pathlib import Path

import lightning as L
import lightning.pytorch.callbacks as cb
import torch
import tyro
from lightning.pytorch.loggers import WandbLogger

from gasp.config.method_configs import AnnotatedBaseConfigUnion, convert_markup_to_ansi
from gasp.config.trainer import TrainingConfig
from gasp.data.joint_datamodule import JointDataModule
from gasp.model.base import BaseModel
from gasp.model.gasp_pre_train import GaspPreTrainConfig


def setup():
    torch.set_float32_matmul_precision("medium")  # heighest, high, medium


def is_large_node():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory > 35e9
    return False


def validate_loaded_checkpoint_config(config: TrainingConfig):
    assert config.load_ckpt_from is not None and config.load_ckpt_from.exists(), "Checkpoint not found"
    config_path = config.load_ckpt_from.parent.parent.parent.parent / "config.yml"
    assert config_path.exists(), f"Config not found at {config_path}"
    loaded_config = TrainingConfig.load_config_from_yaml(config_path)

    assert loaded_config.model.gasp_model.lidar_encoder == config.model.gasp_model.lidar_encoder, "Lidar encoder mismatch"

    assert loaded_config.model.roi == config.model.roi, "ROI mismatch"

    assert loaded_config.datamodule.splits == config.datamodule.splits, "Splits mismatch"

    ld = loaded_config.datamodule.dataset
    nd = config.datamodule.dataset

    if config.model.gasp_model.lidar_encoder:
        assert ld.n_prev_lidar == nd.n_prev_lidar, (
            f"n_future_images mismatch. Expected {ld.n_prev_lidar} but got {nd.n_prev_lidar}"
        )
        assert ld.prev_lidar_max_time == nd.prev_lidar_max_time, (
            f"prev_lidar_max_time mismatch. Expected {ld.prev_lidar_max_time} but got {nd.prev_lidar_max_time}"
        )

    del loaded_config


def scale_with_num_gpus(config: TrainingConfig) -> None:
    keys_to_scale = (
        "datamodule.batch_size",
        "model.max_lr",
    )

    keys_to_divide = ("trainer.max_steps",)

    num_gpus = torch.cuda.device_count()
    num_nodes = config.trainer.num_nodes
    num_acc = config.trainer.accumulate_grad_batches

    if num_gpus <= 1 and num_nodes <= 1:
        return

    def rsetattr(obj, attr, val):
        pre, _, post = attr.rpartition(".")
        return setattr(rgetattr(obj, pre) if pre else obj, post, val)

    def rgetattr(obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split("."))

    for key in keys_to_scale:
        rsetattr(config, key, rgetattr(config, key) * num_gpus * num_nodes)

    for key in keys_to_divide:
        rsetattr(config, key, rgetattr(config, key) / num_gpus / num_nodes / num_acc)

    if "model.max_lr" in keys_to_scale:
        rsetattr(config, "model.max_lr", rgetattr(config, "model.max_lr") * num_acc)


def _clean_config_for_wandb(config: TrainingConfig):
    def __recursive_clean(atr):
        if isinstance(atr, dict):
            for k, v in atr.items():
                if hasattr(v, "__dict__"):
                    v = v.__dict__
                if k.startswith("_"):
                    continue
                atr[k] = __recursive_clean(v)
        else:
            if isinstance(atr, Path):
                atr = str(atr)

        return atr

    _config = deepcopy(config.__dict__)
    _config = __recursive_clean(_config)
    return _config


def get_wandb_logger(config: TrainingConfig, model_name: str) -> WandbLogger:
    wandb_group_name = f"{model_name}-{config.model.roi}"

    if not config.train:
        wandb_group_name = "Eval-" + wandb_group_name

    if config.exp_name == "unnamed":
        config.exp_name = config.timestamp

    possible_tasks = [
        "semantic_4d_occupancy",
    ]
    for task in possible_tasks:
        if getattr(config.model, task, False):
            new_name: list[str] = [wandb_group_name, task]
            wandb_group_name = "-".join(new_name)

    wandb_logger = WandbLogger(
        name=config.exp_name,
        project="gasp",
        offline=config.debug,
        save_dir=config.get_base_dir(),
        config=_clean_config_for_wandb(config),
        group=wandb_group_name,
    )

    return wandb_logger


def get_checkpoint(checkpoint_folder: Path, best: bool = True) -> Path:
    # find the "last.ckpt" can be in any subfolder
    ckpts = glob.glob(str(checkpoint_folder / "**/*.ckpt"), recursive=True)

    # print(checkpoint_folder, ckpts)
    last_ckpt = [c for c in ckpts if "last" in c][0]
    best_ckpt = [c for c in ckpts if "last" not in c][0]

    ckpt = best_ckpt if best else last_ckpt
    return Path(ckpt)


def load_model_weights_from_checkpoint(config: TrainingConfig) -> BaseModel:
    assert config.load_ckpt_from.exists(), f"Resume weights not found at {config.load_ckpt_from}"

    if config.load_ckpt_from.is_dir():
        print(
            f"Loading the {'best' if config.load_ckpt_from_best else 'last'} model weights from the folder you provided"
        )
        config.load_ckpt_from = get_checkpoint(config.load_ckpt_from, config.load_ckpt_from_best)

    print("Loading the model weights from the checkpoint, but NOT the optimizers etc.")
    print(f"Loading from {config.load_ckpt_from}")
    # get the checkpoint config and validate
    if config.validate_config:
        validate_loaded_checkpoint_config(config)
        strict = True
        config_path = config.load_ckpt_from.parent.parent.parent.parent / "config.yml"
        assert config_path.exists(), f"Config not found at {config_path}"
        loaded_config = TrainingConfig.load_config_from_yaml(config_path)
        is_from_pretrain = isinstance(loaded_config.model, GaspPreTrainConfig)
    else:
        print("Skipping config validation, I hope you know what you are doing!")
        strict = False
        is_from_pretrain = True

    model = config.model._target.load_from_checkpoint(
        config.load_ckpt_from,
        config=config.model,
        strict=strict,
        is_from_pretrain=is_from_pretrain,
        visualization_mode=config.visualization_mode,
    )

    return model


def main(config: TrainingConfig):
    L.seed_everything(config.seed, workers=True)

    config.model.configure_dataset(config.datamodule.dataset)
    setup()
    scale_with_num_gpus(config)
    if config.debug:
        config.datamodule.debug = True

    if is_large_node() and config.disable_gradient_checkpointing_on_large_node:
        print("Disabling gradient checkpointing due to large node")
        config.model.gasp_model.gradient_checkpointing = False

    if config.load_ckpt_from is not None and not config.resume_training:
        model: BaseModel = load_model_weights_from_checkpoint(config)
    else:
        model: BaseModel = config.model.setup()

    dm: JointDataModule = config.datamodule.setup()

    wandb_logger = get_wandb_logger(config, model.__class__.__name__)

    callbacks = [
        cb.ModelCheckpoint(monitor=config.monitor_key, save_top_k=1, mode=config.monitor_mode, save_last=True),
        cb.LearningRateMonitor(logging_interval="step"),
    ]

    if config.monitor_patience > 0:
        callbacks.append(
            cb.EarlyStopping(
                monitor=config.monitor_key,
                patience=config.monitor_patience,
                mode=config.monitor_mode,
                min_delta=0.00,
                strict=config.monitor_strict,
            )
        )

    trainer = L.Trainer(
        logger=wandb_logger if config.use_logging else None,
        callbacks=callbacks,
        **config.trainer.__dict__,
    )

    # save the config
    config.save_config()

    if config.train:
        if config.resume_training:
            assert config.load_ckpt_from is not None and config.load_ckpt_from.exists(), "Checkpoint not found"
            if config.load_ckpt_from.is_dir():
                print("Loading the last model weights from the folder you provided")
                config.load_ckpt_from = get_checkpoint(config.load_ckpt_from, best=False)
            print(f"Resuming training from {config.load_ckpt_from}")

        trainer.fit(model, datamodule=dm, ckpt_path=config.load_ckpt_from if config.resume_training else None)

        if config.test:
            trainer.test(model, datamodule=dm, ckpt_path=config.test_after_fit_ckpt_path)

    if config.test and not config.train:
        trainer.test(model=model, datamodule=dm)

    return model


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )


if __name__ == "__main__":
    entrypoint()
