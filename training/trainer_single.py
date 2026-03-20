import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from training.optimizer import construct_optimizer
from training.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.logger import Logger, setup_logging
from training.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_resume_checkpoint,
    human_readable_time,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
)

CORE_LOSS_KEY = "core_loss"


# =========================================================
#  MOCK DISTRIBUTED UTILS
# =========================================================
def barrier():
    pass


def get_rank():
    return 0


def unwrap_ddp_if_wrapped(model):
    return model


def all_reduce_max(tensor):
    return tensor


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None

    def __post_init__(self):
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)


@dataclass
class DistributedConf:
    backend: Optional[str] = None
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    matmul_allow_tf32: Optional[bool] = None
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False


class Trainer:
    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model

        # Set up LoggingConf and CheckpointConf directly to avoid Omegaconf attribute issues
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()

        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else None
        self.meters_conf = meters
        self.loss_conf = loss

        # Override distributed config to ensure no distributed code is accidentally triggered
        distributed = DistributedConf()
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        # single rank
        self.rank = 0
        self.local_rank = 0
        self.distributed_rank = 0

        self._setup_device(accelerator)

        # Remove distributed backend setup and set CUDA configurations
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda.matmul_allow_tf32
                if cuda.matmul_allow_tf32 is not None
                else cuda.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda.cudnn_allow_tf32
                if cuda.cudnn_allow_tf32 is not None
                else cuda.allow_tf32
            )

        # Logger Setup
        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        self._setup_components()
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")
        # Resume
        if self.checkpoint_conf.resume_from is not None:
            if os.path.exists(self.checkpoint_conf.resume_from):
                dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
                if not os.path.exists(dst):
                    makedir(self.checkpoint_conf.save_dir)
                    g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            else:
                logging.warning(
                    f"Resume checkpoint {self.checkpoint_conf.resume_from} not found!"
                )

        self.load_checkpoint()
        # Remove DDP wrapping and distributed setup
        # self._setup_ddp_distributed_training(...)

    def _setup_timers(self):
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        pass  # Do nothing

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_device(self, accelerator):
        # Avoid get_machine_local_and_dist_rank() to prevent potential deoendencies on dist
        self.local_rank = 0
        self.distributed_rank = 0

        if accelerator == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
            torch.cuda.set_device(0)
        else:
            self.device = torch.device("cpu")

    def _move_to_device(self):
        logging.info(f"Moving components to device {self.device}...")
        self.model.to(self.device)
        logging.info("Done moving components.")

    def save_checkpoint(self, epoch, checkpoint_names=None):
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = self.model.state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        if g_pathmgr.exists(checkpoint_path):
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")
        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )
        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        self.best_meter_values = checkpoint.get("best_meter_values", {})
        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        return epoch % self.val_epoch_freq == 0 and epoch < self.max_epochs - 1

    def _step(self, batch: BatchedVideoDatapoint, model: nn.Module, phase: str):
        outputs = model(batch)
        targets = batch.masks
        batch_size = len(batch.img_batch)
        key = batch.dict_key
        loss = self.loss[key](outputs, targets)
        loss_str = f"Losses/{phase}_{key}_loss"
        loss_log_str = os.path.join("Step_Losses", loss_str)

        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(loss_log_str, loss, self.steps[phase])

        self.steps[phase] += 1
        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase in self.meters and key in self.meters[phase]:
            meters_dict = self.meters[phase][key]
            if meters_dict is not None:
                for _, meter in meters_dict.items():
                    meter.update(find_stages=outputs, find_metadatas=batch.metadata)

        return ret_tuple

    def run(self):
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None
        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))
        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            # barrier()  <-- REMOVED
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)

            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "train_stats.json"), "a"
            ) as f:
                f.write(json.dumps(outs) + "\n")

            self.save_checkpoint(self.epoch + 1)
            del dataloader
            gc.collect()

            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()

            self.best_meter_values.update(self._get_trainer_state("train"))
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "best_stats.json"), "a"
            ) as f:
                f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return
        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)
        with g_pathmgr.open(
            os.path.join(self.logging_conf.log_dir, "val_stats.json"), "a"
        ) as f:
            f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)
        curr_phases = [phase]
        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        self.model.eval()
        if hasattr(self.model, "on_validation_epoch_start"):
            self.model.on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()
        for data_iter, batch in enumerate(val_loader):
            data_time.update(time.time() - end)
            batch = batch.to(self.device, non_blocking=True)
            with torch.no_grad():
                with torch.amp.autocast(
                    "cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        batch, self.model, phase
                    )
                    assert len(loss_dict) == 1
                    loss_key, loss = loss_dict.popitem()
                    loss_mts[loss_key].update(loss.item(), batch_size)
                    for k, v in extra_losses.items():
                        if k not in extra_loss_mts:
                            extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                        extra_loss_mts[k].update(v.item(), batch_size)

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )
            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)
            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)
            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            # if data_iter % 10 == 0: dist.barrier() <-- REMOVED

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        if hasattr(self.model, "on_validation_epoch_end"):
            self.model.on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN
        iters_per_epoch = len(train_loader)
        loss_names = [
            f"Losses/{phase}_{batch_key}_loss" for batch_key in self.loss.keys()
        ]
        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            batch = batch.to(self.device, non_blocking=True)

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs

                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(self.model, rank=0, where=self.where)

                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                batch_time_meter.update(time.time() - end)
                end = time.time()
                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )
                mem_meter.update(reset_peak_usage=True)

                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        # self._log_sync_data_times(Phase.TRAIN, data_times) # Masked out sync

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])
        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss

    def _run_step(self, batch, phase, loss_mts, extra_loss_mts, raise_on_error=True):
        self.optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            "cuda",
            enabled=self.optim_conf.amp.enabled,
            dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
        ):
            loss_dict, batch_size, extra_losses = self._step(batch, self.model, phase)

        assert len(loss_dict) == 1
        loss_key, loss = loss_dict.popitem()

        if not math.isfinite(loss.item()):
            error_msg = f"Loss is {loss.item()}, attempting to stop training"
            logging.error(error_msg)
            if raise_on_error:
                raise FloatingPointError(error_msg)
            else:
                return

        self.scaler.scale(loss).backward()
        loss_mts[loss_key].update(loss.item(), batch_size)
        for extra_loss_key, extra_loss in extra_losses.items():
            if extra_loss_key not in extra_loss_mts:
                extra_loss_mts[extra_loss_key] = AverageMeter(
                    extra_loss_key, self.device, ":.2e"
                )
            extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            # 單機版不 sync
            meter_output = (
                {k: v for k, v in zip(meter.keys, meter.val) if meter.val is not None}
                if hasattr(meter, "keys")
                else {}
            )
            # Fallback for simple meters
            if not meter_output and hasattr(meter, "val"):
                meter_output = {"val": meter.val}

            is_better_check = getattr(meter, "is_better", None)
            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value
                if is_better_check is None:
                    continue
                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value, self.best_meter_values[tracked_meter_key]
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value
                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)
        return out_dict

    def _log_timers(self, phase):
        # 簡化 Timer log
        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        pass  # 簡化檢查

    def _setup_components(self):
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components...")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}
        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        print_model_summary(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.optim_conf.amp.enabled if self.optim_conf else False
        )
        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )
        logging.info("Finished setting up components.")

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    logging.info(
        f"Model Summary: Total params: {total_parameters}, Trainable: {trainable_parameters}"
    )
