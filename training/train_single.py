import logging
import os
import sys
from argparse import ArgumentParser
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf
from training.utils.train_utils import makedir, register_omegaconf_resolvers

# Disable NCCL's attempts to use multiple GPUs
os.environ["world_size"] = "1"
os.environ["rank"] = "0"
os.environ["local_rank"] = "0"
os.environ["master_addr"] = "localhost"
os.environ["master_port"] = "12345"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def main(args) -> None:
    cfg = compose(config_name=args.config)

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(
            os.getcwd(), "sam2_logs", args.config
        )
    print(f"Log Dir: {cfg.launcher.experiment_log_dir}")
    add_pythonpath_to_sys_path()
    makedir(cfg.launcher.experiment_log_dir)

    with g_pathmgr.open(
        os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w"
    ) as f:
        f.write(OmegaConf.to_yaml(cfg))

    cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
    cfg_resolved = OmegaConf.create(cfg_resolved)

    if "distributed" in cfg.trainer:
        cfg.trainer.distributed.backend = None

    #
    try:
        trainer = instantiate(cfg.trainer, _recursive_=False)
        trainer.run()
    except Exception as e:
        print(f"Training crashed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=str,
        help="path to config file",
    )
    parser.add_argument("--use-cluster", type=int, default=0)
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--account", type=str, default=None)
    parser.add_argument("--qos", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)

    args = parser.parse_args()
    register_omegaconf_resolvers()

    # execute w/o calling single_node_runner
    main(args)
