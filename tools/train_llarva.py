#!/usr/bin/env python3

"""Train a policy with MMA."""

import os

import hydra
import omegaconf
# import submitit
import torch
# import torch.distributed as dist
import wandb
# from omegaconf import OmegaConf

# import mvp.mma.mma as mma
import llava.train.train_mem as train
# from mvp.mma.utils import suppress_wandb # suppress_print
from utils.sys_utils import (
    dump_cfg,
    omegaconf_to_dict,
    print_dict,
    set_np_formatting,
    set_seed,
    suppress_wandb
)


def add_env_vars_to_config(cfg: omegaconf.DictConfig):
    with omegaconf.open_dict(cfg):
        for env_var_key, env_var_value in os.environ.items():
            env_var_key = env_var_key.upper()
            if (
                env_var_key.startswith("BC_")
                or env_var_key.startswith("SLURM_")
                or env_var_key.startswith("PBS_")
                or "RANK" in env_var_key
                or "NODE" in env_var_key
                or "WORLD" in env_var_key
                or "GLOBAL" in env_var_key
                or "MASTER" in env_var_key
                or "PORT" in env_var_key
                or "ADDR" in env_var_key
                or "GPU" in env_var_key
            ) and not (
                "VSCODE" in env_var_key
            ):
                cfg[f"HPC_{env_var_key}"] = env_var_value


def init_distributed(cfg):
    # Set up distributed env
    print("Start init_distributed()")
    print(f"cfg.num_gpus: {cfg.num_gpus}")

    if cfg.num_gpus > 1:
        print("Initializing distributed training.")
        print("Starting init_process_group")

        master_address = os.environ["MASTER_ADDR"]
        master_port = os.environ["MASTER_PORT"]
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        print(f"master_address: {master_address}, master_port: {master_port}, world_size: {world_size}, rank(global): {global_rank}, local_rank: {local_rank}" )

        # dist.init_process_group("nccl", world_size=world_size, rank=global_rank)
        print("Finished init_process_group")
        print("Available device count: ", torch.cuda.device_count())
        with omegaconf.open_dict(cfg):
            cfg.distributed = True
            cfg.master_address = master_address
            cfg.master_port = master_port
            cfg.local_rank = local_rank
            cfg.global_rank = global_rank
            cfg.world_size = world_size

        torch.cuda.set_device(local_rank)

    else:
        print("Not using distributed training")
        with omegaconf.open_dict(cfg):
            cfg.distributed = False
            cfg.local_rank = 0
            cfg.global_rank = 0
    print(f"cfg.disttributed:{cfg.distributed}, local_rank:{cfg.local_rank}, global_rank:{cfg.global_rank}")


@hydra.main(version_base=None, config_name="config_llarva", config_path="../configs/")
def train(cfg: omegaconf.DictConfig):


    for env_var_key, env_var_value in os.environ.items():
        print(f"{env_var_key}: {env_var_value}")

    # cfg.num_gpus = int(cfg.num_gpus)

    # if cfg.resuming_logs_from_id != "":

    #     # Adjust to previous logs if exits
    #     logdir_temp = cfg.logdir.split('/')
    #     logdir_temp[-2] = cfg.resuming_logs_from_id
    #     cfg.logdir = '/' + os.path.join(*logdir_temp)

        # # Adjust wandb to previous logs if exits
        # wandb_name_temp = cfg.wandb.name.split('_')
        # wandb_name_temp[-1] = cfg.resuming_logs_from_id
        # cfg.wandb.name = '_'.join(wandb_name_temp)

    # if cfg.submission_type == "submitit":
    #     init_distributed_submitit(cfg)
    # elif cfg.submission_type == "manual":
    init_distributed(cfg)
    # else:
    #     print('Please check submission type: submitit or manual on slurm and manual only on pbs')
    
    # Set up logging
    if cfg.local_rank == 0:
        add_env_vars_to_config(cfg)
        print_dict(omegaconf_to_dict(cfg))
        os.makedirs(cfg.output_dir, exist_ok=True)
        dump_cfg(cfg, cfg.output_dir)
        wandb.init(
            **cfg.wandb,
            dir=cfg.output_dir,
            config=omegaconf.OmegaConf.to_container(cfg),
            resume='allow'
        )
    else:
        # suppress_print()

        suppress_wandb()

    # Set rng seed
    # seed = cfg.seed * cfg.num_gpus + local_rank
    seed = cfg.seed
    set_np_formatting()
    set_seed(seed)

    # Perform training
    print("starting train()")

    # Write environment variables to the file
    with open(os.path.join(cfg.logdir, "env.log"), "w") as file:
        for key, value in os.environ.items():
            file.write(f"{key}={value}\n")

    train(cfg)

    # Clean up
    # if cfg.num_gpus > 1:
    #     dist.destroy_process_group()


if __name__ == "__main__":
    for k, v in os.environ.items():
        k = k.upper()
        if ("RANK" in k or "GPU" in k or "TASK" in k or "WORLD" in k or "NODE" in k or "SLURM" in k) and not ("VSCODE" in k):
            print(f"{k}: {v}")
    train()
