# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Union, cast

# from hydra import compose, initialize, DictConfig
from hydra import main as hydra_main
from omegaconf import OmegaConf, DictConfig
from submitit import AutoExecutor, JobEnvironment
from submitit.helpers import DelayedSubmission, TorchDistributedEnvironment


def parse_args():
    parser = argparse.ArgumentParser("Submitit for RPT Train+Val")
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--timeout", default=10080, type=int, help="Duration of the job (min)"
    )
    parser.add_argument(
        "--job_dir",
        default="",
        type=str,
        help="Job dir. Leave empty for automatic.",
    )

    parser.add_argument(
        "--qos",
        default="qos",
        type=str,
        choices=("frontier", "general", "debug"),
        help="Queue to use",
    )
    parser.add_argument(
        "--comment", default="", type=str, help="Comment to pass to scheduler"
    )
    parser.add_argument(
        "--account", default="", type=str, help="The Account string to use"
    )
    parser.add_argument(
        "--constraint",
        default="mla",
        type=str,
        help="Which constraint to use",
        # TODO: This list is for raider. See if we need to adjust for other centers:
        choices=("standard", "viz", "mla", "xfer", "bigmem", "highclock"),
    )
    parser.add_argument(
        "--config-path",
        default="config/base.yaml",
        type=str,
        help="Path to config file",
    )

    return parser.parse_args()


def load_config(_config_path: Union[Path, str]):
    """"""
    config_path = Path(_config_path).resolve()
    # cfg = cast(DictConfig, OmegaConf.load(config_path))

    assert config_path.is_file, f"Config_path should be a file. You said: '{config_path}'"

    @hydra_main(version_base=None, config_name=config_path.name, config_path=str(config_path.parent))
    def _load_conf(cfg: DictConfig):
        return cfg

    config = _load_conf()
    print("Loaded config: {config_path}")
    print(config)
    sys.exit(0)
    return config


def get_shared_folder() -> Path:
    """
    Returns a path on shared storage. submitit uses this path to store job files
    """
    jobs_dir = Path("/p/work1/projects/nga-frontier/rptx/jobs")
    if not jobs_dir.is_dir():
        raise RuntimeError("No shared folder available")
    jobs_dir.mkdir(exist_ok=True, parents=True)
    print("jobs_dir: ", jobs_dir)
    return jobs_dir


def get_init_file():
    """
    _summary_

    :return: _description_
    :rtype: _type_
    """
    # Init file must not exist, but it's parent dir must exist.
    shared_dir = get_shared_folder()
    init_file = shared_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        init_file.unlink(missing_ok=True)
    return init_file


class Trainer(object):
    """
    _summary_

    :param object: _description_
    :type object: _type_
    """

    def __init__(self, config):
        self.config = config
        print(self.config)

    def __call__(self):
        import mvp.mma.trainer as runner

        self._setup_gpu_args()
        runner.train(self.config)

    def checkpoint(self):
        self.config.dist_url = get_init_file().as_uri()
        print("Requeuing ", self.config)
        empty_trainer = type(self)(self.config)
        return DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        job_env = JobEnvironment()
        dist_env = TorchDistributedEnvironment().export(set_cuda_visible_devices=False)
        self.config.logdir = Path(
            str(self.config.logdir).replace("%j", f"{job_env.job_id}")
        )
        # self.config.model.resume = Path(
        #     str(self.config.model.resume).replace(
        #         "%j", f"{job_env.job_id}"
        #     )
        # )

        # These are needed because submitit errors out otherwise.
        # I thought these were deprecated? Who knows.
        # os.environ["RANK"] = str(self.args.rank)
        # os.environ["WORLD_SIZE"] = str(self.args.world_size)

        print(f"master: {dist_env.master_addr}:{dist_env.master_port}")
        print(
            f"World size: {dist_env.world_size}, Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}"
        )


def main():
    """
    _summary_
    """
    args = parse_args()

    print("args.job_dir1: ", args.job_dir)
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"
        print("args.job_dir2: ", args.job_dir)
    elif "%j" not in args.job_dir:
        args.job_dir = f"{args.job_dir}/%j"
    print("args.job_dir3: ", args.job_dir)

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    if args.constraint == "mla":
        num_gpus_per_node = 4
    elif args.constraint == "viz":
        num_gpus_per_node = 1
    else:
        num_gpus_per_node = 0

    print(f"Num GPUs per node: {num_gpus_per_node}")

    nodes = args.nodes
    timeout_min = args.timeout
    qos = args.qos
    account = args.account
    constraint = args.constraint

    kwargs = {}
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_qos=qos,
        slurm_signal_delay_s=120,
        slurm_account=account,
        slurm_constraint=constraint,
        **kwargs,
    )

    executor.update_parameters(name=os.environ["EXP_NAME"])

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir
    config = load_config(args.config_path)
    print(OmegaConf.to_yaml(config))

    trainer = Trainer(config)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
