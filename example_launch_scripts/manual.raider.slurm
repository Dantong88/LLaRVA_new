#!/bin/bash

# Parameters
#SBATCH --output=/p/home/gbiamby/proj/rpt/created_dirs/output/manual/%j/%j_0_log.out
#SBATCH --error=/p/home/gbiamby/proj/rpt/created_dirs/output/manual/%j/%j_0_log.err
#SBATCH --account=ODEFN5169CYFZ
#SBATCH --job-name=dist_test_rptx
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --time=60
##SBATCH --wckey=submitit
#SBATCH --constraint=mla
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100:4

# setup
export EXP_NAME="pretrain-rptx_vitb_multinode_test"
export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${BC_HOST}"
PROJ_ROOT="${HOME}/proj/rpt"
ENV_NAME="rpt"
OUTPUT_DIR="${PROJ_ROOT}/created_dirs/output/${EXP_NAME}"
LOG_DIR="${OUTPUT_DIR}/${SLURM_JOB_ID}"
echo "${PROJ_ROOT}"
pushd "${PROJ_ROOT}"
cp "${PROJ_ROOT}/launch_scripts/manual.raider.slurm" "${LOG_DIR}/"

# Source conda env
if [[ -d "${HOME}/mambaforge" ]]; then
    CONDA_FN="mamba"
    CONDA_DIR="${HOME}/mambaforge"
elif [[ -d "${HOME}/anaconda3" ]]; then
    CONDA_FN="conda"
    CONDA_DIR="${HOME}/anaconda3"
elif [[ -d "${HOME}/miniconda3" ]]; then
    CONDA_FN="conda"
    CONDA_DIR="${HOME}/miniconda3"
fi
echo "CONDA_FN: $CONDA_FN"
echo "CONDA_DIR: $CONDA_DIR"
## Activate Conda (or Miniconda, or Mamba)
echo "Sourcing CONDA_FN: '$CONDA_FN' from location: '${CONDA_DIR}'"
if [ -d "${CONDA_DIR}/etc/profile.d" ]; then
    echo "Sourcing ${CONDA_DIR}/etc/profile.d/conda.sh"
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
if [ -f "${CONDA_DIR}/etc/profile.d/mamba.sh" ]; then
    echo "Sourcing ${CONDA_DIR}/etc/profile.d/mamba.sh"
    source "${CONDA_DIR}/etc/profile.d/mamba.sh"
fi
$CONDA_FN activate "${ENV_NAME}"
$CONDA_FN info --envs
env > "${LOG_DIR}/env.log"


# Job settings
export NGPUS_TOTAL=${SLURM_NTASKS}
# TODO: See here: https://github.com/Lightning-AI/pytorch-lightning/issues/18650
# Read that link and optimize our settings accordingly
export SRUN_CPUS_PER_TASK=8

# * Distributed Setup
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=29500
export WORLD_SIZE="${NGPUS_TOTAL}"

### Get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Debugging vars for the run
export HYDRA_FULL_ERROR=1           # Hydra full error
export OC_CAUSE=1                   # OmegaConf full trace

# command
srun \
    --unbuffered \
    --output ${OUTPUT_DIR}/%j/%j_%t_log.out \
    --error ${OUTPUT_DIR}/%j/%j_%t_log.err \
    python -u \
        ${PROJ_ROOT}/tools/train_mma.py \
            wandb.mode=disabled \
            train.mb_size=256 \
            num_gpus="${NGPUS_TOTAL}" \
            --config-name "mma/rptx/config_vitl.yaml"


