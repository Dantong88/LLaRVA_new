#!/bin/bash
# * what tags can you use?
#   * "%j", "%t", "%A", "%a"
export EXP_NAME="pretrain-rptx_vitb_multinode_test"
export EXP_ID="${EXP_NAME}"
# * Auto-detect the PROJ_ROOT based on where this script is:
# shellcheck disable=SC2155
export SCRIPTS_DIR="$(builtin cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC2155
export PROJ_ROOT="$(realpath "${SCRIPTS_DIR}"/../)"
# * All output for a job will go somewhere under this folder (logs, wandb, etc):
export OUTPUT_DIR="${PROJ_ROOT}/created_dirs/output/${EXP_ID}"
export LOG_DIR="${OUTPUT_DIR}/logs"
export PRETRAINED_CKPT_PATH="${PROJ_ROOT}/pretrained_weights"
export ENV_NAME="rpt"
export WANDB_MODE=disabled

CONSTRAINT="${1}"
QUEUE_NAME="${2:-debug}"

### init virtual environment if needed
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
if [ -d "${CONDA_DIR}/etc/profile.d" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/conda.sh"
fi
if [ -f "${CONDA_DIR}/etc/profile.d/mamba.sh" ]; then
    # shellcheck disable=SC1091
    source "${CONDA_DIR}/etc/profile.d/mamba.sh"
fi
$CONDA_FN activate rpt

cd "${PROJ_ROOT}"

# # CUDA_LAUNCH_BLOCKING=1 \
# # TORCH_DISTRIBUTED_DEBUG=DETAIL \
# # shellcheck disable=SC2097
# NUMEXPR_MAX_THREADS=128 \
#     PYTHONUNBUFFERED=1 \
#     python $PROJ_ROOT/launch_scripts/submitit_train_cluster.py \
#     --account "${ACCOUNT}" \
#     --job_dir "${OUTPUT_DIR}/jobs" \
#     --constraint "${CONSTRAINT}" \
#     --qos "${QUEUE_NAME}" \
#     --timeout 60 \
#     --nodes 1 \
#     --config-path "${PROJ_ROOT}/configs/mma/rptx/config_vitl.yaml"

# CUDA_LAUNCH_BLOCKING=1 \
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# shellcheck disable=SC2097
NUMEXPR_MAX_THREADS=128 \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1 \
    python tools/train_mma.py \
        hydra/launcher=submitit_slurm \
        --multirun \
        --config-name "mma/rptx/config_vitl.yaml"
