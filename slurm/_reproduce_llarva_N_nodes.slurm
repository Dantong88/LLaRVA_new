#!/bin/bash

# Parameters
#SBATCH --exclude=raider-viz01,raider-viz02
#SBATCH --output=created_dirs/slurm_dir/train_${SLURM_JOB_NUM_NODES}_nodes_llarva_%N/%j/%j_0_log.out
#SBATCH --error=created_dirs/slurm_dir/train_${SLURM_JOB_NUM_NODES}_nodes_llarva_%N/%j/%j_0_log.err
#SBATCH --account=ODEFN5169CYFZ
#SBATCH --qos=frontier
#SBATCH --job-name=llarva
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
#SBATCH --time=168:00:00
#SBATCH --constraint=mla
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a40:1
#SBATCH -C viz
#SBATCH --mail-type=ALL
#SBATCH --mail-user=$USER@berkeley.edu
#SBATCH --export=ALL
#SBATCH --propagate=STACK

export WANDB_MODE=offline

ulimit -s unlimited

create_folder_if_not_exists() {
    local folder_name="$1"
    if [ ! -d "$folder_name" ]; then
        mkdir "$folder_name"
        echo "Folder '$folder_name' created."
    else
        echo "Folder '$folder_name' already exists."
    fi
}

# setup
export EXP_NAME="llarva"
# export EXP_ID="${EXP_NAME}_${SLURM_JOB_ID}_${BC_HOST}"
PROJ_ROOT="${HOME}/Projects/${EXP_NAME}"
ENV_NAME=${EXP_NAME}

OUTPUT_DIR="${WORKDIR}/outputs/llarva/llarva_training_dir_${SLURM_JOB_NUM_NODES}_nodes_${BC_HOST}/${SLURM_JOB_ID}"
LOG_DIR="${OUTPUT_DIR}/logs"

echo "PROJ_ROOT ${PROJ_ROOT}"
echo "OUTPUT_DIR ${OUTPUT_DIR}"
echo "LOG_DIR ${LOG_DIR}"

# Check and create folder if the folder does not exists
create_folder_if_not_exists "$OUTPUT_DIR"
create_folder_if_not_exists "$LOG_DIR"

pushd "${PROJ_ROOT}"
cp "${PROJ_ROOT}/example_launch_scripts/manual.raider.slurm" "${LOG_DIR}/"

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

echo "START TIME: $(date)"

# Job settings
# export NGPUS_TOTAL=${SLURM_NTASKS}
# TODO: See here: https://github.com/Lightning-AI/pytorch-lightning/issues/18650
# Read that link and optimize our settings accordingly
# export SRUN_CPUS_PER_TASK=128

# * Distributed Setup
### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=29500
# export WORLD_SIZE="${NGPUS_TOTAL}"

### Get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Prepare hostfile
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_GPUS_ON_NODE"};
$slots=1 if $slots==0; # workaround for gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$b$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile

chmod +r ${HOME}/llarva/hostfile

    # --num_gpus ${SLURM_GPUS_ON_NODE} \
    # --num_nodes ${SLURM_JOB_NUM_NODES} \
# OTHER LAUNCHERS CAN BE USED HERE
export LAUNCHER="deepspeed \
    --hostfile ${HOME}/llarva/hostfile \
    "


# CHANGE HERE THE SCRIPT AND WHATEVER ARGS IT NEEDS
PROGRAM="\
llava/train/train_mem.py \
    --deepspeed ${HOME}/Projects/llarva/scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path ${WORKDIR}/datasets/llarva/exp4/train-34053947.json::${WORKDIR}/datasets/llarva/exp4/val-36743.json \
    --image_folder ${WORKDIR}/datasets/llarva/v2 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${WORKDIR}/outputs/llarva/llarva_training_dir_${SLURM_JOB_NUM_NODES}_nodes_${BC_HOST}/${SLURM_JOB_ID}/logs \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 0.001 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
"

echo "PROGRAM $PROGRAM"

# Debugging vars for the run
export NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1           # Hydra full error
export OC_CAUSE=1                   # OmegaConf full trace


# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --unbuffered \
    --output ${OUTPUT_DIR}/%j_%t_log.out \
    --error ${OUTPUT_DIR}/%j_%t_log.err \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# command
srun $SRUN_ARGS bash -c "$LAUNCHER $PROGRAM"

echo "END TIME: $(date)"