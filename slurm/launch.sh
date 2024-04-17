#!/bin/bash

# * ===============================================================================================
# * ===============================================================================================
# * Variables/Settings:

# * Auto-detect the PROJ_ROOT based on where the git repo root is. We use this to
# * reference any files within the repo so that this script is agnostic to where the user
# * calls it from
export PROJ_ROOT="$(git rev-parse --show-toplevel)"

# * EXP_NAME will always be used to build EXP_ID. EXP_ID might simply have more unique
# * identifiers like SLURM_JOB_ID. This should be part of the EXP_ID. It is also OK to
# * make EXP_ID and EXP_NAME have the same value. The only difference is that EXP_ID
# * might have more specific values as part of the name in addition to EXP_NAME.
export EXP_NAME="deepspeed_test_python_distinit"

# * EXP_ID should be used as part of the OUTPUT_DIR, and WANDB_RUN name. It is a unique
# * identifier for an experiment/run and should mostly be human readable. In some cases we
# * want to build a folder structure within OUTPUT_DIR that has sub-folders with the
# * SLURM_JOB_ID for example, to capture the idea that a single experiment might involve
# * multiple SLURM_JOB_IDS (which is why we don't include the SLURM_JOB_ID in the EXP_ID,
# * but we could if we wanted).
export EXP_ID="${EXP_NAME}"

# * Coordinate the value for OUTPUT_DIR, LOG_DIR, and the --output and --error SBATCH params from
# * the .slurm script so they both log to the same place:
export OUTPUT_DIR="${PROJ_ROOT}/output/${EXP_ID}"
export LOG_DIR="${OUTPUT_DIR}/logs"
export DATA_DIR="${PROJECTS_HOME}/nga-frontier/llarva/data"

# * Job settings
export NPROCS_PER_NODE=128
export NNODES=2
export NGPUS_PER_NODE=4
export NGPUS_TOTAL=$((NNODES * NGPUS_PER_NODE))
export NCPUS_PER_TASK=$((NPROCS_PER_NODE / NGPUS_PER_NODE))
echo "NGPUS_TOTAL: $NGPUS_TOTAL"
echo "NCPUS_PER_TASK: $NCPUS_PER_TASK"


# * ===============================================================================================
# * ===============================================================================================
# * Prepare the launch script (setup folders, etc)
echo "${PROJ_ROOT}"
pushd "${PROJ_ROOT}"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}/checkpoints"


# * ===============================================================================================
# * ===============================================================================================
# * Build the SLURM script:
# * https://slurm.schedmd.com/sbatch.html
# * filename replacements: common ones listed here,
# * see https://slurm.schedmd.com/sbatch.html#SECTION_%3CB%3Efilename-pattern%3C/B%3E for full list):
# *   %j: job_id, %N: short hostname, %n node_id relative to current job, %t task num, %x: job name
cat - >launch.slurm <<EOT
#!/bin/bash

##  NOTE!! You cannot use env vars in the #SBATCH params section. So make sure to do all
##  bash variable expansions for "#SBATCH ..." options before you write the .slurm file

# Parameters
#SBATCH --output="${LOG_DIR}/%j/node%n-task%t-slurm.out.log"
#SBATCH --error="${LOG_DIR}/%j/node%n-task%t-slurm.err.log"
#SBATCH --account="${ACCOUNT}"
#SBATCH --job-name="${EXP_NAME}"
#SBATCH --open-mode=append
#SBATCH --signal=USR2@120
## time: format is: d-hhh:mm:ss
#SBATCH --time=000:30:00
#SBATCH --constraint=mla
## qos: Get a list of possible values using: "sacctmgr show qos"
##  - raider: background | debug | frontier | hie | high | normal | standard | transfer | urgent
##  - nautilus:
##      standard | transfer | background | debug | frontier | frontier_long | hie | high | int_transfer
##      | normal | short-mla-frontier | short-mla-high | short-mla-standard  | urgent
#SBATCH --qos=debug
## gres:
##  - raider/nautilus have 1x(A40 48GB) 's for viz nodes, and 4x(A100 40GB)'s for mla nodes:
#SBATCH --gres=gpu:a100:${NGPUS_PER_NODE}
##SBATCH --gres=gpu:a40:${NGPUS_PER_NODE}

#SBATCH --nodes=${NNODES}
#SBATCH --ntasks-per-node=${NGPUS_PER_NODE}
#SBATCH --ntasks=${NGPUS_TOTAL}
#SBATCH --gpu-bind=none                   # NCCL can't deal with task-binding...
#SBATCH --cpus-per-task=${NCPUS_PER_TASK}

# Pass values from outside script to the launch.slurm file we're building:
export PROJ_ROOT="${PROJ_ROOT}"
export EXP_NAME="${EXP_NAME}"
export EXP_ID="${EXP_ID}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export LOG_DIR="${LOG_DIR}"
export DATA_DIR="${DATA_DIR}"
export NNODES="${NNODES}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE}"
export NGPUS_TOTAL="${NGPUS_TOTAL}"
export NCPUS_PER_TASK="${NCPUS_PER_TASK}"


# * TODO: See here: https://github.com/Lightning-AI/pytorch-lightning/issues/18650
# * Read that link and optimize our settings accordingly
export SRUN_CPUS_PER_TASK=\$((SLURM_CPUS_ON_NODE / NGPUS_PER_NODE))
echo "SRUN_CPUS_PER_TASK: \${SRUN_CPUS_PER_TASK}"

# * Don't base this off of EXP_NAME an/or EXP_ID, the two values have nothing to do
# * with each other. This is an independent setting whose value is just the name of your
# * conda  environment:
export ENV_NAME="llarva"

# Source conda env
if [[ -d "\${HOME}/mambaforge" ]]; then
    CONDA_FN="mamba"
    CONDA_DIR="\${HOME}/mambaforge"
elif [[ -d "\${HOME}/anaconda3" ]]; then
    CONDA_FN="conda"
    CONDA_DIR="\${HOME}/anaconda3"
elif [[ -d "\${HOME}/miniconda3" ]]; then
    CONDA_FN="conda"
    CONDA_DIR="\${HOME}/miniconda3"
fi
echo "CONDA_FN: \$CONDA_FN"
echo "CONDA_DIR: \$CONDA_DIR"
## Activate Conda (or Miniconda, or Mamba)
echo "Sourcing CONDA_FN: '\$CONDA_FN' from location: '\${CONDA_DIR}'"
if [ -d "\${CONDA_DIR}/etc/profile.d" ]; then
    echo "Sourcing \${CONDA_DIR}/etc/profile.d/conda.sh"
    source "\${CONDA_DIR}/etc/profile.d/conda.sh"
fi
if [ -f "\${CONDA_DIR}/etc/profile.d/mamba.sh" ]; then
    echo "Sourcing \${CONDA_DIR}/etc/profile.d/mamba.sh"
    source "\${CONDA_DIR}/etc/profile.d/mamba.sh"
fi
echo "Activating environment"
\$CONDA_FN activate "\${ENV_NAME}"
echo "environment activated"
\$CONDA_FN info --envs
env > "\${LOG_DIR}/env.log"
env

# * Distributed Setup

### Get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=\$(scontrol show hostnames "\$SLURM_NODELIST" | head -n 1)

echo ""
# Thanks for shell-ideas to https://github.com/PrincetonUniversity/multi_gpu_training
export ZMASTER_PORT=\$(expr 10000 + \$(echo -n \$SLURM_JOBID | tail -c 4))
export ZMASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
echo "ZMASTER_ADDR:ZMASTER_PORT="\${ZMASTER_ADDR}:${ZMASTER_PORT}

### Change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
export MASTER_ADDR="\${master_addr}"
export MASTER_PORT=29500
export WORLD_SIZE="\${NGPUS_TOTAL}"
export RANK="\${SLURM_PROCID}"          # global rank
export LOCAL_RANK="\${SLURM_LOCALID}"

echo "SLURM_NODELIST=\${SLURM_NODELIST}"
echo "MASTER_ADDR: \${MASTER_ADDR}"
echo "MASTER_PORT: \${MASTER_PORT}"
echo "WORLD_SIZE: \${WORLD_SIZE}"
echo "RANK: \${RANK}"
echo "LOCAL_RANK: \${LOCAL_RANK}"
echo "SLURM_PROCID: \${SLURM_PROCID}"
echo "SLURM_LOCALID: \${SLURM_LOCALID}"

# Debugging vars for the run
export HYDRA_FULL_ERROR=1           # Hydra full error
export OC_CAUSE=1                   # OmegaConf full trace
export NCCL_DEBUG=WARN

# command
pushd "\${PROJ_ROOT}"

srun \\
    --unbuffered \\
    --output "\${LOG_DIR}/%j/node%n-task%t-srun.out.log" \\
    --error "\${LOG_DIR}/%j/node%n-task%t-srun.err.log" \\
    --jobid \$SLURM_JOBID \\
    python ${PROJ_ROOT}/llava/train/train_mem.py \\
        --deepspeed "\${PROJ_ROOT}/scripts/zero2.json" \\
        --model_name_or_path lmsys/vicuna-7b-v1.5 \\
        --version plain \\
        --data_path "\${DATA_DIR}/process_dataset/large_scape_training/new_action_version/exp4/val-tiny.json" \\
        --image_folder "\${DATA_DIR}/v2" \\
        --vision_tower openai/clip-vit-large-patch14-336 \\
        --mm_projector_type mlp2x_gelu \\
        --tune_mm_mlp_adapter True \\
        --mm_vision_select_layer -2 \\
        --mm_use_im_start_end False \\
        --mm_use_im_patch_token False \\
        --bf16 True \\
        --output_dir "\${OUTPUT_DIR}/checkpoints/llava-v1.5-7b-pretrain_exp4" \\
        --num_train_epochs 1 \\
        --per_device_train_batch_size 8 \\
        --per_device_eval_batch_size 16 \\
        --gradient_accumulation_steps 2 \\
        --evaluation_strategy "steps" \\
        --eval_steps 1000 \\
        --save_strategy "steps" \\
        --save_steps 1000 \\
        --save_total_limit 1 \\
        --learning_rate 1e-3 \\
        --weight_decay 0.0 \\
        --warmup_ratio 0.03 \\
        --lr_scheduler_type "cosine" \\
        --logging_steps 1 \\
        --tf32 True \\
        --model_max_length 2048 \\
        --gradient_checkpointing True \\
        --dataloader_num_workers 4 \\
        --lazy_preprocess True \\
        --report_to "none"

EOT

# * ===============================================================================================
# * ===============================================================================================
# * Run the SLURM script:

# * Copy a snapshot of the launch script to the output dir:
cp launch.slurm "${OUTPUT_DIR}/launch.slurm"

# * Add to SLURM queue:
# sbatch "${OUTPUT_DIR}/launch.slurm"
