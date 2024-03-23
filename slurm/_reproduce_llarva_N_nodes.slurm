#!/bin/bash

## Required Slurm Directives --------------------------------------
#SBATCH --account=ODEFN5169CYFZ
#SBATCH --qos=frontier
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --gres=gpu:a100:4
#SBATCH --time=168:00:00
#SBATCH -C mla
#
## Optional Slurm Directives --------------------------------------
#SBATCH --job-name=llarva
#SBATCH --output="created_dirs/slurm_dir/train_N_nodes_llarva.out"
#SBATCH --error="created_dirs/slurm_dir/train_N_nodes_llarva.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jquenum@berkeley.edu
#SBATCH --export=ALL
#
echo "----------------------------------------"
JOBID=`echo $SLURM_JOB_ID`
echo "----------------------------------------"

source activate rpt
echo "-----------------------------------------"
echo "In Conda envrironment:" $CONDA_DEFAULT_ENV
echo "-----------------------------------------"
nvidia-smi
echo "-----------------------------------------"
echo "Starting at: $(date)"
echo "-----------------------------------------"
echo "-----------------------------------------"




nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | xargs)

## Extract individual nodes from the node range
read -ra node_array <<< "$nodes"
head_node=${node_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')


echo "___________________________"
echo "___________________________"
echo "all nodes:" ${node_array[@]}
echo "head_node:" $head_node
echo "head_node_ip:" $head_node_ip
echo "Num nodes:" $SLURM_NNODES
echo "Num tasks:" $SLURM_NTASKS
echo "Job ID:" $SLURM_JOB_ID
echo "___________________________"
echo "___________________________"

export LOGLEVEL=INFO
export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56

export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export WANDB_MODE=offline

deepspeed --num_nodes 5 --num_gpus 20 \
    /p/home/jquenum/Projects/llarva/llava/train/train_mem.py \
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
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain_exp4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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


echo "-----------------------------------------"
echo "Done at: $(date)"
echo "-----------------------------------------"