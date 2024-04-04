#!/bin/bash

#SBATCH --account=ODEFN5169CYFZ
#SBATCH --qos=frontier
#SBATCH --nodes=8
#SBATCH --ntasks=8
#SBATCH --gres=gpu:a100:4
#SBATCH --time=168:00:00
#SBATCH -C mla
#SBATCH --job-name=rpt
#SBATCH --output="created_dirs/slurm_dir/train_N_nodes_rpt_%N_%n_%j.out"
#SBATCH --error="created_dirs/slurm_dir/train_N_nodes_rpt_%N_%n_%j.err"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jquenum@berkeley.edu
#SBATCH --export=ALL
#SBATCH --propagate=STACK

ulimit -s unlimited

show_separator() {
        echo "________________________________________"
}


show_separator
JOBID=`echo $SLURM_JOB_ID`
show_separator
source activate llarva
show_separator
echo "In Conda envrironment:" $CONDA_DEFAULT_ENV
show_separator
nvidia-smi
show_separator
echo "Starting at: $(date)"
show_separator
show_separator

# Set the port number
rendezvous_port=29500

# # Function to find an available port within a range
# find_available_port() {
#     local port_range_start=$1
#     local port_range_end=$2
#     for port in $(seq "$port_range_start" "$port_range_end"); do
#         if ! nc -zvw3 localhost "$port"; then
#             echo "$port"
#             return
#         fi
#     done
#     echo "Error: No available port found in the specified range." >&2
#     exit 1
# }

# # Set the port range
# port_range_start=29399
# port_range_end=55000

# # Find an available port within the specified range
# rendezvous_port=$(find_available_port "$port_range_start" "$port_range_end")
# echo "Using port $rendezvous_port for rendezvous endpoint."

# # Function to check firewall rules
# check_firewall_rules() {
#     echo "Checking firewall rules..."
#     if iptables -C INPUT -p tcp --dport "$rendezvous_port" -s "$1" -j ACCEPT &>/dev/null; then
#         echo "Firewall rule allowing traffic on port $rendezvous_port from $1 exists."
#     else
#         echo "Error: No firewall rule allowing traffic on port $rendezvous_port from $1 found."
#         echo "Please ensure that the firewall allows incoming connections on port $rendezvous_port."
#     fi
# }


# # Function to check connectivity to the rendezvous endpoint
# check_rendezvous_connectivity() {
#     echo "Checking connectivity to rendezvous endpoint: $1"
#     if nc -zvw3 "$1" "$rendezvous_port"; then
#         echo "Rendezvous endpoint is reachable."
#     else
#         echo "Error: Unable to connect to rendezvous endpoint."
#         check_firewall_rules "$1"
#         exit 1
#     fi
# }


nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | xargs)

## Extract individual nodes from the node range
read -ra node_array <<< "$nodes"
head_node=${node_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')


show_separator
show_separator
echo "all nodes:" ${node_array[@]}
echo "head_node:" $head_node
echo "head_node_ip:" $head_node_ip
echo "Num nodes:" $SLURM_NNODES
echo "Num tasks:" $SLURM_NTASKS
echo "Job ID:" $SLURM_JOB_ID
show_separator
show_separator


# Check rendezvous endpoint connectivity
# check_rendezvous_connectivity "$head_node_ip"

export LOGLEVEL=INFO
export OMP_NUM_THREADS=56
export MKL_NUM_THREADS=56
export WANDB_MODE=offline

deepspeed --num_nodes 5 --num_gpus 20 \
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


show_separator
echo "Done at: $(date)"
show_separator