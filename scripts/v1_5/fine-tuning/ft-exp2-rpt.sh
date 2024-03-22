# this is for the exp4: OXE without trajectory

#!/bin/bash

deepspeed --include localhost:7 llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /home/niudt/LLaVA/checkpoints/merge/llava-v1.5-7b-exp4_quater \
    --version v1 \
    --data_path '/home/niudt/LLaVA/process_dataset/finetune/finetuning_anns/berkeley_rpt_converted_externally_to_rlds_new-pick blue cube-14706-train.json::/home/niudt/LLaVA/process_dataset/finetune/finetuning_anns/berkeley_rpt_converted_externally_to_rlds_new-pick blue cube-78399-val.json' \
    --image_folder '/scratch/partial_datasets/llarva/rtx/v2' \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/niudt/LLaVA/checkpoints/llava-v1.5-7b-pretrain/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/lora/llava-v1.5-7b-lora_ft_rpt_pickbluebin \
    --num_train_epochs 30 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_steps 1000 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
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
