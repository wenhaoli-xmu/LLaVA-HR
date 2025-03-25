#!/bin/bash


MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


function makehostfile() {
    > hostfile
    slots=8
    nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
    for node in $nodes; do
        echo "$node slots=$slots" >> hostfile
    done
}
makehostfile


deepspeed \
    --launcher SLURM \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --hostfile='hostfile' \
    --no_ssh_check \
    llava_hr/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path unsloth/llama-3-8b-Instruct \
    --version llama3 \
    --data_path playground/data/llava_v1_5_mix665k.json \
    --image_folder playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --pretrain_mm_mlp_adapter checkpoints/pretrain-8b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir checkpoints/profile \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2496 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --is_multipath_encoder True \
    --freeze_vision False \
    --input_image_size 1024 \
    --modify v1-profile-spaco2

# bash scripts/v1_5/eval.sh ./checkpoints/llava-hr-7b-sft-1024 2>&1 | tee log-llava-hr-7b-sft-1024.txt