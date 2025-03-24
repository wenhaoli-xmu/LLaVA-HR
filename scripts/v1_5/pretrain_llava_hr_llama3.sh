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
    --data_path playground/data/pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder playground/data/pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/pretrain-8b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
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
    --report_to wandb \
    --is_multipath_encoder True \
    --input_image_size 384