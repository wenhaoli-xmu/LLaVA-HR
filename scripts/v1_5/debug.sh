#!/bin/bash


MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


srun \
    -p Intern5 \
    --job-name=03051200 \
    --nodes=1 \
    --gres=gpu:1 \
    --ntasks=1 \
    --cpus-per-task=12 \
    --quotatype=reserved \
    --kill-on-bad-exit=1 \
    python llava_hr/train/train_mem.py \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path playground/data/llava_v1_5_mix665k.json \
    --image_folder playground/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --vision_tower_slow convnext_large_mlp.clip_laion2b_ft_320 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --pretrain_mm_mlp_adapter ./checkpoints/pretrain-7b/mm_projector.bin \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/debug \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
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
    --modify v1-spaco2
