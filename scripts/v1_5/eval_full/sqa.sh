#!/bin/bash
MODEL_PATH=$1
CKPT='llava-8b-baseline'

python -m llava_hr.eval.model_vqa_science \
    --model-path $1 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llama3

python llava_hr/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${CKPT}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
