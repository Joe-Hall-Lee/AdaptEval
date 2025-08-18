#!/bin/bash
python -u src/evaluate_gpt.py \
    --model-name "gpt-4-1106-preview" \
    --prompt-type "vanilla" \
    --data-type "judgelm" \
    --multi-process False \
    --max-new-token 1024