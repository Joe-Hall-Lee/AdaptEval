#!/bin/bash
python -u src/evaluate_gpt.py \
    --model-name "gpt-4-1106-preview" \
    --prompt-type "vanilla" \
    --data-type "halu-eval-qa" \
    --multi-process True \
    --max-new-token 1024