MODEL_TYPE="prometheus"
DATA_TYPE="prometheus-ood"

python -u src/cascaded_eval.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --logit-file-gpt "outputs/${DATA_TYPE}-gpt-4-1106-preview-vanilla.jsonl"