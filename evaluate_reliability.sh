MODEL_TYPE="prometheus"
DATA_TYPE="halu-eval-qa"

python -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type ${DATA_TYPE} \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"