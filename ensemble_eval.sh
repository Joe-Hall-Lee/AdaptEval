MODEL_TYPE1="llama-3_general-public"
MODEL_TYPE2="llama-3_critic"
DATA_TYPE="salad-bench"

python -u src/roles.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE1}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE1}/${DATA_TYPE}-relia.json" \
    --logit-file2 "relia_scores/${MODEL_TYPE2}/${DATA_TYPE}-logit.jsonl" \
    --output-file2 "relia_scores/${MODEL_TYPE2}/${DATA_TYPE}-relia.json" \