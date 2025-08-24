MODEL_TYPE="pandalm"
DATA_TYPE="pandalm"

python src/evaluate_threshold.py \
    --data_type ${DATA_TYPE} \
    --judge_confidence_file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --judge_logit_file_with_token "token_costs/${MODEL_TYPE}/${DATA_TYPE}-logit-with-tokens.jsonl" \
    --gpt4_logit_file_with_token "token_costs/gpt4/${DATA_TYPE}-logit-with-tokens.jsonl" \
    --lambdas 0.0 0.15 0.3