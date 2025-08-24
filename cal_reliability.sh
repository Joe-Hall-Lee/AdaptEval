export CUDA_VISIBLE_DEVICES=6

MODEL_TYPE="judgelm"
DATA_TYPE="llmbar-natural"

python -u src/cal_reliability.py \
    --model-name-or-path "./models/JudgeLM-7B" \
    --cali-model-name-or-path "./models/vicuna-7b/" \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"