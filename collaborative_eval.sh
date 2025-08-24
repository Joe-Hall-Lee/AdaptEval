MODEL_TYPE1="llama-3_general-public"
MODEL_TYPE2="llama-3_critic"
DATA_TYPE="llmbar-natural"

python src/collaborative_eval.py \
    --data-type ${DATA_TYPE} \
    --model1-output relia_scores/${MODEL_TYPE1}/${DATA_TYPE}-logit.jsonl \
    --model2-output relia_scores/${MODEL_TYPE2}/${DATA_TYPE}-logit.jsonl \
    --model1-relia relia_scores/${MODEL_TYPE1}/${DATA_TYPE}-relia.json \
    --model2-relia relia_scores/${MODEL_TYPE2}/${DATA_TYPE}-relia.json
