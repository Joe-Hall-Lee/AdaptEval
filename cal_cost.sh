export CUDA_VISIBLE_DEVICES=2

MODEL_TYPE="pandalm"
DATA_TYPE="pandalm"
MODEL_PATH="./models/PandaLM-7B"
python src/cal_cost.py \
    --data_type ${DATA_TYPE} \
    --judge_model_path ${MODEL_PATH} \
    --judge_model_type ${MODEL_TYPE} \
    --tokenizer_path ${MODEL_PATH} \
    --batch_size 1 \
    --judge_logit_file_in "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --gpt4_logit_file_in "outputs/${DATA_TYPE}-gpt-4-1106-preview-vanilla.jsonl" \
    --judge_logit_file_out "token_costs/${MODEL_TYPE}/${DATA_TYPE}-logit-with-tokens.jsonl" \
    --gpt4_logit_file_out "token_costs/gpt4/${DATA_TYPE}-logit-with-tokens.jsonl"