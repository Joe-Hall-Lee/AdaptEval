# A Confidence-Based Adaptive Hierarchical Evaluation Method for Large Language Models

This is the official repository for paper **A Confidence-Based Adaptive Hierarchical Evaluation Method for Large Language Models**.

## ⚡️ Usage
### Preparation
Please refer to the following command to prepare your environment.

```shell
conda create -n adapteval python=3.11
pip install -r requirements.txt
```
Please download pre-trained LLMs and put them under ``models``. Specifically, our study are based on the following four finetuned models:

* [JudgeLM-7B](https://huggingface.co/BAAI/JudgeLM-7B-v1.0)

* [PandaLM-7B](https://huggingface.co/WeOpenML/PandaLM-7B-v1)

* [Prometheus-7b-v1.0](https://huggingface.co/kaist-ai/prometheus-7b-v1.0)

* [Auto-J-13b](https://huggingface.co/GAIR/autoj-13b)

To obtain the calibrated reliability scores, or to finetune your own judge model for comparison, you also need to download the following base models:

* [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.3)

* [Llama-7B](https://huggingface.co/huggyllama/llama-7b)

* [Llama2-chat-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)

* [Llama2-chat-13B](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)

Our study are based on the following data, and we have downloaded the respective testsets and put them under ``data``. 

* [JudgeLM-test](https://huggingface.co/datasets/BAAI/JudgeLM-100K/)

* [PandaLM-test](https://github.com/WeOpenML/PandaLM/blob/main/data/testset-v1.json)

* [Auto-J-test](https://github.com/GAIR-NLP/auto-j/blob/main/data/test/testdata_pairwise.jsonl)

* [Prometheus-test](https://github.com/kaistAI/prometheus/blob/main/evaluation/benchmark/data)

* [MT-bench](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)

* [LLMBar](https://github.com/princeton-nlp/LLMBar/tree/main/Dataset/LLMBar)

* [Halu-Eval](https://github.com/RUCAIBox/HaluEval/tree/main/data)

* [Toxic-Chat](https://huggingface.co/datasets/lmsys/toxic-chat)

* [SALAD-Bench](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)

## Evaluate judges on different benchmarks

Run the following script to evaluate the open-source judge model on different testsets.

```shell
MODEL_PATH=./models/JudgeLM-7B
MODEL_TYPE=judgelm
PROMPT_TYPE=vanilla
DATA_TYPE=judgelm
python3 -u src/evaluate_judge.py \
    --model-name-or-path $MODEL_PATH \
    --prompt-type $PROMPT_TYPE \
    --model-type $MODEL_TYPE \
    --data-type $DATA_TYPE \
    --max-new-token 1024
```

Run the following script to evaluate the proprietary model on different testsets.

```shell
MODEL_NAME=gpt-3.5-turbo-0613
PROMPT_TYPE=vanilla
DATA_TYPE=judgelm
python3 -u src/evaluate_gpt.py \
    --model-name $MODEL_NAME \
    --prompt-type $PROMPT_TYPE \
    --data-type $DATA_TYPE \
    --multi-process True \
    --max-new-token 1024 \
    --rewrite-output True
```

## Integrated evaluate with *AdaptEval*
Run the following script to obtain the confidence scores.

```shell
MODEL_PATH=./models/JudgeLM-7B
BASE_MODEL_PATH=./models/Vicuna-7B
MODEL_TYPE=judgelm
DATA_TYPE=salad-bench

python3 -u src/cal_reliability.py \
    --model-name-or-path $MODEL_PATH \
    --cali-model-name-or-path $BASE_MODEL_PATH \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --max-new-token 1024 \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"

```

After that, you can run the following script to perform *AdaptEval*, by allocating the less confident samples to GPT-4 for re-evaluation.

```shell
MODEL_TYPE="judgelm"
DATA_TYPE="salad-bench"
python3 -u src/cascaded_eval.py \
    --data-type $DATA_TYPE \
    --logit-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file1 "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json" \
    --logit-file-gpt "outputs/${DATA_TYPE}-gpt-4-turbo-128k-vanilla.jsonl"
```

You can also run the following script to evaluate the effectiveness of the scores, by bucketing the testset according to the score:

```shell
MODEL_TYPE=judgelm
DATA_TYPE=salad-bench
python3 -u src/evaluate_reliability.py \
    --model-type ${MODEL_TYPE} \
    --data-type $DATA_TYPE \
    --logit-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-logit.jsonl" \
    --output-file "relia_scores/${MODEL_TYPE}/${DATA_TYPE}-relia.json"
```
