from build_prompt_gpt import parse_score_gpt, create_prompt_gpt
from build_dataset import build_dataset, calculate_metrics
import json
import argparse
import random
import time
import os
import multiprocessing
import openai
import tqdm
from dotenv import load_dotenv
load_dotenv()


# -------------------- 参数 --------------------

def build_params_gpt():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--prompt-type", choices=("vanilla", "cot"), default=None)
    p.add_argument("--data-type", choices=("judgelm", "pandalm", "auto-j", "prometheus-ind",
                                           "prometheus-ood", "mt-bench", "halu-eval-summary",
                                           "halu-eval-qa", "halu-eval-dialogue", "salad-bench",
                                           "toxic-chat", "llmbar-neighbor", "llmbar-natural",
                                           "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
                   default=None)
    p.add_argument("--data-path", default="./data")
    p.add_argument("--max-new-token", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--logit-file", type=str, default=None)
    p.add_argument("--pool-number", type=int, default=10)
    p.add_argument("--multi-process", type=str, default="False")
    p.add_argument("--rewrite-output", type=str, default="False")
    return p.parse_args()



def request_gpt(prompt, model, temperature, max_new_tokens):
    model = "gpt-4-1106-preview"
    client = openai.OpenAI(api_key=os.getenv("API_KEY"),
                           base_url="https://api.shubiaobiao.cn/v1/")
    for _ in range(5):
        try:
            return client.chat.completions.create(
                model=model, temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            ).choices[0].message.content
        except Exception as e:
            time.sleep(5)
    return "0 0"


def _worker(args_tuple):
    prompt, answer, model, temp, max_tok, data_t, prompt_t = args_tuple
    pred = request_gpt(prompt, model, temp, max_tok)
    score = parse_score_gpt(pred, data_t, prompt_t)
    return score, pred, prompt, answer


if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    dataset = build_dataset(args.data_type, args.data_path)
    instruction = create_prompt_gpt(args.data_type, args.prompt_type)

    prompts, answers = [], []
    for ex in dataset:
        if args.data_type in ["prometheus-ind", "prometheus-ood", "halu-eval-summary",
                              "halu-eval-dialogue", "halu-eval-qa", "toxic-chat"]:
            prompt = instruction.format(question_body=ex["question_body"],
                                        rubric=ex["rubric"],
                                        answer_body=ex["answer_body"])
        else:
            ex["rubric"] = ("Please rate the helpfulness, relevance, "
                            "accuracy, level of details of their responses.")
            prompt = instruction.format(question_body=ex["question_body"],
                                        rubric=ex["rubric"],
                                        answer1_body=ex["answer1_body"],
                                        answer2_body=ex["answer2_body"])
        prompts.append(prompt)
        answers.append(ex["score"])

    if args.logit_file is None:
        args.logit_file = (f"./outputs/{args.data_type}-{args.model_name}"
                           f"-{args.prompt_type}.jsonl")
    os.makedirs(os.path.dirname(args.logit_file), exist_ok=True)

    # 先读取已有的结果，不清空
    pred_scores = [None] * len(prompts)
    finished = 0
    if os.path.exists(args.logit_file) and args.rewrite_output != "True":
        with open(args.logit_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                pred_scores[idx] = data["score"]
        finished = idx + 1
        print(f"Loaded {finished} existing scores.")
    else:
        open(args.logit_file, "w").close()   # 完全重跑
        finished = 0

    # 剩余任务
    remaining_indices = list(range(finished, len(prompts)))
    task_iter = ((prompts[i], answers[i],
                  args.model_name, args.temperature,
                  args.max_new_token, args.data_type,
                  args.prompt_type) for i in remaining_indices)

    # 计算
    if args.multi_process == "False":
        results = map(_worker, task_iter)
    else:
        pool = multiprocessing.Pool(processes=args.pool_number)
        results = pool.imap(_worker, task_iter, chunksize=1)

    # 边算边写
    with open(args.logit_file, "a", encoding="utf-8") as fout:
        for idx, (score, pred, _, _) in zip(
                remaining_indices,
                tqdm.tqdm(results, total=len(remaining_indices))):
            pred_scores[idx] = score
            fout.write(json.dumps({"score": score,
                                   "prediction": pred,
                                   "answer": answers[idx]}) + "\n")
            fout.flush()

    if args.multi_process != "False":
        pool.close()
        pool.join()

    # 打印指标
    metrics = calculate_metrics(answers, pred_scores, args.data_type)
    print("**********************************************")
    print(
        f"Model: {args.model_name}, Data: {args.data_type}, Prompt: {args.prompt_type}")
    print(metrics)
    print("**********************************************")
