import json
import argparse
import random
import time
import json
import openai
import os
import multiprocessing
from functools import partial
from dotenv import load_dotenv

from evaluate_judge import build_dataset
from build_prompt_gpt import parse_score_gpt, create_prompt_gpt

load_dotenv()  # 加载 .env 文件中的变量

def build_params_gpt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        choices=("vanilla", "cot"),
        default=None,
    )
    parser.add_argument(
        "--data-type",
        type=str,
        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood", "mt-bench",
                 "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue", "salad-bench", "toxic-chat",
                 "llmbar-neighbor", "llmbar-natural", "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"),
        default=None,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=None,
        help="The maximum number of new tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="The temperature for sampling.",
    )
    parser.add_argument(
        "--logit-file",
        type=str,
        default=None
    )
    parser.add_argument(
        "--pool-number",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--multi-process",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--rewrite-output",
        type=str,
        default="False",
    )
    args = parser.parse_args()
    return args

def request_gpt(prompt, model, temperature, max_new_tokens):
    model = "gpt-4-1106-preview"
    api_key = os.getenv("API_KEY")
    client = openai.OpenAI(
        api_key=api_key, base_url="https://api.shubiaobiao.cn/v1/")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    }
    max_tries = 5
    res = ''
    for i in range(max_tries):
        try:
            chat_completion = client.chat.completions.create(
                model=payload['model'], temperature=temperature, messages=payload['messages'])
            res = chat_completion.choices[0].message.content
            break
        except Exception as e:
            if i == max_tries - 1:
                print("MAX_RETRY exceeded! Please check your codes!")
                return "0 0"
            print("Exception! The exception is "+str(e))
            time.sleep(5)
            continue
    return res


def gpt_scoring(prompt, answer, model, temperature, max_new_tokens, data_type, prompt_type, logit_file, lock, counter):
    prediction = request_gpt(
        prompt, model, temperature=temperature, max_new_tokens=max_new_tokens)
    score = parse_score_gpt(
        prediction, data_type=data_type, prompt_type=prompt_type)

    with lock:  # 保证写文件时不会冲突
        with open(logit_file, "a", encoding="utf-8") as fout:
            fout.write(json.dumps(
                {"score": score, "prediction": prediction, "answer": answer}) + "\n")

    counter.value += 1
    print(f"[{counter.value}] sample finished and written to {logit_file}")
    return score


def init(c, l):
    global counter, lock
    counter = c
    lock = l


if __name__ == "__main__":
    args = build_params_gpt()
    random.seed(42)

    if "prometheus" in args.data_type:
        args.prompt_type = "cot"

    if args.max_new_token is None:
        args.max_new_token = 1024 if args.prompt_type == "cot" else 16

    dataset = build_dataset(args.data_type, args.data_path)
    instruction = create_prompt_gpt(args.data_type, args.prompt_type)

    prompts, answers = [], []
    for example in dataset:
        if args.data_type in ["prometheus-ind", "prometheus-ood", "halu-eval-summary", "halu-eval-dialogue", "halu-eval-qa", "toxic-chat"]:
            prompt = instruction.format(question_body=example["question_body"],
                                        rubric=example["rubric"],
                                        answer_body=example["answer_body"])
        else:
            example["rubric"] = "Please rate the helpfulness, relevance, accuracy, level of details of their responses."
            if args.prompt_type == "icl":
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            demonstrations=example["demonstrations"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
            else:
                prompt = instruction.format(question_body=example["question_body"],
                                            rubric=example["rubric"],
                                            answer1_body=example["answer1_body"],
                                            answer2_body=example["answer2_body"])
        prompts.append(prompt)
        answers.append(example["score"])

    if args.logit_file is None:
        args.logit_file = f"./outputs/{args.data_type}-{args.model_name}-{args.prompt_type}.jsonl"

    finished = 0
    if os.path.exists(args.logit_file) and args.rewrite_output != "True":
        with open(args.logit_file, "r", encoding="utf-8") as fin:
            finished = len(fin.readlines())
        print(
            f"Found {finished} finished samples, will resume from {finished}.")
    else:
        # 如果 rewrite_output=True 就清空文件
        open(args.logit_file, "w", encoding="utf-8").close()
        print("Starting fresh run, cleared old results.")

    os.makedirs(os.path.dirname(args.logit_file), exist_ok=True)

    manager = multiprocessing.Manager()
    counter = manager.Value("counter", finished)  # 从已有数量开始计数
    lock = manager.Lock()

    remaining_prompts = prompts[finished:]
    remaining_answers = answers[finished:]

    if args.multi_process == "False":
        pred_scores = [
            gpt_scoring(p, a, model=args.model_name, temperature=args.temperature,
                        max_new_tokens=args.max_new_token, data_type=args.data_type,
                        prompt_type=args.prompt_type, logit_file=args.logit_file,
                        lock=lock, counter=counter)
            for p, a in zip(remaining_prompts, remaining_answers)
        ]
    else:
        pool = multiprocessing.Pool(
            processes=args.pool_number, initializer=init, initargs=(counter, lock))
        pool_fn = partial(gpt_scoring, model=args.model_name, temperature=args.temperature,
                          max_new_tokens=args.max_new_token, data_type=args.data_type,
                          prompt_type=args.prompt_type, logit_file=args.logit_file, lock=lock, counter=counter)
        pred_scores = pool.starmap(pool_fn, zip(
            remaining_prompts, remaining_answers))
