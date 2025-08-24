import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import tiktoken 

from build_dataset import build_dataset
from build_prompt_judge import create_prompt as create_prompt_judge
from build_prompt_gpt import create_prompt_gpt



def process_files(args):
    """计算 token 数并保存到新文件中"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 加载 Tokenizer 和数据集 ---
    print("--- 正在加载 Tokenizer 和数据集... ---")
    # 这个 tokenizer 主要用于微调裁判模型
    judge_tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, padding_side='left')
    if judge_tokenizer.pad_token is None:
        judge_tokenizer.pad_token = judge_tokenizer.eos_token

    # 初始化用于 GPT-4 计数的 tiktoken 编码器
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")

    dataset = build_dataset(args.data_type, "./data")
    print(f"加载数据集 {args.data_type} 完成，共 {len(dataset)} 条样本。")

    # --- 处理 GPT-4 的日志文件 ---
    print(f"\n--- 正在处理 GPT-4 日志文件: {args.gpt4_logit_file_in} ---")
    with open(args.gpt4_logit_file_in, 'r', encoding='utf-8') as f:
        gpt4_logs = [json.loads(line) for line in f]

    # 重新构建 GPT-4 的 prompt
    gpt4_instruction = create_prompt_gpt(args.data_type, prompt_type="vanilla")
    gpt4_prompts = []
    for example in dataset:
        # 确保 rubric 存在
        example.setdefault(
            "rubric", "Please rate the helpfulness, relevance, accuracy, level of details of their responses.")
        prompt = gpt4_instruction.format(
            rubric=example["rubric"],
            question_body=example["question_body"],
            answer1_body=example["answer1_body"],
            answer2_body=example["answer2_body"]
        )
        gpt4_prompts.append(prompt)

    # 为 GPT-4 日志添加 token 计数字段和 answer 字段并保存
    with open(args.gpt4_logit_file_out, 'w', encoding='utf-8') as f_out:
        for i, log_entry in enumerate(tqdm(gpt4_logs, desc="Processing GPT-4 logs")):
            prompt_text = gpt4_prompts[i]
            response_text = log_entry.get("prediction", "")

            # 使用 tiktoken 进行精确计数
            log_entry['num_in_token'] = len(gpt_tokenizer.encode(prompt_text))
            log_entry['num_out_token'] = len(
                gpt_tokenizer.encode(response_text))

            # 添加标准答案
            log_entry['answer'] = dataset[i]['score']

            f_out.write(json.dumps(log_entry) + '\n')
    print(f"已将带 Token 计数和答案的结果保存至: {args.gpt4_logit_file_out}")

    # --- 重新推理微调裁判模型并保存 ---
    print(f"\n--- 正在为微调裁判模型 {args.judge_model_path} 计算Token成本... ---")
    judge_model = AutoModelForCausalLM.from_pretrained(
        args.judge_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    judge_model.eval()

    # 构建裁判模型的 prompts
    judge_instruction = create_prompt_judge(
        args.judge_model_type, args.data_type)
    judge_prompts = []
    for example in dataset:
        prompt = judge_instruction.format(
            question_body=example["question_body"],
            answer1_body=example["answer1_body"],
            answer2_body=example["answer2_body"]
        )
        judge_prompts.append(prompt)

    # 加载裁判模型的原始评分
    with open(args.judge_logit_file_in, 'r', encoding='utf-8') as f:
        original_judge_scores = [json.loads(line) for line in f]

    # 批量生成并保存结果
    batch_size = args.batch_size if args.batch_size else 8
    with open(args.judge_logit_file_out, 'w', encoding='utf-8') as f_out:
        for i in tqdm(range(0, len(judge_prompts), batch_size), desc="Regenerating and saving Judge outputs"):
            batch_prompts = judge_prompts[i:i+batch_size]
            inputs = judge_tokenizer(
                batch_prompts, return_tensors="pt", padding=True).to(device)

            outputs = judge_model.generate(**inputs, max_new_tokens=1024)

            responses_text = judge_tokenizer.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

            # 逐条写入
            for j, response in enumerate(responses_text):
                global_idx = i + j
                ground_truth_score = dataset[global_idx]["score"]

                new_log_entry = {
                    "score": original_judge_scores[global_idx],
                    "prediction": response.strip(),
                    "answer": ground_truth_score,
                    # 减去 padding
                    "num_in_token": len(inputs.input_ids[j]) - inputs.attention_mask[j].tolist().count(0),
                    "num_out_token": len(outputs[j, inputs.input_ids.shape[1]:])
                }
                f_out.write(json.dumps(new_log_entry) + '\n')

    print(f"已将带 Token 计数和答案的结果保存至: {args.judge_logit_file_out}")
    print("\n所有文件处理完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add token counts to log files for AdaptEval.")
    parser.add_argument("--data_type", type=str, required=True, help="数据集类型")
    parser.add_argument("--batch_size", type=int, default=8)

    # 输入文件路径
    parser.add_argument("--judge_model_path", type=str,
                        required=True, help="微调裁判模型的路径")
    parser.add_argument("--judge_model_type", type=str,
                        required=True, help="微调裁判模型的类型")
    parser.add_argument("--judge_logit_file_in", type=str,
                        required=True, help="原始微调裁判模型日志文件路径（只含分数）")
    parser.add_argument("--gpt4_logit_file_in", type=str,
                        required=True, help="原始 GPT-4 日志文件路径")
    parser.add_argument("--tokenizer_path", type=str,
                        required=True, help="用于微调模型计数的 Tokenizer 的路径")

    # 输出文件路径
    parser.add_argument("--judge_logit_file_out", type=str,
                        required=True, help="[输出] 新的、带 token 计数的微调模型日志文件")
    parser.add_argument("--gpt4_logit_file_out", type=str,
                        required=True, help="[输出] 新的、带 token 计数的 GPT-4 日志文件")

    args = parser.parse_args()

    # 自动创建输出目录
    for file_path in [args.judge_logit_file_out, args.gpt4_logit_file_out]:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    process_files(args)
