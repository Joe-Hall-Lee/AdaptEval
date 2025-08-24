import json
import numpy as np
import argparse
from build_dataset import build_dataset, calculate_metrics


def load_results(file_path):
    """从文件加载分数结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def load_model_output(file_path):
    """加载模型输出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line.strip()) for line in f]
    return outputs


def get_relia_scores(results):
    """如果有 entropy_cali，用 Entropy - entropy_cali；否则用 Entropy"""
    entropies = np.array(results["Entropy"])
    if "entropy_cali" in results:
        entropy_cali = np.array(results["entropy_cali"])
        return entropies - entropy_cali
    else:
        return entropies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", type=str, required=True,
                        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood",
                                 "mt-bench", "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue",
                                 "salad-bench", "toxic-chat", "llmbar-neighbor", "llmbar-natural",
                                 "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"))
    parser.add_argument("--output-file1", type=str,
                        required=True, help="评估结果文件1 (含 Entropy 信息)")
    parser.add_argument("--logit-file1", type=str,
                        required=True, help="评估结果文件1 (logit 输出)")
    parser.add_argument("--output-file2", type=str,
                        required=True, help="评估结果文件2 (含 Entropy 信息)")
    parser.add_argument("--logit-file2", type=str,
                        required=True, help="评估结果文件2 (logit 输出)")
    args = parser.parse_args()

    # --- 加载数据 ---
    dataset = build_dataset(args.data_type, "./data")
    answers = [example["score"] for example in dataset]

    # --- 模型 1 ---
    results1 = load_results(args.output_file1)
    relia1 = get_relia_scores(results1)
    outputs1 = load_model_output(args.logit_file1)

    # --- 模型 2 ---
    results2 = load_results(args.output_file2)
    relia2 = get_relia_scores(results2)
    outputs2 = load_model_output(args.logit_file2)

    # --- 直接比较置信度，谁高用谁 ---
    hybrid_outputs = []
    for r1, r2, o1, o2 in zip(relia1, relia2, outputs1, outputs2):
        if r1 >= r2:
            hybrid_outputs.append(o1)
        else:
            hybrid_outputs.append(o2)

    # --- 各自与融合的指标 ---
    acc1 = calculate_metrics(answers, outputs1, args.data_type)
    acc2 = calculate_metrics(answers, outputs2, args.data_type)
    acc_hybrid = calculate_metrics(answers, hybrid_outputs, args.data_type)

    print(f"样本总数: {len(dataset)}")
    print(f"评估结果 1 指标: {acc1}")
    print(f"评估结果 2 指标: {acc2}")
    print(f"融合策略 指标: {acc_hybrid}")


if __name__ == "__main__":
    main()
