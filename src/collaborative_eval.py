import json
import numpy as np
import argparse
from build_dataset import build_dataset, calculate_metrics


def load_results(file_path):
    """从文件加载置信度结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def load_model_output(file_path):
    """加载模型输出"""
    with open(file_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line.strip()) for line in f]
    return outputs


def get_top_half_indices(scores):
    """获取置信度最高的前 50% 索引"""
    sorted_indices = np.argsort(-np.array(scores))
    top_half_indices = sorted_indices[:len(sorted_indices) // 2]
    return set(top_half_indices)


def normalize_scores(scores):
    """对置信度分数做归一化 (min-max)"""
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", type=str, required=True,
                        choices=("judgelm", "pandalm", "auto-j", "prometheus-ind", "prometheus-ood",
                                 "mt-bench", "halu-eval-summary", "halu-eval-qa", "halu-eval-dialogue",
                                 "salad-bench", "toxic-chat", "llmbar-neighbor", "llmbar-natural",
                                 "llmbar-gptinst", "llmbar-gptout", "llmbar-manual"))
    parser.add_argument("--model1-output", type=str,
                        required=True, help="模型 1 logit 文件路径")
    parser.add_argument("--model2-output", type=str,
                        required=True, help="模型 2 logit 文件路径")
    parser.add_argument("--model1-relia", type=str,
                        required=True, help="模型 1 置信度文件路径")
    parser.add_argument("--model2-relia", type=str,
                        required=True, help="模型 2 置信度文件路径")
    args = parser.parse_args()

    # --- 加载数据 ---
    dataset = build_dataset(args.data_type, "./data")
    answers = [example["score"] for example in dataset]

    model1_output = load_model_output(args.model1_output)
    model2_output = load_model_output(args.model2_output)

    model1_conf = load_results(args.model1_relia)["Entropy"]
    model2_conf = load_results(args.model2_relia)["Entropy"]

    # --- 归一化置信度 ---
    model1_conf_norm = normalize_scores(model1_conf)
    model2_conf_norm = normalize_scores(model2_conf)

    # --- 获取前 50% 置信度的索引 ---
    top_model1 = get_top_half_indices(model1_conf)
    top_model2 = get_top_half_indices(model2_conf)

    # --- 合并集合 ---
    combined_indices = top_model1.union(top_model2)

    # --- 计算准确率 ---
    # 模型 1 单独
    model1_acc = calculate_metrics(
        [answers[i] for i in combined_indices],
        [model1_output[i] for i in combined_indices],
        args.data_type
    )

    # 模型 2 单独
    model2_acc = calculate_metrics(
        [answers[i] for i in combined_indices],
        [model2_output[i] for i in combined_indices],
        args.data_type
    )

    # 混合策略
    hybrid_predictions = []
    for i in combined_indices:
        if i in top_model1 and i not in top_model2:
            hybrid_predictions.append(model1_output[i])
        elif i in top_model2 and i not in top_model1:
            hybrid_predictions.append(model2_output[i])
        else:
            # 都在 top50%，选归一化置信度更高的
            if model1_conf_norm[i] >= model2_conf_norm[i]:
                hybrid_predictions.append(model1_output[i])
            else:
                hybrid_predictions.append(model2_output[i])

    hybrid_acc = calculate_metrics(
        [answers[i] for i in combined_indices],
        hybrid_predictions,
        args.data_type
    )

    print(f"总样本数: {len(dataset)}")
    print(f"取并集后的样本数: {len(combined_indices)}")
    print(f"模型 1 准确率: {model1_acc}")
    print(f"模型 2 准确率: {model2_acc}")
    print(f"混合策略准确率: {hybrid_acc}")


if __name__ == "__main__":
    main()
