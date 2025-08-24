import json
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import random

from build_dataset import build_dataset, calculate_metrics

def load_jsonl(file_path, key_name=None):
    """加载 jsonl 文件并提取指定键的值"""
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if key_name:
                data_list.append(data[key_name])
            else:
                data_list.append(data)
    return data_list


def get_calibrated_confidence(confidence_file_path):
    """计算校准后的置信度分数"""
    with open(confidence_file_path, 'r', encoding='utf-8') as f:
        confidence_data = json.load(f)
    se_judge = np.array(confidence_data["Entropy"])
    se_base = np.array(confidence_data["entropy_cali"])
    return se_judge - se_base


def run_threshold_evaluation(args):
    """主函数"""

    # 设置随机种子以保证可复现性
    random.seed(42)
    np.random.seed(42)

    # 加载所有原始数据
    print("--- 正在加载所有数据... ---")
    dataset = build_dataset(args.data_type, "./data")
    answers = np.array([item['score'] for item in dataset])

    judge_predictions = np.array(load_jsonl(
        args.judge_logit_file_with_token, key_name="score"))
    confidence_scores = np.array(
        get_calibrated_confidence(args.judge_confidence_file))

    gpt4_logs = load_jsonl(args.gpt4_logit_file_with_token)
    gpt4_predictions = np.array([item['score'] for item in gpt4_logs])

    print("数据加载完毕。")

    # 划分验证集 (20%) 和测试集 (80%)
    print("\n--- 正在划分验证集与测试集... ---")
    indices = np.arange(len(dataset))

    val_indices, test_indices = train_test_split(
        indices, test_size=0.8, random_state=42)

    # 验证集数据
    val_answers = answers[val_indices]
    val_judge_preds = judge_predictions[val_indices]
    val_gpt4_preds = gpt4_predictions[val_indices]
    val_conf_scores = confidence_scores[val_indices]

    # 测试集数据
    test_answers = answers[test_indices]
    test_judge_preds = judge_predictions[test_indices]
    test_gpt4_preds = gpt4_predictions[test_indices]
    test_conf_scores = confidence_scores[test_indices]
    print(f"划分完毕: {len(val_indices)} 条用于验证调参, {len(test_indices)} 条用于最终测试。")

    # 在验证集上寻找最优阈值
    print("\n--- 正在验证集上为每个 λ 搜索最优阈值... ---")

    # 仅需加载带 token 计数的文件即可
    val_judge_logs = np.array(load_jsonl(
        args.judge_logit_file_with_token))[val_indices]
    val_gpt4_logs = np.array(load_jsonl(args.gpt4_logit_file_with_token))[
        val_indices]

    val_judge_tokens_in = np.array([log['num_in_token'] for log in val_judge_logs])
    val_judge_tokens_out = np.array(
        [log['num_out_token'] for log in val_judge_logs])
    val_gpt4_tokens_in = np.array([log['num_in_token']
                                for log in val_gpt4_logs])
    val_gpt4_tokens_out = np.array(
        [log['num_out_token'] for log in val_gpt4_logs])

    optimal_thresholds = {}
    thresholds_to_test = np.linspace(
        val_conf_scores.min(), val_conf_scores.max(), num=100)

    for lambda_val in args.lambdas:
        best_objective_value = -np.inf
        best_threshold_for_lambda = val_conf_scores.min()

        for threshold in thresholds_to_test:
            use_judge_mask = val_conf_scores >= threshold
            use_gpt4_mask = ~use_judge_mask

            hybrid_preds = np.zeros_like(val_judge_preds)  # 先创建一个同样形状的全零数组
            hybrid_preds[use_judge_mask] = val_judge_preds[use_judge_mask]
            hybrid_preds[use_gpt4_mask] = val_gpt4_preds[use_gpt4_mask]

            metrics = calculate_metrics(
                val_answers.tolist(), hybrid_preds.tolist(), args.data_type)
            accuracy = metrics['accuracy']

            cost_gpt = np.sum(val_gpt4_tokens_in[use_gpt4_mask]) + \
                args.gamma * np.sum(val_gpt4_tokens_out[use_gpt4_mask])

            cost_judge = np.sum(val_judge_tokens_in[use_judge_mask]) + \
                args.gamma * np.sum(val_judge_tokens_out[use_judge_mask])

            total_cost = cost_gpt + cost_judge

            # 避免除以零
            cost_ratio = cost_gpt / total_cost if total_cost > 0 else 0

            objective_value = accuracy - lambda_val * cost_ratio

            if objective_value > best_objective_value:
                best_objective_value = objective_value
                best_threshold_for_lambda = threshold

        optimal_thresholds[lambda_val] = best_threshold_for_lambda
        print(
            f"  - 为 λ={lambda_val} 找到最优阈值 τ* = {best_threshold_for_lambda:.4f}")

    # 在测试集上报告最终结果
    print("\n" + "="*50)
    print("       最终性能报告（在 80% 未见过的测试集上）")
    print("="*50)

    baseline_judge_metrics = calculate_metrics(
        test_answers.tolist(), test_judge_preds.tolist(), args.data_type)
    baseline_gpt4_metrics = calculate_metrics(
        test_answers.tolist(), test_gpt4_preds.tolist(), args.data_type)
    print(f"基线 - Judge 模型独立性能: {baseline_judge_metrics}")
    print(f"基线 - GPT-4 模型独立性能: {baseline_gpt4_metrics}")
    print("-" * 50)

    for lambda_val, optimal_tau in optimal_thresholds.items():
        use_judge_mask = test_conf_scores >= optimal_tau
        use_gpt4_mask = ~use_judge_mask

        hybrid_preds = np.zeros_like(test_judge_preds)  # 先创建一个同样形状的全零数组
        hybrid_preds[use_judge_mask] = test_judge_preds[use_judge_mask]
        hybrid_preds[use_gpt4_mask] = test_gpt4_preds[use_gpt4_mask]

        final_metrics = calculate_metrics(
            test_answers.tolist(), hybrid_preds.tolist(), args.data_type)
        routing_ratio = np.sum(use_gpt4_mask) / len(test_indices)

        print(f"AdaptEval (λ = {lambda_val}, τ* = {optimal_tau:.4f}):")
        print(f"  - 最终评估指标: {final_metrics}")
        print(f"  - GPT-4 路由比例: {routing_ratio:.2%}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find optimal threshold for AdaptEval using a validation set.")
    parser.add_argument("--data_type", type=str, required=True)
    parser.add_argument("--judge_confidence_file", type=str, required=True)
    parser.add_argument("--judge_logit_file_with_token", type=str, required=True)
    parser.add_argument("--gpt4_logit_file_with_token",
                        type=str, required=True)
    parser.add_argument("--gamma", type=float, default=3.0,
                        help="Cost factor for output tokens.")
    parser.add_argument("--lambdas", type=float, nargs='+',
                        default=[0.0, 0.3, 0.5], help="List of lambda values to test.")

    args = parser.parse_args()
    run_threshold_evaluation(args)
