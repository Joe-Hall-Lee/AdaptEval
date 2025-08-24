import json
import numpy as np
import random
from evaluate_judge import calculate_metrics, build_dataset, build_params


def load_results(file_path):
    """从文件加载分数结果"""
    with open(file_path, 'r') as f:
        results = json.load(f)
    return results


def get_average_scores(scores):
    """计算正向和逆向的平均分数"""
    mid_index = len(scores) // 2
    forward_scores = scores[:mid_index]
    backward_scores = scores[mid_index:]
    average_scores = [(forward_scores[i] + backward_scores[i]
                       ) / 2 for i in range(mid_index)]
    return average_scores


def normalize_scores(scores):
    """归一化分数"""
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0] * len(scores)
    normalized_scores = [(score - min_score) /
                         (max_score - min_score) for score in scores]
    return normalized_scores


def compute_calibrated_score(scores, cali_scores, cali_factor=1.0):
    """计算校准后的组合分数"""
    calibrated_score = [(scores[i] - cali_scores[i] *
                         cali_factor) for i in range(len(scores))]
    return calibrated_score


def select_top_half_indices(average_scores, total_length):
    """获取得分最高的前 50% 数据的索引，包括正向和逆向"""
    sorted_indices = np.argsort(-np.array(average_scores))
    top_half_indices = sorted_indices[:len(sorted_indices) // 2]
    indices_with_reverse = np.concatenate(
        [top_half_indices, top_half_indices + total_length // 2])
    return indices_with_reverse


def compute_accuracy_rate(metric_results, answers, judge_output, dataset_type):
    """根据分数结果计算置信度最高的前 50% 样本的准确率"""
    total_length = len(metric_results)
    if dataset_type == "auto-j":
        average_scores = get_average_scores(metric_results)
        top_half_indices = select_top_half_indices(
            average_scores, total_length)
    else:
        sorted_indices = np.argsort(-np.array(metric_results))
        top_half_indices = sorted_indices[:len(sorted_indices) // 2]

    accuracy_rate = calculate_metrics(
        [answers[i] for i in top_half_indices],
        [judge_output[i] for i in top_half_indices],
        dataset_type
    )
    return accuracy_rate


def compute_bucketing_rate(metric_results, answers, judge_output, dataset_type):
    """根据分数将样本分为 5 组，并计算每组的准确率"""
    if dataset_type == "auto-j":
        average_scores = get_average_scores(metric_results)
        sorted_forward_indices = np.argsort(-np.array(average_scores))
        num_forward_samples = len(sorted_forward_indices)
        bucket_size = num_forward_samples // 5
        for i in range(5):
            start_index = bucket_size * i
            end_index = bucket_size * (i + 1) if i < 4 else num_forward_samples
            forward_bucket_indices = sorted_forward_indices[start_index:end_index]
            full_bucket_indices = np.concatenate(
                [forward_bucket_indices, forward_bucket_indices + num_forward_samples])

            accuracy_rate = calculate_metrics(
                [answers[i] for i in full_bucket_indices],
                [judge_output[i] for i in full_bucket_indices],
                dataset_type
            )
            print(f"  Bucket {i+1}/5: {accuracy_rate}")
    else:
        sorted_indices = np.argsort(-np.array(metric_results))
        num_samples = len(sorted_indices)
        bucket_size = num_samples // 5
        for i in range(5):
            start_index = bucket_size * i
            end_index = bucket_size * (i + 1) if i < 4 else num_samples
            bucket_indices = sorted_indices[start_index:end_index]

            accuracy_rate = calculate_metrics(
                [answers[i] for i in bucket_indices],
                [judge_output[i] for i in bucket_indices],
                dataset_type
            )
            print(f"  Bucket {i+1}/5: {accuracy_rate}")


def main():
    random.seed(42)
    np.random.seed(42)

    parser = build_params()
    parser.add_argument("--output-file", type=str, default=None)
    args = parser.parse_args()

    dataset = build_dataset(args.data_type, "./data")
    answers = [example["score"] for example in dataset]

    results = load_results(args.output_file)

    with open(args.logit_file, 'r') as f:
        judge_output = [json.loads(line.strip()) for line in f.readlines()]

    if "Entropy" in results:
        entropy_results = results["Entropy"]
        if "entropy_cali" in results:
            entropy_cali_results = results["entropy_cali"]
            for cali_factor in [0.0, 0.5, 1.0]:
                relia_scores = compute_calibrated_score(
                    entropy_results, entropy_cali_results, cali_factor=cali_factor)
                accuracy_rate = compute_accuracy_rate(
                    relia_scores, answers, judge_output, args.data_type)
                print(
                    f"Accuracy Rate (Entropy, cali={cali_factor}): {accuracy_rate}")

                # 仅在 cali_factor 为 0.0 或 1.0 时执行分桶分析
                if cali_factor in [0.0, 1.0]:
                    compute_bucketing_rate(
                        relia_scores, answers, judge_output, args.data_type)
        else:
            accuracy_rate = compute_accuracy_rate(
                entropy_results, answers, judge_output, args.data_type)
            print(f"Accuracy Rate (Uncalibrated Entropy): {accuracy_rate}")
            # 如果没有校准数据，也对原始熵进行分桶分析
            compute_bucketing_rate(
                entropy_results, answers, judge_output, args.data_type)

    if "Perplexity" in results:
        ppl_results = results["Perplexity"]
        ppl_accuracy_rate = compute_accuracy_rate(
            ppl_results, answers, judge_output, args.data_type)
        print(f"Accuracy Rate (Perplexity): {ppl_accuracy_rate}")

    total_samples = len(answers)
    num_samples_to_select = total_samples // 2
    if args.data_type == "auto-j":
        total_forward_samples = total_samples // 2
        num_forward_to_select = num_samples_to_select // 2
        random_forward_indices = np.random.choice(
            total_forward_samples, num_forward_to_select, replace=False)
        random_indices = np.concatenate(
            [random_forward_indices, random_forward_indices + total_forward_samples])
    else:
        random_indices = np.random.choice(
            total_samples, num_samples_to_select, replace=False)

    random_accuracy_rate = calculate_metrics(
        [answers[i] for i in random_indices],
        [judge_output[i] for i in random_indices],
        args.data_type
    )
    print(f"Random 50% Selection Accuracy Rate: {random_accuracy_rate}")


if __name__ == "__main__":
    main()
