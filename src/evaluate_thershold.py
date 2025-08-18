import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from build_dataset import build_dataset, calculate_metrics
import scipy.interpolate as spi
from matplotlib import font_manager as fm

# --- 设置字体：中文黑体，英文 Times New Roman ---
zh_font = fm.FontProperties(fname=fm.findfont("SimHei"), size=20)   # 中文黑体
en_font = fm.FontProperties(
    fname=fm.findfont("Times New Roman"), size=20)  # 英文

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_jsonl_scores(file_path, key_name=None):
    """加载 jsonl 文件并提取分数"""
    scores = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if key_name:
                scores.append(data[key_name])
            else:
                scores.append(data)
    return scores


def calculate_curve_data(gpt4_file, judge_file, confidence_file, dataset_type, data_path):
    """
    为单个数据集计算性能曲线的核心逻辑。
    返回用于绘图的阈值和准确率数据。
    """
    # 加载数据
    gpt4_scores = load_jsonl_scores(gpt4_file, key_name="score")
    judge_scores = load_jsonl_scores(judge_file)
    with open(confidence_file, 'r', encoding='utf-8') as f:
        confidence_data = json.load(f)
    se_judge = np.array(confidence_data["Entropy"])
    se_base = np.array(confidence_data["entropy_cali"])
    dataset = build_dataset(dataset_type, data_path)
    answers = [item['score'] for item in dataset]

    # 计算校准置信度分数
    calibrated_confidence_scores = se_judge - se_base

    # 遍历阈值计算准确率
    min_conf, max_conf = np.min(calibrated_confidence_scores), np.max(
        calibrated_confidence_scores)
    thresholds_to_test = np.linspace(min_conf, max_conf, num=50)
    accuracies = []

    for threshold in thresholds_to_test:
        hybrid_predictions = []
        for i in range(len(calibrated_confidence_scores)):
            if calibrated_confidence_scores[i] >= threshold:
                hybrid_predictions.append(judge_scores[i])
            else:
                hybrid_predictions.append(gpt4_scores[i])

        metrics = calculate_metrics(answers, hybrid_predictions, dataset_type)
        accuracies.append(metrics["accuracy"])

    return thresholds_to_test, np.array(accuracies)


def plot_curves(pandalm_data, judgelm_data):
    """主绘图函数"""
    p_thresholds, p_accuracies = pandalm_data
    j_thresholds, j_accuracies = judgelm_data

    # --- 取二者公共阈值区间 ---
    common_min = max(p_thresholds.min(), j_thresholds.min())
    common_max = min(p_thresholds.max(), j_thresholds.max())

    p_mask = (p_thresholds >= common_min) & (p_thresholds <= common_max)
    j_mask = (j_thresholds >= common_min) & (j_thresholds <= common_max)

    p_thresholds, p_accuracies = p_thresholds[p_mask], p_accuracies[p_mask]
    j_thresholds, j_accuracies = j_thresholds[j_mask], j_accuracies[j_mask]

    # --- 插值光滑曲线 ---
    p_interp = spi.make_interp_spline(p_thresholds, p_accuracies)
    p_thresholds_smooth = np.linspace(common_min, common_max, 300)
    p_accuracies_smooth = p_interp(p_thresholds_smooth)

    j_interp = spi.make_interp_spline(j_thresholds, j_accuracies)
    j_thresholds_smooth = np.linspace(common_min, common_max, 300)
    j_accuracies_smooth = j_interp(j_thresholds_smooth)

    # --- 绘图 ---
    plt.figure(figsize=(10, 7))

    # 绘制 PandaLM 曲线
    plt.plot(p_thresholds_smooth, p_accuracies_smooth,
             color='black', linestyle='-', linewidth=2, label='PandaLM 测试集')

    # 绘制 JudgeLM 曲线
    plt.plot(j_thresholds_smooth, j_accuracies_smooth,
             color='black', linestyle='--', linewidth=2, label='JudgeLM 测试集')

    # --- 获取当前坐标轴范围 ---
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    # --- 只在最高点加标记并画虚线 ---
    p_best_acc = max(p_accuracies)
    p_best_thr = p_thresholds[np.argmax(p_accuracies)]
    plt.plot(p_best_thr, p_best_acc, 'o', color='black', markersize=10)
    plt.axvline(x=p_best_thr, ymin=0, ymax=(p_best_acc-ymin)/(ymax-ymin),
                color="black", linestyle=":", linewidth=1)

    j_best_acc = max(j_accuracies)
    j_best_thr = j_thresholds[np.argmax(j_accuracies)]
    plt.plot(j_best_thr, j_best_acc, 's', color='black', markersize=10)
    plt.axvline(x=j_best_thr, ymin=0, ymax=(j_best_acc-ymin)/(ymax-ymin),
                color="black", linestyle=":", linewidth=1)

    # --- 美化 ---
    plt.xlabel("置信度阈值", fontsize=20, labelpad=10, fontproperties=zh_font)
    plt.ylabel("AdaptEval 评估准确率", fontsize=20,
               labelpad=10, fontproperties=zh_font)
    plt.xticks(fontsize=16, ticks=np.linspace(
        common_min, common_max, 5), fontproperties=en_font)
    plt.yticks(fontsize=16, fontproperties=en_font)
    plt.grid(True, axis='y', linestyle=':', color='gray')
    plt.legend(fontsize=18, loc="upper left", prop=zh_font)
    plt.tight_layout()

    output_filename = "AdaptEval_performance_curves.png"
    plt.savefig(output_filename, dpi=300)
    print(f"绘图完成，已保存至 {output_filename}")

if __name__ == "__main__":
    # --- PandaLM 数据集文件 ---
    pandalm_files = {
        "gpt4_file": 'outputs/pandalm-gpt-4-1106-preview-vanilla.jsonl',
        "judge_file": 'relia_scores/judgelm/pandalm-logit.jsonl',
        "confidence_file": 'relia_scores/judgelm/pandalm-relia.json',
        "dataset_type": "pandalm",
        "data_path": "./data"
    }

    # --- JudgeLM 数据集文件 ---
    judgelm_files = {
        "gpt4_file": 'outputs/judgelm-gpt-4-1106-preview-vanilla.jsonl',
        "judge_file": 'relia_scores/judgelm/judgelm-logit.jsonl',
        "confidence_file": 'relia_scores/judgelm/judgelm-relia.json',
        "dataset_type": "judgelm",
        "data_path": "./data"
    }

    print("--- 开始处理 PandaLM 数据集 ---")
    pandalm_data = calculate_curve_data(**pandalm_files)

    print("\n--- 开始处理 JudgeLM 数据集 ---")
    judgelm_data = calculate_curve_data(**judgelm_files)

    plot_curves(pandalm_data, judgelm_data)
