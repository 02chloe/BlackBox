#!/usr/bin/env python3
# compare_recall_tmm_soft_hard.py
"""
对比 BlackBox 不同 TMM / KL 配置的 Recall 结果（多版本、多模型折线图）

读取路径示例（和你现有的 recall_summary 一致）：
  /opt/data/private/BlackBox/save-<VERSION>/attack/res/<model_name>/recall_attack_report.json

支持的实验组合：
1) Hard TMM（无 KL）：
    VERSION: "0.1", "0.5", "base0.3"
    p_base : 0.1, 0.2, 0.3

2) Hard TMM + KL：
    VERSION: "base0.1_kl0.001", "kl", "basekl0.001"
    p_base : 0.1, 0.2, 0.3
    --kl_beta 0.001

3) Soft TMM（无 KL）：
    VERSION: "soft-0.1", "soft-0.3", "soft-0.5", "soft-0.7", "soft-0.9"
    soft_mask_ratio: 0.1, 0.3, 0.5, 0.7, 0.9
    --soft_tau 1.0 --patch_ratio 0.5

4) Soft TMM + KL：
    VERSION: "kl_soft0.1", "kl_soft0.3", "kl_soft0.5", "kl_soft0.7", "kl_soft0.9"
    soft_mask_ratio: 0.1, 0.3, 0.5, 0.7, 0.9
    --soft_tau 1.0 --kl_beta 0.001 --patch_ratio 0.5
"""

import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 基本配置
# =========================
BASE_DIR = "/opt/data/private/BlackBox"
MODEL_NAMES = ["detr", "deformable-detr", "sparse-detr", "anchor-detr", "dn-detr"]

# 输出目录：所有对比图统一放在这里
COMPARE_DIR = os.path.join(BASE_DIR, "compare_recall_plots")
os.makedirs(COMPARE_DIR, exist_ok=True)

# =========================
# 实验版本配置（可以按需要改）
# =========================

# 1) Hard TMM（无 KL）
HARD_NO_KL = {
    0.1: "0.1",       # p_base = 0.1
    0.2: "0.5",       # p_base = 0.2
    0.3: "base0.3",   # p_base = 0.3
}

# 2) Hard TMM + KL（kl_beta = 0.001）
HARD_WITH_KL = {
    0.1: "base0.1_kl0.001",  # p_base = 0.1
    0.2: "kl",               # p_base = 0.2
    0.3: "basekl0.001",      # p_base = 0.3
}

# 3) Soft TMM（无 KL）
SOFT_NO_KL = {
    0.1: "soft0.1",
    0.3: "soft-0.3",
    0.5: "soft0.5",
    0.7: "soft-0.7",
    0.9: "soft-0.9",
}

# 4) Soft TMM + KL（kl_beta = 0.001）
SOFT_WITH_KL = {
    0.1: "kl_soft0.1",
    0.3: "kl_soft0.3",
    0.5: "kl_soft0.5",   # ⚠️ 如你的目录实际是 soft-0.5，就改成 "soft-0.5"
    0.7: "kl_soft0.7",
    0.9: "kl_soft0.9",
}

EXPERIMENT_GROUPS = {
    "hard_no_kl": {
        "label": "Hard TMM",
        "color": "#1f77b4",
        "marker": "o",
        "mapping": HARD_NO_KL,
        "x_label": "p_base (hard TMM)",
    },
    "hard_with_kl": {
        "label": "Hard TMM + KL",
        "color": "#ff7f0e",
        "marker": "s",
        "mapping": HARD_WITH_KL,
        "x_label": "p_base (hard TMM)",
    },
    "soft_no_kl": {
        "label": "Soft TMM",
        "color": "#2ca02c",
        "marker": "D",
        "mapping": SOFT_NO_KL,
        "x_label": "soft_mask_ratio (soft TMM)",
    },
    "soft_with_kl": {
        "label": "Soft TMM + KL",
        "color": "#d62728",
        "marker": "^",
        "mapping": SOFT_WITH_KL,
        "x_label": "soft_mask_ratio (soft TMM)",
    },
}

# =========================
# 工具函数
# =========================

def load_recall(version: str, model_name: str) -> float:
    """
    读取某个版本、某个模型的 recall_percent（%）
    若文件不存在，返回 np.nan
    """
    res_base = os.path.join(BASE_DIR, f"save-{version}", "attack", "res")
    path = os.path.join(res_base, model_name, "recall_attack_report.json")
    if not os.path.exists(path):
        print(f"⚠️ 未找到报告: {path}")
        return math.nan

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        recall_str = metrics.get("recall_percent", "nan")
        return float(recall_str)
    except Exception as e:
        print(f"⚠️ 读取或解析失败: {path} | {e}")
        return math.nan


def collect_all_results():
    """
    遍历上面定义的 4 类实验，收集所有模型的 recall 数据。
    返回一个嵌套 dict:
      results[exp_key][model_name][strength] = recall_percent(float)
    """
    results = {}
    for exp_key, cfg in EXPERIMENT_GROUPS.items():
        mapping = cfg["mapping"]
        results[exp_key] = {}
        for model in MODEL_NAMES:
            results[exp_key][model] = {}
            for strength, version in mapping.items():
                r = load_recall(version, model)
                results[exp_key][model][strength] = r
    return results


def plot_per_model(results):
    """
    对每个模型画一张图：
      x 轴：掩码强度（p_base 或 soft_mask_ratio）
      y 轴：Recall (%)
      4 条线：Hard / Hard+KL / Soft / Soft+KL
    """
    for model in MODEL_NAMES:
        plt.figure(figsize=(7, 5))

        # 每个实验组一条线
        for exp_key, cfg in EXPERIMENT_GROUPS.items():
            strength_to_rec = results[exp_key][model]
            # 去掉 nan 的点
            strengths = []
            recalls = []
            for s, r in strength_to_rec.items():
                if r == r:  # 过滤 NaN
                    strengths.append(s)
                    recalls.append(r)
            if not strengths:
                continue

            # 按 x 排序
            strengths, recalls = zip(*sorted(zip(strengths, recalls), key=lambda x: x[0]))
            plt.plot(
                strengths,
                recalls,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                linestyle="-",
            )

        plt.xlabel("Mask Strength (p_base / soft_mask_ratio)")
        plt.ylabel("Recall (%)")
        plt.title(f"Recall vs Mask Strength - {model}")
        plt.grid(alpha=0.3)
        plt.legend()
        out_path = os.path.join(COMPARE_DIR, f"recall_vs_strength_{model}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 保存图像: {out_path}")


def plot_overall_avg(results):
    """
    额外再画一张“整体平均”的图：
      - 把所有模型的 recall 平均到一起，对每个实验组画一条线
      - 用于给出一句 summary：哪种配置整体平均更好
    """
    plt.figure(figsize=(7, 5))

    for exp_key, cfg in EXPERIMENT_GROUPS.items():
        strength_to_rec = results[exp_key]
        # strength_to_rec: {model_name: {strength: recall}}
        # 我们要先按 strength 聚合所有模型的平均 recall
        strength_values = sorted({s for m in strength_to_rec.values() for s in m.keys()})
        xs = []
        ys = []
        for s in strength_values:
            vals = []
            for model in MODEL_NAMES:
                r = strength_to_rec.get(model, {}).get(s, math.nan)
                if r == r:
                    vals.append(r)
            if not vals:
                continue
            xs.append(s)
            ys.append(float(np.mean(vals)))
        if not xs:
            continue
        plt.plot(
            xs,
            ys,
            label=cfg["label"],
            color=cfg["color"],
            marker=cfg["marker"],
            linestyle="-",
        )

    plt.xlabel("Mask Strength (p_base / soft_mask_ratio)")
    plt.ylabel("Average Recall (%)")
    plt.title("Average Recall vs Mask Strength (across all models)")
    plt.grid(alpha=0.3)
    plt.legend()
    out_path = os.path.join(COMPARE_DIR, "recall_vs_strength_overall_avg.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ 保存整体平均图像: {out_path}")


def export_summary_csv(results):
    """
    把所有结果整理成一个长表 CSV，方便你之后做更多分析。
    表头：
      Method, With_KL, Model, Strength, Version, Recall
    """
    rows = []
    for exp_key, cfg in EXPERIMENT_GROUPS.items():
        label = cfg["label"]
        mapping = cfg["mapping"]

        method = "hard" if "hard" in exp_key else "soft"
        with_kl = "with_kl" in exp_key

        for model in MODEL_NAMES:
            strength_to_rec = results[exp_key][model]
            for strength, rec in strength_to_rec.items():
                version = mapping.get(strength, "")
                rows.append({
                    "ExpKey": exp_key,
                    "Method": method,
                    "With_KL": with_kl,
                    "Model": model,
                    "Strength": strength,
                    "Version": version,
                    "Recall(%)": rec,
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(COMPARE_DIR, "recall_comparison_all.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 导出汇总 CSV: {csv_path}")
    # 也顺便打印一下简要统计
    print("\n=== 简要统计：按 Method & With_KL 分组的平均 Recall ===")
    print(df.groupby(["Method", "With_KL"])["Recall(%)"].mean())


def main():
    print("=== 📊 收集所有版本的 Recall 结果用于软/硬 TMM & KL 对比 ===")
    results = collect_all_results()
    print("=== 📈 绘制每个模型的 Recall–强度 折线图 ===")
    plot_per_model(results)
    print("=== 📈 绘制整体平均 Recall–强度 折线图 ===")
    plot_overall_avg(results)
    print("=== 🧾 导出全量对比 CSV ===")
    export_summary_csv(results)
    print("✅ 全部完成！图像与表格保存在:", COMPARE_DIR)


if __name__ == "__main__":
    main()
