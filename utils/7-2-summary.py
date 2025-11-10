#!/usr/bin/env python3
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# ✅ 配置
# =====================================================
VERSION = os.getenv("PATCH_VERSION", "0.5")
BASE_DIR = "/opt/data/private/BlackBox"
RES_BASE = f"{BASE_DIR}/save-{VERSION}/attack/res"
MODEL_NAMES = ["detr", "deformable-detr", "sparse-detr", "anchor-detr", "dn-detr"]

# =====================================================
# ✅ 读取报告
# =====================================================
def load_report(model_name):
    path = os.path.join(RES_BASE, model_name, "recall_attack_report.json")
    if not os.path.exists(path):
        print(f"⚠️ 未找到报告: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    return {
        "Model": model_name,
        "Version": data.get("version", "-"),
        "Clean Boxes": metrics.get("Clean Boxes", 0),
        "Patched Boxes": metrics.get("Patched Boxes", 0),
        "Reduce": metrics.get("reduce", 0),
        "Increase": metrics.get("increase", 0),
        "Recall (%)": metrics.get("recall_percent", "0.00"),
        "Success Images": metrics.get("success_images", 0),
    }

# =====================================================
# ✅ 绘制表格
# =====================================================
def save_table_as_png(df, output_path, title="Model Recall Summary"):
    col_widths = [max(df[col].astype(str).map(len).max(), len(col)) * 0.12 for col in df.columns]
    total_width = sum(col_widths)
    fig_height = 1.2 + len(df) * 0.4
    fig, ax = plt.subplots(figsize=(total_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    # 设置列宽
    for i, width in enumerate(col_widths):
        for j in range(len(df) + 1):
            cell = table[(j, i)]
            cell.set_width(width / total_width)

    for _, cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        cell.set_facecolor("white")

    plt.title(title, fontsize=11, pad=10)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"✅ 已保存表格图片: {output_path}")

# =====================================================
# ✅ 主逻辑
# =====================================================
def main():
    print(f"=== 📊 汇总 Recall 结果 (VERSION={VERSION}) ===")
    records = [load_report(m) for m in MODEL_NAMES]
    records = [r for r in records if r]

    if not records:
        print("⚠️ 无有效报告文件。")
        return

    df = pd.DataFrame(records)
    df["Model"] = pd.Categorical(df["Model"], categories=MODEL_NAMES, ordered=True)
    df = df.sort_values("Model").reset_index(drop=True)

    # 输出路径
    csv_path = os.path.join(RES_BASE, "recall_summary.csv")
    md_path = os.path.join(RES_BASE, "recall_summary.md")
    png_path = os.path.join(RES_BASE, "recall_summary.png")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    save_table_as_png(df, png_path, title=f"Model Recall Summary (Version {VERSION})")

    print("\n✅ 汇总完成！")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
