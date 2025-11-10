#!/usr/bin/env python3
import os
import json
from tqdm import tqdm
from collections import defaultdict

# =====================================================
# ✅ 版本控制变量 —— 只需修改这一行
# =====================================================
VERSION = os.getenv("PATCH_VERSION", "kl")  # ← 优先使用外部传入的版本号
SCORE_THRESHOLD = 0.7  # 置信度阈值

# =====================================================
# 路径与模型配置
# =====================================================
BASE_DIR = "/opt/data/private/BlackBox"
CLEAN_SAVE_DIR = f"{BASE_DIR}/save/attack/detection"               # clean结果源路径
PATCH_SAVE_DIR = f"{BASE_DIR}/save-{VERSION}/attack/detection"     # 输出路径目标
PATCH_ANNOT_PATH = f"{BASE_DIR}/data/coco-patch-{VERSION}/annotations/instances_val2017.json"

MODEL_NAMES = ["detr", "deformable-detr", "sparse-detr", "anchor-detr", "dn-detr"]

# =====================================================
# 工具函数
# =====================================================
def is_patch_inside_box(box, patch):
    """判断 patch 是否完全在检测框内部"""
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = patch
    return x1 <= px1 and y1 <= py1 and x2 >= px2 and y2 >= py2


def load_patch_mapping(annotations_path):
    """加载 image_id → patch_regions 映射"""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    patch_mapping = {img["id"]: img.get("patch_regions", []) for img in data["images"]}
    print(f"✅ 已加载 patch 区域映射: {len(patch_mapping)} 张图片")
    return patch_mapping


def filter_json(input_path, patch_mapping, score_thr=0.7):
    """执行 person + score + patch包含筛选"""
    if not os.path.exists(input_path):
        print(f"⚠️ 文件不存在: {input_path}")
        return None, None

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"⚠️ 文件格式异常(非 list): {input_path}")
        return None, None

    filtered = []
    for det in data:
        if det.get("category_id") != 1 or det.get("score", 0) < score_thr:
            continue

        img_id = det.get("image_id")
        if img_id not in patch_mapping:
            continue

        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue

        patches = patch_mapping[img_id]
        keep = any(is_patch_inside_box(bbox, patch) for patch in patches)
        if keep:
            filtered.append(det)

    grouped = defaultdict(list)
    for det in filtered:
        grouped[det["image_id"]].append(det)
    return filtered, grouped


def save_filtered(filtered, output_path):
    """保存筛选结果"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    print(f"✅ 已保存筛选结果: {output_path} ({len(filtered)} 条)")


# =====================================================
# 主逻辑
# =====================================================
def main():
    print(f"=== Clean 图 person+score+patch包含筛选开始 (VERSION={VERSION}, SCORE≥{SCORE_THRESHOLD}) ===")

    patch_mapping = load_patch_mapping(PATCH_ANNOT_PATH)

    for model_name in MODEL_NAMES:
        clean_input = os.path.join(CLEAN_SAVE_DIR, model_name, "ori", "res-std.json")
        output_dir = os.path.join(PATCH_SAVE_DIR, model_name, "patch")
        output_path = os.path.join(output_dir, "clean-filter.json")

        if not os.path.exists(clean_input):
            print(f"⚠️ {model_name}: 未找到 {clean_input}，跳过。")
            continue

        print(f"\n--- {model_name.upper()} ---")
        filtered, grouped = filter_json(clean_input, patch_mapping, SCORE_THRESHOLD)

        if filtered is None:
            continue

        save_filtered(filtered, output_path)

        total = len(filtered)
        img_count = len(grouped)
        avg_boxes = total / img_count if img_count else 0
        print(f"📊 图片数: {img_count} | 保留检测框数: {total} | 平均每图检测框: {avg_boxes:.2f}")

    print(f"\n✅ 所有 Clean 模型筛选完成，结果已保存在带版本号的 patch 文件夹中。")


if __name__ == "__main__":
    main()
