#!/usr/bin/env python3
import os
import json
from tqdm import tqdm

# ==================================================
# 配置
# ==================================================
VERSION = os.getenv("PATCH_VERSION", "kl")  # ← 优先使用外部传入的版本号
BASE_DIR = "/opt/data/private/BlackBox"
ANNOT_PATH = f"{BASE_DIR}/data/coco-patch-{VERSION}/annotations/instances_val2017.json"

# 模型结果路径列表
MODEL_RESULTS = [
    f"{BASE_DIR}/save-{VERSION}/attack/detection/anchor-detr/patch/res.json",
    f"{BASE_DIR}/save-{VERSION}/attack/detection/dn-detr/patch/res.json",
]

# ==================================================
# 辅助函数
# ==================================================
def load_image_and_patch_mapping(annotations_path):
    """
    从 patched annotations 文件中加载：
      - image_id → file_name
      - image_id → patch_regions（可能为 None）
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_file = {}
    id_to_patch = {}
    for img in data.get("images", []):
        img_id = img["id"]
        id_to_file[img_id] = img.get("file_name", f"unknown_{img_id}.png")
        id_to_patch[img_id] = img.get("patch_regions", [])
    return id_to_file, id_to_patch


def convert_to_standard_format(input_path, image_mapping, patch_mapping, version):
    """将原始 JSON 转换为带 patch 信息的标准格式"""
    if not os.path.exists(input_path):
        print(f"⚠️ 文件不存在: {input_path}")
        return

    model_name = input_path.split("/")[-3]
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 支持两种结构：纯 list 或 dict{"annotations": [...]}
    detections = data.get("annotations", data) if isinstance(data, dict) else data

    results = []
    for det in detections:
        if det.get("category_id") != 1:
            continue

        image_id = int(det.get("image_id", -1))
        file_name = image_mapping.get(image_id, f"unknown_{image_id}.png")
        patch_regions = patch_mapping.get(image_id, [])

        bbox = det.get("bbox", [])
        if len(bbox) == 4:
            # 自动判断是否 xywh
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                x, y, w, h = bbox
                bbox = [x, y, x + w, y + h]

        score = float(det.get("score", 0.0))
        area = abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        results.append({
            "image_id": image_id,
            "file_name": file_name,
            "category_id": 1,
            "category_name": "person",
            "bbox": bbox,
            "score": score,
            "area": area,
            "patch_regions": patch_regions,  # ✅ 新增字段
            "model": model_name,
            "version": version
        })

    # 排序：image_id, category_id, score(desc)
    results = sorted(results, key=lambda x: (x["image_id"], x["category_id"], -x["score"]))

    # 保存结果
    output_path = os.path.join(os.path.dirname(input_path), "res-std.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ {model_name}: {len(results)} 条记录 → {output_path}")

    # 打印前 30 行
    try:
        print("-" * 50)
        print(f"📄 {model_name} - {os.path.basename(output_path)} 前 30 行内容:")
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:30]):
                print(line.rstrip("\n"))
            if len(lines) > 30:
                print("... (已省略更多行)")
        print("-" * 50)
    except Exception as e:
        print(f"❌ 打印 JSON 内容失败: {e}")


# ==================================================
# 主程序
# ==================================================
def main():
    image_mapping, patch_mapping = load_image_and_patch_mapping(ANNOT_PATH)
    for path in tqdm(MODEL_RESULTS, desc="标准化转换中（含 patch）"):
        convert_to_standard_format(path, image_mapping, patch_mapping, VERSION)
    print("\n✅ 全部模型 patched JSON 已转换为标准格式（含 patch_regions）。")


if __name__ == "__main__":
    main()
