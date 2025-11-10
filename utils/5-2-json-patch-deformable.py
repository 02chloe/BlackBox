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

# Deformable-DETR 原始检测结果路径
INPUT_PATH = f"{BASE_DIR}/save-{VERSION}/attack/detection/deformable-detr/patch/res.json"

# ==================================================
# 辅助函数
# ==================================================
def load_image_and_patch_mapping(annotations_path):
    """
    从 annotations 文件加载:
      - id_to_file: image_id → file_name
      - id_to_patch: image_id → patch_regions (list[list])
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_file, id_to_patch = {}, {}
    for img in data.get("images", []):
        img_id = int(img["id"])
        id_to_file[img_id] = img.get("file_name", f"unknown_{img_id}.png")
        id_to_patch[img_id] = img.get("patch_regions", [])
    return id_to_file, id_to_patch


def coco_category_name(cat_id: int) -> str:
    """COCO 类别名映射"""
    names = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
        11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
        16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
        22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
        27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    }
    return names.get(cat_id, f"cls_{cat_id}")


def save_preview(model_name, output_path, max_lines=30):
    """打印前 max_lines 行 JSON 内容"""
    try:
        print("-" * 50)
        print(f"📄 {model_name} - {os.path.basename(output_path)} 前 {max_lines} 行内容:")
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:max_lines]):
                print(line.rstrip("\n"))
            if len(lines) > max_lines:
                print("... (已省略更多行)")
        print("-" * 50)
    except Exception as e:
        print(f"❌ 打印 JSON 内容失败: {e}")

# ==================================================
# 主转换
# ==================================================
def main():
    print("=== Deformable-DETR (patched) → 标准化转换开始 ===")

    if not os.path.exists(INPUT_PATH):
        print(f"⚠️ 未找到输入文件: {INPUT_PATH}")
        return

    # 加载映射信息（包含 patch 信息）
    id_to_file, id_to_patch = load_image_and_patch_mapping(ANNOT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    detections = data if isinstance(data, list) else data.get("annotations", [])
    model_name = "deformable-detr"

    results = []
    for det in tqdm(detections, desc="标准化中"):
        try:
            image_id = int(det.get("image_id"))
        except Exception:
            continue

        file_name = id_to_file.get(image_id, f"unknown_{image_id}.png")
        patch_regions = id_to_patch.get(image_id, [])

        cat_id = int(det.get("category_id", -1))
        cat_name = coco_category_name(cat_id)

        bbox = det.get("bbox", [])
        if not bbox or len(bbox) != 4:
            continue

        # bbox 已是 [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            continue  # 跳过异常框

        score = float(det.get("score", 0.0))
        area = abs((x2 - x1) * (y2 - y1))

        results.append({
            "image_id": image_id,
            "file_name": file_name,
            "category_id": cat_id,
            "category_name": cat_name,
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "area": area,
            "patch_regions": patch_regions,  # ✅ 新增字段
            "model": model_name,
            "version": VERSION
        })

    # ✅ 排序
    results = sorted(results, key=lambda x: (x["image_id"], x["category_id"], -x["score"]))

    # ✅ 保存
    output_path = os.path.join(os.path.dirname(INPUT_PATH), "res-std.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ {model_name}: {len(results)} 条记录 → {output_path}")
    save_preview(model_name, output_path)
    print("=== 转换完成 ===")


if __name__ == "__main__":
    main()
