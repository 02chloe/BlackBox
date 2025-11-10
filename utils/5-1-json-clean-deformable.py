#!/usr/bin/env python3
import os
import json
from tqdm import tqdm

# ==================================================
# 配置
# ==================================================
VERSION = "ori"  # 原图版本（clean）
BASE_DIR = "/opt/data/private/BlackBox"
ANNOT_PATH = f"{BASE_DIR}/data/coco/annotations/instances_val2017.json"

# Deformable-DETR 原始检测结果路径
INPUT_PATH = f"{BASE_DIR}/save/attack/detection/deformable-detr/ori/res.json"

# ==================================================
# 辅助函数
# ==================================================
def load_image_mapping(annotations_path):
    """加载 image_id → file_name 映射"""
    with open(annotations_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {img["id"]: img["file_name"] for img in data["images"]}

def coco_category_name(cat_id: int) -> str:
    """COCO 类别名映射（可按需扩展）"""
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
    """打印前 max_lines 行 JSON 内容，供人工快速检查"""
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
    print("=== Deformable-DETR → 标准化转换开始 ===")

    if not os.path.exists(INPUT_PATH):
        print(f"⚠️ 未找到输入文件: {INPUT_PATH}")
        return

    image_mapping = load_image_mapping(ANNOT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 原始数据通常是 list[dict]
    detections = data if isinstance(data, list) else data.get("annotations", [])
    model_name = "deformable-detr"

    results = []
    for det in tqdm(detections, desc="标准化中"):
        try:
            image_id = int(det.get("image_id"))
        except Exception:
            continue

        file_name = image_mapping.get(image_id, f"unknown_{image_id}.png")
        cat_id = int(det.get("category_id", -1))
        cat_name = coco_category_name(cat_id)

        bbox = det.get("bbox", [])
        if not bbox or len(bbox) != 4:
            continue

        # 已是 [x1, y1, x2, y2]，无需转换
        x1, y1, x2, y2 = bbox
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
            "model": model_name,
            "version": VERSION
        })

    # ✅ 在保存前排序
    results = sorted(results, key=lambda x: (x["image_id"], x["category_id"], -x["score"]))

    # 输出保存
    output_path = os.path.join(os.path.dirname(INPUT_PATH), "res-std.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ {model_name}: {len(results)} 条记录 → {output_path}")
    save_preview(model_name, output_path)
    print("=== 转换完成 ===")

if __name__ == "__main__":
    main()
