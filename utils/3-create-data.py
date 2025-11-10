#!/usr/bin/env python3
import os
import shutil
import subprocess

# =============================================
# 1️⃣ 复制原始 COCO 数据到新版本 coco-patch-{VERSION}
# =============================================
VERSION = os.getenv("PATCH_VERSION", "kl")  # 外部传入优先
BASE_DIR = "/opt/data/private/BlackBox/data"
SOURCE_NAME = "coco"
TARGET_NAME = f"coco-patch-{VERSION}"

SOURCE_PATH = os.path.join(BASE_DIR, SOURCE_NAME)
TARGET_PATH = os.path.join(BASE_DIR, TARGET_NAME)

print(f"--- 步骤 1: 正在复制 {SOURCE_PATH} 到 {TARGET_PATH} ...")

try:
    shutil.copytree(SOURCE_PATH, TARGET_PATH, dirs_exist_ok=True)
    print(f"✅ 步骤 1 完成：{TARGET_PATH} 已更新。")
except Exception as e:
    print(f"❌ 步骤 1 失败：复制数据集时出错 → {e}")
    exit(1)

# =============================================
# 2️⃣ 替换 test2017 与 val2017 中的图片为带补丁图像
# =============================================
PATCH_IMG_DIR = f"/opt/data/private/BlackBox/save-{VERSION}/attack/detection/img/img-patch/"
TARGET_DATA_DIRS = [
    f"{TARGET_PATH}/test2017/",
    f"{TARGET_PATH}/val2017/"
]

for target_dir in TARGET_DATA_DIRS:
    print(f"\n--- 步骤 2: 正在替换 {target_dir} ---")
    try:
        # 清空旧文件夹
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            print(f"🗑️ 已清空旧目录: {target_dir}")

        # 重新创建
        os.makedirs(target_dir, exist_ok=True)

        # 拷贝新图片
        if not os.path.exists(PATCH_IMG_DIR):
            raise FileNotFoundError(f"补丁图片目录不存在: {PATCH_IMG_DIR}")

        print(f"📁 正在复制 {PATCH_IMG_DIR} → {target_dir}")
        # 使用 shell cp -r . 比较快
        subprocess.run(["cp", "-r", f"{PATCH_IMG_DIR}/.", target_dir], check=True)
        print(f"✅ 步骤 2 完成：{target_dir} 已更新为补丁图像版本。")

    except Exception as e:
        print(f"❌ 步骤 2 失败：替换 {target_dir} 时出错 → {e}")
        exit(1)

# =============================================
# 3️⃣ 替换 annotations 文件夹中的 JSON 文件内容
# =============================================
PATCH_REGION_JSON_PATH = f"/opt/data/private/BlackBox/save-{VERSION}/attack/detection/img/patch_regions.json"
TARGET_ANN_DIR = f"{TARGET_PATH}/annotations/"

TARGET_JSON_FILES = [
    "image_info_test-dev2017_info.json",
    "image_info_test-dev2017.json",
    "instances_val2017_info.json",
    "instances_val2017.json",
]

print(f"\n--- 步骤 3: 开始替换 {TARGET_ANN_DIR} 中的 JSON 文件 ---")

try:
    if not os.path.exists(PATCH_REGION_JSON_PATH):
        raise FileNotFoundError(f"未找到补丁 JSON: {PATCH_REGION_JSON_PATH}")

    # 读取 patch_regions.json
    with open(PATCH_REGION_JSON_PATH, "r", encoding="utf-8") as f:
        patch_json_data = f.read()

    os.makedirs(TARGET_ANN_DIR, exist_ok=True)

    for json_name in TARGET_JSON_FILES:
        target_json_path = os.path.join(TARGET_ANN_DIR, json_name)

        if os.path.exists(target_json_path):
            os.remove(target_json_path)
            print(f"🗑️ 已删除旧文件：{target_json_path}")

        with open(target_json_path, "w", encoding="utf-8") as f:
            f.write(patch_json_data)
        print(f"✅ 已替换文件：{target_json_path}")

    print(f"\n✅ 步骤 3 完成：{len(TARGET_JSON_FILES)} 个 JSON 文件已更新。")
    print(f"📄 新内容来源：{PATCH_REGION_JSON_PATH}")
    print(f"📁 目标 annotations 目录：{TARGET_ANN_DIR}")

except Exception as e:
    print(f"❌ 步骤 3 失败：更新 JSON 文件时出错 → {e}")
    exit(1)

# =============================================
# ✅ 结束提示
# =============================================
print(f"\n🎉 所有步骤执行完成！")
print(f"✅ 新数据集：{TARGET_PATH}")
print(f"包含补丁图片与补丁标注，可用于后续检测与筛选。")
