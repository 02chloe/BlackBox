import os
import subprocess

# --- 0. 定义全局参数 ---
# 您想要运行的实验版本号
VERSION = os.getenv("PATCH_VERSION", "clean")  # ← 优先使用外部传入的版本号

# 基础目录
BASE_MODELS_DIR = "/opt/data/private/BlackBox/models"
BASE_DATA_PATH = "/opt/data/private/BlackBox/data/coco"
BASE_SAVE_DIR = "/opt/data/private/BlackBox/save/attack/detection"


# --- 1. 定义所有模型的配置 ---
# 字典存储每个模型的执行信息
# 注意：MODEL_CONFIGS 必须在这里定义
MODEL_CONFIGS = {
    "DN-DETR": {
        "dir": "DN-DETR",
        "command": [
            "python", "main.py", "-m", "dn_dab_detr",
            "--output_dir", f"{BASE_SAVE_DIR}/dn-detr/ori/",
            "--batch_size", "1",
            "--coco_path", BASE_DATA_PATH,
            "--resume", f"{BASE_MODELS_DIR}/weights/dn_detr_r50_50ep.pth",
            "--use_dn",
            "--eval",
            "--save_results",
            "--num_workers", "0"
        ]
    },
    "AnchorDETR": {
        "dir": "anchor_detr",
        "command": [
            "python", "main.py",
            "--eval",
            "--coco_path", BASE_DATA_PATH,
            "--eval_set", "test",
            "--resume", f"{BASE_MODELS_DIR}/weights/AnchorDETR_r50_c5.pth",
            "--output_dir", f"{BASE_SAVE_DIR}/anchor-detr/ori/"
        ]
    },
    "Sparse-DETR": {
        # 注意：Sparse-DETR 使用 bash 脚本，需要特殊处理
        "dir": "sparse_detr",
        "command": [
            "bash", "./configs/r50_sparse_detr_rho_0.1.sh",
            "--resume", f"{BASE_MODELS_DIR}/weights/sparse_detr_r50_10.pth",
            "--eval",
            "--coco_path", BASE_DATA_PATH,
            "--output_dir", f"{BASE_SAVE_DIR}/sparse-detr/ori"
        ]
    },
    "Deformable-DETR": {
        "dir": "Deformable-DETR",
        "command": [
            "python", "main.py",
            "--eval",
            "--resume", f"{BASE_MODELS_DIR}/weights/r50_deformable_detr_single_scale-checkpoint.pth",
            "--output_dir", f"{BASE_SAVE_DIR}/deformable-detr/ori",
            "--coco_path", BASE_DATA_PATH,
            "--num_feature_levels", "1"
        ]
    },
    "DETR": {
        "dir": "detr",
        "command": [
            "python", "main.py",
            "--batch_size", "2",
            "--no_aux_loss",
            "--eval",
            "--resume", "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth",
            "--coco_path", BASE_DATA_PATH,
            "--output_dir", f"{BASE_SAVE_DIR}/detr/ori"
        ]
    }
}

# --- 2. 在定义 MODEL_CONFIGS 之后再打印 ---
print(f"--- 准备运行 {len(MODEL_CONFIGS)} 个模型，使用数据版本: {VERSION} ---")


# --- 3. 循环执行所有模型的评估 ---
# 保存初始工作目录，以便执行完毕后返回
original_cwd = os.getcwd()

try:
    for model_name, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(BASE_MODELS_DIR, config["dir"])
        
        print("-" * 50)
        print(f"开始执行模型: {model_name}")
        print(f"切换到目录: {model_dir}")
        
        # 切换工作目录 (对应于 'cd ...')
        os.chdir(model_dir)
        
        # 确保输出目录存在
        # output_dir是command list中的一个参数，我们手动提取第一个作为检查对象
        # 查找 --output_dir 参数的位置，并获取其后的路径
        try:
            output_dir_param_index = config["command"].index("--output_dir")
            output_dir_path = config["command"][output_dir_param_index + 1]
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"输出结果将保存到: {output_dir_path}")
        except ValueError:
            # 如果没有 --output_dir 参数，则跳过目录创建
            pass


        # 执行命令 (对应于 'python main.py ...' 或 'bash ...')
        # 打印执行的命令（易于调试）
        print(f"执行命令: {' '.join(config['command'])}")
        
        # subprocess.run 运行命令，check=True 表示如果命令返回非零退出码（通常表示错误），则抛出异常
        # 这里为了简化Jupyter输出，暂时不捕获大量输出，让它直接流向Notebook
        subprocess.run(config["command"], check=True)
        
        print(f"{model_name} 执行成功。")


finally:
    # 无论脚本是否出错，都确保回到初始工作目录
    os.chdir(original_cwd)
    print("-" * 50)
    print("所有模型评估流程已结束。工作目录已恢复。")
