import os, importlib, pathlib

def build_child_env():
    env = os.environ.copy()

    # 1) LD_LIBRARY_PATH: nvjitlink → cusparse → torch/lib → cuda/lib64 → old
    try:
        m = importlib.import_module("nvidia.nvjitlink")
        nvjitlink_dir = str(pathlib.Path(m.__file__).with_name("lib"))
    except Exception:
        nvjitlink_dir = "/usr/local/lib/python3.8/dist-packages/nvidia/nvjitlink/lib"
    cusparse_dir = "/usr/local/lib/python3.8/dist-packages/nvidia/cusparse/lib"
    torch_lib_dir = "/usr/local/lib/python3.8/dist-packages/torch/lib"
    cuda_home    = env.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib64   = os.path.join(cuda_home, "lib64")

    ld_parts = []
    for d in (nvjitlink_dir, cusparse_dir, torch_lib_dir, cuda_lib64):
        if os.path.isdir(d):
            ld_parts.append(d)
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts)

    # 2) PYTHONPATH: 三个 ops 放最前
    ops = [
        "/opt/data/private/BlackBox/models/DN-DETR/models/dn_dab_deformable_detr/ops",
        "/opt/data/private/BlackBox/models/Deformable-DETR/models/ops",
        "/opt/data/private/BlackBox/models/sparse_detr/models/ops",
    ]
    py_parts = []
    if env.get("PYTHONPATH"):
        py_parts = [p for p in env["PYTHONPATH"].split(":") if p]
    for d in reversed(ops):  # insert 到最前
        if os.path.isdir(d) and d not in py_parts:
            py_parts.insert(0, d)
    env["PYTHONPATH"] = ":".join(py_parts)

    return env






import os
import subprocess
import importlib
import pathlib

# --- 0. 定义全局参数 ---
# 您想要运行的实验版本号
VERSION = os.getenv("PATCH_VERSION", "kl")  # ← 优先使用外部传入的版本号
# 基础目录
BASE_MODELS_DIR = "/opt/data/private/BlackBox/models"
BASE_DATA_PATH = f"/opt/data/private/BlackBox/data/coco-patch-{VERSION}"
BASE_SAVE_DIR = f"/opt/data/private/BlackBox/save-{VERSION}/attack/detection"


# --- 1. 定义所有模型的配置 ---
# 字典存储每个模型的执行信息
# 注意：MODEL_CONFIGS 必须在这里定义
MODEL_CONFIGS = {
    "DN-DETR": {
        "dir": "DN-DETR",
        "command": [
            "python", "main.py", "-m", "dn_dab_detr",
            "--output_dir", f"{BASE_SAVE_DIR}/dn-detr/patch/",
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
            "--output_dir", f"{BASE_SAVE_DIR}/anchor-detr/patch/"
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
            "--output_dir", f"{BASE_SAVE_DIR}/sparse-detr/patch"
        ]
    },
    "Deformable-DETR": {
        "dir": "Deformable-DETR",
        "command": [
            "python", "main.py",
            "--eval",
            "--resume", f"{BASE_MODELS_DIR}/weights/r50_deformable_detr_single_scale-checkpoint.pth",
            "--output_dir", f"{BASE_SAVE_DIR}/deformable-detr/patch",
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
            "--output_dir", f"{BASE_SAVE_DIR}/detr/patch"
        ]
    }
}

# --- 2. 在定义 MODEL_CONFIGS 之后再打印 ---
print(f"--- 准备运行 {len(MODEL_CONFIGS)} 个模型，使用数据版本: {VERSION} ---")


# --- 3. 循环执行所有模型的评估 ---
# 保存初始工作目录，以便执行完毕后返回
original_cwd = os.getcwd()

def build_child_env():
    """
    构造子进程环境，优先使用：
      1) pip 安装的 nvidia nvjitlink (nvidia.nvjitlink)
      2) torch 自带的 lib 目录
      3) CUDA lib64
    返回一个 env 字典（从 os.environ.copy() 克隆并修改 LD_LIBRARY_PATH、PYTHONPATH）
    """
    env = os.environ.copy()

    # 1) nvjitlink from pip (if installed)
    nvjitlink_dir = ""
    try:
        m = importlib.import_module("nvidia.nvjitlink")
        nvjitlink_dir = str(pathlib.Path(m.__file__).with_name("lib"))
    except Exception:
        nvjitlink_dir = ""

    # 2) torch lib dir (prefer import, else fallback common path)
    torch_lib_dir = ""
    try:
        import torch as _torch
        torch_lib_dir = str(pathlib.Path(_torch.__path__[0]) / "lib")
    except Exception:
        possible = "/usr/local/lib/python3.8/dist-packages/torch/lib"
        if os.path.isdir(possible):
            torch_lib_dir = possible

    # 3) cuda lib64
    cuda_home = env.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib64 = os.path.join(cuda_home, "lib64")

    # Build LD_LIBRARY_PATH with preferred ordering
    ld_parts = []
    if nvjitlink_dir:
        ld_parts.append(nvjitlink_dir)
    if torch_lib_dir:
        ld_parts.append(torch_lib_dir)
    ld_parts.append(cuda_lib64)
    # append existing LD_LIBRARY_PATH if present
    if env.get("LD_LIBRARY_PATH"):
        ld_parts.append(env["LD_LIBRARY_PATH"])

    env["LD_LIBRARY_PATH"] = ":".join([p for p in ld_parts if p])

    # Add current model dir (ops) to PYTHONPATH to help importing local .so
    cwd = os.getcwd()
    existing_py = env.get("PYTHONPATH", "")
    if cwd not in existing_py.split(":"):
        env["PYTHONPATH"] = cwd + ((":" + existing_py) if existing_py else "")

    return env


try:
    for model_name, config in MODEL_CONFIGS.items():
        model_dir = os.path.join(BASE_MODELS_DIR, config["dir"])
        
        print("-" * 50)
        print(f"开始执行模型: {model_name}")
        print(f"切换到目录: {model_dir}")
        
        # 切换工作目录 (对应于 'cd ...')
        os.chdir(model_dir)
        
        # 确保输出目录存在
        try:
            output_dir_param_index = config["command"].index("--output_dir")
            output_dir_path = config["command"][output_dir_param_index + 1]
            os.makedirs(output_dir_path, exist_ok=True)
            print(f"输出结果将保存到: {output_dir_path}")
        except ValueError:
            pass

        # 打印将要执行的命令（便于调试）
        print(f"执行命令: {' '.join(config['command'])}")

        # 构造子进程环境并打印 LD_LIBRARY_PATH（便于诊断）
        env = build_child_env()
        print(f"[env] LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH')}")
        print(f"[env] PYTHONPATH={env.get('PYTHONPATH')}\n")

        # 启动子进程，使用构造好的 env（如果命令需要 bash + args，env 也会生效）
        subprocess.run(config["command"], check=True, env=env)
        
        print(f"{model_name} 执行成功。")


finally:
    # 无论脚本是否出错，都确保回到初始工作目录
    os.chdir(original_cwd)
    print("-" * 50)
    print("所有模型评估流程已结束。工作目录已恢复。")
