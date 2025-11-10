# 创建 /opt/data/private/BlackBox/prepare_environment.py
import os
import sys
import subprocess
import importlib

def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    dependencies = [
        "torch",
        "torchvision", 
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python",
        "timm",
        "pycocotools"
    ]
    
    print("检查依赖...")
    missing_deps = []
    
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} 已安装")
        except ImportError:
            missing_deps.append(dep)
            print(f"✗ {dep} 未安装")
    
    if missing_deps:
        print(f"安装缺失依赖: {missing_deps}")
        for dep in missing_deps:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
                print(f"✓ {dep} 安装成功")
            except subprocess.CalledProcessError:
                print(f"✗ {dep} 安装失败")
                return False
    else:
        print("所有依赖已就绪")
    
    return True

def setup_offline_mode():
    """设置离线模式，避免网络请求"""
    # 设置环境变量避免下载
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TORCH_HUB_OFFLINE'] = '1'
    
    print("✓ 已设置离线模式环境变量")

def main():
    print("=" * 50)
    print("环境准备脚本")
    print("=" * 50)
    
    success = check_and_install_dependencies()
    if success:
        setup_offline_mode()
        print("\n🎉 环境准备完成")
    else:
        print("\n❌ 环境准备失败")
        sys.exit(1)

if __name__ == "__main__":
    main()
