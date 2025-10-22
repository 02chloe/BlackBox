# /opt/data/private/BlackBox/utils/models_loader/__init__.py
import sys
import importlib
from contextlib import contextmanager

@contextmanager
def use_repo(repo_path: str):
    """临时切换sys.path并清除冲突模块"""
    backup_path = list(sys.path)
    backup_modules = {}
    for name in ["models", "util", "ops"]:
        if name in sys.modules:
            backup_modules[name] = sys.modules.pop(name)
    sys.path.insert(0, repo_path)
    try:
        yield
    finally:
        sys.path = backup_path
        # 不恢复models防止污染
