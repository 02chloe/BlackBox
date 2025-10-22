# /opt/data/private/BlackBox/utils/load_model.py

# ✅ 改成延迟导入：
def load_deformable_detr(*args, **kwargs):
    from .models_loader.deformable_detr_loader import load_deformable_detr as f
    return f(*args, **kwargs)

def load_dn_detr(*args, **kwargs):
    from .models_loader.dn_detr_loader import load_dn_detr as f
    return f(*args, **kwargs)

def load_sparse_detr(*args, **kwargs):
    from .models_loader.sparse_detr_loader import load_sparse_detr as f
    return f(*args, **kwargs)

def load_anchor_detr(*args, **kwargs):
    from .models_loader.anchor_detr_loader import load_anchor_detr as f
    return f(*args, **kwargs)

def load_detr_r50(*args, **kwargs):
    from .models_loader.detr_r50_loader import load_detr_r50 as f
    return f(*args, **kwargs)


__all__ = [
    "load_detr_r50",
    "load_deformable_detr",
    "load_anchor_detr",
    "load_sparse_detr",
    "load_dn_detr",
]
