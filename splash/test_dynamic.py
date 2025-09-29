import torch
from .dynamic import DynamicManager

def test_dynamic_manager_identity_when_empty():
    B,T,d = 2, 8, 16
    h = [torch.randn(B,T,d) for _ in range(3)]  # 0 is embeddings
    mgr = DynamicManager(d_model=d)
    out = mgr.apply_layers(h)
    assert all((o==x).all().item() for o,x in zip(out,h))

def test_dynamic_manager_shapes_with_modules():
    B,T,d = 2, 8, 32
    h = [torch.randn(B,T,d) for _ in range(3)]
    mgr = DynamicManager(d_model=d)
    mgr.add_layer(1, use_sala=True, use_pos=True, pos_mode="diag", pos_alpha=0.1)
    out = mgr.apply_layers(h)
    assert out[0].shape == h[0].shape
    assert out[1].shape == h[1].shape
