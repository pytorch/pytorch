python
import torch, gc, os, pytest
from torch.nn import GradBank
def test_wrap_linear():
    layer = GradBank.wrap(torch.nn.Linear(10, 5))
    x = torch.randn(4, 10).cuda()
    out = layer(x).sum()
    out.backward()
    # sanity: no NaN/inf
    assert torch.isfinite(layer.layer.weight.grad).all()
def test_memory_stable():
    layer = GradBank.wrap(torch.nn.Linear(1024, 1024)).cuda()
    x = torch.randn(64, 1024).cuda()
    base = torch.cuda.memory_allocated()
    for _ in range(1000):
        layer(x).sum().backward()
        del _
    gc.collect(); torch.cuda.empty_cache()
    peak = torch.cuda.max_memory_allocated()
    assert (peak - base) < 2 * 1024**3  # < 2 GB growth
def test_checkpoint_amp():
    layer = GradBank.wrap(torch.nn.Linear(10, 10)).cuda()
    x = torch.randn(2, 10).cuda()
    with torch.autocast("cuda"):
        out = torch.utils.checkpoint.checkpoint(layer, x).sum()
    out.backward()
    assert torch.isfinite(layer.layer.weight.grad).all()
def test_disabled_context():
    layer = GradBank.wrap(torch.nn.Linear(10, 10))
    x = torch.randn(2, 10)
    with GradBank.disabled():
        layer(x).sum().backward()
    # Hook ran but produced no scaling
    assert layer.step == -1
