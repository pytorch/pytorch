import torch
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional, _wrap_all_tensors_to_functional, functionalize
import torch.nn.functional as F
import torch.fx as fx
from torch._ops import HigherOrderOperator
from torch._C import DispatchKey, DispatchKeySet, _ExcludeDispatchKeyGuard
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    make_fx,
    ProxyTorchDispatchMode,
    track_tensor_tree,
)
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.compile_fx import compile_fx_inner

sdpa = HigherOrderOperator("sdpa")

def math_attention(q, k, v, score_mod, *other_buffers):
    scores = q @ k.transpose(-2, -1)

    from functorch.dim import dims
    b, h, m, n = dims()

    scores = scores[b, h, m, n]
    scores = score_mod(scores, b, h, m, n, *other_buffers)
    scores = scores.order(b, h, m, n)

    scores = scores.softmax(dim=-1)
    return scores @ v

@sdpa.py_impl(DispatchKey.CompositeExplicitAutograd)
def sdpa_dense(q, k, v, score_mod, *other_buffers):
    return math_attention(q, k, v, score_mod, *other_buffers).contiguous()
    # out = F.scaled_dot_product_attention(q, k, v, scale=1).contiguous()
    # return out

@sdpa.py_impl(DispatchKey.Autograd)
def sdpa_autograd(*args, **kwargs):
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.AutogradCPU)):
        return sdpa(*args, **kwargs)

def trace_sdpa(proxy_mode, q, k, v, score_mod, *other_buffers):
    if score_mod is None:
        with proxy_mode:
            return F.scaled_dot_product_attention(q, k, v)

    with disable_proxy_modes_tracing():
        example_out = F.scaled_dot_product_attention(q, k, v)
    example_vals = [torch.zeros((), dtype=q.dtype)] + [torch.zeros((), dtype=torch.int) for _ in range(4)]
    score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("sdpa_score", score_graph)
    node_args = (q, k, v, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy('call_function', sdpa, proxy_args, {}, name="sdpa_impl")
    return track_tensor_tree(example_out, out_proxy, constant=None, tracer=proxy_mode.tracer)

@sdpa.py_impl(ProxyTorchDispatchMode)
def sdpa_proxy_torch_dispatch_mode(q, k, v, score_mod, *other_buffers):
    mode = _get_current_dispatch_mode()
    assert (mode is not None), "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_sdpa(mode, q, k, v, score_mod, *other_buffers)
        else:
            return sdpa(q, k, v, score_mod, *other_buffers)

@sdpa.py_impl(DispatchKey.Functionalize)
def sdpa_functionalize(q, k, v, score_mod, *other_buffers):
    reapply_views = torch._C._functionalization_reapply_views_tls()

    q, k, v, *other_buffers = _unwrap_all_tensors_from_functional((q, k, v, *other_buffers), reapply_views=reapply_views)
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        out = sdpa(q, k, v, score_mod, *other_buffers)
        return _wrap_all_tensors_to_functional(out, level=0)

@sdpa.py_impl(FakeTensorMode)
def sdpa_fake_tensor_mode(*args, **kwargs):
    return sdpa_dense(*args, **kwargs)

sdpa.fallthrough(DispatchKey.PythonDispatcher)
sdpa.fallthrough(DispatchKey.PythonTLSSnapshot)
sdpa.fallthrough(DispatchKey.ADInplaceOrView)
sdpa.fallthrough(DispatchKey.BackendSelect)
sdpa.fallthrough(DispatchKey.AutocastCPU)
sdpa.fallthrough(DispatchKey.AutocastCPU)

def bench(f, name=None, iters=100, warmup=5, display=True, profile=False):
    import time
    from triton.testing import do_bench

    for _ in range(warmup):
        f()
    if profile:
        with torch.profiler.profile() as prof:
            f()
        prof.export_chrome_trace(f"{name if name is not None else 'trace'}.json")

    us_per_iter = do_bench(lambda: f())*1000

    if name is None:
        res = us_per_iter
    else:
        res= f"{name}: {us_per_iter:.3f}us"

    if display:
        print(res)
    return res

import torch

Z = 4
H = 8
N_CTX = 2048
D_HEAD = 64
dtype = torch.float16
torch.manual_seed(0)
q = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
k = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
v = torch.randn((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
torch.set_default_device('cuda')

vals = torch.randn(N_CTX, dtype=dtype)
def identity(score, b, h, m, n):
    return score

def causal_mask(score, b, h, m, n):
    return torch.where(m <= n, score, float("-inf"))

def rel_bias(score, b, h, m, n):
    bias = (m - n)
    return score + bias

def compose(*fs):
    def new_func(score, b, h, m, n):
        for f in fs:
            score = f(score, b, h, m, n)
        return score
    return new_func

def alibi_bias(score, b, h, m, n):
    bias = (m - n) * h
    return score + bias

def create_attention(score_mod):
    def foo(q, k, v):
        return sdpa(q, k, v, score_mod)
    return foo

score_mods = {
    "nop": identity,
    "causal": causal_mask,
    "rel": rel_bias,
    "alibi": alibi_bias,
    "rel + causal": compose(rel_bias, causal_mask),
    "alibi + causal": compose(alibi_bias, causal_mask),
    }

for name, score_mod in score_mods.items():
    foo = create_attention(score_mod)
    compiled = torch.compile(foo)

    ref_out = foo(q.to(torch.float64), k.to(torch.float64), v.to(torch.float64))
    compiled_out = compiled(q, k, v)
    torch.testing.assert_close(ref_out.to(dtype=torch.float16), compiled_out, atol=1e-1, rtol=0)
    bench(lambda: foo(q, k, v), "eager")
    bench(lambda: compiled(q, k, v), "compiled")


# bench(lambda: foo(q, k, v))
# exit(0)