from typing import Callable

import torch
import torch.nn.functional as F
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._functorch.eager_transforms import (
    _unwrap_all_tensors_from_functional,
    _wrap_all_tensors_to_functional,
)
from torch._ops import HigherOrderOperator
from torch._subclasses import FakeTensorMode
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

sdpa = HigherOrderOperator("templated_attention")

# maybe remove kwargs


def math_attention(q, k, v, score_mod: Callable, *other_buffers):
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
    example_vals = [torch.zeros((), dtype=q.dtype)] + [
        torch.zeros((), dtype=torch.int) for _ in range(4)
    ]
    score_graph = make_fx(score_mod)(*example_vals, *other_buffers)
    proxy_mode.tracer.root.register_module("sdpa_score", score_graph)
    node_args = (q, k, v, score_graph, *other_buffers)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", sdpa, proxy_args, {}, name="sdpa_impl"
    )
    return track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=proxy_mode.tracer
    )


@sdpa.py_impl(ProxyTorchDispatchMode)
def sdpa_proxy_torch_dispatch_mode(q, k, v, score_mod, *other_buffers):
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if mode.enable_tracing:
            return trace_sdpa(mode, q, k, v, score_mod, *other_buffers)
        else:
            return sdpa(q, k, v, score_mod, *other_buffers)


@sdpa.py_impl(DispatchKey.Functionalize)
def sdpa_functionalize(q, k, v, score_mod, *other_buffers):
    reapply_views = torch._C._functionalization_reapply_views_tls()

    q, k, v, *other_buffers = _unwrap_all_tensors_from_functional(
        (q, k, v, *other_buffers), reapply_views=reapply_views
    )
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        out = sdpa(q, k, v, score_mod, *other_buffers)
        return _wrap_all_tensors_to_functional(out, level=0)


@sdpa.py_impl(FakeTensorMode)
def sdpa_fake_tensor_mode(*args, **kwargs):
    return sdpa_dense(*args, **kwargs)


# sdpa.fallthrough(DispatchKey.PythonDispatcher)
# sdpa.fallthrough(DispatchKey.PythonTLSSnapshot)
# sdpa.fallthrough(DispatchKey.ADInplaceOrView)
# sdpa.fallthrough(DispatchKey.BackendSelect)
# sdpa.fallthrough(DispatchKey.AutocastCPU)
# sdpa.fallthrough(DispatchKey.AutocastCPU)
