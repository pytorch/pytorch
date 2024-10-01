# mypy: allow-untyped-defs
import functools
from typing import Dict, Set, Tuple

import torch
from torch import Tensor
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket

from ..pattern_matcher import fwd_only, register_replacement


aten = torch.ops.aten

def norm_pattern(x, weight, bias, eps):
    return torch.ops.aten.native_layer_norm(x, [x.shape[-1]], weight, bias, eps)

from torch._decomp.decompositions import *

from typing import Any, List, Optional, Tuple

# Copied and modified from decompositions.py
def native_layer_norm_backward(
    grad_out: Tensor,
    output: Tensor,
    normalized_shape: List[int],
    mean: Tensor,
    rstd: Tensor,
    weight: Optional[Tensor],
    bias: Optional[Tensor],
    output_mask: List[bool],
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    import torch._prims_common as utils
    def _unsqueeze_to_dim(x: Tensor, dim: int) -> Tensor:
        for _ in range(dim - x.dim()):
            x = x.unsqueeze(-1)
        return x

    def prod(x: List[int]):
        r = 1
        for i in x:
            r *= i
        return r
    def _maybe_cast(x: Optional[Tensor], dtype) -> Optional[Tensor]:
        if x is not None:
            return x.to(dtype)
        return x

    input_shape = output.shape
    input_ndim = output.dim()
    computation_dtype = utils.get_computation_dtype(output.dtype)
    grad_out_cast, output_cast, weight_cast, bias_cast = (
        x.to(computation_dtype).contiguous() if x is not None else x
        for x in (grad_out, output, weight, bias)
    )
    assert grad_out_cast is not None

    axis = input_ndim - len(normalized_shape)
    inner_dims = input_shape[axis:]
    outer_dims = input_shape[:axis]
    inner_dim_indices: List[int] = []
    outer_dim_indices: List[int] = []
    for i in range(input_ndim):
        if i >= axis:
            inner_dim_indices.append(i)
        else:
            outer_dim_indices.append(i)

    N = prod(inner_dims)  # type: ignore[arg-type]
    M = prod(outer_dims)  # type: ignore[arg-type]
    if M <= 0 or N <= 0:
        return (
            output.new_zeros(input_shape) if output_mask[0] else None,
            output.new_zeros(input_shape[axis:]) if output_mask[1] else None,
            output.new_zeros(input_shape[axis:]) if output_mask[2] else None,
        )
    mean = _unsqueeze_to_dim(mean, output_cast.dim())  # type: ignore[union-attr]
    rstd = _unsqueeze_to_dim(rstd, output_cast.dim())  # type: ignore[union-attr]
    x_hat = (output_cast - bias_cast) / weight_cast
    if weight_cast is not None:
        grad_x_hat = grad_out_cast * weight_cast
    else:
        grad_x_hat = grad_out_cast
    a = grad_x_hat * N
    b = torch.sum(grad_x_hat, inner_dim_indices, True)
    c1 = torch.mul(grad_x_hat, x_hat)
    c2 = torch.sum(c1, inner_dim_indices, True)
    c3 = torch.mul(x_hat, c2)

    inner = a - b - c3
    d_input: Optional[Tensor] = None
    d_weight: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        # breakpoint()
        d_input = (rstd / N) * inner

    if output_mask[1] and weight_cast is not None:
        if len(outer_dim_indices) > 0:
            d_weight = torch.sum(grad_out_cast * x_hat, outer_dim_indices, False)
        else:
            d_weight = grad_out_cast * x_hat

    if output_mask[2] and bias_cast is not None:
        if len(outer_dim_indices) > 0:
            d_bias = torch.sum(grad_out_cast, outer_dim_indices, False)
        else:
            d_bias = grad_out_cast.clone()

    return (
        _maybe_cast(d_input, output.dtype),
        _maybe_cast(d_weight, output.dtype),
        _maybe_cast(d_bias, output.dtype),
    )

class CustomLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        out, mean, rstd = torch.ops.aten.native_layer_norm(x, [x.shape[-1]], weight, bias, eps)
        ctx.save_for_backward(out, mean, rstd, weight, bias)
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        out, mean, rstd, weight, bias = ctx.saved_tensors
        return *native_layer_norm_backward(grad_out, out, [out.shape[-1]], mean, rstd, weight, bias, [True, True, True]), None

def norm_replacement(x, weight, bias, eps):
    return CustomLayerNorm.apply(x, weight, bias, eps), None, None

from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, joint_fwd_bwd, fwd_only


@functools.lru_cache(None)
def _misc_patterns_init():
    from .joint_graph import patterns as joint_graph_patterns
    from .post_grad import pass_patterns as post_grad_patterns_all

    post_grad_patterns = post_grad_patterns_all[1]  # medium priority

    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"

    # These patterns do 2 things
    # 1. Since we know that index is completely unique, we can codegen it using
    # stores instead of atomic adds, which is quite a bit faster.
    # 2. Also, since we are guaranteed that they are completely within bounds,
    # we can use unsafe indexing and skip debug asserts
    def randperm_index_add_pattern(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return torch.index_add(x, dim=0, source=y, index=index), index

    def randperm_index_add_replacement(x, y):
        index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
        return (
            torch.ops.aten._unsafe_index_put(
                x, (index,), aten._unsafe_index(x, (index,)) + y, accumulate=False
            ),
            index,
        )

    register_replacement(
        randperm_index_add_pattern,
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        fwd_only,
        [post_grad_patterns, joint_graph_patterns],
    )

    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        randperm_index_pattern,
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        fwd_only,
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
    )
    my_patterns = PatternMatcherPass()
    def improves_partition(match):
        num_output_users = set(match.output_node().users) - set(match.nodes)
        num_input_users = set(match.kwargs['x'].users) - set(match.nodes)
        print("trying!")
        if len(num_output_users) > len(num_input_users):
            print("replacing input-dependent norm_backward with output-dependent!")
            print(num_input_users, num_output_users)
            return True
        return False

    register_replacement(
        norm_pattern,
        norm_replacement,
        [torch.empty(4, 4, device='cuda', requires_grad=True, dtype=torch.float16), torch.randn(4, device='cuda', requires_grad=True), torch.randn(4, device='cuda', requires_grad=True)],
        joint_fwd_bwd,
        [joint_graph_patterns],
        extra_check=improves_partition,
        scalar_workaround={'eps': 42}
    )


class NumpyCompatNormalization:
    numpy_compat: Dict[str, Tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: Dict[str, str]
    cache: Dict["torch.fx.graph.Target", Set[str]]

    def __init__(self) -> None:
        self.cache = {}  # callable -> tuple of replaceable args e.g. ["axis"]
        self.inverse_mapping = {}
        for actual_kwarg, numpy_kwargs in self.numpy_compat.items():
            for numpy_kwarg in numpy_kwargs:
                assert numpy_kwarg not in self.inverse_mapping
                self.inverse_mapping[numpy_kwarg] = actual_kwarg

    def __call__(self, graph: torch.fx.Graph):
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if isinstance(node.target, (OpOverload, OpOverloadPacket)):
                # only applies to torch ops; e.g. torch.stack(axis=1) works, torch.ops.aten.stack(axis=1) doesn't.
                continue
            kwargs = node.kwargs

            if node.target in self.cache:
                replaceable_kwargs = self.cache[node.target]
            else:
                signatures = torch.fx.operator_schemas.get_signature_for_torch_op(
                    node.target
                )
                signatures = () if signatures is None else signatures
                replaceable_kwargs = set()
                for sig in signatures:
                    for param_name in sig.parameters.keys():
                        if param_name in self.numpy_compat:
                            replaceable_kwargs.update(self.numpy_compat[param_name])

                self.cache[node.target] = replaceable_kwargs

            if not replaceable_kwargs:
                continue

            new_kwargs = {}
            kwargs_changed = False
            for k, v in kwargs.items():
                if k in replaceable_kwargs:
                    kwargs_changed = True
                    new_kwargs[self.inverse_mapping[k]] = v
                else:
                    new_kwargs[k] = v

            if kwargs_changed:
                node.kwargs = torch.fx.immutable_collections.immutable_dict(new_kwargs)
                counters["inductor"]["numpy_compat_normalization"] += 1


numpy_compat_normalization = NumpyCompatNormalization()
