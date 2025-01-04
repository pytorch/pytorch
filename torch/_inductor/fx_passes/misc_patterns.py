# mypy: allow-untyped-defs
import functools
from collections import deque
from typing import Dict, Tuple

import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement
from ..utils import partialize_and_update_signature


aten = torch.ops.aten


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

    # Float8 training patterns
    E4M3_MAX_POS = torch.finfo(torch.float8_e4m3fn).max
    E5M2_MAX_POS = torch.finfo(torch.float8_e5m2).max

    def amax_with_scaling_pattern(tensor_x_inp, scale_x, IS_E5M2):
        tensor_x = tensor_x_inp.to(torch.float32) * scale_x
        if IS_E5M2:
            tensor_x = tensor_x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
            tensor_x = tensor_x.to(torch.float8_e5m2)
        else:
            tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_x = tensor_x.to(torch.float8_e4m3fn)
        amax = torch.max(torch.abs(tensor_x_inp))
        return (tensor_x, amax)

    def amax_with_scaling_tiled_replacement(tensor_x_inp, scale_x, IS_E5M2):
        tensor_x = tensor_x_inp.to(torch.float32) * scale_x
        if IS_E5M2:
            tensor_x = tensor_x.clamp(min=-1 * E5M2_MAX_POS, max=E5M2_MAX_POS)
            tensor_x = tensor_x.to(torch.float8_e5m2)
        else:
            tensor_x = tensor_x.clamp(min=-1 * E4M3_MAX_POS, max=E4M3_MAX_POS)
            tensor_x = tensor_x.to(torch.float8_e4m3fn)
        amax_1 = torch.max(torch.abs(tensor_x_inp), dim=-1).values
        amax = torch.max(amax_1)
        return (tensor_x, amax)

    # The amax_with_scaling_pattern will also match dynamic scaling cases, we want to avoid that.
    # `scale_x` of delayed scaling comes from the previous iteration, instead of from `tensor_x_inp`.
    # We check that `scale_x` is not a dependency of `tensor_x_inp`
    def fp8_delayed_scaling_extra_check(match):
        scale_x_inputs = deque([match.kwargs["scale_x"]])
        max_num_node_to_check = 50  # Don't traverse too many nodes
        current_num_node = 0
        while len(scale_x_inputs) > 0 and current_num_node < max_num_node_to_check:
            current_node = scale_x_inputs.popleft()
            for n in current_node.all_input_nodes:
                if n == match.kwargs["tensor_x_inp"]:
                    return False
                scale_x_inputs.append(n)
                current_num_node += 1
        return True

    if torch.cuda.is_available():
        for IS_E5M2 in [True, False]:
            # torch.float16 has the same pattern as torch.bfloat16, because they both needs `tensor_x_inp.to(torch.float32)`
            # It will cause errors in `assert pattern_repr not in _seen_patterns`
            for dtype in [torch.float32, torch.bfloat16]:
                device = "cuda"
                register_replacement(
                    partialize_and_update_signature(
                        amax_with_scaling_pattern, IS_E5M2=IS_E5M2
                    ),
                    partialize_and_update_signature(
                        amax_with_scaling_tiled_replacement, IS_E5M2=IS_E5M2
                    ),
                    [
                        torch.tensor((16, 16), device=device, dtype=dtype),
                        torch.tensor(2.0, device=device, dtype=torch.float32),
                    ],
                    fwd_only,
                    post_grad_patterns,
                    extra_check=fp8_delayed_scaling_extra_check,
                )


class NumpyCompatNormalization:
    numpy_compat: Dict[str, Tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: Dict[str, str]
    cache: Dict["torch.fx.graph.Target", OrderedSet[str]]

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
                replaceable_kwargs = OrderedSet()
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
