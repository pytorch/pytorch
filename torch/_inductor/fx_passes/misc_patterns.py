# mypy: allow-untyped-defs
import functools

import torch
from torch._dynamo.utils import counters
from torch._ops import OpOverload, OpOverloadPacket
from torch.utils._ordered_set import OrderedSet

from ..pattern_matcher import fwd_only, register_replacement


aten = torch.ops.aten


@functools.cache
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
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_add_replacement,
        [torch.empty(4, 8, device=device), torch.empty(2, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
    )

    def randperm_index_pattern(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten.index(x, (index,)), index

    def randperm_index_replacement(x, slice_shape):
        index = torch.randperm(x.shape[0], device=x.device)[:slice_shape]
        return torch.ops.aten._unsafe_index(x, (index,)), index

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        randperm_index_pattern,
        # pyrefly: ignore [bad-argument-type]
        randperm_index_replacement,
        [torch.empty(4, 8, device=device)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        [post_grad_patterns, joint_graph_patterns],
        scalar_workaround={"slice_shape": 42},
    )

    # Pattern: e8m0 extraction with ceiling rounding (for MX format scaling)
    # Only register on SM100+ where the PTX instruction is available
    if device == "cuda" and torch.cuda.get_device_capability() >= (10, 0):
        from .. import inductor_prims

        # Pattern 1: Bit manipulation approach
        def e8m0_rceil_pattern(inp):
            inp_bits = inp.view(torch.int32)
            biased_exp = (inp_bits >> 23) & 0xFF
            mantissa = inp_bits & 0x7FFFFF
            needs_round_up = mantissa != 0
            e8m0_biased = biased_exp + needs_round_up.to(torch.int32)
            e8m0_biased = torch.clamp(e8m0_biased, 0, 255)
            return e8m0_biased.to(torch.uint8)

        def e8m0_rceil_replacement(inp):
            return inductor_prims.cvt_e8m0_rceil(inp)

        def e8m0_extra_check(match):
            inp = match.kwargs.get("inp")
            if inp is None:
                return False
            inp_val = inp.meta.get("val")
            return (
                inp_val is not None
                and inp_val.device.type == "cuda"
                and inp_val.dtype == torch.float32
            )

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32)],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

        # Pattern 2: log2 + ceil approach (used by torchao MX formats)
        # Matches: (clamp(ceil(log2(x)), -127, 127) + 127).to(uint8)
        E8M0_BIAS = 127

        def e8m0_rceil_log2_pattern(inp):
            log2_val = torch.log2(inp)
            ceil_val = torch.ceil(log2_val)
            clamped = torch.clamp(ceil_val, min=-E8M0_BIAS, max=E8M0_BIAS)
            biased = clamped + E8M0_BIAS
            return biased.to(torch.uint8)

        def e8m0_rceil_log2_replacement(inp):
            # The PTX instruction expects the raw float value, not log2
            # So we need to convert: if inp is log2(x), then 2^inp is x
            # But actually our pattern matches on the value before log2
            return inductor_prims.cvt_e8m0_rceil(inp)

        register_replacement(
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_pattern,
            # pyrefly: ignore [bad-argument-type]
            e8m0_rceil_log2_replacement,
            [torch.randn(32, device="cuda", dtype=torch.float32).abs() + 1e-10],
            # pyrefly: ignore [bad-argument-type]
            fwd_only,
            # pyrefly: ignore [bad-argument-type]
            [post_grad_patterns],
            extra_check=e8m0_extra_check,
        )

    # TODO: Add pattern for cvt.rn.bf16x2.ue8m0x2 (e8m0 -> bf16 conversion)
    # This is the inverse operation for MX format dequantization


class NumpyCompatNormalization:
    numpy_compat: dict[str, tuple[str, ...]] = {
        "dim": ("axis",),
        "keepdim": ("keepdims",),
        "input": ("x", "a", "x1"),
        "other": ("x2",),
    }
    inverse_mapping: dict[str, str]
    cache: dict["torch.fx.graph.Target", OrderedSet[str]]

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
                    for param_name in sig.parameters:
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
