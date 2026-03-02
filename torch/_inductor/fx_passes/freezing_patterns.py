# mypy: allow-untyped-defs
import functools

import torch
from torch._inductor.compile_fx import fake_tensor_prop
from torch._inductor.utils import GPU_TYPES

from ..._dynamo.utils import counters
from .. import config
from ..pattern_matcher import (
    _return_true,
    CallFunction,
    fwd_only,
    Ignored,
    init_once_fakemode,
    KeywordArg,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    stable_topological_sort,
)


aten = torch.ops.aten

# First pass_patterns[0] are applied, then [1], then [2]
pass_patterns = [
    PatternMatcherPass(),
    PatternMatcherPass(),
    PatternMatcherPass(),
]

binary_folding_pass = PatternMatcherPass()


def freezing_passes(gm: torch.fx.GraphModule, aot_example_inputs):
    """
    Passes that are applied to the graph to freeze pass.
    """

    from ..freezing import constant_fold

    lazy_init()
    # We need a few rounds of binary folding to get rid of all the
    # unnecessary nodes, but may need a good method to chose the rounds number.
    # works like: conv+binary+binary.
    binary_folding = counters["inductor"]["binary_folding"]
    fake_tensor_prop(gm, aot_example_inputs, True)

    torch._inductor.fx_passes.binary_folding.mark_mixed_dtype_allowed_computation_ops(
        gm
    )
    for _ in range(4):
        constant_fold(gm)
        # Make sure meta['val'] is properly set for all nodes
        fake_tensor_prop(gm, aot_example_inputs, True)
        binary_folding_pass.apply(gm.graph)  # type: ignore[arg-type]
        # If we don't have binary folding, we don't need to run the pass again.
        # TODO: remove the need to run fake_tensor_prop on the whole model.
        if counters["inductor"]["binary_folding"] == binary_folding:
            break
        binary_folding = counters["inductor"]["binary_folding"]

    torch._inductor.fx_passes.binary_folding.recover_original_precision_folded_computation_ops(
        gm
    )

    constant_fold(gm)
    fake_tensor_prop(gm, aot_example_inputs, True)

    for pattern in pass_patterns:
        pattern.apply(gm.graph)  # type: ignore[arg-type]

    # The CPU weight packing always assume the conv's weight is channels last,
    # So make sure the layout_optimization is on when doing it.
    if (
        torch._C._has_mkldnn
        and config.cpp.weight_prepack
        and config.layout_optimization
    ):
        from .mkldnn_fusion import _eliminate_duplicate_packed_nodes

        _eliminate_duplicate_packed_nodes(gm)

    stable_topological_sort(gm.graph)
    gm.recompile()
    gm.graph.lint()


@init_once_fakemode
def lazy_init(input_device: torch.device | None = None):
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    from .binary_folding import binary_folding_init

    addmm_patterns_init()
    binary_folding_init()


def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    while pass_number > len(pass_patterns) - 1:
        pass_patterns.append(PatternMatcherPass())
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=pass_patterns[pass_number],
    )


def register_binary_folding_pattern(pattern, extra_check=_return_true):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        # pyrefly: ignore [bad-argument-type]
        pass_dict=binary_folding_pass,
    )


def _register_int4_mm_fusion_patterns(device, val):
    """
    Register INT4 GPU matmul fusion patterns for _weight_int4pack_mm.

    Fuses multiple INT4 matmuls with the same input into a single matmul + split:
        (int4_mm(x, w1, gs, sz1), int4_mm(x, w2, gs, sz2), ...)
        -> split(int4_mm(x, cat(w1,w2,...), gs, cat(sz1,sz2,...)), sizes)

    This is beneficial for patterns like Q/K/V projections in attention layers.

    For INT4 packed weight format:
    - Packed weight shape: [N/8, K/(inner_k*16), 32, inner_k/2]
    - Scale_zeros shape: [K/group_size, N, 2]
    We concatenate packed weights on dim=0 and scale_zeros on dim=1.

    Note: Scalar group_size must be hardcoded in patterns (not a variable).
    We register patterns for common group_size values.
    """
    if device not in ("cuda", "mps"):
        return

    # Create example tensors for INT4 packed weight format
    # These are representative shapes for pattern matching; actual shapes vary.
    # Packed weight shape: [N/8, K/(inner_k*16), 32, inner_k/2] where inner_k=8
    # Scale_zeros shape: [K/group_size, N, 2] (scale and zero packed together)
    # Note: scale_zeros dtype can be bfloat16, float16, or float32 in practice;
    # the example dtype here is just for tracing the pattern.
    int4_weight = functools.partial(
        torch.empty,
        (8, 1, 32, 4),
        device=device,
        dtype=torch.int32,
        requires_grad=False,
    )
    int4_scale_zeros = functools.partial(
        torch.empty,
        (1, 64, 2),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=False,
    )

    def check_int4_gpu_concat_weights(match):
        """Check if INT4 GPU weights can be concatenated."""
        # Must be on CUDA or MPS
        inp_val = match.kwargs["inp"].meta["val"]
        if not (inp_val.is_cuda or inp_val.is_mps):
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        scale_inputs = ["sz1", "sz2"]
        if "sz3" in match.kwargs:
            scale_inputs.append("sz3")

        # All weights must be get_attr (constants)
        for wgt in weight_inputs:
            if match.kwargs[wgt].op != "get_attr":
                return False

        # All scale_zeros must be get_attr (constants)
        for sz in scale_inputs:
            if match.kwargs[sz].op != "get_attr":
                return False

        # Weights must have same K dimension (dims 1-3 must match)
        first_w = match.kwargs["w1"].meta["val"]
        for wgt in weight_inputs[1:]:
            w = match.kwargs[wgt].meta["val"]
            if w.shape[1:] != first_w.shape[1:]:
                return False

        # Scale_zeros must have compatible shapes (same dims except dim=1)
        # and same dtype for safe concatenation
        first_sz = match.kwargs["sz1"].meta["val"]
        for sz in scale_inputs[1:]:
            sz_val = match.kwargs[sz].meta["val"]
            # Shape is [K/gs, N, 2] - dims 0 and 2 must match
            if (
                sz_val.shape[0] != first_sz.shape[0]
                or sz_val.shape[2] != first_sz.shape[2]
            ):
                return False
            # Dtype must match for concatenation
            if sz_val.dtype != first_sz.dtype:
                return False

        return True

    # 3-way fusion pattern (e.g., Q/K/V projections)
    def int4_pattern_three(inp, w1, w2, w3, sz1, sz2, sz3, group_size):
        return (
            aten._weight_int4pack_mm.default(inp, w1, group_size, sz1),
            aten._weight_int4pack_mm.default(inp, w2, group_size, sz2),
            aten._weight_int4pack_mm.default(inp, w3, group_size, sz3),
        )

    def int4_replacement_three(inp, w1, w2, w3, sz1, sz2, sz3, group_size):
        cat_w = torch.cat((w1, w2, w3), dim=0)
        cat_sz = torch.cat((sz1, sz2, sz3), dim=1)
        mm = aten._weight_int4pack_mm.default(inp, cat_w, group_size, cat_sz)
        # Packed weight shape is [N/8, ...], so N = w.size(0) * 8
        # (assumes inner_k_tiles=8 which is the standard int4pack format)
        n1, n2 = w1.size(0) * 8, w2.size(0) * 8
        return mm.tensor_split([n1, n1 + n2], dim=-1)

    # 2-way fusion pattern (e.g., K/V in cross-attention)
    def int4_pattern_two(inp, w1, w2, sz1, sz2, group_size):
        return (
            aten._weight_int4pack_mm.default(inp, w1, group_size, sz1),
            aten._weight_int4pack_mm.default(inp, w2, group_size, sz2),
        )

    def int4_replacement_two(inp, w1, w2, sz1, sz2, group_size):
        cat_w = torch.cat((w1, w2), dim=0)
        cat_sz = torch.cat((sz1, sz2), dim=1)
        mm = aten._weight_int4pack_mm.default(inp, cat_w, group_size, cat_sz)
        # Packed weight shape is [N/8, ...], so N = w.size(0) * 8
        n1 = w1.size(0) * 8
        return mm.tensor_split([n1], dim=-1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        int4_pattern_three,
        # pyrefly: ignore [bad-argument-type]
        int4_replacement_three,
        [
            val(),
            int4_weight(),
            int4_weight(),
            int4_weight(),
            int4_scale_zeros(),
            int4_scale_zeros(),
            int4_scale_zeros(),
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_int4_gpu_concat_weights,
        scalar_workaround={"group_size": 128},
        exclusive_arg_names=("w1", "w2", "w3", "sz1", "sz2", "sz3"),
    )
    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        int4_pattern_two,
        # pyrefly: ignore [bad-argument-type]
        int4_replacement_two,
        [
            val(),
            int4_weight(),
            int4_weight(),
            int4_scale_zeros(),
            int4_scale_zeros(),
        ],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_int4_gpu_concat_weights,
        scalar_workaround={"group_size": 128},
        exclusive_arg_names=("w1", "w2", "sz1", "sz2"),
    )


@functools.cache
def addmm_patterns_init():
    """
    addmm related patterns.
    To avoid duplication, also includes int8 WoQ GEMM pattern without bias.
    """
    device = next(
        (gpu for gpu in GPU_TYPES if getattr(torch, gpu).is_available()), "cpu"
    )
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)
    scale = functools.partial(torch.empty, (10,), device=device, requires_grad=False)

    def check_int8_woq_concat_linear_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if not is_cpu or not config.cpp.enable_concat_linear:
            # Currently, this pattern is only supported on CPU
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        if not all(
            match.kwargs[wgt].target is torch.ops.prims.convert_element_type.default
            for wgt in weight_inputs
        ):
            return False

        if not all(
            next(iter(match.kwargs[wgt]._input_nodes.keys())).meta["val"].dtype
            is torch.int8
            for wgt in weight_inputs
        ):
            return False

        if not all(
            match.kwargs[wgt].meta["val"].dtype is torch.bfloat16
            for wgt in weight_inputs
        ):
            return False

        return True

    def check_concat_weights(match):
        is_cpu = match.kwargs["inp"].meta["val"].is_cpu
        if is_cpu and not config.cpp.enable_concat_linear:
            return False

        weight_inputs = ["w1", "w2"]
        if "w3" in match.kwargs:
            weight_inputs.append("w3")

        equal_shape_inputs = [weight_inputs]

        if "b1" in match.kwargs:
            bias_inputs = ["b1", "b2"]
            if "b3" in match.kwargs:
                bias_inputs.append("b3")

            equal_shape_inputs.append(bias_inputs)

        for equal_shape_group in equal_shape_inputs:
            inps = [match.kwargs[name] for name in equal_shape_group]

            if not all(
                inp.op == "get_attr"
                and inp.meta["val"].shape == inps[0].meta["val"].shape
                for inp in inps
            ):
                return False
        return True

    def int8_woq_fusion_pattern(inp, w1, w2, w3, s1, s2, s3):
        return ((inp @ w1) * s1, (inp @ w2) * s2, (inp @ w3) * s3)

    def int8_woq_fusion_replacement(inp, w1, w2, w3, s1, s2, s3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_s = torch.cat((s1, s2, s3), dim=0)
        mm = (inp @ cat_w).mul(cat_s)
        n1, n2 = w1.size(1), w2.size(1)
        return mm.tensor_split([n1, n1 + n2], dim=-1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        int8_woq_fusion_pattern,
        # pyrefly: ignore [bad-argument-type]
        int8_woq_fusion_replacement,
        [val(), val(), val(), val(), scale(), scale(), scale()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_int8_woq_concat_linear_weights,
        exclusive_arg_names=("w1", "w2", "w3", "s1", "s2", "s3"),
    )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        matmul_fuse_pattern,
        # pyrefly: ignore [bad-argument-type]
        matmul_replacement,
        [val(), val(), val(), val()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
    )

    def matmul_fuse_pattern_two(inp, w1, w2):
        return (inp @ w1, inp @ w2)

    def matmul_replacement_two(inp, w1, w2):
        cat_t = torch.cat((w1, w2), dim=1)
        mm = inp @ cat_t
        return mm.chunk(2, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        matmul_fuse_pattern_two,
        # pyrefly: ignore [bad-argument-type]
        matmul_replacement_two,
        [val(), val(), val()],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2"),
    )

    def addmm_fuse_pattern_second(inp, w1, w2, w3, b1, b2, b3):
        return (
            aten.addmm(b1, inp, w1),
            aten.addmm(b2, inp, w2),
            aten.addmm(b3, inp, w3),
        )

    def addmm_fuse_replacement_second(inp, w1, w2, w3, b1, b2, b3):
        cat_w = torch.cat((w1, w2, w3), dim=1)
        cat_b = torch.cat((b1, b2, b3))
        return aten.addmm(cat_b, inp, cat_w).chunk(3, dim=1)

    register_replacement(
        # pyrefly: ignore [bad-argument-type]
        addmm_fuse_pattern_second,
        # pyrefly: ignore [bad-argument-type]
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        # pyrefly: ignore [bad-argument-type]
        fwd_only,
        # pyrefly: ignore [bad-argument-type]
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )

    # Register INT4 GPU matmul fusion patterns
    _register_int4_mm_fusion_patterns(device, val)


def same_dtype(match):
    return match.output_node().args[0].meta["val"].dtype == match.kwargs["dtype"]


@register_graph_pattern(
    CallFunction(
        torch.ops.prims.convert_element_type.default,
        Ignored(),
        KeywordArg("dtype"),
    ),
    # pyrefly: ignore [bad-argument-type]
    pass_dict=pass_patterns[0],
    extra_check=same_dtype,
)
def unnecessary_dtype_convert(match: Match, **kwargs):
    """Remove unnecessary dtype conversion op, probably left as a result of Conv-Bn folding"""
    graph = match.graph
    node = match.output_node()
    node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
    graph.erase_node(node)
