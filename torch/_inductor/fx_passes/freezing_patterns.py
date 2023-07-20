import functools

import torch
from .. import config
from ..pattern_matcher import (
    _return_true,
    inference_graph,
    init_once_fakemode,
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


def freezing_passes(gm: torch.fx.GraphModule):
    """
    Passes that are applied to the graph to freeze pass.
    """

    lazy_init()
    for patterns in pass_patterns:
        patterns.apply(gm.graph)

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
def lazy_init():
    if torch._C._has_mkldnn and config.cpp.weight_prepack:
        from .mkldnn_fusion import _mkldnn_weight_pack_init

        _mkldnn_weight_pack_init()

    addmm_patterns_init()


def register_freezing_graph_pattern(pattern, extra_check=_return_true, pass_number=0):
    return register_graph_pattern(
        pattern,
        extra_check=extra_check,
        pass_dict=pass_patterns[pass_number],
    )


@functools.lru_cache(None)
def addmm_patterns_init():
    if torch.cuda.is_available():
        # workaround https://github.com/pytorch/pytorch/issues/97894
        device = "cuda"
    else:
        device = "cpu"
    val = functools.partial(torch.empty, (10, 10), device=device, requires_grad=False)

    def check_concat_weights(match):
        weights = [
            match.kwargs["w1"],
            match.kwargs["w2"],
            match.kwargs["w3"],
        ]
        return all(
            w.op == "get_attr" and w.meta["val"].shape == weights[0].meta["val"].shape
            for w in weights
        )

    def matmul_fuse_pattern(inp, w1, w2, w3):
        return (inp @ w1, inp @ w2, inp @ w3)

    def matmul_replacement(inp, w1, w2, w3):
        cat_t = torch.cat((w1, w2, w3), dim=1)
        mm = inp @ cat_t
        return mm.chunk(3, dim=1)

    register_replacement(
        matmul_fuse_pattern,
        matmul_replacement,
        [val(), val(), val(), val()],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3"),
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
        addmm_fuse_pattern_second,
        addmm_fuse_replacement_second,
        [val() for _ in range(7)],
        inference_graph,
        pass_patterns[0],
        extra_check=check_concat_weights,
        exclusive_arg_names=("w1", "w2", "w3", "b1", "b2", "b3"),
    )
