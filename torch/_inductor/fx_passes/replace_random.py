import collections
import functools
import logging

import torch
from ..._dynamo.utils import counters
from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunctionVarArgs,
    inference_graph,
    init_once_fakemode,
    Match,
    PatternMatcherPass,
    register_graph_pattern,
    register_replacement,
    training_graph,
)
from ..virtualized import V

log = logging.getLogger(__name__)
patterns = PatternMatcherPass()
aten = torch.ops.aten


def replace_random_passes(gm: torch.fx.GraphModule):
    """Modify the given FX graph to use backend-native random ops"""
    if config.fallback_random:
        return 0

    lazy_init()

    count = patterns.apply(gm)
    count += fuse_seed_creation_pass(gm.graph)

    return count


def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)

    if not device_seeds:
        return 0

    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            with V.fake_mode:
                combined.meta["val"] = torch.empty(
                    [len(seeds)], device=device, dtype=torch.int64
                )

        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(
                    inductor_prims.lookup_seed, (combined, idx)
                )
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)

    return len(device_seeds)


@init_once_fakemode
def lazy_init():
    if not torch.cuda.is_available():
        return

    # workaround https://github.com/pytorch/pytorch/issues/97894
    device = "cuda"
    # sizes/values don't actually matter for initial trace
    # once we get a possible match we re-trace with the actual values and verify the match still holds
    t = functools.partial(torch.empty, [1], device=device)
    # workaround https://github.com/pytorch/pytorch/issues/97894
    # 0.113377 is a "magic" value that lets us recover the lost input arg relationship
    workaround = {"dropout_p": 0.113377}

    register_replacement(
        _dropout_pattern,
        _dropout_replacement,
        [t(requires_grad=True), *workaround.values()],
        inference_graph,
        patterns,
        scalar_workaround=workaround,
        prepend=True,
    )
    register_replacement(
        _dropout_pattern,
        _dropout_replacement,
        [t(requires_grad=True), *workaround.values()],
        training_graph,
        patterns,
        scalar_workaround=workaround,
        prepend=True,
    )


def default_kwargs(device):
    return {}


def get_device(device):
    if device is not None:
        return device
    return torch.empty([]).device  # default device


def _dropout_pattern(x: torch.Tensor, dropout_p: float):
    return torch.dropout(x, dropout_p, True)


def _dropout_replacement(x: torch.Tensor, dropout_p: float):
    assert 0 < dropout_p < 1, "should have been handled in decomps"
    counters["inductor"]["replace_dropout"] += 1
    seed = inductor_prims.seed(x.device)
    scale = float(1.0 / (1.0 - dropout_p))

    # We could do low memory dropout here by recomputing the bool_mask
    # in both forward and backward.  Unfortunately, the latest version of
    # the partitioner doesn't play nice with this.  The CSE pass combines
    # the randomness calls, and if you fix that it decides to put both
    # bool_mask computations in the forwards.
    bool_mask = (
        inductor_prims.random([*x.size()], seed, "rand", **default_kwargs(x.device))
        > dropout_p
    )

    class Dropout(torch.autograd.Function):
        @staticmethod
        def forward(_, x):
            return bool_mask.to(x.dtype) * x * scale

        @staticmethod
        def backward(_, grad_output):
            return bool_mask.to(grad_output.dtype) * grad_output * scale

    return Dropout.apply(x)


@register_graph_pattern(CallFunctionVarArgs(aten.rand.default), pass_dict=patterns)
@register_graph_pattern(CallFunctionVarArgs(aten.randn.default), pass_dict=patterns)
def replace_random(
    match: Match, size, *, dtype=None, device=None, layout=None, pin_memory=None
):
    def replacement():
        result = inductor_prims.random(
            size, inductor_prims.seed(device), mode, **default_kwargs(device)
        )
        if dtype is not None:
            result = result.to(dtype)
        return result

    mode = {
        aten.rand.default: "rand",
        aten.randn.default: "randn",
    }[match.output_node().target]
    device = get_device(device)
    match.replace_by_example(replacement, [])


@register_graph_pattern(CallFunctionVarArgs(aten.randint.low), pass_dict=patterns)
def replace_randint(
    match: Match,
    low,
    high,
    size,
    *,
    dtype=torch.int64,
    device=None,
    layout=None,
    pin_memory=None,
):
    def replacement():
        result = inductor_prims.randint(low, high, size, inductor_prims.seed(device))
        return result.to(dtype)

    device = get_device(device)
    match.replace_by_example(replacement, [])
