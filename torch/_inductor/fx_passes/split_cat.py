import functools
import logging

import torch
from torch._dynamo.utils import counters
from ..pattern_matcher import (
    Arg,
    CallFunction,
    CallMethod,
    config_flag,
    get_arg_value,
    MULTIPLE,
    PatternEntry,
)

log = logging.getLogger(__name__)


# Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
# subsequent optimizations
class NormalizeSplit(PatternEntry):
    def _get_split_args(self, split_node):
        input_kwarg = "tensor"
        split_size_kwarg = "split_size_or_sections"
        dim_kwarg = "dim"
        if split_node.op == "call_method":
            split_size_kwarg = "split_size"
        return (
            get_arg_value(split_node, 0, input_kwarg),
            get_arg_value(split_node, 1, split_size_kwarg),
            get_arg_value(split_node, 2, dim_kwarg),
        )

    def apply(self, match, graph, node):
        split_node = match.nodes[0]
        split_input, split_size, split_dim = self._get_split_args(split_node)
        if split_input is None or split_dim is None or split_size is None:
            log.warning("couldn't find split args")
            return
        if isinstance(split_size, (list, tuple)):
            return
        if "example_value" not in split_node.meta:
            log.warning("example value absent for node", split_node)
            return
        assert isinstance(split_node.meta["example_value"], (list, tuple))
        split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

        if any(isinstance(section, torch.SymInt) for section in split_sections):
            # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
            return
        with graph.inserting_after(split_node):
            new_split_node = graph.call_function(
                torch.split, args=(split_input, split_sections, split_dim)
            )
        split_node.replace_all_uses_with(new_split_node)
        new_split_node.meta.update(split_node.meta)
        graph.erase_node(split_node)
        counters["inductor"]["split_cat_norm"] += 1


@functools.lru_cache(None)
def _split_cat_init():
    from .pre_grad import patterns

    for pattern in [
        CallFunction(torch.split, Arg(), Arg(), Arg(), _users=MULTIPLE),
        CallFunction(torch.split, Arg(), Arg(), dim=Arg(), _users=MULTIPLE),
        CallFunction(
            torch.split, Arg(), split_size_or_sections=Arg(), dim=Arg(), _users=MULTIPLE
        ),
        CallFunction(
            torch.split,
            tensor=Arg(),
            split_size_or_sections=Arg(),
            dim=Arg(),
            _users=MULTIPLE,
        ),
        CallMethod("split", Arg(), Arg(), Arg(), _users=MULTIPLE),
        CallMethod("split", Arg(), Arg(), dim=Arg(), _users=MULTIPLE),
        CallMethod("split", Arg(), split_size=Arg(), dim=Arg(), _users=MULTIPLE),
    ]:
        pattern = NormalizeSplit(
            pattern=pattern, extra_check=config_flag("split_cat_fx_passes")
        )
        pattern.register(patterns)
