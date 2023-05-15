import functools
import logging
import operator
from typing import List

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import (
    Arg,
    CallFunction,
    CallMethod,
    config_flag,
    FailedMatch,
    get_arg_value,
    Ignored,
    KeywordArg,
    Match,
    MatchContext,
    MULTIPLE,
    PatternEntry,
    register_graph_pattern,
)
from .pre_grad import merge_splits_pass

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
        if "example_value" not in split_node.meta:
            log.warning("example value absent for node: %s", split_node)
            return
        assert isinstance(split_node.meta["example_value"], (list, tuple))
        split_sections = [t.size()[split_dim] for t in split_node.meta["example_value"]]

        if any(isinstance(section, torch.SymInt) for section in split_sections):
            # TODO dynamic_shapes with assume_static_by_default=False fails while AOT Autograd tracing.
            return
        if split_dim < 0:  # Normalize split dim
            split_dim += split_input.meta["example_value"].dim()
        with graph.inserting_after(split_node):
            new_split_node = graph.call_function(
                torch.split,
                args=(split_input, split_sections),
                kwargs={"dim": split_dim},
            )
        split_node.replace_all_uses_with(new_split_node)
        new_split_node.meta.update(split_node.meta)
        graph.erase_node(split_node)
        counters["inductor"]["split_cat_norm"] += 1


class TorchSplit(CallFunction):
    """
    Matches a call to torch.split if it is in a normalized form. Ensures that all users of
    splits are unique getitems.
    """

    def __init__(self, arg, sizes):
        # using KeywordArg("dim") for `dim` checks they all match
        super().__init__(
            torch.split, arg, sizes, _users=MULTIPLE, dim=KeywordArg("dim")
        )

    def _match(self, node: torch.fx.Node, ctx: MatchContext):
        m = super()._match(node, ctx)
        if not m:
            return m
        split_sections = node.args[1]
        if not isinstance(split_sections, (list, tuple)):
            return FailedMatch("split not normalized")
        # check users are all unique getitems
        seen_idxs = set()
        for user in node.users:
            if not CallFunction(operator.getitem, Arg(), Arg()).match(user):
                # This should ideally never happen. Split user should always be a getitem
                return FailedMatch(f"user of split not a getitem: {user}")
            if not isinstance(user.args[1], int):
                return FailedMatch("only integer getitems are handled")
            if user.args[1] in seen_idxs:
                return FailedMatch(f"duplicate getitem {user.args[1]}")
            if user.args[-1] < 0:
                # This shouldn't ideally happen as dynamo normalizes indexes to positive
                return FailedMatch("negative index")
            seen_idxs.add(user.args[1])
        return m


@register_graph_pattern(
    TorchSplit(
        CallFunction(
            operator.getitem,
            TorchSplit(
                KeywordArg("first_split_input"),
                KeywordArg("first_split_sections"),
            ),
            Ignored(),
        ),
        KeywordArg("next_split_sections"),
    ),
    pass_dict=merge_splits_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def merge_splits(
    match: Match,
    first_split_input: torch.fx.Node,
    first_split_sections: List[int],
    next_split_sections: List[int],
    # Note: dim is implicitly passed by TorchSplit, as it internally uses a pattern with dim
    dim: int,
):
    node = match.output_node()
    graph = match.graph
    first_split = node.args[0].args[0]
    next_split_index = node.args[0].args[1]

    new_split_sections = list(first_split_sections)
    new_split_sections[next_split_index : next_split_index + 1] = next_split_sections

    first_split_dim = first_split.kwargs["dim"]

    to_remove = []

    with graph.inserting_before(first_split):
        # Add the new split node
        new_split = graph.call_function(
            torch.split,
            args=(first_split_input, new_split_sections),
            kwargs={"dim": first_split_dim},
        )
        first_split_num_to_user = {
            user.args[1]: user for user in first_split.users.keys()
        }

        new_split_num = 0
        for split_num in range(len(first_split_sections)):
            if split_num not in first_split_num_to_user:
                new_split_num += 1
                continue
            old_getitem = first_split_num_to_user[split_num]
            if split_num != next_split_index:
                old_getitem.update_arg(0, new_split)
                old_getitem.update_arg(1, new_split_num)
                new_split_num += 1
            else:
                next_split_num_to_user = {
                    user.args[1]: user for user in node.users.keys()
                }
                for next_split_num in range(len(next_split_sections)):
                    with graph.inserting_after(new_split):
                        new_getitem = graph.call_function(
                            operator.getitem, args=(new_split, new_split_num)
                        )
                    new_split_num += 1
                    next_getitem = next_split_num_to_user[next_split_num]
                    new_getitem.meta.update(next_getitem.meta)
                    next_getitem.replace_all_uses_with(new_getitem)
                    to_remove.append(next_getitem)
                to_remove.append(node)
                to_remove.append(old_getitem)

        to_remove.append(first_split)
    for node in to_remove:
        graph.erase_node(node)

    counters["inductor"]["consecutive_split_merged"] += 1


@functools.lru_cache(None)
def _split_cat_init():
    from .pre_grad import normalize_split_pass

    # Pass 1: Normalize split cats
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
        pattern.register(normalize_split_pass)
