import logging
import operator
from typing import Callable, List, Tuple, Union

import numpy

import torch
from torch._dynamo.utils import counters

from ..pattern_matcher import (
    Arg,
    CallFunction,
    CallFunctionVarArgs,
    CallMethodVarArgs,
    config_flag,
    FailedMatch,
    get_arg_value,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MatchContext,
    MULTIPLE,
    PatternExpr,
    register_graph_pattern,
    RepeatedExpr,
)
from .pre_grad import (
    merge_splits_pass,
    normalize_split_pass,
    split_cat_pass,
    unbind_stack_pass,
)

log = logging.getLogger(__name__)


def _get_split_args_default(split_node):
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


def normalize_split_base(match: Match, _get_split_args: Callable):
    """
    Normalize split with split_size into split_with_sizes, so that we only deal with one type of split in
    subsequent optimizations
    """
    split_node = match.nodes[0]
    graph = match.graph
    split_input, split_size, split_dim = _get_split_args(split_node)
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


@register_graph_pattern(
    CallFunctionVarArgs(torch.split, users=MULTIPLE),
    pass_dict=normalize_split_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallMethodVarArgs("split", users=MULTIPLE),
    pass_dict=normalize_split_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def normalize_split_default(match: Match, *args, **kwargs):
    return normalize_split_base(match, _get_split_args_default)


def find_next_users(split_node):
    next_users = []
    for getitem_node in split_node.users.keys():
        for getitem_user in getitem_node.users.keys():
            if getitem_user not in next_users:
                next_users.append(getitem_user)
    return next_users


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


class SplitCatSimplifier:
    """
    Helper class to simplify split-cat pattern. In simple cases, both split and cat node can be removed in a "split->cat"
    pattern. However, there are various cases where they can't and we need to simplify split/ add transforms before cat.
    Some such cases are:
        1. Final node has additional args (not coming from the initial split)
        2. Shuffling of args between split/cat
        3. Some final nodes are non-(cat/stack)
        4. Split-dim != cat-dim (but equal split)

    Note that any combination of the above cases can happen.

    To deal with 1, 2, & 3 - we iterate over all users of split. And figure out common "ranges" that can be merged.
    Then, we simplify the split accordingly. In the best case, split can be entirely removed.

    To deal with 4, we add some transformations (unflatten + movedim) (See `get_transform_params`).

    Finally, depending on final node being cat or stack, unsqueeze/flatten needs to be added.

    """

    def simplify(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: List[int],
    ):
        # Find the next users (i.e. users after the getitem)
        next_users = find_next_users(split_node)
        # Gather inputs of the next users. When inputs come from `split_node`, they are instead represented by
        # a tuple indicating the split ranges. See `get_user_input_list` for more details
        user_inputs_list = self.get_user_input_list(split_node, next_users)
        # Simplify the split_sections based on user_inputs_list. In simpler cases, len(simplified_split_ranges) == 1 and
        # we can simply replace the split node. Otherwise, we simplify it.
        simplified_split_ranges = self.get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        if not simplified_split_ranges:  # Simplification not possible
            return
        transform_params_list = self.get_transform_params(
            split_node, next_users, user_inputs_list
        )
        if not transform_params_list:
            return

        # Start actual replacement
        user_inputs_list_new = self.replace_split(
            graph, split_node, split_sections, user_inputs_list, simplified_split_ranges
        )
        self.replace_cat(
            graph, split_node, next_users, user_inputs_list_new, transform_params_list
        )
        self.erase_old_nodes(graph, split_node, next_users)

    def get_user_input_list(
        self, split_node, next_users
    ) -> List[List[Union[torch.fx.Node, Tuple[int, int]]]]:
        """
        Returns list of inputs to the following user nodes, in order. The outer list represents the user node. The inner
        list represents the inputs to that particular node. This list can either contain
          - a tuple representing the ranges of get_items that should go into the cat (closed interval)
          - torch.fx.Node representing "other" inputs (which are not coming from our split)
        """
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]] = []
        for user in next_users:
            if user.target in {torch.cat, torch.stack}:
                user_inputs_list.append(self.get_merged_user_inputs(split_node, user))
            else:
                user_inputs_list.append(self.get_non_cat_node_input(split_node, user))
        return user_inputs_list

    def get_merged_user_inputs(
        self, split_node: torch.fx.Node, cat_node: torch.fx.Node
    ) -> List[Union[torch.fx.Node, Tuple[int, int]]]:
        user_inputs = get_arg_value(cat_node, 0, "tensors")
        simplified_user_inputs = []
        split_users = set(split_node.users.keys())
        for user_input in user_inputs:
            if user_input not in split_users:
                simplified_user_inputs.append(user_input)
            else:
                # Add which "getitem" cat depends on
                simplified_user_inputs.append(user_input.args[1])
        return self.merge_consecutive_inputs(simplified_user_inputs)

    def get_non_cat_node_input(
        self, split_node: torch.fx.Node, node: torch.fx.Node
    ) -> List[Tuple[int, int]]:
        """
        Get input for a non cat node in the same format as `get_merged_user_inputs`
        """
        node_input = []
        split_users = set(split_node.users.keys())
        for node_arg in (*node.args, *node.kwargs.values()):
            if node_arg in split_users:
                getitem_num = get_arg_value(node_arg, 1)
                node_input.append((getitem_num, getitem_num))
        return node_input

    def merge_consecutive_inputs(
        self, inputs: List[Union[torch.fx.Node, int]]
    ) -> List[Union[torch.fx.Node, Tuple[int, int]]]:
        """
        Merge consecutive inputs going into a user node.

        For e.g.
        [arg0, 0, 1, 2, arg1] -> [arg0, (0, 2), arg1]
        """
        merged_ranges = []
        cur_range = None
        for input_ in inputs:
            if isinstance(input_, int):
                if not cur_range:
                    cur_range = [input_, input_]
                elif input_ == cur_range[1] + 1:
                    cur_range[1] += 1
                else:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = [input_, input_]
            else:
                if cur_range:
                    merged_ranges.append(tuple(cur_range))
                    cur_range = None
                merged_ranges.append(input_)
        if cur_range:
            merged_ranges.append(tuple(cur_range))
        return merged_ranges

    def get_simplified_split_ranges(
        self,
        split_sections,
        next_users,
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]],
    ) -> List[Tuple[int, int]]:
        ranges = set()
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            ranges |= {
                user_input
                for user_input in user_inputs
                if isinstance(user_input, tuple)
            }

        cumulative_sizes = [0] + list(numpy.cumsum(split_sections))
        split_ranges = sorted(
            [(cumulative_sizes[r[0]], cumulative_sizes[r[1] + 1]) for r in ranges]
        )

        if not self.has_non_overlapping_ranges(
            split_ranges,
        ):  # This need not be a strict condition
            # However, we keep it now for simplicity.
            return
        split_ranges = self.fill_gaps(split_ranges, 0, cumulative_sizes[-1])
        if len(split_sections) == len(split_ranges):  # Simplification not possible
            return
        counters["inductor"]["scmerge_split_sections_removed"] = len(
            split_sections
        ) - len(split_ranges)
        return split_ranges

    def has_non_overlapping_ranges(self, ranges: List[Tuple[int, int]]):
        for range_, next_range in zip(ranges, ranges[1:]):
            if range_[1] > next_range[0]:
                return False
        return True

    def fill_gaps(self, ranges, min_, max_):
        cur = min_
        filled_ranges = []
        for a, b in ranges:
            if cur < a:
                filled_ranges.append((cur, a))
            filled_ranges.append((a, b))
            cur = b
        if filled_ranges[-1][1] < max_:
            filled_ranges.append((filled_ranges[-1][1], max_))
        return filled_ranges

    def get_transform_params(
        self,
        split_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]],
    ) -> List[List[Tuple]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        We replace a split node with an unflatten followed by a movedim
        """
        split_dim = split_node.kwargs["dim"]
        split_sections = split_node.args[1]
        transform_params_list = []
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            if user_node.target not in {torch.cat, torch.stack}:
                transform_params_list.append(None)
                continue

            cat_dim = get_arg_value(user_node, 1, "dim")
            transform_params = []
            for user_input in user_inputs:
                if split_dim == cat_dim and user_node.target == torch.cat:
                    # No transform needed
                    transform_params.append((None, None, None, None))
                elif isinstance(user_input, tuple):  # Split being simplified
                    # Verify equal split
                    subset_split_sections = split_sections[
                        user_input[0] : user_input[1] + 1
                    ]
                    # All sections should be equal
                    if len(set(subset_split_sections)) != 1:
                        return

                    num_splits = len(subset_split_sections)
                    unflatten_params = (split_dim, (num_splits, -1))
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    transform_params.append(
                        (unflatten_params, movedim_params, None, None)
                    )
                elif (
                    user_node.target == torch.stack or split_dim != cat_dim
                ):  # We need to unsqueeze inputs not coming through split
                    transform_params.append((None, None, (cat_dim,), None))
                else:  # Non-split inputs
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list

    def replace_split(
        self,
        graph: torch.fx.Graph,
        split_node: torch.fx.Node,
        split_sections: List[int],
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]],
        split_ranges: List[Tuple[int, int]],
    ) -> List[List[torch.fx.Node]]:
        """
        Replace the split node. It can either remove the split node if len(split_ranges) == 1, or simplify it
        into a split with lesser sections if len(split_ranges) > 1.

        Returns the new `user_inputs_list`, with tuples replaced with new getitems from the newer split node.
        """
        split_input = split_node.args[0]
        split_dim = split_node.kwargs["dim"]
        if len(split_ranges) == 1:  # We can completely eliminate the split node
            split_items = [split_input]
        else:
            with graph.inserting_after(split_node):
                new_split = graph.call_function(
                    torch.split,
                    args=(
                        split_input,
                        [r[1] - r[0] for r in split_ranges],
                        split_dim,
                    ),
                )
                new_split.meta.update(split_node.meta)
                counters["inductor"]["scmerge_split_added"] += 1
            with graph.inserting_after(new_split):
                split_items = [
                    graph.call_function(operator.getitem, args=(new_split, i))
                    for i in range(len(split_ranges))
                ]
        # Now assign the right getitem to the right input
        cumulative_sizes = [0] + list(numpy.cumsum(split_sections))
        new_user_inputs_list = []
        for user_inputs in user_inputs_list:
            new_user_inputs = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # Find the correct new getitem (present in split_items)
                    new_user_inputs.append(
                        split_items[
                            split_ranges.index(
                                (
                                    cumulative_sizes[user_input[0]],
                                    cumulative_sizes[user_input[1] + 1],
                                )
                            )
                        ]
                    )
                else:
                    new_user_inputs.append(user_input)
            new_user_inputs_list.append(new_user_inputs)
        return new_user_inputs_list

    def replace_cat(
        self,
        graph,
        split_node,
        next_users,
        user_inputs_list_new,
        transform_params_list,
    ):
        split_dim = split_node.kwargs["dim"]

        split_users = split_node.users.keys()
        new_cats = []
        for user_node, user_inputs_new, transform_params in zip(
            next_users, user_inputs_list_new, transform_params_list
        ):
            if user_node.target not in {torch.cat, torch.stack}:
                # Change the args and kwargs of non-cat/stack nodes. Replace old getitems (belonging to
                # the original split node) with the newer getitems
                next_cat_input = 0
                new_args = []
                for arg_num, arg in enumerate(user_node.args):
                    if arg in split_users:
                        new_args.append(user_inputs_new[next_cat_input])
                        next_cat_input += 1
                    else:
                        new_args.append(arg)
                user_node.args = tuple(new_args)
                for key, arg in user_node.kwargs.items():
                    if arg in split_users:
                        user_node.kwargs[key] = user_inputs_new[next_cat_input]
                        next_cat_input += 1
                continue

            # Handle cat/stack user nodes
            cat_dim = get_arg_value(user_node, 1, "dim")
            user_inputs_new_transformed = []
            # For `unsqueeze` transform, we will combine consecutive inputs with the same unsqueeze params, and stack them
            to_stack = []
            stack_dim = None
            with graph.inserting_before(user_node):
                for user_input_new, transform_param in zip(
                    user_inputs_new, transform_params
                ):
                    # Apply transforms
                    (
                        unflatten_params,
                        movedim_params,
                        unsqueeze_params,
                        flatten_params,
                    ) = transform_param
                    if unsqueeze_params and (
                        stack_dim is None or stack_dim == unsqueeze_params[0]
                    ):
                        to_stack.append(user_input_new)
                        stack_dim = unsqueeze_params[0]
                        continue
                    elif to_stack:
                        stacked_input = graph.call_function(
                            torch.stack, args=(to_stack, stack_dim)
                        )
                        to_stack = []
                        stack_dim = None
                        user_inputs_new_transformed.append(stacked_input)
                        if unsqueeze_params:
                            to_stack.append(user_input_new)
                            stack_dim = unsqueeze_params[0]
                            continue

                    if unflatten_params:
                        user_input_new = graph.call_function(
                            torch.unflatten, args=(user_input_new, *unflatten_params)
                        )
                    if movedim_params:
                        user_input_new = graph.call_function(
                            torch.movedim, args=(user_input_new, *movedim_params)
                        )
                    if flatten_params:
                        user_input_new = graph.call_function(
                            torch.flatten, args=(user_input_new, *flatten_params)
                        )
                    user_inputs_new_transformed.append(user_input_new)
                if to_stack:
                    stacked_input = graph.call_function(
                        torch.stack, args=(to_stack, stack_dim)
                    )
                    user_inputs_new_transformed.append(stacked_input)

            with graph.inserting_after(user_node):
                if len(user_inputs_new_transformed) > 1:
                    new_cat_node = graph.call_function(
                        torch.cat, args=(user_inputs_new_transformed, cat_dim)
                    )
                    new_cat_node.meta.update(user_node.meta)
                    counters["inductor"]["scmerge_cat_added"] += 1
                else:
                    new_cat_node = user_inputs_new_transformed[-1]

            if (
                user_node.target == torch.cat
                and split_dim != cat_dim
                and split_node.target == torch.split
            ):
                with graph.inserting_after(new_cat_node):
                    new_cat_node = graph.call_function(
                        torch.flatten, args=(new_cat_node, cat_dim, cat_dim + 1)
                    )
            user_node.replace_all_uses_with(new_cat_node)
            new_cats.append(new_cat_node)

    def erase_old_nodes(self, graph, split_node, next_users):
        to_remove = [split_node]
        counters["inductor"]["scmerge_split_removed"] += 1
        for getitem_node in split_node.users.keys():
            to_remove.append(getitem_node)
        for next_user in next_users:
            if next_user.target not in {torch.cat, torch.stack}:
                continue
            counters["inductor"]["scmerge_cat_removed"] += 1
            to_remove.append(next_user)
        for node in reversed(to_remove):
            graph.erase_node(node)


class UnbindCatRemover(SplitCatSimplifier):
    """
    Helper class to merge Unbind->Cat/Stack. Many of the cases are similar to SplitCatSimplifier.

    Unbind can't be simplified like splits. So, we can only remove the unbind node. Other than this,
    other cases like multiple users, additional args, dim mismatch are similar to `SplitCatSimplifier`,
    hence we extend that class.
    """

    def remove_unbind(
        self,
        graph: torch.fx.Graph,
        unbind_node: torch.fx.Node,
    ):
        num_unbind = (
            max(getitem_node.args[1] for getitem_node in unbind_node.users.keys()) + 1
        )
        split_sections = [1 for _ in range(num_unbind)]

        super().simplify(graph, unbind_node, split_sections)

    def get_simplified_split_ranges(
        self,
        split_sections,
        next_users,
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]],
    ) -> List[Tuple[int, int]]:
        simplified_split_ranges = super().get_simplified_split_ranges(
            split_sections, next_users, user_inputs_list
        )
        if not simplified_split_ranges or len(simplified_split_ranges) != 1:
            return None
        return simplified_split_ranges

    def get_transform_params(
        self,
        unbind_node: torch.fx.Node,
        next_users: List[torch.fx.Node],
        user_inputs_list: List[List[Union[torch.fx.Node, Tuple[int, int]]]],
    ) -> List[List[Tuple]]:
        """
        Figure out what transforms are needed for each input to each cat node.

        Here is the rough transforms we apply:

        x -> unbind -> stack => x -> movedim

        x -> unbind -> cat => x -> movedim -> flatten

        When cat/stack nodes have additional args:

             addn ---|              addn -> unsqueeze ---|
        x -> unbind -> stack  =>           x -> movedim  -> cat

             addn ---|                            addn ---|
        x -> unbind -> cat  =>   x -> movedim -> flatten  -> cat

        (Note application of these depends on the dims as well)


        """
        split_dim = unbind_node.kwargs["dim"]
        transform_params_list = []
        for user_node, user_inputs in zip(next_users, user_inputs_list):
            cat_dim = get_arg_value(user_node, 1, "dim")
            transform_params = []
            for user_input in user_inputs:
                if isinstance(user_input, tuple):
                    # User input is coming from unbind
                    movedim_params = (
                        (split_dim, cat_dim) if split_dim != cat_dim else None
                    )
                    flatten_params = None
                    if user_node.target == torch.cat:
                        flatten_params = (cat_dim, cat_dim + 1)
                    transform_params.append(
                        (None, movedim_params, None, flatten_params)
                    )
                elif (
                    user_node.target == torch.stack
                ):  # We need to unsqueeze inputs not coming through unbind into cat
                    transform_params.append((None, None, (cat_dim,), None))
                else:  # Non-unbind inputs
                    transform_params.append((None, None, None, None))
            transform_params_list.append(transform_params)
        return transform_params_list


class GetItem(CallFunction):
    def __init__(self, arg, index, _users=1):
        super().__init__(operator.getitem, arg, index, _users=_users)

    def find_anchor_nodes(self, ctx: MatchContext, searched):
        # We generally match GetItem with arg being an Arg(). So, we never return the anchor
        # nodes as the stored node in ctx.pattern_to_node is returned. Here we override find_anchor_nodes
        # to not use ctx.pattern_to_node
        for pattern in self.flat_args_kwargs[0]:
            if isinstance(pattern, PatternExpr):
                for other_node in pattern.find_anchor_nodes(ctx, searched):
                    if not isinstance(other_node, torch.fx.Node):
                        continue
                    for node in other_node.users:
                        if node not in searched:
                            if self._match_fns(node):
                                yield node
                                searched.add(node)


@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(
                    KeywordArg("split_input"),
                    KeywordArg("split_sizes"),
                ),
                Ignored(),
            ),
            KeywordArg("dim"),
            _users=MULTIPLE,
        ),
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    RepeatedExpr(
        CallFunction(
            torch.squeeze,
            GetItem(
                TorchSplit(
                    KeywordArg("split_input"),
                    KeywordArg("split_sizes"),
                ),
                Ignored(),
            ),
            dim=KeywordArg("dim"),
            _users=MULTIPLE,
        )
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def merge_split_squeeze(
    match: Match, split_input: torch.fx.Node, split_sizes: List[int], dim: int
):
    graph = match.graph
    split = next(node for node in match.nodes if node.target == torch.split)
    if not all(s == 1 for s in split_sizes):
        return
    next_users = find_next_users(split)
    if not all(node.target == torch.squeeze for node in next_users):
        return
    with graph.inserting_before(match.output_node()):
        unbind = graph.call_function(
            torch.unbind, args=(split_input,), kwargs={"dim": dim}
        )
        for item_index, getitem_node in sorted(
            [
                (getitem_node.args[1], getitem_node)
                for getitem_node in split.users.keys()
            ]
        ):
            squeeze = next(iter(getitem_node.users.keys()))
            new_get_item = graph.call_function(
                operator.getitem, args=(unbind, item_index)
            )
            squeeze.replace_all_uses_with(new_get_item)
            new_get_item.meta.update(squeeze.meta)
            graph.erase_node(squeeze)
            graph.erase_node(getitem_node)
    graph.erase_node(split)
    counters["inductor"]["split_squeeze_replaced"] += 1


getitem_unbind = ListOf(
    GetItem(
        CallFunction(
            torch.unbind,
            KeywordArg("unbind_input"),
            dim=KeywordArg("dim"),
            _users=MULTIPLE,
        ),
        Ignored(),
        _users=MULTIPLE,
    ),
    partial=True,
)


@register_graph_pattern(
    CallFunction([torch.stack, torch.cat], getitem_unbind, Ignored(), _users=MULTIPLE),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat], tensors=getitem_unbind, dim=Ignored(), _users=MULTIPLE
    ),
    pass_dict=unbind_stack_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def merge_unbind_stack(match: Match, unbind_input: torch.fx.Node, dim: int):
    unbind_node = next(node for node in match.nodes if node.target == torch.unbind)
    UnbindCatRemover().remove_unbind(match.graph, unbind_node)


getitem_split = ListOf(
    CallFunction(
        operator.getitem,
        TorchSplit(
            Ignored(),
            KeywordArg("split_sections"),
        ),
        Ignored(),
        _users=MULTIPLE,
    ),
    partial=True,
)


@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        tensors=getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        dim=Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
@register_graph_pattern(
    CallFunction(
        [torch.stack, torch.cat],
        getitem_split,
        Ignored(),
        _users=MULTIPLE,
    ),
    pass_dict=split_cat_pass,
    extra_check=config_flag("split_cat_fx_passes"),
)
def simplify_split_cat(match: Match, split_sections: List[int], dim: int):
    if not isinstance(split_sections, (list, tuple)):  # Unnormalized split
        return
    split_node = next(node for node in match.nodes if node.target == torch.split)
    SplitCatSimplifier().simplify(match.graph, split_node, split_sections)
