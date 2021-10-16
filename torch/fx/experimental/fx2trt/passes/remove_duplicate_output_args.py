#!/usr/bin/env python3

import operator
import typing as t
import logging
import torch.fx as fx
import dataclasses as dc


_LOGGER = logging.getLogger(__name__)


RemoveDuplicateOutputArgsFunc = t.Callable[
    [
        fx.GraphModule,
        t.Collection[str],
    ],
    t.Mapping[str, "RemoveDuplicateResult"]
]


def remove_duplicate_output_args(
    top_level: fx.GraphModule,
    target_subnets: t.Collection[str]
) -> t.Mapping[str, "RemoveDuplicateResult"]:
    """Removes duplicate output args.

    This pass removes duplicate output args from the target subnets and fixes
    their uses in the top level module where the subnets are called. This pass
    must be called after acc split on the top-level net and subsequent calls to
    the acc trace on the subnets.

    This pass will change both the subnets and top level module.

    Returns:
        a mapping of the target subnet name to its dedupcate result
    """

    processed_subnets = {}
    for node in top_level.graph.nodes:  # type: fx.Node
        if node.op == "call_module" and node.name in target_subnets:
            assert isinstance(node.target, str)
            sub_gm = top_level.get_submodule(node.target)
            assert isinstance(sub_gm, fx.GraphModule)

            replace_res = _remove_duplicate_output_args(sub_gm)
            processed_subnets[node.name] = replace_res
            if replace_res.replacement_map is None:
                continue
            sub_gm.recompile()

            needs_recompile = False
            # iterate on the copy since we will be changing elements of node.users
            for user in list(node.users):
                idx = _ensure_proper_output_use(user, node)
                idx_new = replace_res.replacement_map[idx]
                if idx_new != idx:
                    user.args = (user.args[0], idx_new)
                    needs_recompile = True

            if needs_recompile:
                top_level.recompile()
    return processed_subnets


@dc.dataclass(frozen=True)
class RemoveDuplicateResult:
    replacement_map: t.Optional[t.List[int]]
    module: fx.GraphModule


def _ensure_proper_output_use(user: fx.Node, target_node: fx.Node) -> int:
    """
    Ensures the node looks in proper form of calling the output of an fx2trt
    splitter sub-net. Specifically:

    1. op is call function, target: operator.getitem
    2. args is a 2-element tuple
    3. args[0] is the name of the subnet's output
    4. args[1] is the index into the subnet output tuple

    E.g.:

        %getitem_4 : [#users=1] = call_function[target=operator.getitem](args = (%_run_on_acc_1, 4), kwargs = {})

    returns the index into the subnet output tuple
    """
    _LOGGER.info(f"Checking user node: {user.format_node()}")
    assert (
        user.op == "call_function"
        and user.target == operator.getitem
        and len(user.args) == 2
        and isinstance(user.args[0], fx.Node)
        and user.args[0].name == target_node.name
        and isinstance(user.args[1], int)
    ), f"Node is not a proper user of splitter output: {user.format_node()}"

    return user.args[1]


def _remove_duplicate_output_args(gm: fx.GraphModule) -> RemoveDuplicateResult:
    output_nodes = [n for n in gm.graph.nodes if n.op == "output"]
    assert len(output_nodes) == 1, \
           f"Expecting exactly one `output` node, but got {len(output_nodes)}"

    changed = False
    # arg node name to its index in the new output args tuple
    name_to_idx: t.Dict[str, int] = {}
    output_node = output_nodes[0]

    # Output op only uses its `args[0]`, and it does not have `kwargs`.
    # https://pytorch.org/docs/stable/fx.html#torch.fx.Node
    args: t.Sequence[t.Any] = output_node.args[0]

    # Only concern outselves to the case where the args is an iterable of fx.Node.
    # Other return cases (e.g., a single value) is possible and we don't handle
    # that in this pass.
    if not (isinstance(args, t.Iterable) and all(isinstance(a, fx.Node) for a in args)):
        return RemoveDuplicateResult(replacement_map=None, module=gm)

    # Map old index of the arg node to the remaining node's idx,
    # initialized to `i => i`
    replacement_map: t.List[int] = list(range(len(args)))
    args_new = []
    for idx, a in enumerate(args):
        assert isinstance(a, fx.Node), \
               f"Expecting fx.Node instance, but got: {type(a)}"

        if a.name not in name_to_idx:
            args_new.append(a)
            name_to_idx[a.name] = len(args_new) - 1
        else:
            changed = True
            _LOGGER.warning(
                f"Replaced duplicate output arg '{a.name}': "
                f"{idx} -> {name_to_idx[a.name]}"
            )
        replacement_map[idx] = name_to_idx[a.name]

    output_node.args = (tuple(args_new),)
    if changed:
        gm.recompile()
    return RemoveDuplicateResult(replacement_map, module=gm)
