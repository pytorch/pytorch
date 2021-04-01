import operator
from typing import Dict, Set, List, Optional

import torch.fx
from torch.fx.passes.split_module import split_module
import re


def _make_tuple(x):
    """
    Helper to convert x into a one item tuple if it's not a tuple already.
    """
    return x if isinstance(x, tuple) else (x,)


class FoldedGraphModule(torch.fx.GraphModule):
    """
    FoldedGraphModule is a GraphModule which also contains another
    `const_subgraph_module` representing a subgraph which has all const attr
    inputs and which can be run once before running the main standard
    `graph`. The `const_output_names` are the ordered list names of attrs which
    represent what each respective output from the const_subgraph should be set
    on which attrs.
    """

    def __init__(
        self,
        root: torch.nn.Module,
        graph: torch.fx.Graph,
        const_subgraph: Optional[torch.fx.Graph] = None,
        const_output_names: Optional[List[str]] = None,
    ):
        super().__init__(root, graph)
        self.const_subgraph_module = (
            None
            if const_subgraph is None
            else torch.fx.GraphModule(root, const_subgraph)
        )
        self.const_output_names = const_output_names
        self.has_folding_been_run = False

    def __call__(self, *args, **kwargs):
        if not self.has_folding_been_run:
            self.run_folding()
        return super().__call__(*args)

    def run_folding(self):
        # If there's no const subgraph module or attr output names to use, return
        # early as there is no const folding to perform.
        if self.const_subgraph_module is None or self.const_output_names is None:
            return

        assert not self.has_folding_been_run
        self.has_folding_been_run = True

        # Actually run const folding subgraph. We _make_tuple here because
        # single attr const fold subgraphs output a single Tensor while
        # multiple outputs are returned as Tuple[Tensor,].
        folded_attrs = _make_tuple(self.const_subgraph_module())

        # Look for output node from const folding subgraph and set attrs on the
        # module with the results.
        for i in range(len(folded_attrs)):
            setattr(
                self, self.const_output_names[i], torch.nn.Parameter(folded_attrs[i])
            )


def split_const_subgraphs(
    module: torch.nn.Module,
) -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    """
    mod_traced = torch.fx.symbolic_trace(module)

    # Build up a list of const_nodes, defined as nodes that are themselves
    # get_attrs, or have all get_attr or other constant node inputs.
    const_nodes: Set[torch.fx.Node] = set()
    found_const_folding = False
    for node in mod_traced.graph.nodes:
        # Skip over placeholders/outputs because they can't be const folded and
        # we don't want to add tags to them.
        if node.op in {"placeholder", "output"}:
            continue

        # If the node itself is constant, or all of its inputs are constant,
        # then tag it as constant.
        if node.op == "get_attr" or set(node.all_input_nodes).issubset(const_nodes):
            const_nodes.add(node)
            if node.op != "get_attr":
                found_const_folding = True

    # If we did not find any const folding then return early without a const fold subgraph.
    if not found_const_folding:
        return FoldedGraphModule(mod_traced, mod_traced.graph)

    # Partition the module into two: submod_0 for constant folding subgraph, and
    # submod_1 for the rest.
    def mod_partition(node: torch.fx.Node):
        return 0 if node in const_nodes else 1

    split = split_module(mod_traced, module, mod_partition)

    # Gather all names that are output from the const folding subgraph, which we
    # will need to set dummy params on the module.
    const_output_names: List[str] = []
    for node in split.submod_0.graph.nodes:
        if node.op == "output":
            # Note: we _make_tuple here because the output Node either contains
            # a single output Node, or Tuple[Node], so this simplifies things.
            const_output_names = [o.name for o in _make_tuple(node.args[0])]
            break

    # Make sure the attr name we want to use is uniquely named in the module.
    for i in range(len(const_output_names)):
        # Add a suffix to make it easier to tell these were the result of const folding.
        name = const_output_names[i] + "__CF"
        # Delete all characters that are illegal in a Python identifier.
        name = re.sub("[^0-9a-zA-Z_]+", "_", name)
        if name[0].isdigit():
            name = f"_{name}"
        # Now make sure it is in fact unique to the module by incrementing suffix value.
        while hasattr(mod_traced, name):
            match = re.match(r"(.*)_(\d+)$", name)
            if match is None:
                name = name + "_1"
            else:
                base, num = match.group(1, 2)
                name = f"{base}_{int(num) + 1}"
        const_output_names[i] = name

    # Now track the const_output_names to what name is used in the parent graph
    # from the split via call_function getitem, to see what order it is passed
    # into the non-const subgraph submod_1. First look to the parent module
    # containing/calling into the const/non-const submodules to determine what
    # the inputs are to each. Note if submod_0 had a single output then there is
    # no getitem, and we can simply use the output from the call to submoid_0.
    call_submod_0_args, call_submod_1_args = None, None
    orig_ph_targets: List[str] = []
    for node in split.graph.nodes:
        if node.op == "placeholder":
            orig_ph_targets.append(node.target)

        if node.op == "call_module":
            if node.target == "submod_0":
                call_submod_0_args = node.args
                continue
            elif node.target == "submod_1":
                call_submod_1_args = node.args
                continue
    assert call_submod_0_args is not None and call_submod_1_args is not None

    # Look through the args for the call into submod_1, and find the args that
    # come from submod_0. Also look for get_attrs fed directly from the parent
    # split into submod_1, i.e. those attrs that are not constant folded.
    submod_1_input_idx_to_folded_attr_name: Dict[int, str] = {}
    submod_1_input_idx_to_unfolded_attr_name: Dict[int, str] = {}
    for i, node in enumerate(call_submod_1_args):
        const_output_name = None
        # If we only had a single output from submod_0 then we simply look for
        # the call_module into it.
        if len(const_output_names) == 1:
            if node.op == "call_module" and node.target == "submod_0":
                const_output_name = const_output_names[0]

        # Else we had multiple outputs from submod_0, so we need to look for all
        # getitems from the call to it.
        else:
            if (
                node.op == "call_function"
                and node.target == operator.__getitem__
                and node.args[0].target == "submod_0"
            ):
                const_output_name = const_output_names[node.args[1]]

        # Now map from the index of the constant into calling submod_1 and map
        # to the constant output name, which we use for swapping in getattrs
        # instead of placeholders in submod_1.
        if const_output_name is not None:
            submod_1_input_idx_to_folded_attr_name[i] = const_output_name
        elif node.op == "get_attr":
            submod_1_input_idx_to_unfolded_attr_name[i] = node.target

    assert len(submod_1_input_idx_to_folded_attr_name) == len(const_output_names)

    # Now we have a mapping from const output names to the index they are passed
    # into submod_1, so swap in getattrs for placeholders.
    ph_idx = 0
    for node in split.submod_1.graph.nodes:
        if node.op != "placeholder":
            continue
        is_folded_attr = ph_idx in submod_1_input_idx_to_folded_attr_name.keys()
        is_unfolded_attr = ph_idx in submod_1_input_idx_to_unfolded_attr_name.keys()
        if not is_folded_attr and not is_unfolded_attr:
            ph_idx += 1
            continue

        const_output_name = (
            submod_1_input_idx_to_folded_attr_name[ph_idx]
            if is_folded_attr
            else submod_1_input_idx_to_unfolded_attr_name[ph_idx]
        )
        if is_folded_attr:
            assert not hasattr(mod_traced, const_output_name)
            # Use a dummy param, which will be overwritten when we run const folding.
            setattr(
                mod_traced,
                const_output_name,
                torch.nn.Parameter(torch.randn(1)),
            )
        with split.submod_1.graph.inserting_before(node):
            node.replace_all_uses_with(split.submod_1.graph.get_attr(const_output_name))
        split.submod_1.graph.erase_node(node)
        ph_idx += 1

    # We may need to reorder placeholders to ensure they have the same order as
    # they do in the original split.
    ph_idx = 0
    node = next(iter(split.submod_1.graph.nodes))
    while node.op != "root":
        if node.op != "placeholder":
            node = node.next
            continue

        curr_orig_ph_target = orig_ph_targets[ph_idx]
        ph_idx += 1
        # If this ph is in the correct position, nothing to do.
        if curr_orig_ph_target == node.target:
            node = node.next
            continue

        # This ph is not in the correct order, so search the rest of the graph
        # for the ph we expected and prepend it before the current ph.
        later_node = node.next
        while later_node.op != "root":
            if (
                later_node.op == "placeholder"
                and curr_orig_ph_target == later_node.target
            ):
                break
            later_node = later_node.next
        assert later_node.op != "root"
        node.prepend(later_node)
        # Note we do not increment node here, as it still may be in the wrong
        # place (we just prepended the ph that should have come before it).

    # split_module currently does not use get_attrs for attrs. Instead it passes
    # them in as args from the parent module, which used get_attrs. Here we set
    # them as get_attrs inside submod_0, allowing for running folding without
    # somehow a priori knowing the attrs that should be passed as args. We can
    # unconditionally do this for all placeholders because we know all
    # placeholders to submod_0 must be constants accessible via get_attr.
    for node in split.submod_0.graph.nodes:
        if node.op != "placeholder":
            continue
        in_node = next(n for n in call_submod_0_args if n.name == node.target)
        assert in_node.op == "get_attr"
        with split.submod_0.graph.inserting_before(node):
            node.replace_all_uses_with(split.submod_0.graph.get_attr(in_node.target))
        split.submod_0.graph.erase_node(node)

    return FoldedGraphModule(
        mod_traced, split.submod_1.graph, split.submod_0.graph, const_output_names
    )
