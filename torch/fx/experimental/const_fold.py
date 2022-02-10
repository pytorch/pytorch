from typing import Callable, Dict, Set, Optional, Union

import torch.fx
import torch.fx.experimental.fx_acc.acc_utils as acc_utils
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module


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
        fx_const_folded_attrs_name: str = None,
    ):
        # In init, we set graph's owning module to root which will make graph's
        # owning module be None because graph already have a owning module. We
        # need owning module to run DCE. To work around we set the number of
        # graph's owners to 0.
        graph._owners = 0
        super().__init__(root, graph)
        self.const_subgraph_module = (
            None
            if const_subgraph is None
            else torch.fx.GraphModule(root, const_subgraph)
        )
        self.has_folding_been_run = False
        self.fx_const_folded_attrs_name = fx_const_folded_attrs_name

    def __call__(self, *args, **kwargs):
        if not self.has_folding_been_run:
            self.run_folding()
        return super().__call__(*args)

    def run_folding(self):
        # If there's no const subgraph module or attr output names to use, return
        # early as there is no const folding to perform.
        if (
            self.const_subgraph_module is None
            or self.fx_const_folded_attrs_name is None
        ):
            return

        assert not self.has_folding_been_run
        self.has_folding_been_run = True

        # Actually run const folding subgraph. Note that single attr const fold
        # subgraphs output a single Tensor while multiple outputs are returned as
        # Tuple[Tensor,].
        folded_attrs = self.const_subgraph_module()
        params = (
            torch.nn.ParameterList([torch.nn.Parameter(i) for i in folded_attrs])
            if isinstance(folded_attrs, tuple)
            else torch.nn.Parameter(folded_attrs)
        )
        setattr(self, self.fx_const_folded_attrs_name, params)


def _inline_module(gm: torch.fx.GraphModule, inline_mod_name: str):
    """
    Given `gm` and some graph module which is called with target name `inline_mod_name`,
    this helper will inline all of the nodes from that called graph module into `gm`.
    """
    # Fetch the inner graph module that we want to inline inside `gm`.
    inline_mod = dict(gm.named_modules())[inline_mod_name]
    assert isinstance(inline_mod, torch.fx.GraphModule)
    call_mod_node_to_replace = None
    for node in gm.graph.nodes:
        if node.op == "call_module" and node.target == inline_mod_name:
            call_mod_node_to_replace = node
            break
    assert call_mod_node_to_replace is not None

    # Now actually do the swap. Note that we have to keep track of new nodes that are
    # copied into `gm` -- we do this via replacement_mapping.
    call_mod_args = call_mod_node_to_replace.args
    replacement_mapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0

    def replacement_fn(node):
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node

    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == "placeholder":
            replacement_mapping[inline_node] = call_mod_args[ph_count]
            ph_count += 1
            continue

        if inline_node.op == "output":
            outputs = inline_node.args[0]
            output_replacements = map_arg(outputs, replacement_fn)
            call_mod_node_to_replace.replace_all_uses_with(output_replacements)
            continue

        with gm.graph.inserting_before(call_mod_node_to_replace):
            new_node = gm.graph.node_copy(inline_node, replacement_fn)
        replacement_mapping[inline_node] = new_node

    gm.graph.eliminate_dead_code()


def split_const_subgraphs(
    module: Union[torch.nn.Module, torch.fx.GraphModule],
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None
) -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    """
    if not isinstance(module, torch.fx.GraphModule):
        mod_traced = torch.fx.symbolic_trace(module)
    else:
        mod_traced = module

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
        if node.op != "get_attr" and not set(node.all_input_nodes).issubset(const_nodes):
            continue

        # If provided skip folding function says to skip, then skip.
        if skip_folding_node_fn and skip_folding_node_fn(node):
            continue

        # Must be a constant foldable node at this point.
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

    const_gm, non_const_gm = split.submod_0, split.submod_1
    const_mod_name, non_const_mod_name = "submod_0", "submod_1"

    # The module that a call_module node refers to gets copied to submodules during split.
    # The path to the module also gets inlined, i.e. mod.a.b -> mod_a_b. Here we need to
    # attach inlined modules to `split` as it's the owning module now.
    for node in non_const_gm.graph.nodes:
        if node.op == "call_module":
            setattr(split, node.target, getattr(non_const_gm, node.target))
    for node in const_gm.graph.nodes:
        if node.op == "call_module":
            setattr(split, node.target, getattr(const_gm, node.target))

    # split_module currently does not use get_attrs for attrs. Instead it passes
    # them in as args from the parent module, which used get_attrs. Here we set
    # them as get_attrs inside const_gm, allowing for running folding without
    # somehow a priori knowing the attrs that should be passed as args. We can
    # unconditionally do this for all placeholders because we know all
    # placeholders to const_gm must be constants accessible via get_attr.
    call_const_gm_args = None
    for node in split.graph.nodes:
        if node.op == "call_module":
            if node.target == const_mod_name:
                call_const_gm_args = node.args
                break
    assert call_const_gm_args is not None

    # Here we do the actual replacement of placeholders to get_attrs. Note that here we
    # set the const_gm.graph into a new root_const_gm with split as the root module,
    # because we are fetching attributes directly from the root module, instead of
    # fetching them from const_gm. Example: The const_gm must have some format like:
    # graph():
    #    %inp : [#users=1] = placeholder[target=const_inp]
    #    %add : [#users=1] = call_function[target=operator.add](args = (%inp, %inp), kwargs = {})
    #    return add
    # We replace that with the following, which does not have any placeholders:
    # graph():
    #    %inp_1 : [#users=1] = get_attr[target=const_inp]
    #    %add : [#users=1] = call_function[target=operator.add](args = (%inp_1, %inp_1), kwargs = {})
    #    return add
    root_const_gm = torch.fx.GraphModule(split, const_gm.graph)
    for node in root_const_gm.graph.nodes:
        if node.op == "output":
            multiple_outputs = isinstance(node.args[0], tuple)
            continue
        if node.op != "placeholder":
            continue
        in_node = next(n for n in call_const_gm_args if n.name == node.target)
        assert in_node.op == "get_attr"
        with root_const_gm.graph.inserting_before(node):
            new_node = root_const_gm.graph.get_attr(in_node.target)
        new_node.meta = node.meta.copy()
        node.replace_all_uses_with(new_node)
        root_const_gm.graph.erase_node(node)
    assert "multiple_outputs" in locals()

    # Now find the call to const_gm inside split, and replace it with a getattr to the
    # folded tensor(s) that result from constant folding. Note that we don't need to
    # worry about whether this is one or more tensors because the original graph
    # correctly uses getitem to extract individual tensors if there are multiple folded.
    fx_const_folded_attrs_name = acc_utils.get_unique_attr_name_in_module(
        split, "_FX_CONST_FOLDED_ATTRS"
    )
    setattr(
        split,
        fx_const_folded_attrs_name,
        torch.nn.ParameterList() if multiple_outputs else torch.nn.Parameter(),
    )
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == const_mod_name:
            with node.graph.inserting_before(node):
                folded_attrs = node.graph.get_attr(fx_const_folded_attrs_name)
            folded_attrs.meta = node.meta.copy()
            node.replace_all_uses_with(folded_attrs)
            break

    split.graph.eliminate_dead_code()

    # Finally, inline the non-constant submod into the split submod. This is so that the
    # original caller who may have passed in a graph module will get back out a graph
    # module whose graph is traced to the same granularity.
    _inline_module(split, non_const_mod_name)

    return FoldedGraphModule(
        split, split.graph, root_const_gm.graph, fx_const_folded_attrs_name
    )
