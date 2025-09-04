# mypy: allow-untyped-defs
import re
from typing import Callable, Optional, Union

import torch.fx
from torch.fx.node import map_arg
from torch.fx.passes.split_module import split_module


__all__ = [
    "FoldedGraphModule",
    "get_unique_attr_name_in_module",
    "split_const_subgraphs",
]


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
        fx_const_folded_attrs_name: Optional[str] = None,
        device_for_folded_attrs: str = "cuda",
    ):
        super().__init__(root, graph)
        self.const_subgraph_module = (
            None
            if const_subgraph is None
            else torch.fx.GraphModule(root, const_subgraph)
        )
        self.has_folding_been_run = False
        self.fx_const_folded_attrs_name = fx_const_folded_attrs_name
        self.device_for_folded_attrs = device_for_folded_attrs

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

        def _create_param(i):
            return torch.nn.Parameter(
                i.detach().clone()
                if not isinstance(i, int)
                else torch.Tensor([i]).to(device=self.device_for_folded_attrs),
                requires_grad=i.requires_grad if isinstance(i, torch.Tensor) else False,
            )

        params = (
            torch.nn.ParameterList([_create_param(i) for i in folded_attrs])
            if isinstance(folded_attrs, tuple)
            else _create_param(folded_attrs)
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
    call_mod_kwargs = call_mod_node_to_replace.kwargs

    replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}
    ph_count = 0

    def replacement_fn(node):
        new_node = replacement_mapping[node]
        new_node.meta = node.meta.copy()
        return new_node

    for inline_node in inline_mod.graph.nodes:
        if inline_node.op == "placeholder":
            replacement_mapping[inline_node] = (
                call_mod_kwargs[inline_node.name]
                if inline_node.name in call_mod_kwargs
                else call_mod_args[ph_count]
            )

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


def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    """
    Make sure the name is unique (in a module) and can represents an attr.
    """
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

    return name


def split_const_subgraphs(
    module: Union[torch.nn.Module, torch.fx.GraphModule],
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
    device_for_folded_attrs: str = "cpu",
) -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    """

    import sympy

    if not isinstance(module, torch.fx.GraphModule):
        mod_traced = torch.fx.symbolic_trace(module)
    else:
        mod_traced = module

    # Build up a list of const_nodes, defined as nodes that are themselves
    # get_attrs, or have all get_attr or other constant node inputs.
    const_nodes: set[torch.fx.Node] = set()
    found_const_folding = False
    for node in mod_traced.graph.nodes:
        # Skip over placeholders/outputs because they can't be const folded and
        # we don't want to add tags to them.
        if node.op in {"placeholder", "output"}:
            continue

        # If the node itself is constant, or all of its inputs are constant,
        # then tag it as constant.
        if node.op != "get_attr" and not set(node.all_input_nodes).issubset(
            const_nodes
        ):
            continue

        # If provided skip folding function says to skip, then skip.
        if skip_folding_node_fn and skip_folding_node_fn(node):
            continue

        # Skip folding side-effectful functions
        if node.is_impure():
            continue

        # Skip folding nodes that have symbolic fill_value
        if isinstance(node.kwargs.get("fill_value", None), sympy.Expr):
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

    const_mod_name, non_const_mod_name = "submod_0", "submod_1"
    # Safely get submod_1 in case there are no non-const nodes
    const_gm, non_const_gm = split.submod_0, getattr(split, non_const_mod_name, None)

    # The module that a call_module node refers to gets copied to submodules during split.
    # The path to the module also gets inlined, i.e. mod.a.b -> mod_a_b. Here we need to
    # attach inlined modules to `split` as it's the owning module now.
    for node in non_const_gm.graph.nodes if non_const_gm else []:
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
    #    %inp : [num_users=1] = placeholder[target=const_inp]
    #    %add : [num_users=1] = call_function[target=operator.add](args = (%inp, %inp), kwargs = {})
    #    return add
    # We replace that with the following, which does not have any placeholders:
    # graph():
    #    %inp_1 : [num_users=1] = get_attr[target=const_inp]
    #    %add : [num_users=1] = call_function[target=operator.add](args = (%inp_1, %inp_1), kwargs = {})
    #    return add
    root_const_gm = torch.fx.GraphModule(split, const_gm.graph)

    # The order of placeholders in the const_gm graph should match the order of
    # args in the outer module, so we can simply use an index for the
    # placeholder mapping
    ph_idx = 0
    for node in root_const_gm.graph.nodes:
        if node.op == "output":
            multiple_outputs = isinstance(node.args[0], tuple)
            continue
        if node.op != "placeholder":
            continue
        assert ph_idx < len(call_const_gm_args)
        in_node = call_const_gm_args[ph_idx]
        ph_idx += 1
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
    fx_const_folded_attrs_name = get_unique_attr_name_in_module(
        mod_traced, "_FX_CONST_FOLDED_ATTRS"
    )
    setattr(
        split,
        fx_const_folded_attrs_name,
        torch.nn.ParameterList() if multiple_outputs else torch.nn.Parameter(),  # type: ignore[possibly-undefined]
    )
    for node in split.graph.nodes:
        if node.op == "call_module" and node.target == const_mod_name:
            with node.graph.inserting_before(node):
                folded_attrs = node.graph.get_attr(fx_const_folded_attrs_name)
            folded_attrs.meta = node.meta.copy()
            node.replace_all_uses_with(folded_attrs)
            break

    # Finally, inline the non-constant submod (if it exists) into the split submod.
    # This is so that the original caller who may have passed in a graph module will
    # get back out a graph module whose graph is traced to the same granularity.
    if hasattr(split, non_const_mod_name):
        _inline_module(split, non_const_mod_name)

    split.graph.eliminate_dead_code()

    return FoldedGraphModule(
        split,
        split.graph,
        root_const_gm.graph,
        fx_const_folded_attrs_name,
        device_for_folded_attrs,
    )
