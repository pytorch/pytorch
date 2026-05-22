import inspect
import logging
from collections import defaultdict, OrderedDict
from collections.abc import Callable
from typing import Any

import torch
from torch.fx._compatibility import compatibility
from torch.fx._lazy_graph_module import _LazyGraphModule, _make_graph_module
from torch.fx._utils import lazy_format_graph_code
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node


__all__ = ["Partition", "split_module", "split_module_simple"]
log = _LOGGER = logging.getLogger(__name__)


@compatibility(is_backward_compatible=True)
class Partition:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.submod_name = f"submod_{name}"
        self.node_names: list[str] = []
        self.inputs: dict[str, None] = {}
        self.outputs: dict[str, None] = {}
        self.dependencies: dict[str, None] = {}
        self.dependents: dict[str, None] = {}
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.environment: dict[Node, Node] = {}
        self.targets: dict[str, Any] = {}

    def __repr__(self) -> str:
        return (
            f"name: {self.name},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {self.inputs},\n"
            f" outputs: {self.outputs},\n"
            f" partitions depended on: {self.dependencies},\n"
            f" partition dependents: {self.dependents}"
        )


def _get_attr_from_qualname(mod: torch.nn.Module, qualname: str) -> Any:
    attr_val = mod
    for atom in qualname.split("."):  # type: ignore[union-attr]
        if not hasattr(attr_val, atom):
            raise AttributeError(f"Node target {qualname} not found!")
        attr_val = getattr(attr_val, atom)
    return attr_val


# Creates subgraphs out of main graph
@compatibility(is_backward_compatible=True)
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[Node], int],
    qualname_map: dict[str, str] | None = None,
    keep_original_order: bool | None = False,
    keep_original_node_name: bool | None = False,
    keep_original_input_name: bool = True,
    *,
    partition_affix: str | None = None,
    tuple_return: bool = False,
) -> GraphModule:
    """
    Creates subgraphs out of main graph

    Args:
        m (GraphModule): Graph module to split
        root_m (torch.nn.Module): root nn module. Not currently used. Included
            because the root nn module is usually transformed via
            torch.fx._symbolic_trace.symbolic_trace (see example below)
        split_callback (Callable[[Node], int]): Callable function
            that maps a given Node instance to a numeric partition identifier.
            split_module will use this function as the policy for which operations
            appear in which partitions in the output Module.
        qualname_map: Optional[Dict[str, str]]: optional output parameter that returns a
            mapping from new target names in the module after split to old target
            names in the original module.
        keep_original_order: Optional[bool]: keep the original order of the GraphModule
            or use the Topological order of the new constructed GraphModule
        keep_original_node_name: Optional[bool]: If the partitioned graphs should
            have the same node names as the original graph.
        keep_original_input_name: bool: If the partitioned graphs should
            have the same input names as the original graph.
        partition_affix: Optional[str]: If specified, the submodules' names will contain
            the affix, e.g. "submod_<affix>_<idx>".
        tuple_return: bool: If True, submodule outputs are always wrapped in a tuple,
            even when there is only a single output value.  This makes all subgraphs
            conform to the convention expected by ``torch._inductor.compile_fx``.

    Returns:
        GraphModule: the module after split.

    Example:

        This is a sample setup:

            import torch
            from torch.fx._symbolic_trace import symbolic_trace
            from torch.fx.graph_module import GraphModule
            from torch.fx.node import Node
            from torch.fx.passes.split_module import split_module

            class MyModule(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand(3, 4))
                    self.linear = torch.nn.Linear(4, 5)

                def forward(self, x, y):
                    z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                    w = self.linear(y).clamp(min=0.0, max=1.0)
                    return z + w

            # symbolically trace model
            my_module = MyModule()
            my_module_traced = symbolic_trace(my_module)

            # random mod partitioning
            partition_counter = 0
            NPARTITIONS = 3

            def mod_partition(node: Node):
                global partition_counter
                partition = partition_counter % NPARTITIONS
                partition_counter = (partition_counter + 1) % NPARTITIONS
                return partition

            # split module in module with submodules
            module_with_submodules = split_module(
                my_module_traced, my_module, mod_partition
            )

        Output looks like this. Original graph is broken into partitions

            > print(module_with_submodules)
            GraphModule(
                (submod_0): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_1): GraphModule(
                    (linear): Linear(in_features=4, out_features=5, bias=True)
                )
                (submod_2): GraphModule()
            )

            def forward(self, x, y):
                param = self.param
                submod_0 = self.submod_0(x, param, y);  x = param = y = None
                getitem = submod_0[0]
                getitem_1 = submod_0[1];  submod_0 = None
                submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None
                getitem_2 = submod_1[0]
                getitem_3 = submod_1[1];  submod_1 = None
                submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
                return submod_2

        Output of split module is the same as output of input traced module.
        This is an example within a test setting:

            > orig_out = my_module_traced(x, y)
            > submodules_out = module_with_submodules(x, y)
            > self.assertEqual(orig_out, submodules_out)
            True
    """

    log.debug(
        "%s",
        lazy_format_graph_code("pre split_module", m, colored=True),
    )

    def construct_graph(
        node: Node,
        base_mod_env: dict[str, Node],
        base_mod_attrs: dict[str, torch.fx.graph_module.GraphModule],
    ) -> tuple[dict[str, Node], dict[str, torch.fx.graph_module.GraphModule]]:
        if node.op == "placeholder":
            default_value = (
                node.args[0] if len(node.args) > 0 else inspect.Signature.empty
            )
            if keep_original_node_name:
                args = (
                    () if default_value is inspect.Signature.empty else (default_value,)
                )
                base_mod_env[node.name] = base_mod_graph.create_node(
                    "placeholder",
                    node.name,
                    args=args,  # type: ignore[arg-type]
                    type_expr=node.type,
                )
            else:
                base_mod_env[node.name] = base_mod_graph.placeholder(
                    node.target,  # type: ignore[arg-type]
                    type_expr=node.type,
                    default_value=default_value,
                )
            base_mod_env[node.name].meta = node.meta.copy()
        elif node.op == "get_attr":
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)  # type: ignore[arg-type]
            base_mod_env[node.name].meta = node.meta.copy()
            if not isinstance(node.target, str):
                raise AssertionError(f"Expected str target, got {type(node.target)}")
            attr_val = _get_attr_from_qualname(m, node.target)
            base_mod_attrs[node.target] = attr_val  # type: ignore[index]
        return base_mod_env, base_mod_attrs

    import sympy

    partitions: dict[str, Partition] = {}
    orig_nodes: dict[str, Node] = {}
    symbol_to_node: dict[sympy.Symbol, Node] = {}

    def record_cross_partition_use(def_node: Node, use_node: Node | None) -> None:
        from torch.fx.experimental.symbolic_shapes import free_symbols

        defined = getattr(def_node, "_fx_partition", None)
        used = getattr(use_node, "_fx_partition", None)

        log.debug(
            "record_cross_partition_use %s (%s) %s (%s)",
            def_node.name,
            defined,
            use_node.name if use_node is not None else "-",
            used,
        )

        if defined != used:
            if defined is not None:
                def_partition = partitions[defined]
                def_partition.outputs.setdefault(def_node.name)
                if used is not None:
                    def_partition.dependents.setdefault(used)

            if used is not None:
                use_partition = partitions[used]
                use_partition.inputs.setdefault(def_node.name)
                # We have made def_node an input to the use_partition.  If
                # this input has symbolic symbols in its size, those also must
                # be made as inputs to the partition
                if (def_val := def_node.meta.get("example_value")) is not None:
                    for s in sorted(free_symbols(def_val), key=str):
                        s_node = symbol_to_node[s]
                        use_partition.inputs.setdefault(s_node.name)
                        if symbol_to_node[s].op != "placeholder":
                            # If the node that defines the symbol is not a
                            # placeholder, we must make it an output of the
                            # partition.  Note that this may be in a different
                            # partition than defined!  Although, this doesn't
                            # really make a difference for correctness, since
                            # defined is guaranteed to have the symbol in
                            # scope and can return it; you just get less
                            # optimal codegen in this case.
                            s_defined = getattr(s_node, "_fx_partition", None)
                            if s_defined is not None:
                                s_def_partition = partitions[s_defined]
                                s_def_partition.outputs.setdefault(s_node.name)
                                s_def_partition.dependents.setdefault(used)
                                use_partition.dependencies.setdefault(s_defined)
                if defined is not None:
                    use_partition.dependencies.setdefault(defined)

    def instantiate_node_partition_mapping(node: Node) -> None:
        partition_idx = split_callback(node)
        partition_name = str(partition_idx)
        if partition_affix is not None:
            # For example, if user specifies partition_affix = "pp", then the
            # partition name will be "pp_0", "pp_1", etc
            partition_name = "_".join([partition_affix, partition_name])

        log.debug(
            "instantiate_node_partition_mapping %s (%s)", node.name, partition_name
        )

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        node._fx_partition = partition_name

    # Global State Nodes are nodes which by their global state effects,
    # "taint" all downstream nodes while they are active.
    GLOBAL_STATE_NODES = [
        torch.amp._enter_autocast,
        torch.amp._exit_autocast,
        torch._C._set_grad_enabled,
    ]

    # For grad regions:
    # ------------------------
    # 1. first region: we do nothing
    # 2. subsequent regions: we insert the set_grad at the beginning
    grad_regions: OrderedDict[Node, set[int]] = OrderedDict()

    # For autocast regions:
    # ------------------------
    # 1. first region: we will only insert the _exit at the end
    # 2. intermediate regions: we will insert both the
    #    _enter at the beginning and _exit at the end
    # 3. last region: we will only insert _enter at the beginning
    # We will do so in the order in which the autocasts were instantiated.
    autocast_regions: OrderedDict[Node, set[int]] = OrderedDict()
    autocast_exits: dict[Node, Node | None] = {}

    active_grad = None
    active_autocasts = set()

    for node in m.graph.nodes:
        # This will prefer placeholder bindings, because those come first.
        # This is a little dangerous though: it is possible that an unbacked
        # symbol is used without any binding site for it, in which case we
        # will get a KeyError not able to find it.  I'd like to fix this by
        # having passes.runtime_assert establish some invariants that I can
        # rely on later, but this needs some extra work.  Quick fix first.
        # See https://github.com/pytorch/pytorch/issues/130534
        if (
            (val := node.meta.get("example_value")) is not None
            and isinstance(val, (torch.SymInt, torch.SymFloat))
            and isinstance(s0 := val.node.expr, sympy.Symbol)
            and s0 not in symbol_to_node
        ):
            symbol_to_node[val.node.expr] = node

        if node.op in ["placeholder", "get_attr", "output"]:
            continue

        instantiate_node_partition_mapping(node)

        if node.op == "call_function" and node.target in GLOBAL_STATE_NODES:
            if node.target is torch._C._set_grad_enabled:
                if len(node.args) != 1:
                    raise AssertionError(
                        f"Expected 1 arg for _set_grad_enabled, got {len(node.args)}"
                    )
                if not isinstance(node.args[0], bool):
                    raise AssertionError(f"Expected bool arg, got {type(node.args[0])}")
                active_grad = node
                grad_regions[active_grad] = set({split_callback(node)})
            elif node.target is torch.amp._enter_autocast:
                # Should all be python constants
                if not all(not isinstance(arg, Node) for arg in node.args):
                    raise AssertionError(
                        "Expected all args to be python constants, not Nodes"
                    )
                active_autocasts.add(node)
                autocast_regions[node] = set({split_callback(node)})
                autocast_exits[node] = None
            elif node.target is torch.amp._exit_autocast:
                if len(node.args) != 1:
                    raise AssertionError(
                        f"Expected 1 arg for _exit_autocast, got {len(node.args)}"
                    )
                autocast_regions[node.args[0]].add(split_callback(node))
                active_autocasts.remove(node.args[0])
                autocast_exits[node.args[0]] = node

        if active_grad is not None:
            grad_regions[active_grad].add(split_callback(node))

        for a in active_autocasts:
            autocast_regions[a].add(split_callback(node))

    if not all(v is not None for v in autocast_exits.values()):
        raise AssertionError("autocast must exit")

    # pyrefly: ignore [bad-assignment]
    autocast_regions = {k: sorted(v) for k, v in autocast_regions.items()}
    # pyrefly: ignore [bad-assignment]
    grad_regions = {k: sorted(v) for k, v in grad_regions.items()}

    if _LOGGER.isEnabledFor(logging.DEBUG):
        _LOGGER.debug("autocast_regions: %s", autocast_regions)
        _LOGGER.debug("grad_regions: %s", grad_regions)

    assert_monotonically_increasing = bool(autocast_regions) or bool(grad_regions)

    # split nodes into partitions
    highest_partition = -1
    for node in m.graph.nodes:
        orig_nodes[node.name] = node

        # TODO currently placeholders/parameters aren't put into random partitions,
        # rather they're added to the graphs where they are used down below
        if node.op in ["placeholder", "get_attr"]:
            continue
        if node.op == "output":
            torch.fx.graph.map_arg(
                node.args[0], lambda n: record_cross_partition_use(n, None)
            )
            continue

        if assert_monotonically_increasing:
            pid = split_callback(node)
            if highest_partition > pid:
                raise AssertionError(
                    "autocast or set_grad_enabled require monotonically increasing "
                    f"partitions: highest: {highest_partition}, this node's: {pid}"
                )
            highest_partition = pid

        # do not capture cross-partition dependencies for global state nodes as they will be
        # self-contained - their setup and unwind will be isolated to each partition submodule.
        if node.target not in GLOBAL_STATE_NODES:
            torch.fx.graph.map_arg(
                node.args, lambda def_node: record_cross_partition_use(def_node, node)
            )
            torch.fx.graph.map_arg(
                node.kwargs, lambda def_node: record_cross_partition_use(def_node, node)
            )

    original_partition_order = list(partitions.keys())
    # find partitions with no dependencies
    root_partitions: list[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.dependencies):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions: list[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].dependents:
            partitions[dependent].dependencies.pop(root_partition)  # noqa: B909
            if not partitions[dependent].dependencies:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # Enter prelude
    for regions_mapping in [autocast_regions, grad_regions]:
        for node, regions in regions_mapping.items():
            if len(regions) == 0:
                raise AssertionError("Expected at least one region for node")
            # pyrefly: ignore [bad-index]
            partitions[str(regions[0])].environment[node] = node
            # pyrefly: ignore [bad-index, index-error]
            # pyrefly: ignore [bad-index, index-error]
            for r in regions[1:]:
                partition = partitions[str(r)]
                new_node = partition.graph.create_node(
                    op=node.op,
                    target=node.target,
                    args=tuple(arg for arg in node.args),
                    kwargs={},
                    type_expr=node.type,
                )
                new_node.meta = (
                    node.meta.copy()
                )  # is it really a good idea to copy this?
                partition.environment[node] = new_node

    # add placeholders to partition inputs
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        new_inputs: dict[str, None] = {}

        counter = 0

        # Process inputs that become placeholders before inputs that become
        # get_attr nodes, so that all placeholders precede get_attr in the
        # submodule graph.
        placeholder_inputs: list[str] = []
        get_attr_inputs: list[str] = []
        for inp in partition.inputs:
            orig_node = orig_nodes[inp]
            if (
                orig_node.op == "get_attr"
                and isinstance(orig_node.target, str)
                and isinstance(
                    _get_attr_from_qualname(m, orig_node.target), torch.nn.Module
                )
            ):
                get_attr_inputs.append(inp)
            else:
                placeholder_inputs.append(inp)

        for inp in placeholder_inputs + get_attr_inputs:
            orig_node = orig_nodes[inp]
            # We don't pass in get_attr nodes as inputs to the partition, but
            # instead set them as targets and use getattr within the module

            def add_placeholder() -> Node:
                if keep_original_input_name:
                    name = inp
                else:
                    nonlocal counter
                    name = f"arg_{counter}"
                    counter += 1
                placeholder = partition.graph.placeholder(
                    name,
                    type_expr=orig_nodes[inp].type,
                )
                new_inputs[inp] = None
                return placeholder

            if orig_node.op == "get_attr":
                if not isinstance(orig_node.target, str):
                    raise AssertionError(
                        f"Expected str target, got {type(orig_node.target)}"
                    )

                orig_attr = _get_attr_from_qualname(m, orig_node.target)
                if isinstance(orig_attr, torch.nn.Module):
                    placeholder = partition.graph.get_attr(orig_node.target)
                    partition.targets[orig_node.target] = orig_attr
                else:
                    placeholder = add_placeholder()
            else:
                placeholder = add_placeholder()
            placeholder.meta = orig_nodes[inp].meta.copy()
            partition.environment[orig_nodes[inp]] = placeholder
        partition.inputs = new_inputs

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, "_fx_partition"):
            partition = partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(
                node.kwargs, lambda n: environment[n]
            )

            if node.op not in ["call_module", "get_attr"]:
                target = node.target
            else:
                target_attr = _get_attr_from_qualname(m, node.target)
                target = node.target.replace(".", "_")
                partition.targets[target] = target_attr
                # Fill in the passed-in mapping from new qualname to old qualname
                if qualname_map is not None:
                    # When creating the split module later, the submodules will have
                    # path prefix matching the corresponding partition's submod_name
                    qualname = f"{partition.submod_name}.{target}"
                    qualname_map[qualname] = node.target

            if not isinstance(gathered_args, tuple):
                raise AssertionError(
                    f"Expected tuple for gathered_args, got {type(gathered_args)}"
                )
            if not isinstance(gathered_kwargs, dict):
                raise AssertionError(
                    f"Expected dict for gathered_kwargs, got {type(gathered_kwargs)}"
                )
            name = node.name if keep_original_node_name else None
            new_node = partition.graph.create_node(
                op=node.op,
                target=target,
                args=gathered_args,
                kwargs=gathered_kwargs,
                type_expr=node.type,
                name=name,
            )
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    # Exit epilogue
    for regions_mapping in [autocast_regions]:
        for node in reversed(regions_mapping):
            regions = regions_mapping[node]
            if len(regions) == 0:
                raise AssertionError("Expected at least one region")
            # pyrefly: ignore [bad-index, index-error]
            for r in regions[:-1]:
                partition = partitions[str(r)]
                exit_node = autocast_exits[node]
                if exit_node is None:
                    raise AssertionError("Missing exit node")
                new_node = partition.graph.create_node(
                    op=exit_node.op,
                    target=exit_node.target,
                    args=(partition.environment[node],),
                    kwargs={},
                    type_expr=exit_node.type,
                )
                new_node.meta = (
                    exit_node.meta.copy()
                )  # is it really a good idea to copy this?

    # original module environment dict mapping node names to nodes
    orig_mod_env: dict[str, Node] = {}
    # Set up values to construct base module
    base_mod_env: dict[str, Node] = {}
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    base_mod_attrs: dict[str, torch.fx.graph_module.GraphModule] = {}
    if not keep_original_order:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )

    else:
        # Go through the graph to construct the mapping dict
        for node in m.graph.nodes:
            orig_mod_env[node.name] = node

    # Do some things iterating over the partitions in topological order again:
    # 1) Finish off submodule Graphs by setting corresponding outputs
    # 2) Construct GraphModules for each submodule
    # 3) Construct the base graph by emitting calls to those submodules in
    #    topological order or original order specified by keep_original_order

    construct_order_partitions = (
        sorted_partitions if not keep_original_order else original_partition_order
    )

    already_constructed_attr_nodes = set()

    # We actually need to insert the placeholder nodes in the original order
    # otherwise graph signature will be wrong.
    original_order = [node for node in m.graph.nodes if node.op == "placeholder"]

    for partition_name in construct_order_partitions:
        partition = partitions[partition_name]

        # Set correct output values
        output_vals = tuple(
            partition.environment[orig_nodes[name]] for name in partition.outputs
        )

        if len(output_vals) == 1 and not tuple_return:
            partition.graph.output(output_vals[0])
        else:
            partition.graph.output(output_vals)

        if keep_original_order:
            # first get the attr nodes required by this partition
            orig_mod_attr_nodes: list[Node] = [
                orig_mod_env[key]
                for key in partition.inputs
                if key not in original_order
            ]

            for node in original_order:
                if node in already_constructed_attr_nodes:
                    continue  # already added this attr to the base graph
                base_mod_env, _based_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

            # Construct GraphModule for this partition
            for node in orig_mod_attr_nodes:  # type: ignore[attr-defined]
                if node in already_constructed_attr_nodes:
                    continue
                base_mod_env, base_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

        base_mod_attrs[partition.submod_name] = _make_graph_module(
            partition.targets, partition.graph
        )

        # Emit call in base graph to this submodule
        output_val = base_mod_graph.call_module(
            partition.submod_name,
            tuple(base_mod_env[name] for name in partition.inputs),
        )

        num_outputs = len(partition.outputs)
        if num_outputs > 1 or (num_outputs == 1 and tuple_return):
            # Unpack return values from submodule
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        elif num_outputs == 1:
            base_mod_env[next(iter(partition.outputs))] = output_val

    # When keep_original_order=True and if the graph doesn't have any
    # `call_function` node then `base_mod_graph`, `base_mod_env` and `base_mod_attrs`
    # are never populated.
    # For this case, we call `construct_graph` here which takes care of updating them.
    if keep_original_order and not base_mod_env:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )

    # Add output node to `base_mod_graph` (i.e. the split graph) which will be returned.
    for node in m.graph.nodes:
        if node.op == "output":
            base_mod_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name])
            )

    ret = _make_graph_module(base_mod_attrs, base_mod_graph)
    log.debug(
        "%s",
        lazy_format_graph_code("post split_module", ret, colored=True),
    )
    return ret


def _decompose_size_nodes(m: GraphModule) -> None:
    """Decompose x.size() into per-dim sym_size.int calls.

    torch.Size objects cannot cross split boundaries because aot_autograd
    cannot handle them as submodule outputs. This replaces each size() call
    with individual sym_size.int(x, dim) nodes:
      - Dynamic dims (SymInt) -> new sym_size.int node
      - Static dims (plain int) -> inlined as literal constant
    """
    # Collect upfront since we mutate the graph during iteration.
    size_nodes = list(m.graph.find_nodes(op="call_method", target="size"))

    for node in size_nodes:
        # Skip size(dim) calls -- only decompose full size() with no dim argument.
        if len(node.args) > 1 or node.kwargs:
            continue
        tensor_node = node.args[0]
        ev = tensor_node.meta.get("example_value")
        if ev is None:
            continue

        dims: list[Node | int] = []
        with m.graph.inserting_after(tensor_node):
            for i, dim_val in enumerate(ev.shape):
                if isinstance(dim_val, torch.SymInt):
                    dn = m.graph.call_function(
                        torch.ops.aten.sym_size.int, args=(tensor_node, i)
                    )
                    dn.meta["example_value"] = dim_val
                    dims.append(dn)
                elif isinstance(dim_val, int):
                    dims.append(dim_val)

        for user in list(node.users):
            new_args = []
            for arg in user.args:
                if arg is node:
                    new_args.extend(dims)
                else:
                    new_args.append(arg)
            user.args = tuple(new_args)
        m.graph.erase_node(node)


def _make_lite_graph_module(
    graph: torch.fx.graph.Graph,
    modules: dict[str, GraphModule] | None = None,
) -> GraphModule:
    """Construct a lightweight GraphModule that bypasses expensive init overhead.

    Creates a ``_LazyGraphModule`` instance without going through
    ``GraphModule.__new__`` (MRO traversal + per-instance class creation),
    ``nn.Module.__init__`` (17+ container allocations), or
    ``GraphModule.__init__`` (node iteration + ``recompile()`` codegen).
    Per-instance classes are still created for lazy forward dispatch.

    Only suitable for parameterless graph containers (no parameters, buffers,
    or ``get_attr``/``call_module`` targets to copy from a root module).

    The dict entries below mirror nn.Module.__init__ (torch/nn/modules/module.py).
    Update in lockstep if nn.Module adds or removes instance attributes.
    """
    cls = type("GraphModuleImpl", (_LazyGraphModule,), {})
    inst = object.__new__(cls)
    d = inst.__dict__
    d["training"] = False
    d["_parameters"] = dict[str, Any]()
    d["_buffers"] = dict[str, Any]()
    d["_modules"] = modules if modules is not None else dict[str, Any]()
    d["_non_persistent_buffers_set"] = set[str]()
    d["_backward_pre_hooks"] = OrderedDict()
    d["_backward_hooks"] = OrderedDict()
    d["_is_full_backward_hook"] = None
    d["_forward_hooks"] = OrderedDict()
    d["_forward_hooks_with_kwargs"] = OrderedDict()
    d["_forward_hooks_always_called"] = OrderedDict()
    d["_forward_pre_hooks"] = OrderedDict()
    d["_forward_pre_hooks_with_kwargs"] = OrderedDict()
    d["_state_dict_hooks"] = OrderedDict()
    d["_state_dict_pre_hooks"] = OrderedDict()
    d["_load_state_dict_pre_hooks"] = OrderedDict()
    d["_load_state_dict_post_hooks"] = OrderedDict()
    d["_graph"] = graph
    d["meta"] = dict[str, Any]()
    graph.owning_module = inst
    cls.forward = _LazyGraphModule._lazy_forward  # type: ignore[attr-defined]
    return inst


def _detect_dependencies(
    m: GraphModule,
    node_to_partition: dict[Node, int],
) -> tuple[
    dict[int, torch.fx.graph.Graph],
    dict[int, dict[str, Node]],
    dict[int, dict[str, Node]],
    list[int],
]:
    """Assign nodes to partitions and detect cross-partition dependencies.

    Returns:
        partition_graphs: Maps partition ID to its Graph.
        partition_inputs: Maps partition ID to ordered dict of input names -> source nodes.
        partition_outputs: Maps partition ID to ordered dict of output names -> source nodes.
        seen_partitions: Partition IDs in order of first appearance.
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import free_symbols

    partition_graphs: dict[int, torch.fx.graph.Graph] = {}
    partition_inputs: dict[int, dict[str, Node]] = defaultdict(dict)
    partition_outputs: dict[int, dict[str, Node]] = defaultdict(dict)
    symbol_to_node: dict[sympy.Symbol, Node] = {}
    seen_partitions: list[int] = []
    seen_partitions_set: set[int] = set()

    def _record_output_dep(
        n: Node,
        _outputs: dict[int, dict[str, Node]] = partition_outputs,
    ) -> Node:
        if hasattr(n, "_fx_partition"):
            _outputs[n._fx_partition].setdefault(n.name, n)
        return n

    for node in m.graph.nodes:
        # Track SymInt symbol bindings (prefer earlier bindings).
        val = node.meta.get("example_value")
        if val is not None and hasattr(val, "node") and hasattr(val.node, "expr"):
            s0 = val.node.expr
            if isinstance(s0, sympy.Symbol) and s0 not in symbol_to_node:
                symbol_to_node[s0] = node

        if node.op == "placeholder":
            continue

        if node.op == "output":
            torch.fx.graph.map_arg(node.args[0], _record_output_dep)
            continue

        pid = node_to_partition[node]
        node._fx_partition = pid

        if pid not in seen_partitions_set:
            seen_partitions.append(pid)
            seen_partitions_set.add(pid)
            partition_graphs[pid] = torch.fx.graph.Graph()

        def _record_cross_dep(
            def_node: Node,
            use_pid: int = pid,
            _inputs: dict[int, dict[str, Node]] = partition_inputs,
            _outputs: dict[int, dict[str, Node]] = partition_outputs,
            _sym: dict[sympy.Symbol, Node] = symbol_to_node,
        ) -> Node:
            def_pid = getattr(def_node, "_fx_partition", None)
            if def_pid != use_pid:
                if def_pid is not None:
                    _outputs[def_pid].setdefault(def_node.name, def_node)
                _inputs[use_pid].setdefault(def_node.name, def_node)

                def_val = def_node.meta.get("example_value")
                if def_val is not None and _sym:
                    for s in sorted(free_symbols(def_val), key=str):
                        s_node = _sym.get(s)
                        if s_node is None:
                            continue
                        _inputs[use_pid].setdefault(s_node.name, s_node)
                        if s_node.op != "placeholder":
                            s_pid = getattr(s_node, "_fx_partition", None)
                            if s_pid is not None:
                                _outputs[s_pid].setdefault(s_node.name, s_node)
            return def_node

        torch.fx.graph.map_arg(node.args, _record_cross_dep)
        torch.fx.graph.map_arg(node.kwargs, _record_cross_dep)

    return partition_graphs, partition_inputs, partition_outputs, seen_partitions


def _clone_nodes_into_partitions(
    m: GraphModule,
    partition_graphs: dict[int, torch.fx.graph.Graph],
    partition_inputs: dict[int, dict[str, Node]],
    partition_outputs: dict[int, dict[str, Node]],
    seen_partitions: list[int],
) -> dict[int, dict[Node, Node]]:
    """Create placeholders, clone nodes, and set outputs for each partition.

    Returns:
        partition_env: Maps partition ID to {original_node: cloned_node} dict.
    """
    partition_env: dict[int, dict[Node, Node]] = defaultdict(dict)

    # Create placeholders for cross-partition inputs
    for pid in seen_partitions:
        g = partition_graphs[pid]
        env = partition_env[pid]
        for inp_name, orig_node in partition_inputs[pid].items():
            placeholder = g.placeholder(inp_name, type_expr=orig_node.type)
            placeholder.meta = orig_node.meta.copy()
            env[orig_node] = placeholder

    # Clone operational nodes
    for node in m.graph.nodes:
        if not hasattr(node, "_fx_partition"):
            continue
        pid = node._fx_partition
        g = partition_graphs[pid]
        env = partition_env[pid]

        gathered_args = torch.fx.graph.map_arg(node.args, env.__getitem__)
        gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, env.__getitem__)

        new_node = g.create_node(
            op=node.op,
            target=node.target,
            args=gathered_args,
            kwargs=gathered_kwargs,
            type_expr=node.type,
        )
        new_node.meta = node.meta.copy()
        env[node] = new_node

    # Set output nodes
    for pid in seen_partitions:
        g = partition_graphs[pid]
        env = partition_env[pid]
        out_nodes = partition_outputs[pid]
        output_vals = tuple(env[orig_node] for orig_node in out_nodes.values())
        if len(output_vals) == 1:
            g.output(output_vals[0])
        elif len(output_vals) > 1:
            g.output(output_vals)
        else:
            g.output(())

    return partition_env


def _build_stitching_graph(
    m: GraphModule,
    partition_graphs: dict[int, torch.fx.graph.Graph],
    partition_inputs: dict[int, dict[str, Node]],
    partition_outputs: dict[int, dict[str, Node]],
    seen_partitions: list[int],
    partition_affix: str | None = None,
) -> GraphModule:
    """Build the outer stitching graph that calls each partition submodule."""
    base_graph = torch.fx.graph.Graph()
    base_env: dict[str, Node] = {}
    base_modules: dict[str, GraphModule] = {}

    for node in m.graph.nodes:
        if node.op == "placeholder":
            p = base_graph.placeholder(
                node.target,
                type_expr=node.type,
            )
            p.meta = node.meta.copy()
            base_env[node.name] = p

    for pid in seen_partitions:
        if partition_affix is not None:
            submod_name = f"submod_{partition_affix}_{pid}"
        else:
            submod_name = f"submod_{pid}"
        base_modules[submod_name] = _make_lite_graph_module(partition_graphs[pid])

        input_nodes = tuple(base_env[name] for name in partition_inputs[pid])
        output_val = base_graph.call_module(submod_name, input_nodes)

        out_names = list(partition_outputs[pid].keys())
        if len(out_names) > 1:
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, name in enumerate(out_names):
                base_env[name] = output_val_proxy[i].node  # type: ignore[index]
        elif len(out_names) == 1:
            base_env[out_names[0]] = output_val

    for node in m.graph.nodes:
        if node.op == "output":
            base_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_env[n.name])
            )

    return _make_lite_graph_module(base_graph, modules=base_modules)


@compatibility(is_backward_compatible=False)
def split_module_simple(
    m: GraphModule,
    node_to_partition: dict[Node, int],
    *,
    partition_affix: str | None = None,
) -> GraphModule:
    """Lightweight graph splitter for simple partition patterns.

    A faster alternative to :func:`split_module` for inference-only graphs
    from ``torch.compile``/Dynamo. Because these graphs have no autocast/grad
    regions, no ``get_attr`` nodes, and no non-linear partition dependencies,
    we can skip the topological sort, autocast tracking, and ``get_attr``
    special-casing that ``split_module`` performs. More importantly, we
    construct partition submodules as lightweight ``_LazyGraphModule``
    instances that bypass ``nn.Module.__init__`` and defer ``recompile()``
    codegen — this eliminates the dominant cost when creating 70-100+
    partition submodules for large models.

    Args:
        m: Graph module to split.
        node_to_partition: Maps each operational node to a partition ID.
            Placeholders, get_attr, and output nodes should NOT be included.
        partition_affix: If set, submodule names become
            ``submod_{affix}_{idx}`` instead of ``submod_{idx}``.
    """
    partition_graphs, partition_inputs, partition_outputs, seen_partitions = (
        _detect_dependencies(m, node_to_partition)
    )

    _clone_nodes_into_partitions(
        m, partition_graphs, partition_inputs, partition_outputs, seen_partitions
    )

    return _build_stitching_graph(
        m,
        partition_graphs,
        partition_inputs,
        partition_outputs,
        seen_partitions,
        partition_affix,
    )
