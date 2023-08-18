import inspect
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule

__all__ = ["Partition", "split_module"]


@compatibility(is_backward_compatible=True)
class Partition:
    def __init__(self, name: str):
        self.name: str = name
        self.submod_name = f"submod_{name}"
        self.node_names: List[str] = []
        self.inputs: Dict[str, None] = {}
        self.outputs: Dict[str, None] = {}
        self.partitions_dependent_on: Dict[str, None] = {}
        self.partition_dependents: Dict[str, None] = {}
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.environment: Dict[torch.fx.node.Node, torch.fx.node.Node] = {}
        self.targets: Dict[str, Any] = {}

    def __repr__(self) -> str:
        return (
            f"name: {self.name},\n"
            f" nodes: {self.node_names},\n"
            f" inputs: {self.inputs},\n"
            f" outputs: {self.outputs},\n"
            f" partitions dependent on: {self.partitions_dependent_on},\n"
            f" partition dependents: {self.partition_dependents}"
        )


# Creates subgraphs out of main graph
@compatibility(is_backward_compatible=True)
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[torch.fx.node.Node], int],
    qualname_map: Optional[Dict[str, str]] = None,
    keep_original_order: Optional[bool] = False,
):
    """
    Creates subgraphs out of main graph

    Args:
        m (GraphModule): Graph module to split
        root_m (torch.nn.Module): root nn module. Not currently used. Included
            because the root nn module is usually transformed via
            torch.fx._symbolic_trace.symbolic_trace (see example below)
        split_callback (Callable[[torch.fx.node.Node], int]): Callable function
            that maps a given Node instance to a numeric partition identifier.
            split_module will use this function as the policy for which operations
            appear in which partitions in the output Module.
        qualname_map: Optional[Dict[str, str]]: optional output parameter that returns a
            mapping from new target names in the module after split to old target
            names in the original module.
        keep_original_order: Optional[bool]: keep the original order of the GraphModule
            or use the Topological order of the new constructed GraphModule


    Returns:
        GraphModule: the module after split.

    Example:

        This is a sample setup:

            import torch
            from torch.fx.symbolic_trace import symbolic_trace
            from torch.fx.graph_module import GraphModule
            from torch.fx.node import Node
            from torch.fx.passes.split_module import split_module

            class MyModule(torch.nn.Module):
                def __init__(self):
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

    def construct_graph(
        node: torch.fx.node.Node,
        base_mod_env: Dict[str, torch.fx.node.Node],
        base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule],
    ):
        if node.op == "placeholder":
            default_value = (
                node.args[0] if len(node.args) > 0 else inspect.Signature.empty
            )
            base_mod_env[node.name] = base_mod_graph.placeholder(
                node.target, type_expr=node.type, default_value=default_value
            )
            base_mod_env[node.name].meta = node.meta.copy()
        elif node.op == "get_attr":
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)
            base_mod_env[node.name].meta = node.meta.copy()
            attr_val = m
            for atom in node.target.split("."):  # type: ignore[union-attr]
                if not hasattr(attr_val, atom):
                    raise AttributeError(f"Node target {node.target} not found!")
                attr_val = getattr(attr_val, atom)
            base_mod_attrs[node.target] = attr_val  # type: ignore[index]
        return base_mod_env, base_mod_attrs

    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, torch.fx.node.Node] = {}

    def record_cross_partition_use(
        def_node: torch.fx.node.Node, use_node: Optional[torch.fx.node.Node]
    ):  # noqa: B950
        def_partition_name = getattr(def_node, "_fx_partition", None)
        use_partition_name = getattr(use_node, "_fx_partition", None)
        if def_partition_name != use_partition_name:
            if def_partition_name is not None:
                def_partition = partitions[def_partition_name]
                def_partition.outputs.setdefault(def_node.name)
                if use_partition_name is not None:
                    def_partition.partition_dependents.setdefault(use_partition_name)

            if use_partition_name is not None:
                use_partition = partitions[use_partition_name]
                use_partition.inputs.setdefault(def_node.name)
                if def_partition_name is not None:
                    use_partition.partitions_dependent_on.setdefault(def_partition_name)

    # split nodes into partitions
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
        partition_name = str(split_callback(node))

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        node._fx_partition = partition_name

        torch.fx.graph.map_arg(
            node.args, lambda def_node: record_cross_partition_use(def_node, node)
        )
        torch.fx.graph.map_arg(
            node.kwargs, lambda def_node: record_cross_partition_use(def_node, node)
        )  # noqa: B950

    original_partition_order = list(partitions.keys())
    # find partitions with no dependencies
    root_partitions: List[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.partitions_dependent_on):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions: List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].partition_dependents:
            partitions[dependent].partitions_dependent_on.pop(root_partition)
            if not partitions[dependent].partitions_dependent_on:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # add placeholders to partitions
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        for input in partition.inputs:
            placeholder = partition.graph.placeholder(
                input,
                type_expr=orig_nodes[input].type,
            )
            placeholder.meta = orig_nodes[input].meta.copy()
            partition.environment[orig_nodes[input]] = placeholder

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
                target_atoms = node.target.split(".")
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise AttributeError(f"Operator target {node.target} not found!")
                    target_attr = getattr(target_attr, atom)
                # target = target_atoms[-1]
                target = "_".join(target_atoms)
                partition.targets[target] = target_attr
                # Fill in the passed-in mapping from new qualname to old qualname
                if qualname_map is not None:
                    # When creating the split module later, the submodules will have
                    # path prefix matching the corresponding partition's submod_name
                    qualname = f"{partition.submod_name}.{target}"
                    qualname_map[qualname] = node.target

            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(
                op=node.op,
                target=target,
                args=gathered_args,
                kwargs=gathered_kwargs,
                type_expr=node.type,
            )
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    # original module environment dict mapping node names to nodes
    org_mod_env: Dict[str, torch.fx.node.Node] = {}
    # Set up values to construct base module
    base_mod_env: Dict[str, torch.fx.node.Node] = {}
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule] = {}
    if not keep_original_order:
        for node in m.graph.nodes:
            base_mod_env, base_mod_attrs = construct_graph(
                node, base_mod_env, base_mod_attrs
            )

    else:
        # Go through the graph to construct the mapping dict
        for node in m.graph.nodes:
            org_mod_env[node.name] = node

    # Do some things iterating over the partitions in topological order again:
    # 1) Finish off submodule Graphs by setting corresponding outputs
    # 2) Construct GraphModules for each submodule
    # 3) Construct the base graph by emitting calls to those submodules in
    #    topological order or original order specified by keep_original_order

    construct_order_partitions = (
        sorted_partitions if not keep_original_order else original_partition_order
    )

    already_constructed_attr_nodes = set()
    for partition_name in construct_order_partitions:
        partition = partitions[partition_name]

        # Set correct output values
        output_vals = tuple(
            partition.environment[orig_nodes[name]] for name in partition.outputs
        )

        # skip output node generation if there are no output values
        num_output_vals = len(output_vals)
        if num_output_vals == 1:
            partition.graph.output(output_vals[0])
        elif num_output_vals > 1:
            partition.graph.output(output_vals)

        if keep_original_order:
            # first get the attr nodes required by this partition
            org_mod_attr_nodes: List[torch.fx.node.Node] = [
                org_mod_env[key] for key in partition.inputs
            ]
            # Construct GraphModule for this partition
            for node in org_mod_attr_nodes:  # type: ignore[attr-defined]
                if node in already_constructed_attr_nodes:
                    continue
                base_mod_env, base_mod_attrs = construct_graph(
                    node, base_mod_env, base_mod_attrs
                )
                already_constructed_attr_nodes.add(node)

        base_mod_attrs[partition.submod_name] = torch.fx.graph_module.GraphModule(
            partition.targets, partition.graph
        )  # noqa: B950

        # Emit call in base graph to this submodule
        output_val = base_mod_graph.call_module(
            partition.submod_name,
            tuple(base_mod_env[name] for name in partition.inputs),
        )

        num_outputs = len(partition.outputs)
        if num_outputs > 1:
            # Unpack multiple return values from submodule
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        elif num_outputs == 1:
            base_mod_env[list(partition.outputs)[0]] = output_val

    for node in m.graph.nodes:
        if node.op == "output":
            base_mod_graph.output(
                torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name])
            )  # noqa: B950

    return torch.fx.graph_module.GraphModule(base_mod_attrs, base_mod_graph)
