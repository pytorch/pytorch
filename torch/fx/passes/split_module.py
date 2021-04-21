import torch
from torch.fx.graph_module import GraphModule
from typing import Callable, List, Dict, Any, Optional

class Partition:
    def __init__(self, name: str):
        self.name: str = name
        self.node_names: List[str] = []
        self.inputs: Dict[str, None] = {}
        self.outputs: Dict[str, None] = {}
        self.partitions_dependent_on: Dict[str, None] = {}
        self.partition_dependents: Dict[str, None] = {}
        self.graph : torch.fx.graph.Graph = torch.fx.graph.Graph()  # type: ignore[attr-defined, name-defined]
        self.environment : Dict[torch.fx.node.Node, torch.fx.node.Node] = {}  # type: ignore[name-defined]
        self.targets : Dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"name: {self.name},\n" \
            f" nodes: {self.node_names},\n" \
            f" inputs: {self.inputs},\n" \
            f" outputs: {self.outputs},\n" \
            f" partitions depenent on: {self.partitions_dependent_on},\n" \
            f" parition dependents: {self.partition_dependents}"

# Creates subgraphs out of main graph
def split_module(
    m: GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[torch.fx.node.Node], int],  # type: ignore[name-defined]
):
    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, torch.fx.node.Node] = {}  # type: ignore[name-defined]

    def record_cross_partition_use(def_node : torch.fx.node.Node, use_node : Optional[torch.fx.node.Node]):  # type: ignore[name-defined] # noqa: B950
        def_partition_name = getattr(def_node, '_fx_partition', None)
        use_partition_name = getattr(use_node, '_fx_partition', None)
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

    # split nodes into parititons
    for node in m.graph.nodes:
        orig_nodes[node.name] = node

        # TODO currently placeholders/parameters aren't put into random partitions,
        # rather they're added to the graphs where they are used down below
        if node.op in ["placeholder", "get_attr"]:
            continue
        if node.op == 'output':
            torch.fx.graph.map_arg(node.args[0], lambda n: record_cross_partition_use(n, None))  # type: ignore[attr-defined]
            continue
        partition_name = str(split_callback(node))

        # add node to partitions
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.node_names.append(node.name)
        node._fx_partition = partition_name

        torch.fx.graph.map_arg(node.args, lambda def_node: record_cross_partition_use(def_node, node))  # type: ignore[attr-defined]
        torch.fx.graph.map_arg(node.kwargs, lambda def_node: record_cross_partition_use(def_node, node))  # type: ignore[attr-defined] # noqa: B950

    # find partitions with no dependencies
    root_partitions : List[str] = []
    for partition_name, partition in partitions.items():
        if not len(partition.partitions_dependent_on):
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions : List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].partition_dependents:
            partitions[dependent].partitions_dependent_on.pop(root_partition)
            if not partitions[dependent].partitions_dependent_on:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # add placeholders to parititons
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        for input in partition.inputs:
            placeholder = partition.graph.placeholder(input)
            placeholder.meta = orig_nodes[input].meta.copy()
            partition.environment[orig_nodes[input]] = placeholder

    # Transform nodes and collect targets for partition's submodule
    for node in m.graph.nodes:
        if hasattr(node, '_fx_partition'):
            partition = partitions[node._fx_partition]

            # swap out old graph nodes in kw/args with references to new nodes in this submodule
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n : environment[n])  # type: ignore[attr-defined]
            gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, lambda n : environment[n])  # type: ignore[attr-defined]

            if node.op not in ['call_module', 'get_attr']:
                target = node.target
            else:
                target_atoms = node.target.split('.')
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise RuntimeError(f'Operator target {node.target} not found!')
                    target_attr = getattr(target_attr, atom)
                # target = target_atoms[-1]
                target = '_'.join(target_atoms)
                partition.targets[target] = target_attr

            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(op=node.op, target=target, args=gathered_args,
                                                   kwargs=gathered_kwargs)
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node

    # Set up values to construct base module
    base_mod_env : Dict[str, torch.fx.node.Node] = {}  # type: ignore[name-defined]
    base_mod_graph : torch.fx.graph.Graph = torch.fx.graph.Graph()  # type: ignore[attr-defined, name-defined]
    base_mod_attrs : Dict[str, torch.fx.graph_module.GraphModule] = {}  # type: ignore[name-defined]
    for node in m.graph.nodes:
        if node.op == 'placeholder':
            base_mod_env[node.name] = base_mod_graph.placeholder(node.name)
        elif node.op == 'get_attr':
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)
            attr_val = m
            for atom in node.target.split('.'):
                if not hasattr(attr_val, atom):
                    raise RuntimeError(f'Node target {node.target} not found!')
                attr_val = getattr(attr_val, atom)
            base_mod_attrs[node.target] = attr_val

    # Do some things iterating over the partitions in topological order again:
    # 1) Finish off submodule Graphs by setting corresponding outputs
    # 2) Construct GraphModules for each submodule
    # 3) Construct the base graph by emitting calls to those submodules in
    #    topological order

    for partition_name in sorted_partitions:
        partition = partitions[partition_name]

        # Set correct output values
        output_vals = tuple(partition.environment[orig_nodes[name]] for name in partition.outputs)
        output_vals = output_vals[0] if len(output_vals) == 1 else output_vals  # type: ignore[assignment]
        partition.graph.output(output_vals)

        # Construct GraphModule for this partition
        submod_name = f'submod_{partition_name}'
        base_mod_attrs[submod_name] = torch.fx.graph_module.GraphModule(partition.targets, partition.graph)  # type: ignore[attr-defined] # noqa: B950

        # Emit call in base graph to this submodule

        output_val = base_mod_graph.call_module(submod_name, tuple(base_mod_env[name] for name in partition.inputs))
        if len(partition.outputs) > 1:
            # Unpack multiple return values from submodule
            output_val_proxy = torch.fx.proxy.Proxy(output_val)  # type: ignore[attr-defined]
            for i, output_name in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node  # type: ignore[index]
        else:
            base_mod_env[list(partition.outputs)[0]] = output_val

    for node in m.graph.nodes:
        if node.op == 'output':
            base_mod_graph.output(torch.fx.graph.map_arg(node.args[0], lambda n : base_mod_env[n.name]))  # type: ignore[attr-defined] # noqa: B950

    return torch.fx.graph_module.GraphModule(base_mod_attrs, base_mod_graph)  # type: ignore[attr-defined]
