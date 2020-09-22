import torch
import torch.fx
import operator
from typing import Callable, List, Dict


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x, y):
        z = self.linear(x + self.param).clamp(min=0.0, max=1.0) 
        w = self.linear(y).clamp(min=0.0, max=1.0) 
        return z + w


# Symbolically trace handbuild model for testing
my_module = MyModule()
my_module_traced = torch.fx.symbolic_trace(my_module)



# Creates a module from the nodes and root torch.nn.module of a sybmolically traced torch.fx.GraphModule
#
# module_name: name for new module's root torch.nn.module
# module_node_names: list of node names to add to new graph, need to match names in orig_nodes
# module_outputs: list of node names which are outputs of this submodule
# orig_nodes: nodes from original torch.fx.GraphModule
# orig_root_module: original torch.nn.module which orig_nodes's torch.fx.GraphModule was traced from
# submodule_hints: if adding submodules from modules created by this function, add their names here
#                  and add their names to module_node_names and add their nodes to orig_nodes
# placeholder_hints: if node in module was originally not a placeholder but now is for this graph,
#                    add node name here
def create_module_from_nodes(
    module_name: str,
    module_node_names: List[str],
    module_outputs: List[str],
    orig_nodes: Dict[str, torch.fx.Node],
    orig_root_module: torch.nn.Module,
    submodule_hints: List[str],
    placeholder_hints: List[str],
) -> (torch.fx.GraphModule, torch.fx.Node):
    # create new graph
    graph = torch.fx.Graph()

    # for tracking graph info 
    new_nodes: Dict[str, torch.fx.Node] = {}
    graph_placeholders = []
    graph_outputs = []

    # copy over nodes from original graph
    for node_name in module_node_names:
        orig_node = orig_nodes[node_name]

        if node_name in placeholder_hints:
            # actual node work done in another submodule, result is input into this submodule
            op = 'placeholder' if node_name in placeholder_hints else orig_node.op
            new_args = ()
            new_kwargs = ()
            target = node_name
        else:
            op = orig_node.op

            # perform arg and kwarg transformation
            def replace_args(old_node: torch.fx.Node, new_nodes: Dict[str, torch.fx.Node]):
                return new_nodes[old_node.name]

            new_args = torch.fx.map_arg(
                orig_node.args, lambda n: replace_args(n, new_nodes)
            )
            new_kwargs = torch.fx.map_arg(
                orig_node.kwargs, lambda n: replace_args(n, new_nodes)
            )

            # create target
            target = (
                module_name + "." + orig_node.name
                if orig_node.op in ["get_attr", "call_module"]
                else orig_node.target
            )

        # create new node
        new_nodes[node_name] = graph.create_node(
            op=op,
            target=target,
            args=new_args,
            kwargs=new_kwargs,
            name=orig_node.name,
        )

        if node_name in module_outputs:
            graph_outputs.append(new_nodes[node_name])
        if op == "placeholder":
            graph_placeholders.append(new_nodes[node_name])

    # set graph outputs 
    if module_outputs:
        graph.output(graph_outputs)
    else:
        graph.output(new_nodes[module_node_names[-1]])

    # set up dict for copying over targets
    subgraph_targets = {}
    for node_name, node in new_nodes.items():
        if node.op in ["get_attr", "call_module"]:
            if node_name in submodule_hints:
                # when node is a module created from this function, target is
                # in orig_node's target directly
                node_target = orig_nodes[node_name].target
            else:
                # the actual target still lives in the original torch.nn.module's hiearchy
                node_target = orig_root_module
                for atom in orig_nodes[node_name].target.split("."):
                    node_target = getattr(node_target, atom)
            subgraph_targets[str(node.target)] = node_target

    module: torch.fx.GraphModule = torch.fx.GraphModule(subgraph_targets, graph)
    module_as_node = torch.fx.Node(
        graph=graph,
        name=module_name,
        op="call_module",
        target=module,
        args=tuple(graph_placeholders),
        kwargs={},
    )

    return (module, module_as_node)


class Partition:
    def __init__(self, name: str):
        self.name: str = name
        self.nodes: List[str] = []
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.partitions_dependent_on: List[str] = []
        self.partition_dependents: List[str] = []

    def __repr__(self) -> str:
        return f"name: {self.name},\n" \
            f" nodes: {self.nodes},\n" \
            f" inputs: {self.inputs},\n" \
            f" outputs: {self.outputs},\n" \
            f" partitions depenent on: {self.partitions_dependent_on},\n" \
            f" parition dependents: {self.partition_dependents}"


# Creates subgraphs out of main graph 
def split_module(
    m: torch.fx.GraphModule,
    root_m: torch.nn.Module,
    split_callback: Callable[[torch.fx.Node], int],
):
    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, torch.fx.Node] = {}

    # split nodes into parititons
    for node in m.graph.nodes:
        orig_nodes[node.name] = node

        # TODO currently placeholders/parameters aren't put into random partitions, 
        # rather they're added to the graphs where they are used down below
        if node.op in ["placeholder", "get_attr"]:
            continue
        partition_name = split_callback(node)

        # add node to partitions
        if (partition := partitions.get(partition_name)) is None:
            partitions[partition_name] = partition = Partition(partition_name)

        partition.nodes.append(node.name)

    # analyze parition inputs, is partition input if is a node argument and isn't inside of parition already
    for partition in partitions.values():
        for node_name in partition.nodes:

            def find_external_inputs(node_arg: torch.fx.Node, cur_partition: Partition):
                if node_arg.name not in cur_partition.nodes and node_arg.name not in cur_partition.inputs:
                    cur_partition.inputs.append(node_arg.name)

            torch.fx.map_arg(orig_nodes[node_name].args, lambda n: find_external_inputs(n, partition))
            torch.fx.map_arg(orig_nodes[node_name].kwargs, lambda n: find_external_inputs(n, partition))

    # analyze partition outputs, is an output if it is used outside the partition itself
    for output_partition_name, output_partition in partitions.items():
        for output_node_name in output_partition.nodes:
            # for each node in each partition, compare against all other nodes in other paritition inputs
            # to check if node is an output
            for input_partition_name, input_partition in partitions.items():
                if input_partition_name != output_partition_name:
                    # if node is an input, need to add dependency between paritions
                    if output_node_name in input_partition.inputs:
                        if input_partition_name not in output_partition.partition_dependents:
                            output_partition.partition_dependents.append(input_partition_name)
                            input_partition.partitions_dependent_on.append(output_partition_name)
                        if output_node_name not in output_partition.outputs:
                            # add node to list of output partition's list of output nodes
                            output_partition.outputs.append(output_node_name)

    # find partitions with no dependencies
    root_partitions : List[str] = []
    for partition_name, partition in partitions.items():
        if not partition.partitions_dependent_on:
            root_partitions.append(partition_name)

    # check partitions for circular dependencies and create topological partition ordering
    sorted_partitions : List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].partition_dependents:
            partitions[dependent].partitions_dependent_on.remove(root_partition)
            if not partitions[dependent].partitions_dependent_on:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError("cycle exists between partitions!")

    # info needed to make outer module
    outer_module_nodes : List[str] = []
    submodule_hints = []

    # add original placeholders to outer graph
    for node_name, node in orig_nodes.items():
        if node.op == 'placeholder':
            outer_module_nodes.append(node_name)

    # create submodules from partitions
    for partition_name in sorted_partitions:
        submodule_name = "submod_" + str(partition_name)
        submodule_hints.append(submodule_name)
        submodule_nodes = []

        # track placeholders 
        placeholder_hints = []
        for node_name in partitions[partition_name].inputs:
            node = orig_nodes[node_name]
            if node.op not in ['placeholder', 'get_attr']: 
                # node not originally placeholder, need to convert 
                placeholder_hints.append(node_name)
            submodule_nodes.append(node_name)

        # grab rest of nodes from partition
        submodule_nodes.extend(partitions[partition_name].nodes)

        # create submodule itself
        submodule, submodule_node = create_module_from_nodes(
            module_name=submodule_name,
            module_node_names=submodule_nodes,
            module_outputs=partitions[partition_name].outputs,
            orig_nodes=orig_nodes,
            orig_root_module=root_m,
            submodule_hints=[],
            placeholder_hints=placeholder_hints,
        )

        # prep information for outer module 
        orig_nodes[submodule_name] = submodule_node
        outer_module_nodes.append(submodule_name)

        # need to unpack submodule outputs into new nodes in outer module
        # to allow for them to be passed into other submodules 
        for counter, output_node_name in enumerate(partitions[partition_name].outputs):
            output_node = torch.fx.Node(
                graph=None,
                name=output_node_name,
                op="call_function",
                target=operator.getitem,
                args=(orig_nodes[submodule_name], counter),
                kwargs={},
            )
            orig_nodes[output_node_name] = output_node
            outer_module_nodes.append(output_node_name) 

    # create outer GraphModule
    outer_graph_module, _ = create_module_from_nodes(
        module_name='outer_graph',
        module_node_names=outer_module_nodes,
        module_outputs=[],
        orig_nodes=orig_nodes,
        orig_root_module=root_m,
        submodule_hints=submodule_hints,
        placeholder_hints=[]
    )

    return outer_graph_module

# random mod partitioning
partition_counter = 0
NPARTITIONS = 3
def mod_partition(node: torch.fx.Node):
    global partition_counter
    partition = partition_counter % NPARTITIONS
    partition_counter = (partition_counter + 1) % NPARTITIONS
    return partition


split_graph = split_module(my_module_traced, my_module, mod_partition)

print(split_graph)

x = torch.rand(3, 4)
y = torch.rand(3, 4)

orig_out = my_module_traced(x, y)
subgraphs_out = split_graph(x, y)

print(orig_out)
print()
print(subgraphs_out)
