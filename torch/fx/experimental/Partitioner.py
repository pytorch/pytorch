from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from typing import Dict, List, Union, Set, NamedTuple, Tuple
import torch
from torch.fx.experimental.subgraph_creation_example import split_module
import operator

class DAGNode(NamedTuple):
    """
    DAGNode class maintains useful information for a partition (submodule).
    inputs(submodule node) and outputs(submodule node).
    """
    submodule: Node
    input_nodes: List[Node]
    output_nodes: List[Node]

class DAG:
    """DAG class contains all the DAG nodes"""
    def __init__(self) -> None:
        self.nodes: List[DAGNode] = []

    def create_node(
        self,
        submodule: Node,
        input_nodes: List[Node],
        output_nodes: List[Node]
    ) -> None:
        node = DAGNode(submodule, input_nodes, output_nodes)
        self.nodes.append(node)

class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulation the partition.
    """
    def __init__(self, partition_id: int, fx_module: GraphModule) -> None:
        self.graph_module = fx_module
        self.nodes: Set[Node] = set()
        self.partition_id = partition_id
        self.parents: Set['Partition'] = set()
        self.children: Set['Partition'] = set()
        self.bfs_level: int = -1

    def add_node(self, node: Node) -> None:
        """Append a new node into the partition."""
        self.nodes.add(node)

    def add_parent(self, partition: 'Partition') -> None:
        self.parents.add(partition)

    def add_child(self, partition: 'Partition') -> None:
        self.children.add(partition)

    def __str__(self):
        return str(self.partition_id)

class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new graph module
    """
    dag: DAG
    module_with_submodules: GraphModule

class Device(NamedTuple):
    name: str
    available_mem_bytes: Union[float, int]

class Partitioner:
    """A graph module may not fit into one device.
    Partitioner class helps cut one graph into subgraphs (partitions),
    so that each partition could fit into a different device.
    The main function of this class is self.partition_graph.
    For self.partition_graph, first, it checks the size of the whole graph
    and see if the whole graph can fit into one device.
    If it does, it goes to self.find_single_partition
    If the whole graph is even larger than the combined memory size of all devices,
    a RuntimeError is raised.
    If the whole graph cannot fit into one devices but
    could be split into multiple devices, it goes to self.size_based_partition.
    After the size_based_partition, it checks if the number of partitions exceeds
    the number of devices. If it does, a RuntimeError is raised.
    Otherwise, a DAG structure is returned
    along with a new graph module with partitions as submodule nodes.
    """
    def __init__(self) -> None:
        self.partitions: Set[Partition] = set()
        self.devices: List[Device] = []
        self.node_to_partitions: Dict[Node, List[int]] = {}
        self.partition_to_used_mem_bytes: Dict[Partition, int] = {}

    def partition_graph(
        self,
        fx_module: GraphModule,
        torch_module: torch.nn.Module,
        devices: List[Device]
    ) -> PartitionResult:
        """
        Given the fx module, torch module and devices,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
        self.graph_module = fx_module
        self.devices = devices
        self.torch_module = torch_module
        if len(self.devices) == 0:
            raise RuntimeError('No devices')
        available_mem_bytes = self.devices[0].available_mem_bytes
        # Check if there are op nodes in the graph
        nodes = self.graph_module.graph.nodes
        if all(node.op in {'placeholder', 'get_attr', 'output'} for node in nodes):
            raise RuntimeError('No Partition since no operations in the module')
        # Calculate total size of the graph
        total_size_of_graph = 0
        for node in nodes:
            if node.op == 'output':
                break
            total_size_of_graph += node.size_bytes.total_size
        if total_size_of_graph <= available_mem_bytes:
            self.find_single_partition()
        elif total_size_of_graph > len(self.devices) * available_mem_bytes:
            raise RuntimeError('Devices have no enough memory for the module')
        else:
            if not all(device.available_mem_bytes == available_mem_bytes for device in self.devices):
                raise RuntimeError('All devices must have same memory size!')
            self.size_based_partition(available_mem_bytes)
        # Check if enought devices are provided for all partitions
        if len(self.partitions) > len(self.devices):
            raise RuntimeError('Lack of Devices')
        module_with_submodules = self.do_partition()
        # The DAG contains DAGNodes with info of each partition's input nodes, output nodes
        # and how partitions are connected.
        dag = self.dump_partition_DAG(module_with_submodules)
        ret = PartitionResult(dag, module_with_submodules)
        return ret

    def find_single_partition(self) -> None:
        """Only one partition (one graph on one device)."""
        partition_0 = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op == 'output':
                break
            self.node_to_partitions[node] = [partition_0.partition_id]
            partition_0.add_node(node)
        return

    def size_based_partition(self, available_mem_bytes: Union[float, int]) -> None:
        """This method partitions the graph based on memory size.
           We assume all devices have the same memory size.
           The basic idea is:
           First, create a new partition.
           Then traverse the graph through self.graph_module.graph.nodes
           The traversal only focuses on op nodes
           (call_function, call_module, call_method).
           The placeholder nodes (placeholder) and constant nodes (get_attr) are skipped.
           A placeholder (placeholder) or a constant (get_attr)
           is added into a partition when it is a input node for a op node.
           From one op node to another, check if a op node and its input nodes
           can fit into the current partition.
           If the current partition is full, create a new one
           and continue traversing op nodes.
           Then through self.combine_partition_based_on_size(),
           partitions will be combined to keep
           as less partitions as possible.
           self.check_partition_dependecy checks if the combination of
           partitions leads to a circular dependency
        """
        # Create the first partition
        partition = self.create_partition()
        # Track the used mem for the current partition
        used_mem_bytes = 0
        for node in self.graph_module.graph.nodes:
            if node.op in {'call_module', 'call_method', 'call_function'}:
                # Find all its input nodes
                input_nodes: Dict[Node, None] = {}
                map_arg(node.args, lambda n: input_nodes.setdefault(n))
                map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
                # Calculate total size of related nodes
                total_size_of_input_nodes = 0
                for n in input_nodes:
                    # Make sure this node hasn't been in this partition yet
                    if n not in partition.nodes:
                        size_bytes = getattr(n, 'size_bytes', None)
                        if size_bytes:
                            total_size_of_input_nodes += size_bytes.output_size
                        else:
                            raise RuntimeError('node has no size_bytes attr')
                # Don't forget the op node itself
                size_bytes = getattr(node, 'size_bytes', None)
                if size_bytes:
                    total_size_of_input_nodes += size_bytes.total_size
                else:
                    raise RuntimeError('node has no size_bytes attr')
                # The current node with its inputs cannot fit into the current partition
                if used_mem_bytes + total_size_of_input_nodes > available_mem_bytes:
                    self.partition_to_used_mem_bytes[partition] = used_mem_bytes
                    partition = self.create_partition()
                    used_mem_bytes = 0
                    # The current node may be too large to fit into a whole new partition
                    if total_size_of_input_nodes > available_mem_bytes:
                        raise RuntimeError(node.target + 'is too large to fit into a device')
                # Add the current node into the current partition
                partition.add_node(node)
                # Add all input nodes if they are placeholders or constants
                for n in input_nodes:
                    if (n not in partition.nodes) and (n.op in {'placeholder', 'get_attr'}):
                        partition.add_node(n)
                used_mem_bytes = used_mem_bytes + total_size_of_input_nodes
        # Update used mem mapping for the last partition
        self.partition_to_used_mem_bytes[partition] = used_mem_bytes
        # Find parent partitions and child partitions for each partition.
        self.set_parents_and_children()
        # Combine small partitions
        self.combine_partitions_based_on_size(available_mem_bytes)
        # Reassign partition ids and update self.node_to_partitions.
        self.reorganize_partitions()
        return

    def do_partition(self) -> GraphModule:
        """Return a module with submodules (partitions)."""
        for node in self.graph_module.graph.nodes:
            if node.op == 'output':
                break
        module_with_submodules = split_module(
            self.graph_module,
            self.torch_module,
            lambda node: self.node_to_partitions[node][0]
        )
        return module_with_submodules

    def dump_partition_DAG(self, module_with_submodules: GraphModule) -> DAG:
        dag = DAG()
        for node in module_with_submodules.graph.nodes:
            if node.op == 'output':
                break
            if node.op in {'placeholder', 'get_attr'}:
                continue
            if node.target == operator.__getitem__:
                continue
            input_nodes : Dict[Node, None] = {}
            map_arg(node.args, lambda n: input_nodes.setdefault(n))
            map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
            # When a node has two or more output nodes,
            # it outputs its result to 'getitem' nodes.
            # Those 'getitem' nodes are the output node for this node.
            # Otherwise, the output node is this node itself.
            if len(node.users) > 1:
                output_nodes = list(node.users)
            else:
                output_nodes = [node]
            dag.create_node(node, list(input_nodes), output_nodes)
        return dag

    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        partition_id = len(self.partitions)
        assert isinstance(self.graph_module, GraphModule)
        partition = Partition(partition_id, self.graph_module)
        self.partitions.add(partition)
        return partition

    def combine_partitions_based_on_size(self, available_mem_bytes) -> None:
        """Combining small partitions together to keep as less partitions as possible.
           Here is an example of the algorithm to do this:
           Assume some partitions, we first sort them based on partiiton used memory size.
           [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]
           The available memory is 10.
           step 1: self.find_partition_to_combine_based_on_size()
           First, mark bfs level for each partition
           Second, look the smallest partition, partition_4: 10 - 1 = 9
           It means any partition has a used memory equal or less than 9 could combine this partition
           We go from the largest and selection partition_0.
           Check the bfs level for two partitions, if the level difference is less than 2,
           it can be combined.
           Then repeat step 1.
        """
        find_combination = True
        while find_combination:
            # Sort partitions based on memory size
            sorted_partitions = sorted(self.partition_to_used_mem_bytes.items(), key=lambda item: item[1])
            # Mark bfs level
            self.get_bfs_level_partition()
            find_combination = self.find_partition_to_combine_based_on_size(sorted_partitions, available_mem_bytes)
        return

    def find_partition_to_combine_based_on_size(
        self,
        sorted_partitions: List[Tuple[Partition, int]],
        available_mem_bytes: int
    ) -> bool:
        """step 1 in self.combine_partition_based_on_size()"""
        find_combination = False
        smallest_partition = sorted_partitions.pop(0)[0]
        left_mem = available_mem_bytes - self.partition_to_used_mem_bytes[smallest_partition]
        for t in sorted_partitions[::-1]:
            if t[1] <= left_mem and abs(smallest_partition.bfs_level - t[0].bfs_level) <= 1:
                self.combine_two_partitions(t[0], smallest_partition)
                find_combination = True
                break
        return find_combination

    def combine_two_partitions(self, partition_0: Partition, partition_1: Partition) -> None:
        """Given two partitions, combine them into a new one
        and remove the previous two partitions
        """
        partition = self.create_partition()
        partition.nodes = partition_0.nodes.union(partition_1.nodes)
        partition.parents = partition_0.parents.union(partition_1.parents)
        partition.children = partition_0.children.union(partition_1.children)
        partition.bfs_level = max(partition_0.bfs_level, partition_1.bfs_level)
        if partition_0 in partition.children:
            partition.children.remove(partition_0)
        if partition_0 in partition.parents:
            partition.parents.remove(partition_0)
        if partition_1 in partition.children:
            partition.children.remove(partition_1)
        if partition_1 in partition.parents:
            partition.parents.remove(partition_1)
        self.partition_to_used_mem_bytes[partition] = self.partition_to_used_mem_bytes[partition_0] + \
            self.partition_to_used_mem_bytes[partition_1]
        del self.partition_to_used_mem_bytes[partition_0]
        del self.partition_to_used_mem_bytes[partition_1]
        # Replace partition_0 and partition_1 with the new partition in children and parents
        for p in partition.parents:
            if partition_0 in p.children:
                p.children.remove(partition_0)
                p.children.add(partition)
            if partition_1 in p.children:
                p.children.remove(partition_1)
                p.children.add(partition)
        for p in partition.children:
            if partition_0 in p.parents:
                p.parents.remove(partition_0)
                p.parents.add(partition)
            if partition_1 in p.parents:
                p.parents.remove(partition_1)
                p.parents.add(partition_1)
        self.partitions.remove(partition_0)
        self.partitions.remove(partition_1)
        return

    def set_parents_and_children(self) -> None:
        # Go through all nodes in a partition.
        # If a node's user is in other partition,
        # then the other partition is this partition's children.
        # This partition is the other partition's parent
        for partition in self.partitions:
            for node in partition.nodes:
                # For each node in the current partition, find its users
                users = node.users
                for n in users:
                    # Find which the partition the user belongs to.
                    # Note that if the node itself is also belongs to that partition,
                    # that partition is not the child of the current partition
                    for p in self.partitions:
                        if p != partition and n in p.nodes and node not in p.nodes:
                            if p not in partition.children:
                                partition.add_child(p)
                                if partition not in p.parents:
                                    p.add_parent(partition)
        return

    def reorganize_partitions(self) -> None:
        # Rearrange partition ids
        for i, partition in enumerate(self.partitions):
            partition.partition_id = i
        # Update self.node_to_partitions accordingly
        for partition in self.partitions:
            for node in partition.nodes:
                if node not in self.node_to_partitions:
                    self.node_to_partitions[node] = [partition.partition_id]
                else:
                    self.node_to_partitions[node].append(partition.partition_id)
        return

    def get_bfs_level_partition(self) -> None:
        current_level: Set[Partition] = set()
        visited: Set[Partition] = set()
        for partition in self.partitions:
            # If a partition has no parent, it should be in root level
            if len(partition.parents) == 0:
                current_level.add(partition)
        next_level: Set[Partition] = set()
        level = 0
        # Start bfs
        while current_level:
            partition = current_level.pop()
            partition.bfs_level = level
            visited.add(partition)
            children = partition.children
            for child in children:
                if child not in next_level:
                    next_level.add(child)
            if not current_level:
                current_level = next_level.copy()
                next_level = set()
                level += 1
        return
