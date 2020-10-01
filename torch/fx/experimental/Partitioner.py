from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.experimental import GraphManipulation
from typing import Dict, List, Union

class DAGNode:
    """
    DAGNode class maintains useful information for a partition.
    It includes parent partitions' ids, child partitions' ids, inputs(Node) and outputs(Node) of the partition.
    """
    def __init__(
        self,
        partition_id: int,
        parents_ids: List[int],
        children_ids: List[int],
        input_nodes: List[Node],
        output_nodes: List[Node]
    ) -> None:
        self.partition_id = partition_id
        self.parents = parents_ids
        self.children = children_ids
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

    def __str__(self):
        line: str = 'partition id: ' + str(self.partition_id) + '\n'
        line += 'parent partitions:'
        for parent in self.parents:
            line += 'partition_' + str(parent) + ' '
        line += '\n'
        line += 'children partitions:'
        for child in self.children:
            line += 'partition_' + str(child) + ' '
        line += '\n'
        line += 'input nodes: '
        for node in self.input_nodes:
            line += '(' + node.name + ':' + node.op + ') '
        line += '\n'
        line += 'output nodes: '
        output_nodes = self.output_nodes
        for node in output_nodes:
            line += '(' + node.name + ':' + node.op + ') '
        return line

class DAG:
    """DAG class contains all the DAG nodes"""
    def __init__(self) -> None:
        self.nodes: List[DAGNode] = []

    def create_node(
        self,
        partition_id: int,
        parents_ids: List[int],
        children_ids: List[int],
        input_nodes: List[Node],
        output_nodes: List[Node]
    ) -> None:
        node = DAGNode(partition_id, parents_ids, children_ids, input_nodes, output_nodes)
        self.nodes.append(node)

class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulation the partition.
    """
    def __init__(self, partition_id: int, fx_module: GraphModule) -> None:
        self.graph_module = fx_module
        self.nodes: List[Node] = []
        self.partition_id = partition_id
        self.parents: List[Partition] = []
        self.children: List[Partition] = []

    def add_node(self, node: Node) -> None:
        """Append a new node into the partition."""
        self.nodes.append(node)

    def add_parent(self, partition: 'Partition') -> None:
        self.parents.append(partition)

    def add_child(self, partition: 'Partition') -> None:
        self.children.append(partition)

    def get_children(self) -> List['Partition']:
        return self.children

    def get_parents(self) -> List['Partition']:
        return self.parents

    def get_input_nodes(self) -> List[Node]:
        """Input nodes are coming from two places:
        placeholder and output from its parents output.
        """
        input_nodes: List[Node] = []
        for node in self.nodes:
            if node.op == 'placeholder':
                input_nodes.append(node)
        for partition in self.parents:
            input_nodes += partition.get_output_nodes()
        return input_nodes

    def get_output_nodes(self) -> List[Node]:
        """Output nodes are the nodes that without any user inside this partition."""
        output_nodes: List[Node] = []
        for node in self.nodes:
            index = self.graph_module.graph.nodes.index(node)
            user_indexes = GraphManipulation.get_all_users_of(self.graph_module, index)
            user_nodes = {self.graph_module.graph.nodes[i] for i in user_indexes}
            # check if user nodes has an intersection with self.nodes
            if not set(self.nodes).intersection(user_nodes):
                output_nodes.append(node)
        return output_nodes

    def __str__(self) -> str:
        return str(self.partition_id)

class Partitioner:
    """A graph module may not fit into one device.
    Partitioner class helps cut one graph into subgraphs (partitions),
    so that each partition could fit into a different device.
    """
    def __init__(self) -> None:
        """
        After a partitioner is created,
        it first check if multiple types of devices (backends) are involved.
        So far, we assume that there is only one backend and one device with unlimited memory
        """
        self.partitions: List[Partition] = []
        self.devices: List[Dict[str, Union[str, float]]] = []
        self.node_to_partitions: Dict[Node, List[int]] = {}

    def partition_graph(self, fx_module: GraphModule, devices: List[Dict[str, Union[str, float]]]) -> DAG:
        """
        Given the fx_module and devices, find the partitions, do the partitions,
        and then return a dictionary representing the DAG of partitions
        """
        self.graph_module = fx_module
        self.devices = devices
        # Create a dummy partition as root.
        self.root_partition = self.create_partition()
        # So far, the whole could fit into one device since we assume the memory is unlimited.
        # TODO: Check the available memory size for each device and see if one device is enough or multiple partitions are needed.
        self.find_single_partition()
        self.do_partition()
        dag = self.dump_partition_DAG()
        return dag

    def find_single_partition(self) -> None:
        """Only one partition (one graph on one device)."""
        partition_0 = self.create_partition()
        for i, node in enumerate(self.graph_module.graph.nodes):
            self.node_to_partitions[node] = [partition_0.partition_id]
            partition_0.add_node(node)
        # Connect the partition to the root.
        self.root_partition.add_child(partition_0)
        partition_0.add_parent(self.root_partition)
        return

    def do_partition(self) -> None:
        """Mark the partition on each node in the fx_module."""
        for node in self.graph_module.graph.nodes:
            node.partition_ids = self.node_to_partitions[node]
        return

    def dump_partition_DAG(self) -> DAG:
        dag = DAG()
        for i, partition in enumerate(self.partitions):
            input_nodes = partition.get_input_nodes()
            output_nodes = partition.get_output_nodes()
            parents = partition.parents
            parent_ids = [self.partitions.index(p) for p in parents]
            children = partition.children
            children_ids = [self.partitions.index(c) for c in children]
            dag.create_node(i, parent_ids, children_ids, input_nodes, output_nodes)
        return dag

    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        partition_id = len(self.partitions)
        assert isinstance(self.graph_module, GraphModule)
        partition = Partition(partition_id, self.graph_module)
        self.partitions.append(partition)
        return partition
