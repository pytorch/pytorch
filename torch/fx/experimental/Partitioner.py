from torch.fx.graph_module import GraphModule
from torch.fx.node import Node, map_arg
from typing import Dict, List, Set, NamedTuple, Tuple
import torch
from torch.fx.experimental.subgraph_creation_example import split_module
import operator
from torch.fx.experimental.partitioner_utils import Partition, \
    Device, PartitionerConfig, get_partition_to_latency_mapping,\
    get_latency_of_partitioned_graph, NodeLatency, get_extra_size_of

class DAGNode():
    """
    DAGNode class maintains useful information for a partition (submodule).
    inputs(submodule node) and outputs(submodule node).
    """
    def __init__(
        self,
        submodule_node: Node,
        input_nodes: List[Node],
        output_nodes: List[Node],
        logical_device_ids: List[int],
        size_bytes: int
    ) -> None:
        self.submodule_node: Node = submodule_node
        self.input_nodes: List[Node] = input_nodes
        self.output_nodes: List[Node] = output_nodes
        self.logical_device_ids: List[int] = logical_device_ids
        self.size_bytes = size_bytes

    def __str__(self) -> str:
        return str(self.submodule_node)

class DAG:
    """DAG class contains all the DAG nodes"""
    def __init__(self) -> None:
        self.nodes: List[DAGNode] = []

    def create_node(
        self,
        submodule_node: Node,
        input_nodes: List[Node],
        output_nodes: List[Node],
        logical_devices: List[int],
        size_bytes: int
    ) -> None:
        node = DAGNode(submodule_node, input_nodes, output_nodes, logical_devices, size_bytes)
        self.nodes.append(node)

class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new graph module
    """
    dag: DAG
    module_with_submodules: GraphModule

"""Followings are some helper functions for partition manipulation"""

def combine_two_partitions(
    partition_0: Partition,
    partition_1: Partition,
    partitions: List[Partition]
) -> None:
    """Given a list of partitions and its two partitions,
       combine these two partitions into a new one appending to the partitions
       and remove the previous two partitions from the list of partitions
    """
    partition = Partition(len(partitions))
    partition.nodes = partition_0.nodes.union(partition_1.nodes)
    partition.recalculate_mem_size()
    partitions.append(partition)
    partitions.remove(partition_0)
    partitions.remove(partition_1)
    # Reorganize partitions
    reorganize_partitions(partitions)
    return

def set_parents_and_children(partitions: List[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition
    """
    # Go through all nodes in a partition.
    # If a node's user is in other partition,
    # then the other partition is this partition's children.
    # This partition is the other partition's parent
    for partition in partitions:
        partition.children = set()
        partition.parents = set()
    for partition in partitions:
        for node in partition.nodes:
            # For each node in the current partition, find its users
            users = node.users
            for n in users:
                # Find which the partition the user belongs to.
                # Note that if the node itself is also belongs to that partition,
                # that partition is not the child of the current partition
                for p in partitions:
                    if p != partition and n in p.nodes and node not in p.nodes:
                        partition.children.add(p)
                        p.parents.add(partition)
    return

def reorganize_partitions(partitions: List[Partition]) -> None:
    """Given a list of partitions, reorganzie partiton id,
    its parents and its children for each partition
    """
    # Rearrange partition ids
    for i, partition in enumerate(partitions):
        partition.partition_id = i
    set_parents_and_children(partitions)
    return

def get_bfs_level_partition(partitions: List[Partition]) -> None:
    """Given a list of partitions,
       mark the bfs level for each partition
    """
    current_level: Set[Partition] = set()
    visited: Set[Partition] = set()
    for partition in partitions:
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

def get_node_to_partition_mapping(partitions: List[Partition]) -> Dict[Node, int]:
    """Given a list of partitions,return node to partition mapping
    """
    node_to_partition: Dict[Node, int] = {}
    for partition in partitions:
        for node in partition.nodes:
            node_to_partition[node] = partition.partition_id
    return node_to_partition

def get_device_to_partitions_mapping(partitions: List[Partition], devices: List[Device]):
    """Given a list of partitions and a list of devices,
        map each partition into a device.
    """
    def calculate_extra_mem_bytes_needed_for(partition: Partition, partitions: List[Partition]):
        all_nodes: Set[Node] = set()
        for p in partitions:
            all_nodes = all_nodes.union(p.nodes)
        extra_size_needed = 0
        for node in partition.nodes:
            if node in all_nodes or node.op in {'placeholder', 'get_attr'}:
                continue
            else:
                extra_size_needed += get_extra_size_of(node, all_nodes)
        return extra_size_needed

    def find_device_for(partition: Partition):
        """Given a partition, find a logical device for the partition
            The algorithm is that:
            #1. sort all the devices based on left mem size
            #2. put the partition on the device that has just enought mem
            for that partition
        """
        for d in device_to_left_mem_bytes:
            extra_size_needed = calculate_extra_mem_bytes_needed_for(partition, device_to_partitions[d])
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False
    # logical id to device
    logical_id_to_device: Dict[int, Device] = {}
    # Track partitions on device
    device_to_partitions: Dict[Device, List[Partition]] = {}
    # Track device's left mem size
    device_to_left_mem_bytes: Dict[Device, int] = {}
    for d in devices:
        logical_id_to_device[d.logical_id] = d
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes
    # Deal with the partitions that have a device
    # Find all no device partitions
    no_device_partitions = []
    for partition in partitions:
        if partition.logical_device_ids != []:
            logical_id = partition.logical_device_ids[0]
            device = logical_id_to_device[logical_id]
            device_to_partitions[device] = [partition]
            device_to_left_mem_bytes[device] = d.available_mem_bytes - partition.used_mem_bytes
        else:
            no_device_partitions.append(partition)
    # Find device for each no device partition
    found_device = True
    for partition in no_device_partitions:
        device_to_left_mem_bytes = {
            d: left_mem_bytes for d, left_mem_bytes
            in sorted(device_to_left_mem_bytes.items(), key=lambda item: item[1])
        }
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device

class Partitioner:
    """A graph module may not fit into one device.
    Partitioner class helps cut one graph into subgraphs (partitions),
    so that each partition could fit into a different device.
    The main function of this class is self.partition_graph.
    It will partition the graph based on the scheme specified in partition_config
    A DAG structure is returned
    along with a new graph module with partitions as submodule nodes.
    """
    def __init__(self) -> None:
        self.partitions: List[Partition] = []
        self.node_to_partition: Dict[Node, int] = {}
        self.devices: List[Device] = []

    def partition_graph(
        self,
        fx_module: GraphModule,
        torch_module: torch.nn.Module,
        partitioner_config: PartitionerConfig
    ) -> PartitionResult:
        """
        Given the fx module, torch module and partitioner_config,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
        self.graph_module = fx_module
        self.torch_module = torch_module
        self.devices = partitioner_config.devices
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
        device_with_max_mem = max(self.devices, key=lambda d: d.available_mem_bytes)
        if total_size_of_graph <= device_with_max_mem.available_mem_bytes:
            self.find_single_partition(total_size_of_graph)
        elif total_size_of_graph > sum([d.available_mem_bytes for d in self.devices]):
            raise RuntimeError('Devices have no enough memory for the module')
        else:
            if partitioner_config.is_sparse_nn:
                if not all(device.available_mem_bytes == available_mem_bytes for device in self.devices):
                    raise RuntimeError('All devices must have same memory size!')
                # sparse_nn_partition only support same memory size
                # TODO: add different size support for sparse_nn_partition
                self.sparse_nn_partition(available_mem_bytes)
            elif partitioner_config.is_cost_aware:
                self.cost_aware_partition(
                    partitioner_config.transfer_rate_bytes_per_sec,
                    partitioner_config.node_to_latency_mapping
                )
            else:
                self.size_based_partition(available_mem_bytes)
        module_with_submodules = self.do_partition()
        # The DAG contains DAGNodes with info of each partition's input nodes, output nodes
        # and how partitions are connected.
        dag = self.dump_dag(module_with_submodules)
        ret = PartitionResult(dag, module_with_submodules)
        return ret

    def find_single_partition(self, total_size_of_graph) -> None:
        """Only one partition (one graph on one device)."""
        partition_0 = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op == 'output':
                break
            partition_0.nodes.add(node)
        partition_0.used_mem_bytes = total_size_of_graph
        partition_0.logical_device_ids = [0]
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def size_based_partition(self, available_mem_bytes: int) -> None:
        """This method is to partition the graph based on memory size.
           It uses greedy approach. The result may not be the best.
           The basic idea is:
           Step 1:
           Find a device which has enough memory to fit the first node, create a empty partition
           with the size of that device.
           Then keep adding the following nodes into the partition until the partition is full.
           Step 2:
           Repeat Step 1 until no device left
           Step 3:
           If some nodes are left, create a partition for each left node (single node partition).
           and then try to map those partitions into logical devices with non single node partitions.
        """
        def find_device_based_on_size(node) -> Device:
            """Given a node, this function is to find a logical device
               that could fit the node.
            """
            mem_size_needed = get_extra_size_of(node, set())
            device = Device('', -1, -1)
            for d in self.devices:
                if d not in occupied_devices and d.available_mem_bytes >= mem_size_needed:
                    device = d
                    break
            if device.available_mem_bytes < 0:
                raise RuntimeError(str(node) + 'is too large to fit any device')
            occupied_devices.append(device)
            return device

        # Track partition and its left mem size
        partition_to_left_mem_bytes: Dict[Partition, int] = {}
        # Track all the devices that have been used
        occupied_devices: List[Device] = []
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {'call_module', 'call_method', 'call_function'}:
                # Check if there are devices left
                if len(self.partitions) <= len(self.devices):
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    # Check if the current partition is the very first partition
                    if partition.used_mem_bytes == 0:
                        # Find a device to fit the first node, return available mem size
                        device = find_device_based_on_size(node)
                        occupied_devices.append(device)
                        # Update partition and its left mem size
                        partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                        # Update available mem for the current partitio
                        partition.logical_device_ids.append(device.logical_id)
                    else:
                        # The current partition is not the first partition
                        # Check if the current node can fit into this partition
                        if partition_to_left_mem_bytes[partition] < total_size_of_input_nodes:
                            # Check if no device is left
                            if len(self.partitions) == len(self.devices):
                                # No device left, all the partitions before are non single node partitions
                                non_single_node_partitions = self.partitions[:]
                                # Create the first single node partition for the current node
                                self.create_single_node_partition(node)
                                continue
                            # Some devices are still left
                            device = find_device_based_on_size(node)
                            partition = self.create_partition()
                            total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                            partition_to_left_mem_bytes[partition] = device.available_mem_bytes
                            partition.logical_device_ids.append(device.logical_id)
                    partition.add_node(node)
                    partition_to_left_mem_bytes[partition] -= total_size_of_input_nodes
                    partition.used_mem_bytes += total_size_of_input_nodes
                # No device left, create single node partitions
                else:
                    self.create_single_node_partition(node)
        reorganize_partitions(self.partitions)
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        # Mapping all partitions into device
        found_partition_to_device_mapping = get_device_to_partitions_mapping(self.partitions, self.devices)
        if not found_partition_to_device_mapping:
            raise RuntimeError("Cannot Get a Valid Partition to Logical Device Mapping")
        return

    def do_partition(self) -> GraphModule:
        """Return a module with submodules (partitions)."""
        module_with_submodules = split_module(
            self.graph_module,
            self.torch_module,
            lambda node: self.node_to_partition[node]
        )
        return module_with_submodules

    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
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
            partition_id = int(node.name.rsplit('_', 1)[-1])
            device_ids = self.partitions[partition_id].logical_device_ids
            size_bytes = self.partitions[partition_id].used_mem_bytes
            dag.create_node(node, list(input_nodes), output_nodes, device_ids, size_bytes)
        return dag

    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        partition_id = len(self.partitions)
        partition = Partition(partition_id)
        self.partitions.append(partition)
        return partition

    def create_single_node_partition(self, node):
        """Create a partition for a single node
        """
        partition = self.create_partition()
        partition.add_node(node)
        partition.recalculate_mem_size()
        return

    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        """This method partition a sparse nn module.
           It first traverse all the nodes and do the partitions based on memory size.
           If the current partition has no enough memory left for a new op node
           (call_module, call_method, call_function), a new partition is created.
           Different from size_based_partition, when traversing cross the boundary between
           non-embedding nodes and embedding nodes, a new partition is created regardlessly.
           For example, if the current node is a non-embedding node but the next node is an
           embedding node, a new partition is created for the next node.
           After the partition, the partitions are combined as much as possible.
           The rule is that a non-embedding partition only
           combines with another non-embedding one.
           So as the embedding partitions.
        """
        def combine_partitions_based_on_size(partitions: List[Partition], available_mem_bytes: int) -> None:
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
                sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
                # Mark bfs level
                get_bfs_level_partition(self.partitions)
                find_combination, partitions = \
                    find_partition_to_combine_based_on_size(
                        sorted_partitions,
                        available_mem_bytes,
                        partitions
                    )
            return

        def calculate_mem_bytes_needed(p1, p2):
            """Given two partitions, calculate how many mem bytes
               are needed if two partitions are combined
            """
            nodes = p1.nodes.union(p2.nodes)
            mem_bytes_needed = 0
            for node in nodes:
                mem_bytes_needed += get_extra_size_of(node, nodes)
            return mem_bytes_needed

        def find_partition_to_combine_based_on_size(
            sorted_partitions: List[Partition],
            available_mem_bytes: int,
            partitions: List[Partition]
        ) -> Tuple[bool, List[Partition]]:
            """step 1 in combine_partition_based_on_size()"""
            find_combination = False
            smallest_partition = sorted_partitions.pop(0)
            for p in sorted_partitions[::-1]:
                if abs(smallest_partition.bfs_level - p.bfs_level) <= 1:
                    # Calculate how many bytes needed if combined
                    mem_bytes_needed = calculate_mem_bytes_needed(p, smallest_partition)
                    if mem_bytes_needed <= available_mem_bytes:
                        combine_two_partitions(p, smallest_partition, self.partitions)
                        partitions.remove(smallest_partition)
                        partitions.remove(p)
                        partitions.append(self.partitions[-1])
                        find_combination = True
                        break
            return find_combination, partitions

        def reset_partition_in_sparse_nn(partition, new_partition=True):
            if in_embedding_region:
                embedding_partitions.append(partition)
            else:
                non_embedding_partitions.append(partition)
            if new_partition:
                partition = self.create_partition()
                partition.left_mem_bytes = available_mem_bytes
                return partition
            return None

        def is_embedding_node(node: Node) -> bool:
            """Check if a node is an embedding node"""
            if node.op == 'call_module':
                submodule = self.graph_module
                for atom in str(node.target).split('.'):
                    if not hasattr(submodule, atom):
                        raise RuntimeError(f'Module {submodule} has no attribute {atom}')
                    submodule = getattr(submodule, atom)
                    if 'Embedding' in str(submodule):
                        return True
            return False

        # Track embedding partitons and non-embedding partitions separately
        embedding_partitions: List[Partition] = []
        non_embedding_partitions: List[Partition] = []
        # A Flag to check the boundary
        in_embedding_region: bool = False
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {'call_module', 'call_method', 'call_function'}:
                # Check if crossing the boundary between embedding nodes and non embedding nodes
                if is_embedding_node(node) != in_embedding_region:
                    # Crossing the boundary
                    # Check if the current partition is an empty partition
                    if partition.used_mem_bytes != 0:
                        # The current partition isn't an empty partition. Create a new one.
                        partition = reset_partition_in_sparse_nn(partition)
                    in_embedding_region = not in_embedding_region
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if total_size_of_input_nodes + partition.used_mem_bytes > available_mem_bytes:
                    partition = reset_partition_in_sparse_nn(partition)
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    if total_size_of_input_nodes > available_mem_bytes:
                        raise RuntimeError(node.target + 'is too large to fit into a device')
                partition.add_node(node)
                partition.used_mem_bytes += total_size_of_input_nodes
        reset_partition_in_sparse_nn(partition, new_partition=False)
        # Set parents and children for partitions
        set_parents_and_children(self.partitions)
        # Combining non-embedding partitions
        combine_partitions_based_on_size(non_embedding_partitions, available_mem_bytes)
        # Combining embedding partitions
        combine_partitions_based_on_size(embedding_partitions, available_mem_bytes)
        total_size_of_non_embedding_partitions = 0
        for partition in non_embedding_partitions:
            total_size_of_non_embedding_partitions += partition.used_mem_bytes
        # Check if devices are enough for all partitions
        if len(embedding_partitions) > len(self.devices):
            msg = 'Need ' + str(len(embedding_partitions)) + ' devices, but only ' \
                + str(len(self.devices)) + ' provided'
            raise RuntimeError(msg)
        occupied_devices = []
        for i, partition in enumerate(embedding_partitions):
            # Check if all non-embedding partitions can fit into embedding partition devices
            if total_size_of_non_embedding_partitions + partition.used_mem_bytes > available_mem_bytes:
                raise RuntimeError(
                    'partition_' +
                    str(partition.partition_id) +
                    '(embedding partition) and non embedding partitions can not fit into one device'
                )
            else:
                # Add logical device to the partition
                partition.logical_device_ids = [self.devices[i].logical_id]
                occupied_devices.append(self.devices[i].logical_id)
        # Add logical devices to the non_embedding_partitions
        for partition in non_embedding_partitions:
            partition.logical_device_ids = occupied_devices
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def cost_aware_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: Dict[Node, NodeLatency]
    ) -> None:
        """This method is to partition the fx module based on the cost.
           The cost is the total latency of running the whole graph.
           In partitioner_utils.py, the cost model is built.
           The algorithm is:
           #1. At every begining, each node is a partition.
               Then we map all the partitions to the devices
               and calculate the cost
           #2. Then try to pre-combine any two of the partitions if the two
               partitions can be combined.
               (the bfs level is less than 2 or two partitions are connected and
               can find partition to device mapping)
               See if any partition pair could reduce the current cost.
               Choose the pair that shows the minimum cost and then combine them
           #3. Repeat #2 until the cost cannot be reduced.
        """

        def try_combining_partitions(
            p0_index,
            p1_index,
            partitions
        ) -> float:
            """Given two partitions and a list of partitions, try to combine these two partitions
               and see what is the cost of the modified partition list
            """
            p0 = partitions[p0_index]
            p1 = partitions[p1_index]
            """If two partitions' bfs level are less than 2 or two partitions are connected to each other,
               then they can be combined
            """
            if (abs(p0.bfs_level - p1.bfs_level) <= 1) or (p0 in p1.parents) or p0 in (p1.children):
                combine_two_partitions(p0, p1, partitions)
                # Check if the modified partition list can be mapped to devices after combination
                found_deivce = get_device_to_partitions_mapping(partitions, self.devices)
                if not found_deivce:
                    return float('inf')
                # Calculate the new cost
                partition_to_latency_mapping = get_partition_to_latency_mapping(partitions, node_to_latency_mapping)
                cost = get_latency_of_partitioned_graph(partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
                return cost
            # If two partition can not be combined, the cost is inf
            return float('inf')

        def search_combination(
            transfer_rate_bytes_per_sec,
            node_to_latency_mapping
        ) -> bool:
            """Given transfer rate between partitions and each node's latency,
               find two partitions to combine so the cost of the partitions can
               be reduced.
               The algorithm is :
               1. Going through all the partition pairs and see
               if the pair of partitions can be combined.
               2. If they are combined, the cost is calculated.
               3. Select the minimum cost and combine its cooresponding partition pair
            """
            partition_to_latency_mapping = get_partition_to_latency_mapping(self.partitions, node_to_latency_mapping)
            cost = get_latency_of_partitioned_graph(self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec)
            if len(self.partitions) == 1:
                return False
            partition_pair: List[int] = []
            for i in range(len(self.partitions) - 1):
                for j in range(i + 1, len(self.partitions)):
                    # Try to combine the partition pair
                    # and see the new cost after combination
                    new_cost = try_combining_partitions(
                        i,
                        j,
                        self.partitions[:]
                    )
                    if new_cost <= cost:
                        partition_pair = [i, j]
                        cost = new_cost
            # If a partition pair is found, combine them
            if len(partition_pair) != 0:
                p0 = self.partitions[partition_pair[0]]
                p1 = self.partitions[partition_pair[1]]
                combine_two_partitions(p0, p1, self.partitions)
                get_bfs_level_partition(self.partitions)
                get_device_to_partitions_mapping(self.partitions, self.devices)
                return True
            return False

        for node in self.graph_module.graph.nodes:
            if node.op not in {'placeholder', 'get_attr', 'output'}:
                self.create_single_node_partition(node)
        # Set up parent partitions and children partitions for each partition
        set_parents_and_children(self.partitions)
        # Get bfs level for each partition
        get_bfs_level_partition(self.partitions)
        find_combination = True
        while find_combination:
            # Search for a pair partition to generate the minimum new cost,
            # then combine them
            find_combination = search_combination(
                transfer_rate_bytes_per_sec,
                node_to_latency_mapping
            )
        # Make sure all partitions are set up correctly.
        reorganize_partitions(self.partitions)
        # Set up node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return
