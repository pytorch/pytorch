# mypy: allow-untyped-defs
import operator
from collections import deque
from typing import NamedTuple

import torch
from torch.fx.experimental.partitioner_utils import (
    Device,
    get_extra_size_of,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
    NodeLatency,
    Partition,
    PartitionerConfig,
    PartitionMode,
)
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node
from torch.fx.passes.graph_manipulation import get_size_of_all_nodes
from torch.fx.passes.split_module import split_module


class DAGNode:
    """DAGNode class maintains useful information for a partition (submodule),
    and its input submodules and output submodules.
    """

    def __init__(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_device_ids: list[int],
        size_bytes: int,
    ) -> None:
        self.submodule_node: Node = submodule_node
        self.input_nodes: list[Node] = input_nodes
        self.output_nodes: list[Node] = output_nodes
        self.logical_device_ids: list[int] = logical_device_ids
        self.size_bytes = size_bytes

    def __str__(self) -> str:
        return str(self.submodule_node)


class DAG:
    """DAG class contains all the DAG nodes"""

    def __init__(self) -> None:
        self.nodes: list[DAGNode] = []

    def create_node(
        self,
        submodule_node: Node,
        input_nodes: list[Node],
        output_nodes: list[Node],
        logical_devices: list[int],
        size_bytes: int,
    ) -> None:
        node = DAGNode(
            submodule_node, input_nodes, output_nodes, logical_devices, size_bytes
        )
        self.nodes.append(node)


class PartitionResult(NamedTuple):
    """NameTuple used for returning DAG and a new fx module"""

    dag: DAG
    module_with_submodules: GraphModule


"""Followings are some helper functions for partition manipulation"""


def reset_partition_device(partitions):
    for partition in partitions:
        partition.logical_device_ids = []


def combine_two_partitions(
    partition_0: Partition, partition_1: Partition, partitions: list[Partition]
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
    reorganize_partitions(partitions)
    return


def set_parents_and_children(partitions: list[Partition]) -> None:
    """Given a list of partitions, mark parents and children for each partition"""
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
                # Find which the partition the user node belongs to.
                # Note that if the node itself is also belongs to that partition,
                # that partition is not the child of the current partition
                for p in partitions:
                    if p != partition and n in p.nodes and node not in p.nodes:
                        partition.children.add(p)
                        p.parents.add(partition)
    return


def reorganize_partitions(partitions: list[Partition]) -> None:
    """Given a list of partitions, reorganize partition id,
    its parents and its children for each partition
    """
    # Rearrange partition ids
    for i, partition in enumerate(partitions):
        partition.partition_id = i
    set_parents_and_children(partitions)
    return


def get_bfs_level_partition(partitions: list[Partition]) -> None:
    """Given a list of partitions,
    mark the bfs level for each partition
    """
    current_level: set[Partition] = set()
    visited: set[Partition] = set()
    for partition in partitions:
        # If a partition has no parent, it should be in root level
        if len(partition.parents) == 0:
            current_level.add(partition)
    next_level: set[Partition] = set()
    level = 0
    # bfs
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


def get_node_to_partition_mapping(partitions: list[Partition]) -> dict[Node, int]:
    """Given a list of partitions,return node to partition mapping"""
    node_to_partition: dict[Node, int] = {}
    for partition in partitions:
        for node in partition.nodes:
            node_to_partition[node] = partition.partition_id
    return node_to_partition


def get_logical_id_to_device(devices: list[Device]) -> dict[int, Device]:
    """Get a mapping from device logical ID to Device object."""
    logical_id_to_device: dict[int, Device] = {}
    for d in devices:
        logical_id_to_device[d.logical_id] = d
    return logical_id_to_device


def get_device_partition_stats(
    partitions: list[Partition], devices: list[Device]
) -> tuple[dict[Device, list[Partition]], dict[Device, int], list[Partition]]:
    """Given a list of partitions and a list of devices, returns:
    1. A mapping from device to partitions on it;
    2. A mapping from device to its remaining memory size;
    3. A list of partitions that do not have a device.
    """
    # logical id to device
    logical_id_to_device = get_logical_id_to_device(devices)
    # Track partitions on device
    device_to_partitions: dict[Device, list[Partition]] = {}
    # Track device's left mem size
    device_to_left_mem_bytes: dict[Device, int] = {}
    for d in devices:
        device_to_partitions[d] = []
        device_to_left_mem_bytes[d] = d.available_mem_bytes

    # Deal with the partitions that already have a device
    # and also collect all partitions without a device (no_device_partitions)
    no_device_partitions = []
    for partition in partitions:
        if partition.logical_device_ids != []:
            for logical_id in partition.logical_device_ids:
                device = logical_id_to_device[logical_id]
                device_to_partitions[device].append(partition)
                device_to_left_mem_bytes[device] -= partition.used_mem_bytes
        else:
            no_device_partitions.append(partition)

    return (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    )


def get_device_to_partitions_mapping(
    partitions: list[Partition], devices: list[Device]
):
    """Given a list of partitions and a list of devices,
    map each partition into a device.
    """

    def calculate_extra_mem_bytes_needed_for(
        partition: Partition, partitions: list[Partition]
    ):
        all_nodes: set[Node] = set()
        for p in partitions:
            all_nodes = all_nodes.union(p.nodes)
        if len(all_nodes) == 0:
            return partition.used_mem_bytes
        all_nodes = all_nodes.union(partition.nodes)
        extra_size_needed = 0
        for node in partition.nodes:
            extra_size_needed += get_extra_size_of(node, all_nodes)
        return extra_size_needed

    def find_device_for(partition: Partition):
        """Given a partition, find a logical device for the partition
        The algorithm is to put the partition on the device
        that has just enough mem left for that partition.
        device_to_left_mem_bytes is a dictionary between device and its left mem size
        sorted by its left mem size
        """
        for d in device_to_left_mem_bytes:
            extra_size_needed = calculate_extra_mem_bytes_needed_for(
                partition, device_to_partitions[d]
            )
            if extra_size_needed < device_to_left_mem_bytes[d]:
                device_to_partitions[d].append(partition)
                partition.logical_device_ids.append(d.logical_id)
                device_to_left_mem_bytes[d] -= extra_size_needed
                return True
        return False

    (
        device_to_partitions,
        device_to_left_mem_bytes,
        no_device_partitions,
    ) = get_device_partition_stats(partitions, devices)

    # Find devices for all the partitions without a device
    found_device = True
    for partition in no_device_partitions:
        device_to_left_mem_bytes = dict(
            sorted(device_to_left_mem_bytes.items(), key=operator.itemgetter(1))
        )
        found_device = find_device_for(partition)
        if not found_device:
            break
    return found_device


def check_dependency(partition):
    """Given a partition,check if there is a circular dependency on
    this partition using bfs
    """
    visited: set[Partition] = {partition}
    queue: deque[Partition] = deque([partition])
    while queue:
        p = queue.popleft()
        for child in p.children:
            if child == partition:
                return True
            else:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
    return False


class Partitioner:
    """A fx module may not fit into one device.
    Partitioner class helps partition one fx module into submodules (partitions),
    so that the submodules can be executed crossing different accelerators.
    The main function of this class is self.partition_graph.
    It partitions the fx module based on the scheme specified in partition_config
    A DAG structure is returned
    along with a new fx module with submodule nodes.
    """

    def __init__(self) -> None:
        self.partitions: list[Partition] = []
        self.node_to_partition: dict[Node, int] = {}
        self.devices: list[Device] = []

    def partition_graph(
        self,
        fx_module: GraphModule,
        torch_module: torch.nn.Module,
        partitioner_config: PartitionerConfig,
    ) -> PartitionResult:
        """Given the fx module, torch module and partitioner_config,
        find the partitions, do the partitions,
        and then return a DAG and a new fx module with submodule nodes (partitions)
        """
        self.graph_module = fx_module
        self.torch_module = torch_module
        self.devices = partitioner_config.devices
        if len(self.devices) == 0:
            raise RuntimeError("No devices")
        # Tag the size in bytes to all nodes in the graph_module.
        get_size_of_all_nodes(self.graph_module)
        # Check if there are op nodes in the fx module
        nodes = self.graph_module.graph.nodes
        if all(node.op in {"placeholder", "get_attr", "output"} for node in nodes):
            raise RuntimeError("No Partition since no operations in the module")
        # Calculate total size of the fx module
        total_size_of_graph = 0
        for node in nodes:
            if node.op == "output":
                break
            total_size_of_graph += node.size_bytes.total_size
        # Find the device with the max mem size
        device_with_max_mem = max(self.devices, key=lambda d: d.available_mem_bytes)
        # AOT based partition
        if partitioner_config.mode == PartitionMode.aot_based:
            self.aot_based_partition(
                partitioner_config.node_to_partition_mapping,
                partitioner_config.partition_to_logical_device_mapping,
            )
        # Single partition if the whole module can be fit into one device
        elif total_size_of_graph <= device_with_max_mem.available_mem_bytes:
            self.find_single_partition(
                total_size_of_graph, logical_device_id=device_with_max_mem.logical_id
            )
        elif total_size_of_graph > sum(d.available_mem_bytes for d in self.devices):
            raise RuntimeError("Devices have no enough memory for the module")
        else:
            # Sparse nn based partition
            if partitioner_config.mode == PartitionMode.sparse_nn:
                available_mem_bytes = self.devices[0].available_mem_bytes
                if not all(
                    device.available_mem_bytes == available_mem_bytes
                    for device in self.devices
                ):
                    raise RuntimeError("All devices must have same memory size!")
                # sparse_nn_partition only support same memory size
                # TODO: add different size support for sparse_nn_partition
                self.sparse_nn_partition(available_mem_bytes)
            # Cost aware partition
            elif partitioner_config.mode == PartitionMode.cost_aware:
                self.cost_aware_partition(
                    partitioner_config.transfer_rate_bytes_per_sec,
                    partitioner_config.node_to_latency_mapping,
                )
            # KL based partition
            elif partitioner_config.mode == PartitionMode.kl_based:
                self.kl_based_partition(
                    partitioner_config.transfer_rate_bytes_per_sec,
                    partitioner_config.node_to_latency_mapping,
                )
            else:
                self.size_based_partition()

        # Saturate host if possible.
        if partitioner_config.saturate_host:
            self.saturate_host()

        # Partition the graph module based on the partition assignment.
        module_with_submodules = self.do_partition()

        # The DAG contains DAGNodes with info of each partition's input nodes, output nodes
        # and how partitions are connected.
        dag = self.dump_dag(module_with_submodules)
        ret = PartitionResult(dag, module_with_submodules)
        return ret

    def find_single_partition(
        self, total_size_of_graph, logical_device_id: int = 0
    ) -> None:
        """Fit the whole fx module into one device"""
        partition_0 = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op == "output":
                # Skip the output node, but there can
                # be nodes after the output in certain cases.
                continue
            partition_0.nodes.add(node)
        partition_0.used_mem_bytes = total_size_of_graph
        partition_0.logical_device_ids = [logical_device_id]
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def size_based_partition(self) -> None:
        """This method is to partition the fx module based on memory size.
        It uses greedy approach. The result may not be the best.
        The basic idea is:
        Step 1:
        Find a device which has enough memory to fit the current node, create a empty partition
        with the size of that device.
        Then keep adding the following nodes into the partition until the partition is full.
        Step 2:
        Repeat Step 1 until no device left
        Step 3:
        If some nodes are left, create a partition for each left node (single node partition).
        and then try to map those partitions into logical devices with enough mem left.
        """

        def find_device_based_on_size(node) -> Device:
            """Given a node, this function is to find a logical device
            that could fit the node.
            """
            mem_size_needed = get_extra_size_of(node, set())
            device = Device("", -1, -1)
            for d in self.devices:
                if (
                    d not in occupied_devices
                    and d.available_mem_bytes >= mem_size_needed
                ):
                    device = d
                    break
            if device.available_mem_bytes < 0:
                raise RuntimeError(str(node) + "is too large to fit any device")
            occupied_devices.append(device)
            return device

        # Track partition and its left mem size
        partition_to_left_mem_bytes: dict[Partition, int] = {}
        # Track all the devices that have been used
        occupied_devices: list[Device] = []
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {"call_module", "call_method", "call_function"}:
                # Check if there are devices left
                if len(self.partitions) <= len(self.devices):
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    # Check if the current partition is the very first partition
                    if partition.used_mem_bytes == 0:
                        # Find a device to fit the first node, return available mem size
                        device = find_device_based_on_size(node)
                        occupied_devices.append(device)
                        # Update partition and its left mem size
                        partition_to_left_mem_bytes[partition] = (
                            device.available_mem_bytes
                        )
                        # Update available mem for the current partition
                        partition.logical_device_ids.append(device.logical_id)
                    else:
                        # The current partition is not the first partition
                        # Check if the current node can fit into current partition
                        if (
                            partition_to_left_mem_bytes[partition]
                            < total_size_of_input_nodes
                        ):
                            # Check if no device is left
                            if len(self.partitions) == len(self.devices):
                                # No device is left
                                # Create the first single node partition for the current node
                                self.create_single_node_partition(node)
                                continue
                            # Some devices are still left
                            # Create a new partition with a mem size that is enough for the current node
                            device = find_device_based_on_size(node)
                            partition = self.create_partition()
                            total_size_of_input_nodes = get_extra_size_of(
                                node, partition.nodes
                            )
                            partition_to_left_mem_bytes[partition] = (
                                device.available_mem_bytes
                            )
                            partition.logical_device_ids.append(device.logical_id)
                    partition.add_node(node)
                    partition_to_left_mem_bytes[partition] -= total_size_of_input_nodes
                # Create single node partitions if no device is left
                else:
                    self.create_single_node_partition(node)
        reorganize_partitions(self.partitions)
        # Get the node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        # Mapping all partitions into device
        found_partition_to_device_mapping = get_device_to_partitions_mapping(
            self.partitions, self.devices
        )
        if not found_partition_to_device_mapping:
            raise RuntimeError("Cannot Get a Valid Partition to Logical Device Mapping")
        return

    def saturate_host(self) -> None:
        """Saturate host by assigning replicates to unused devices with enough memory.
        It uses a greedy approach to find a next available set of devices to place all split
        partitions: For each used device, it searches for an idle device with minimal memory
        size that can hold all the partition located on that device; If the search is successful
        for all used devices, it then assigns the new devices' logical ID to the corresponding
        partition.
        """
        (
            device_to_partitions,
            device_to_left_mem_bytes,
            no_device_partitions,
        ) = get_device_partition_stats(self.partitions, self.devices)

        assert len(no_device_partitions) == 0, (
            f"Expect no_device_partitions has 0 device, but get {len(no_device_partitions)}"
        )

        # Devices that hold partitions
        used_devices = [d for d in self.devices if len(device_to_partitions[d]) > 0]
        # Track replicates of the assigned devices
        replicated_device_to_used_device: dict[Device, Device] = {}

        while len(used_devices) * 2 + len(replicated_device_to_used_device) <= len(
            self.devices
        ):
            # Success flag for this round
            success = True
            # Devices that have not been assigned
            idle_devices = [
                d
                for d in self.devices
                if d not in used_devices and d not in replicated_device_to_used_device
            ]
            # Temporary mapping from replicated device to original device
            temp_replicate_mapping = {}

            # Find a new device to replicate all partitions on an used device
            for used_device in used_devices:
                # Idle devices that have enough memory
                available_devices = [
                    d
                    for d in idle_devices
                    if d.available_mem_bytes
                    >= used_device.available_mem_bytes
                    - device_to_left_mem_bytes[used_device]
                ]
                if len(available_devices) == 0:
                    success = False
                    break
                new_device = min(available_devices, key=lambda d: d.available_mem_bytes)
                idle_devices.remove(new_device)
                temp_replicate_mapping[new_device] = used_device

            if not success:
                break
            replicated_device_to_used_device.update(temp_replicate_mapping)

        # Update logical device IDs assigned to the partitions
        for (
            replicate_device,
            original_device,
        ) in replicated_device_to_used_device.items():
            logical_id = replicate_device.logical_id
            for partition in device_to_partitions[original_device]:
                partition.logical_device_ids.append(logical_id)
        for p in self.partitions:
            print(p.logical_device_ids)

    def do_partition(self) -> GraphModule:
        """Return a new fx module with submodule nodes (partitions)."""
        module_with_submodules = split_module(
            self.graph_module,
            self.torch_module,
            lambda node: self.node_to_partition[node],
        )
        return module_with_submodules

    def dump_dag(self, module_with_submodules: GraphModule) -> DAG:
        """Return the dag structure and the new fx module with submodules."""
        dag = DAG()
        for node in module_with_submodules.graph.nodes:
            if node.op == "output":
                break
            if node.op in {"placeholder", "get_attr"}:
                continue
            if node.target is operator.__getitem__:
                continue
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # When a node has two or more output nodes,
            # it outputs its result to 'getitem' nodes.
            # Those 'getitem' nodes are the output node for this node.
            # Otherwise, the output node is this node itself.
            if len(node.users) > 1:
                output_nodes = list(node.users)
            else:
                output_nodes = [node]
            partition_id = int(node.name.rsplit("_", 1)[-1])
            device_ids = self.partitions[partition_id].logical_device_ids
            size_bytes = self.partitions[partition_id].used_mem_bytes
            dag.create_node(
                node, list(input_nodes), output_nodes, device_ids, size_bytes
            )
        return dag

    def create_partition(self) -> Partition:
        """Create a partition and append it to self.partitions."""
        partition_id = len(self.partitions)
        partition = Partition(partition_id)
        self.partitions.append(partition)
        return partition

    def create_single_node_partition(self, node):
        """Create a partition for a single node"""
        partition = self.create_partition()
        partition.add_node(node)
        return

    def sparse_nn_partition(self, available_mem_bytes: int) -> None:
        """This method partition a sparse nn module.
        It is size based partition but different from size_based_partition,
        it only works when all the devices have same memory size (available_mem_bytes).
        In the future, devices with different mem sizes will be supported like size_based_partition.
        It first traverse all the nodes and do the partitions based on the same memory size.
        If the current partition has no enough memory left for a new op node
        (call_module, call_method, call_function), a new partition is created.
        When crossing the boundary between non-embedding nodes and embedding nodes,
        a new partition is created regardlessly.
        For example, if the current node is a non-embedding node but the next node is an
        embedding node, a new partition is created for the next node.
        After the partition, the partitions are combined as much as possible.
        The rule is that a non-embedding partition only
        combines with another non-embedding one.
        So as the embedding partitions.
        """

        def combine_partitions_based_on_size(
            partitions: list[Partition], available_mem_bytes: int
        ) -> None:
            """Combining small partitions together to keep as less partitions as possible.
            Here is an example of the algorithm to do this:
            Assume some partitions, we first sort them based on partition used memory size.
            [(partition_4, 1), (partition_3, 1), (partition_2, 2), (partition_1, 7), (partition_0, 9)]
            The available memory is 10.
            step 1: self.find_partition_to_combine_based_on_size()
            First, mark bfs level for each partition
            Second, look the smallest partition, partition_4: 10 - 1 = 9
            It means any partition has a used memory equal or less than 9 could combine this partition
            We go from the largest and selection partition_0.
            Check the bfs level for two partitions, if the level difference is less than 2,
            it can be combined.
            step 2: repeat step 1 until no partitions can be combined
            """
            find_combination = True
            while find_combination:
                # Sort partitions based on memory size
                sorted_partitions = sorted(partitions, key=lambda p: p.used_mem_bytes)
                # Mark bfs level
                get_bfs_level_partition(self.partitions)
                find_combination, partitions = find_partition_to_combine_based_on_size(
                    sorted_partitions,
                    available_mem_bytes,
                    partitions,
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
            sorted_partitions: list[Partition],
            available_mem_bytes: int,
            partitions: list[Partition],
        ) -> tuple[bool, list[Partition]]:
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
            """If crossing the boundary between non-embedding nodes and
            embedding nodes, create a new partition
            """
            if in_embedding_region:
                embedding_partitions.append(partition)
            else:
                non_embedding_partitions.append(partition)
            if new_partition:
                partition = self.create_partition()
                # pyrefly: ignore [missing-attribute]
                partition.left_mem_bytes = available_mem_bytes
                return partition
            return None

        def is_embedding_node(node: Node) -> bool:
            """Check if a node is an embedding node"""
            if node.op == "call_module":
                submodule = self.graph_module
                for atom in str(node.target).split("."):
                    if not hasattr(submodule, atom):
                        raise RuntimeError(
                            f"Module {submodule} has no attribute {atom}"
                        )
                    submodule = getattr(submodule, atom)
                    if "Embedding" in str(submodule):
                        return True
            return False

        # Track embedding partitions and non-embedding partitions separately
        embedding_partitions: list[Partition] = []
        non_embedding_partitions: list[Partition] = []
        # A Flag to check the boundary
        in_embedding_region: bool = False
        partition = self.create_partition()
        for node in self.graph_module.graph.nodes:
            if node.op in {"call_module", "call_method", "call_function"}:
                # Check if crossing the boundary between embedding nodes and non embedding nodes
                if is_embedding_node(node) != in_embedding_region:
                    # Crossing the boundary
                    # Check if the current partition is an empty partition
                    if partition.used_mem_bytes != 0:
                        # The current partition isn't an empty partition. Create a new one.
                        partition = reset_partition_in_sparse_nn(partition)
                    in_embedding_region = not in_embedding_region
                total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                if (
                    total_size_of_input_nodes + partition.used_mem_bytes
                    > available_mem_bytes
                ):
                    partition = reset_partition_in_sparse_nn(partition)
                    total_size_of_input_nodes = get_extra_size_of(node, partition.nodes)
                    if total_size_of_input_nodes > available_mem_bytes:
                        raise RuntimeError(
                            node.target + "is too large to fit into a device"
                        )
                partition.add_node(node)
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
            msg = (
                "Need "
                + str(len(embedding_partitions))
                + " devices, but only "
                + str(len(self.devices))
                + " provided"
            )
            raise RuntimeError(msg)
        occupied_devices = []
        for i, partition in enumerate(embedding_partitions):
            # Check if all non-embedding partitions can fit into embedding partition devices
            if (
                total_size_of_non_embedding_partitions + partition.used_mem_bytes
                > available_mem_bytes
            ):
                raise RuntimeError(
                    "partition_"
                    + str(partition.partition_id)
                    + "(embedding partition) and non embedding partitions can not fit into one device"
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
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        """This method is to partition the fx module based on the cost.
        The cost is the total latency of running the whole fx module.
        In partitioner_utils.py, the cost model is built.
        The cost aware partition algorithm is:
        #1. At every beginning, each node is a partition.
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

        def try_combining_partitions(p0_index, p1_index, partitions) -> float:
            """Given two partitions and a list of partitions, combine these two partitions
            and see what is the cost of the modified partition list
            """
            p0 = partitions[p0_index]
            p1 = partitions[p1_index]
            """If two partitions' bfs level are less than 2 or two partitions are connected to each other,
               then they can be combined
            """
            if (
                (abs(p0.bfs_level - p1.bfs_level) <= 1)
                or (p0 in p1.parents)
                or p0 in (p1.children)
            ):
                combine_two_partitions(p0, p1, partitions)
                # Check if a circular dependency exists after combining
                if check_dependency(partitions[-1]):
                    return float("inf")
                # Check if the modified partition list can be mapped to devices after combination
                reset_partition_device(partitions)
                found_device = get_device_to_partitions_mapping(
                    partitions, self.devices
                )
                if not found_device:
                    return float("inf")
                # Calculate the new cost
                partition_to_latency_mapping = get_partition_to_latency_mapping(
                    partitions, node_to_latency_mapping
                )
                cost = get_latency_of_partitioned_graph(
                    partitions,
                    partition_to_latency_mapping,
                    transfer_rate_bytes_per_sec,
                )
                return cost
            # If two partition can not be combined, the cost is inf
            return float("inf")

        def search_combination(
            transfer_rate_bytes_per_sec, node_to_latency_mapping
        ) -> bool:
            """Given transfer rate between partitions and each node's latency,
            find two partitions to combine so the cost of the partitions can
            be reduced.
            The algorithm is :
            1. Go through all the partition pairs and see
            if any pair of partitions can be combined.
            2. Calculate the cost after the combination.
            3. Select the minimum cost and combine its corresponding partition pair.
            """
            partition_to_latency_mapping = get_partition_to_latency_mapping(
                self.partitions, node_to_latency_mapping
            )
            cost = get_latency_of_partitioned_graph(
                self.partitions,
                partition_to_latency_mapping,
                transfer_rate_bytes_per_sec,
            )
            if len(self.partitions) == 1:
                return False
            partition_pair: list[int] = []
            for i in range(len(self.partitions) - 1):
                for j in range(i + 1, len(self.partitions)):
                    # Try to combine the partition pair
                    # and see the new cost after combination
                    new_cost = try_combining_partitions(i, j, self.partitions[:])
                    if new_cost <= cost:
                        partition_pair = [i, j]
                        cost = new_cost
                    reorganize_partitions(self.partitions)
            # If a partition pair is found, combine them
            if len(partition_pair) != 0:
                p0 = self.partitions[partition_pair[0]]
                p1 = self.partitions[partition_pair[1]]
                combine_two_partitions(p0, p1, self.partitions)
            get_bfs_level_partition(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return len(partition_pair) != 0

        for node in self.graph_module.graph.nodes:
            if node.op not in {"placeholder", "get_attr", "output"}:
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
                transfer_rate_bytes_per_sec, node_to_latency_mapping
            )
        # Make sure all partitions are set up correctly
        reorganize_partitions(self.partitions)
        # Set up node to partition mapping
        self.node_to_partition = get_node_to_partition_mapping(self.partitions)
        return

    def kl_based_partition(
        self,
        transfer_rate_bytes_per_sec: float,
        node_to_latency_mapping: dict[Node, NodeLatency],
    ) -> None:
        """This function is a cost aware partition based
        on Kernighan-Lin algorithm.
        First, the graph is partitioned using size_based_partition.
        Then, each node is swapped with any other node in a different
        partition, and at the same time, the cost is estimated after
        the swapping.
        For example, we have nodes n0, n1, n2, n3 and n4.
        Using size_based_partition, n0 and n1 are in Partition p0.
        n2, n3 and n4 in Partition p1. The current cost is estimated.
        We first tried using n0 to swap with n2 from the other partition.
        Then we see that swapping n0 and n2 shows a lower cost
        than the current cost and it is the minimum among other pairs like
        (n0, None)(This means moving n0 to Partition without swapping other nodes),
        (n0, n3) and (n0, n4). We swap n0 and n2 and set the new cost
        as the current cost.
        Then We repeat this process for all the other nodes until all swapping pairs
        are tried.
        """

        def swap_nodes(n0, n1, p0, p1):
            # Either n0 or n1 could be None
            # That means we simply move the node
            # to another partition
            if n0 is not None:
                p0.remove_node(n0)
                p1.add_node(n0)
            if n1 is not None:
                p0.add_node(n1)
                p1.remove_node(n1)

        def try_swap_nodes(
            n0, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec
        ):
            cost = float("inf")
            swap_nodes(n0, n1, p0, p1)
            # Reorganize partitions after swapping
            reorganize_partitions(self.partitions)
            # Check if there is a circular dependency after swapping
            if (not check_dependency(p0)) and (not check_dependency(p1)):
                reset_partition_device(self.partitions)
                partition_to_latency_mapping = get_partition_to_latency_mapping(
                    self.partitions, node_to_latency_mapping
                )
                # Check if all partitions can be mapped to logical devices after swapping
                found_device = get_device_to_partitions_mapping(
                    self.partitions, self.devices
                )
                if not found_device:
                    cost = float("inf")
                else:
                    cost = get_latency_of_partitioned_graph(
                        self.partitions,
                        partition_to_latency_mapping,
                        transfer_rate_bytes_per_sec,
                    )
            # Swap back and reset all partitions back to original
            swap_nodes(n1, n0, p0, p1)
            reorganize_partitions(self.partitions)
            reset_partition_device(self.partitions)
            get_device_to_partitions_mapping(self.partitions, self.devices)
            return cost

        def swap_node_to_partition(
            node, p0, p1, node_to_latency_mapping, transfer_rate_per_sec
        ):
            """This function helps to swap one node from partition p0
            with all the nodes in another partition p1
            """
            p1_nodes = list(p1.nodes) + [None]
            min_cost = float("inf")
            node_pair: list[Node] = []
            for n1 in p1_nodes:
                # Ignore the node if it is not a op node
                if n1 is not None and n1.op in {"placeholder", "get_attr"}:
                    continue
                # Try swapping node in p0 with n1 in p1
                cost = try_swap_nodes(
                    node, n1, p0, p1, node_to_latency_mapping, transfer_rate_per_sec
                )
                if cost < min_cost:
                    # pyrefly: ignore [bad-assignment]
                    node_pair = [node, n1]
                    min_cost = cost
            return cost, node_pair  # type: ignore[possibly-undefined]

        # First use size_base_partition
        self.size_based_partition()
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            self.partitions, node_to_latency_mapping
        )
        # Calculate the cost of the partitions
        cost = get_latency_of_partitioned_graph(
            self.partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        # Keep tracking the node pair that shows the better cost
        node_pair: list[Node] = []
        # Keep tracking the partition pair of node pair
        partition_pair: list[Partition] = []
        # Collect all the op nodes from the graph
        op_nodes = [
            n
            for n in self.graph_module.graph.nodes
            if n.op not in {"placeholder", "get_attr", "output"}
        ]
        for node in op_nodes:
            # Find which partition the current node belongs
            p0_index = self.node_to_partition[node]
            p0 = self.partitions[p0_index]
            # Go through all the other partitions to swap
            # with other nodes from those partitions
            for p1_index, _ in enumerate(self.partitions):
                if p0_index != p1_index:
                    p1 = self.partitions[p1_index]
                    new_cost, new_node_pair = swap_node_to_partition(
                        node,
                        p0,
                        p1,
                        node_to_latency_mapping,
                        transfer_rate_bytes_per_sec,
                    )
                    # Update the cost
                    # Track the swapped node pair and their partitions
                    if new_cost < cost:
                        cost = new_cost
                        node_pair = new_node_pair
                        partition_pair = [p0, p1]
            # Do the swapping after trying all the nodes from a partition
            if len(node_pair) != 0:
                swap_nodes(
                    node_pair[0], node_pair[1], partition_pair[0], partition_pair[1]
                )
                reorganize_partitions(self.partitions)
                get_device_to_partitions_mapping(self.partitions, self.devices)
        reorganize_partitions(self.partitions)
        # Mapping the device to the partition
        get_device_to_partitions_mapping(self.partitions, self.devices)
        return

    def aot_based_partition(
        self, node_to_partition_mapping, partition_to_logical_device_mapping
    ):
        """This function helps to rebuild the partitions given the nodes and its
        corresponding partition id
        """
        partition_id_to_partition_mapping: dict[int, Partition] = {}
        self.node_to_partition = node_to_partition_mapping
        for node in self.node_to_partition:
            partition_id = self.node_to_partition[node]
            # If the requested partition has not been created, create the partition
            if partition_id not in partition_id_to_partition_mapping:
                partition = Partition(partition_id)
                self.partitions.append(partition)
                partition_id_to_partition_mapping[partition_id] = partition
                partition.logical_device_ids = partition_to_logical_device_mapping[
                    partition_id
                ]
            else:
                partition = partition_id_to_partition_mapping[
                    self.node_to_partition[node]
                ]
            # Add the current node into the partition
            partition.add_node(node)
