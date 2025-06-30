# mypy: allow-untyped-defs
from enum import Enum
from typing import NamedTuple

from torch.fx.node import map_arg, Node


class Partition:
    """Partition class contains all the information about an individual partition.
    It also provides necessary methods for manipulation the partition.
    """

    def __init__(self, partition_id: int) -> None:
        self.nodes: set[Node] = set()
        self.partition_id = partition_id
        self.parents: set[Partition] = set()
        self.children: set[Partition] = set()
        self.bfs_level: int = -1
        self.used_mem_bytes: int = 0
        self.logical_device_ids: list[int] = []

    def __str__(self):
        return str(self.partition_id)

    def recalculate_mem_size(self):
        self.used_mem_bytes = 0
        for node in self.nodes:
            self.used_mem_bytes += get_extra_size_of(node, self.nodes)

    def add_node(self, node):
        input_nodes: dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        # Add current node's input nodes if they are placeholder or constants
        for n in input_nodes:
            if n.op in {"placeholder", "get_attr"}:
                self.nodes.add(n)
        self.nodes.add(node)
        self.recalculate_mem_size()

    def remove_node(self, node):
        # Remove a node only if the node is in the partition
        if node in self.nodes:
            self.nodes.remove(node)
            # Collect the node's input nodes
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # Check if an input node is a placeholder or get_attr,
            # and this input node is not used by some other nodes in this partition,
            # the remove this input node
            for input_node in input_nodes:
                if all(
                    n not in self.nodes for n in input_node.users
                ) and input_node.op in {"placeholder", "get_attr"}:
                    self.nodes.remove(input_node)
            self.recalculate_mem_size()


class Device(NamedTuple):
    name: str
    available_mem_bytes: int
    logical_id: int


class NodeLatency(NamedTuple):
    # Latency due to the memory bandwidth
    mem_latency_sec: float
    # Latency due to the computation
    computer_latency_sec: float


class PartitionLatency(NamedTuple):
    # Sum of all nodes' memory latency on the critical path
    mem_latency_sec: float
    # Sum of all nodes' compute latency on the critical path
    computer_latency_sec: float
    # Latency of the critical path
    overall_latency_sec: float


class PartitionMode(Enum):
    size_based = 0
    sparse_nn = 1
    cost_aware = 2
    kl_based = 3
    aot_based = 4


class PartitionerConfig(NamedTuple):
    devices: list[Device]
    mode: PartitionMode = PartitionMode.size_based
    transfer_rate_bytes_per_sec: float = 0.0
    node_to_latency_mapping: dict[Node, NodeLatency] = {}
    node_to_partition_mapping: dict[Node, int] = {}
    partition_to_logical_device_mapping: dict[int, list[int]] = {}
    # Saturate host by replicating partitions to the remaining idle devices.
    saturate_host: bool = False


def get_extra_size_of(node: Node, nodes: set[Node]) -> int:
    """Given a node and a set of nodes,
    this function return the extra size that needed
    if this node is included in this set.
    """
    # Find all its input nodes
    input_nodes: dict[Node, None] = {}
    map_arg(node.args, input_nodes.setdefault)
    map_arg(node.kwargs, input_nodes.setdefault)
    # Calculate total size of related nodes
    total_size_of_input_nodes = 0
    for n in input_nodes:
        # Make sure this node hasn't been in this set yet
        if n not in nodes:
            size_bytes = getattr(n, "size_bytes", None)
            if size_bytes:
                total_size_of_input_nodes += size_bytes.output_size
            else:
                raise RuntimeError("node has no size_bytes attr")
    # Don't forget the op node itself
    size_bytes = getattr(node, "size_bytes", None)
    if size_bytes:
        total_size_of_input_nodes += size_bytes.total_size
    else:
        raise RuntimeError("node has no size_bytes attr")
    return total_size_of_input_nodes


def get_latency_of_one_partition(
    partition: Partition, node_to_latency_mapping: dict[Node, NodeLatency]
) -> PartitionLatency:
    """Given a partition and its nodes' latency, return a PartitionLatency for this partition"""

    def get_top_nodes(partition: Partition) -> list[Node]:
        """Given a partition, return a list of nodes on the top bfs level"""
        top_nodes: list[Node] = []
        for node in partition.nodes:
            # Skip placeholder and get_attr nodes
            if node.op in {"placeholder", "get_attr"}:
                continue
            input_nodes: dict[Node, None] = {}
            map_arg(node.args, input_nodes.setdefault)
            map_arg(node.kwargs, input_nodes.setdefault)
            # If a node has no input nodes in this partition,
            # or its input nodes in this partition are placeholders and get_attrs
            # this node is on the top bfs level in this partition
            if not any(
                n in partition.nodes and n.op not in {"placeholder", "get_attr"}
                for n in input_nodes
            ):
                top_nodes.append(node)
        return top_nodes

    def dfs_helper(node: Node, partition_latency) -> PartitionLatency:
        """Given a top node of a partition, this function returns
        the latency of the critical path in the partition
        """
        node_latency = node_to_latency_mapping[node]
        # Calculate the current overall latency of the partition
        overall_latency_sec = partition_latency.overall_latency_sec + max(
            node_latency.computer_latency_sec, node_latency.mem_latency_sec
        )
        # Update the mem latency of this path
        mem_latency_sec = (
            partition_latency.mem_latency_sec + node_latency.mem_latency_sec
        )
        # Update the compute latency of this path
        computer_latency_sec = (
            partition_latency.computer_latency_sec + node_latency.computer_latency_sec
        )
        # Get all users of this node that are in this partition
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            )
            for n in users:
                # Get new partition latency recursively
                new_partition_latency = dfs_helper(
                    n,
                    PartitionLatency(
                        mem_latency_sec, computer_latency_sec, overall_latency_sec
                    ),
                )
                if (
                    new_partition_latency.overall_latency_sec
                    > max_latency.overall_latency_sec
                ):
                    max_latency = new_partition_latency
            return max_latency
        # If there is no user, the node is at bottom of the partition
        return PartitionLatency(
            mem_latency_sec, computer_latency_sec, overall_latency_sec
        )

    # Main part starts
    # Get all top level nodes of this partition
    top_nodes = get_top_nodes(partition)
    critical_path_latency = PartitionLatency(
        mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
    )
    # Go through all top nodes and find the largest latency (critical pass latency)
    for node in top_nodes:
        partition_latency = dfs_helper(
            node,
            PartitionLatency(
                mem_latency_sec=0.0, computer_latency_sec=0.0, overall_latency_sec=0.0
            ),
        )
        if (
            partition_latency.overall_latency_sec
            > critical_path_latency.overall_latency_sec
        ):
            critical_path_latency = partition_latency
    return critical_path_latency


def get_partition_to_latency_mapping(
    partitions: list[Partition], node_to_latency_mapping: dict[Node, NodeLatency]
) -> dict[Partition, PartitionLatency]:
    """Given all the partitions and node_to_latency_mapping dictionary,
    return a mapping dictionary of each partition to its overall latency
    """
    partition_to_latency_mapping: dict[Partition, PartitionLatency] = {}
    # Go through each partition and get its latency
    for partition in partitions:
        partition_latency = get_latency_of_one_partition(
            partition, node_to_latency_mapping
        )
        partition_to_latency_mapping[partition] = partition_latency
    return partition_to_latency_mapping


def get_comm_latency_between(
    parent_partition: Partition,
    child_partition: Partition,
    transfer_rate_bytes_per_sec: float,
):
    """Given two partitions (parent and child),
    calculate the communication latency between the two.
    """
    # If two partitions are on the same device, the comm latency is 0.
    if (
        parent_partition.logical_device_ids != []
        and child_partition.logical_device_ids != []
        and parent_partition.logical_device_ids == child_partition.logical_device_ids
    ):
        return 0.0
    # Keep tracking the communication size between parent and child
    comm_size = 0
    # Keep tracking all the counted node
    visited_nodes = set()
    # Go through all nodes in the child partition
    # If a node has input nodes from the parent partition,
    # the output size of those input nodes will be counted
    # and added to comm_size
    for node in child_partition.nodes:
        input_nodes: dict[Node, None] = {}
        map_arg(node.args, input_nodes.setdefault)
        map_arg(node.kwargs, input_nodes.setdefault)
        for n in input_nodes:
            if n in parent_partition.nodes and n not in visited_nodes:
                size_bytes = getattr(n, "size_bytes", None)
                if size_bytes is not None:
                    comm_size += size_bytes.output_size
                visited_nodes.add(n)
    return comm_size / transfer_rate_bytes_per_sec


def get_latency_of_partitioned_graph(
    partitions: list[Partition],
    partition_to_latency_mapping: dict[Partition, PartitionLatency],
    transfer_rate_bytes_per_sec: float,
):
    """Given all partitions in a graph, find the critical path among all partitions
    and return its latency as the latency of the whole graph
    """

    def dfs_helper(partition: Partition, latency_so_far_sec: float) -> float:
        """This function helps to recursively get the latency of a path of partitions"""
        # Update latency by adding current partition's latency
        latency_so_far_sec += partition_to_latency_mapping[
            partition
        ].overall_latency_sec

        if partition.children:
            max_latency_sec = 0.0
            for child in partition.children:
                # Calculate latency between
                comm_latency_sec = get_comm_latency_between(
                    partition, child, transfer_rate_bytes_per_sec
                )
                new_latency_sec = dfs_helper(
                    child, latency_so_far_sec + comm_latency_sec
                )
                if new_latency_sec > max_latency_sec:
                    max_latency_sec = new_latency_sec
            return max_latency_sec
        return latency_so_far_sec

    def get_top_partitions(partitions: list[Partition]) -> list[Partition]:
        """This function is to return all the partitions without parents
        as the starting points of all the paths
        """
        # If a partition has no parents, then it is a top partition
        top_partitions = [
            partition for partition in partitions if len(partition.parents) == 0
        ]
        return top_partitions

    top_partitions = get_top_partitions(partitions)
    critical_path_latency_sec = 0.0
    for partition in top_partitions:
        latency_sec = dfs_helper(partition, 0.0)
        if latency_sec > critical_path_latency_sec:
            critical_path_latency_sec = latency_sec
    return critical_path_latency_sec
