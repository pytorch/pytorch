from typing import NamedTuple, Dict, List
from torch.fx.node import Node, map_arg
from torch.fx.experimental.Partitioner import Partition

class NodeLatency(NamedTuple):
    # Latency due to the memory bandwidth
    mem_latency: float
    # Latency due to the computation
    compute_latency: float

class PartitionLatency(NamedTuple):
    # Sum of all nodes' memory latency on the critical path
    mem_latency: float
    # Sum of all nodes' compute latency on the critical path
    compute_latency: float
    # Latency of the critical path
    overall_latency: float

def get_latency_of_one_partition(
    partition: Partition,
    node_to_latency_mapping: Dict[Node, NodeLatency]
) -> PartitionLatency:
    """Given a partiton and its nodes' latency, return a PartitionLatency for this partition"""

    def get_top_nodes(partition: Partition) -> List[Node]:
        """Given a partition, return a list of nodes on the top bfs level"""
        top_nodes: List[Node] = []
        for node in partition.nodes:
            # Skip placeholder and get_attr nodes
            if node.op in {'placeholder', 'get_attr'}:
                continue
            input_nodes: Dict[Node, None] = {}
            map_arg(node.args, lambda n: input_nodes.setdefault(n))
            map_arg(node.kwargs, lambda n: input_nodes.setdefault(n))
            # If a node has no input nodes in this partition,
            # or its input nodes in this partition are placeholders and get_attrs
            # this node is on the top bfs level in this partition
            if not any([n in partition.nodes and n.op not in {'placeholder', 'get_attr'} for n in input_nodes]):
                top_nodes.append(node)
        return top_nodes

    def dfs_helper(node: Node, partition_latency) -> PartitionLatency:
        """Given a top node of a partition, this function returns
           the latency of the critical path in the partition
        """
        node_latency = node_to_latency_mapping[node]
        # Calculate the current overall latency of the partition
        overall_latency = partition_latency.overall_latency + max(node_latency.compute_latency, node_latency.mem_latency)
        # Update the mem latency of this path
        mem_latency = partition_latency.mem_latency + node_latency.mem_latency
        # Update the compute latency of this path
        compute_latency = partition_latency.compute_latency + node_latency.compute_latency
        # Get all users of this node that are in this partition
        users = set(node.users).intersection(partition.nodes)
        if users:
            max_latency = PartitionLatency(mem_latency=0., compute_latency=0., overall_latency=0.)
            for n in users:
                # Get new partition latency recursively
                new_partition_latency = dfs_helper(n, PartitionLatency(mem_latency, compute_latency, overall_latency))
                if new_partition_latency.overall_latency > max_latency.overall_latency:
                    max_latency = new_partition_latency
            return max_latency
        # If there is no user, the node is at bottom of the partition
        return PartitionLatency(mem_latency, compute_latency, overall_latency)
    # Main part starts
    # Get all top level nodes of this partition
    top_nodes = get_top_nodes(partition)
    critical_path_latency = PartitionLatency(mem_latency=0., compute_latency=0., overall_latency=0.)
    # Go through all top nodes and find the largest latency (critical pass latency)
    for node in top_nodes:
        partition_latency = dfs_helper(node, PartitionLatency(mem_latency=0., compute_latency=0., overall_latency=0.))
        if partition_latency.overall_latency > critical_path_latency.overall_latency:
            critical_path_latency = partition_latency
    return critical_path_latency
