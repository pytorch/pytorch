"""
DAG node classes for representing trace execution graphs.
"""

from typing import Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from .json_profile import JsonProfile


try:
    import graphviz
except ImportError:
    pass

NodeName = Literal["op", "kernel"]
RooflineName = Literal["compute", "memory"]


class TraceDAGNode:
    """Represents a node in the DAG - either an operation or a kernel."""

    def __init__(self, name: str, node_type: NodeName):
        self.name = name
        self.node_type: NodeName = node_type
        # List of (duration_us, thread_id) for kernels
        self.kernel_instances: List[Tuple[float, int]] = []

        # Number of times this operation appears in the trace
        self.instance_count: int = 0

        # List of achieved FLOPS % for each instance
        self.achieved_flops_list: List[float] = []

        # List of achieved bandwidth % for each instance
        self.achieved_bandwidth_list: List[float] = []

        # List of "compute" or "memory" for each instance based on roofline analysis
        self.bound_type_list: List[RooflineName] = []

        # Multi-trace support: maps trace_id to trace-specific data
        self.trace_data: Dict[int, Dict] = {}


class MultiTraceDAGNode:
    """Represents a composite node in the multi-trace DAG."""

    def __init__(self, name: str, node_type: NodeName):
        self.name = name
        self.node_type: NodeName = node_type

        # Maps trace_id to node data
        self.trace_instances: Dict[int, TraceDAGNode] = {}

        # Which traces contain this node
        self.present_in_traces: OrderedSet[int] = OrderedSet()

    def add_trace_instance(self, trace_id: int, node: TraceDAGNode):
        """Add data for this node from a specific trace."""
        self.trace_instances[trace_id] = node
        self.present_in_traces.add(trace_id)


class BaseDAG:
    """Base class for DAG implementations with common functionality."""

    def _create_gradient_color(
        self, base_color: str, percentage: float, max_percentage: float
    ) -> str:
        """Create a gradient color from light to dark based on percentage."""
        base_color = base_color.lstrip("#")
        r, g, b = tuple(int(base_color[i : i + 2], 16) for i in (0, 2, 4))

        if max_percentage > 0:
            intensity = percentage / max_percentage
        else:
            intensity = 0

        min_intensity = 0.01
        max_intensity = 1.2

        scaled_intensity = min_intensity + (max_intensity - min_intensity) * intensity

        final_r = int(255 * (1 - scaled_intensity) + r * scaled_intensity)
        final_g = int(255 * (1 - scaled_intensity) + g * scaled_intensity)
        final_b = int(255 * (1 - scaled_intensity) + b * scaled_intensity)

        return f"#{final_r:02x}{final_g:02x}{final_b:02x}"


class MultiTraceDAG(BaseDAG):
    """Directed Acyclic Graph representing multiple collapsed trace trees."""

    def __init__(self):
        self.nodes: Dict[str, MultiTraceDAGNode] = {}
        # (parent, child, trace_id) relationships
        self.edges: OrderedSet[Tuple[str, str, int]] = OrderedSet()
        self.trace_colors: Dict[int, str] = {}
        self.trace_names: Dict[int, str] = {}

    def add_trace_dag(self, trace_id: int, dag: "TraceDAG", trace_name: str):
        """Add a single trace's DAG to the multi-trace DAG."""
        self.trace_names[trace_id] = trace_name

        for node_name, node in dag.nodes.items():
            if node_name not in self.nodes:
                self.nodes[node_name] = MultiTraceDAGNode(node_name, node.node_type)
            self.nodes[node_name].add_trace_instance(trace_id, node)

        for parent, child in dag.edges:
            self.edges.add((parent, child, trace_id))

    def assign_colors(self):
        """Assign distinct colors to each trace."""
        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FECA57",
            "#FF9F43",
            "#686DE0",
            "#F8B500",
        ]
        for i, trace_id in enumerate(sorted(self.trace_names.keys())):
            self.trace_colors[trace_id] = colors[i % len(colors)]

    def calculate_kernel_time_gradients(self):
        """Calculate gradient colors based on kernel time percentages for each trace."""
        self.trace_kernel_gradients = {}

        for trace_id in self.trace_names.keys():
            total_kernel_time = 0.0
            kernel_times = {}

            for node_name, multi_node in self.nodes.items():
                if (
                    multi_node.node_type == "kernel"
                    and trace_id in multi_node.trace_instances
                ):
                    node = multi_node.trace_instances[trace_id]
                    kernel_time = sum(dur for dur, _ in node.kernel_instances)
                    kernel_times[node_name] = kernel_time
                    total_kernel_time += kernel_time

            gradients = {}
            if total_kernel_time > 0:
                max_percentage = (
                    max(kernel_times.values()) / total_kernel_time * 100
                    if kernel_times
                    else 0
                )

                base_color = self.trace_colors[trace_id]

                for node_name, kernel_time in kernel_times.items():
                    percentage = (kernel_time / total_kernel_time) * 100
                    gradient_color = self._create_gradient_color(
                        base_color, percentage, max_percentage
                    )
                    gradients[node_name] = gradient_color

            self.trace_kernel_gradients[trace_id] = gradients

    def calculate_kernel_colors(
        self,
        color_mode: str,
        baseline_profile: Optional["JsonProfile"] = None,
    ) -> Dict[int, Dict[str, str]]:
        """
        Calculate colors for kernel nodes based on the specified color mode for each trace.
        Returns a dictionary mapping trace_id -> {kernel_name: color_string}.
        """
        from .utils import calculate_kernel_colors as calc_kernel_colors

        if color_mode not in [
            "time",
            "diff",
            "mem-utilization", 
            "compute-utilization",
            "roofline",
        ]:
            return {}

        trace_kernel_colors = {}

        if color_mode == "time":
            return self.trace_kernel_gradients

        for trace_id in self.trace_names.keys():
            base_gradients = self.trace_kernel_gradients.get(trace_id, {})
              
            # Collect kernel nodes for this trace
            trace_kernel_nodes = {}
            for kernel_name, multi_node in self.nodes.items():
                if (
                    multi_node.node_type == "kernel"
                    and trace_id in multi_node.trace_instances
                ):
                    trace_kernel_nodes[kernel_name] = multi_node.trace_instances[trace_id]

            # Use the common function to calculate colors
            trace_kernel_colors[trace_id] = calc_kernel_colors(
                color_mode, base_gradients, trace_kernel_nodes, baseline_profile
            )

        return trace_kernel_colors

    def get_color_legend_text(self, color_mode: str) -> Optional[str]:
        """Get legend text for the specified color mode."""
        from .utils import get_color_legend_text
        return get_color_legend_text(color_mode)

    def filter_by_height(self, height: int) -> "MultiTraceDAG":
        """
        Filter the multi-trace DAG to only show nodes up to a specified height above kernel nodes.

        Args:
            height: Maximum levels of non-kernel nodes to show above kernels
                   (0 = only kernels, 1 = kernels + direct parents, etc.)

        Returns:
            A new filtered MultiTraceDAG containing only nodes within the height limit
        """
        if height < 0:
            return self

        filtered_dag = MultiTraceDAG()

        filtered_dag.trace_colors = self.trace_colors.copy()
        filtered_dag.trace_names = self.trace_names.copy()

        kernel_nodes = set()
        for node_name, multi_node in self.nodes.items():
            if multi_node.node_type == "kernel":
                kernel_nodes.add(node_name)

        if height == 0:
            for kernel_name in kernel_nodes:
                if kernel_name in self.nodes:
                    multi_node = self.nodes[kernel_name]
                    new_multi_node = MultiTraceDAGNode(kernel_name, "kernel")
                    for trace_id, node in multi_node.trace_instances.items():
                        new_multi_node.add_trace_instance(trace_id, node)
                    filtered_dag.nodes[kernel_name] = new_multi_node
            return filtered_dag

        reverse_edges_by_trace = {}
        for trace_id in self.trace_names.keys():
            reverse_edges_by_trace[trace_id] = {}

        for parent, child, trace_id in self.edges:
            if trace_id not in reverse_edges_by_trace:
                reverse_edges_by_trace[trace_id] = {}
            if child not in reverse_edges_by_trace[trace_id]:
                reverse_edges_by_trace[trace_id][child] = []
            reverse_edges_by_trace[trace_id][child].append(parent)

        # For each trace, perform BFS from kernels going backwards
        nodes_to_include_by_trace = {}
        for trace_id in self.trace_names.keys():
            nodes_to_include = set()
            current_level = set()

            # Start with kernels that exist in this trace
            for kernel_name in kernel_nodes:
                if (
                    kernel_name in self.nodes
                    and trace_id in self.nodes[kernel_name].present_in_traces
                ):
                    nodes_to_include.add(kernel_name)
                    current_level.add(kernel_name)

            # BFS backwards for specified height
            for level in range(height):
                next_level = set()
                for node in current_level:
                    # Add all parents of current level nodes for this trace
                    if node in reverse_edges_by_trace[trace_id]:
                        for parent in reverse_edges_by_trace[trace_id][node]:
                            if parent not in nodes_to_include:
                                next_level.add(parent)
                                nodes_to_include.add(parent)

                if not next_level:
                    break

                current_level = next_level

            nodes_to_include_by_trace[trace_id] = nodes_to_include

        # Union all nodes to include across all traces
        all_nodes_to_include = set()
        for nodes_set in nodes_to_include_by_trace.values():
            all_nodes_to_include.update(nodes_set)

        # Build the filtered DAG with only the nodes we want to include
        for node_name in all_nodes_to_include:
            if node_name in self.nodes:
                original_multi_node = self.nodes[node_name]
                new_multi_node = MultiTraceDAGNode(
                    node_name, original_multi_node.node_type
                )

                # Only include trace instances that should be included for this node
                for trace_id, node in original_multi_node.trace_instances.items():
                    if node_name in nodes_to_include_by_trace.get(trace_id, set()):
                        new_multi_node.add_trace_instance(trace_id, node)

                # Only add the multi-node if it has at least one trace instance
                if new_multi_node.trace_instances:
                    filtered_dag.nodes[node_name] = new_multi_node

        # Add edges that connect nodes within our filtered set
        for parent, child, trace_id in self.edges:
            if (
                parent in filtered_dag.nodes
                and child in filtered_dag.nodes
                and parent in nodes_to_include_by_trace.get(trace_id, set())
                and child in nodes_to_include_by_trace.get(trace_id, set())
            ):
                filtered_dag.edges.add((parent, child, trace_id))

        return filtered_dag


class TraceDAG(BaseDAG):
    """Directed Acyclic Graph representing the collapsed trace tree."""

    def __init__(self):
        self.nodes: Dict[str, TraceDAGNode] = {}
        # (parent, child) relationships
        self.edges: OrderedSet[Tuple[str, str]] = OrderedSet()

    def add_node(self, name: str, node_type: NodeName) -> TraceDAGNode:
        """Add a node to the DAG if it doesn't exist."""
        if name not in self.nodes:
            self.nodes[name] = TraceDAGNode(name=name, node_type=node_type)
        return self.nodes[name]

    def add_edge(self, parent: str, child: str):
        """Add an edge from parent to child."""
        self.edges.add((parent, child))

    def add_kernel_instance(self, kernel_name: str, duration_us: float, thread_id: int):
        """Add a kernel instance to a kernel node."""
        if kernel_name in self.nodes:
            node = self.nodes[kernel_name]
            if node.node_type == "kernel":
                node.kernel_instances.append((duration_us, thread_id))

    def calculate_kernel_time_gradients(
        self, base_color: str = "#4ECDC4"
    ) -> Dict[str, str]:
        """Calculate gradient colors based on kernel time percentages."""
        # Calculate total kernel time
        total_kernel_time = 0.0
        kernel_times = {}

        for node_name, node in self.nodes.items():
            if node.node_type == "kernel":
                kernel_time = sum(dur for dur, _ in node.kernel_instances)
                kernel_times[node_name] = kernel_time
                total_kernel_time += kernel_time

        # Calculate percentages and create gradient mapping
        gradients = {}
        if total_kernel_time > 0:
            max_percentage = (
                max(kernel_times.values()) / total_kernel_time * 100
                if kernel_times
                else 0
            )

            for node_name, kernel_time in kernel_times.items():
                percentage = (kernel_time / total_kernel_time) * 100
                # Create gradient from light to dark based on percentage
                gradient_color = self._create_gradient_color(
                    base_color, percentage, max_percentage
                )
                gradients[node_name] = gradient_color

        return gradients
