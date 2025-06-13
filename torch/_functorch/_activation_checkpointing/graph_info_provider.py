from typing import Any, Optional

import networkx as nx

from torch.fx import Graph, Node


class GraphInfoProvider:
    """
    This class provides information about the graph, such as the nodes, edges, and their runtime and memory requirements.
    It also provides methods to create graphs from the information provided.
    """

    __RECOMPUTABLE_NODE_ONLY_GRAPH = "recomputable_node_only_graph"
    __RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT = (
        "recomputable_node_only_graph_with_larger_graph_context"
    )
    __FULL_NX_JOINT_GRAPH = "full_nx_joint_graph"
    __SIMPLIFIED_FX_JOINT_GRAPH = "fx_joint_graph"

    def __init__(
        self,
        graph_nodes_in_order: list[str],
        graph_edges: list[tuple[str, str]],
        all_recomputable_banned_nodes: list[str],
        all_node_runtimes: Optional[dict[str, float]] = None,
        all_node_memories: Optional[dict[str, float]] = None,
        recorded_knapsack_input_memories: Optional[list[float]] = None,
        recorded_knapsack_input_runtimes: Optional[list[float]] = None,
        joint_graph: Optional[Graph] = None,
    ):
        self.graph_nodes_in_order = graph_nodes_in_order
        self.graph_edges = graph_edges
        self.all_node_runtimes: dict[str, float] = dict()
        if all_node_runtimes is None:
            if recorded_knapsack_input_runtimes is None:
                raise ValueError(
                    "Either all_node_runtimes or recorded_knapsack_input_runtimes must be provided."
                )
            self.all_node_runtimes = {
                node: recorded_knapsack_input_runtimes[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_runtimes.update(all_node_runtimes)
        self.all_node_memories: dict[str, float] = dict()
        if all_node_memories is None:
            if recorded_knapsack_input_memories is None:
                raise ValueError(
                    "Either all_node_memories or recorded_knapsack_input_memories must be provided."
                )
            self.all_node_memories = {
                node: recorded_knapsack_input_memories[i]
                for i, node in enumerate(all_recomputable_banned_nodes)
            }
        else:
            self.all_node_memories.update(all_node_memories)
        self.all_recomputable_banned_nodes = all_recomputable_banned_nodes
        self.all_recomputable_banned_nodes_set = set(all_recomputable_banned_nodes)
        self.recorded_knapsack_input_memories = recorded_knapsack_input_memories
        self.recorded_knapsack_input_runtimes = recorded_knapsack_input_runtimes
        self._lazily_initialized_graphs: dict[str, Any] = {
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH: None,
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT: None,
            self.__FULL_NX_JOINT_GRAPH: None,
            self.__SIMPLIFIED_FX_JOINT_GRAPH: None,
        }

    @classmethod
    def inialize_from_graph(
        cls,
        joint_graph: Graph,
        all_recomputable_banned_nodes: list[Node],
        recorded_knapsack_input_memories: list[float],
        recorded_knapsack_input_runtimes: list[float],
    ) -> "GraphInfoProvider":
        """
        Enables initialization from a joint graph.
        """
        graph_nodes_in_order = [node.name for node in joint_graph.nodes]
        graph_edges = [
            (node.name, user.name) for node in joint_graph.nodes for user in node.users
        ]
        all_recomputable_banned_node_names = [
            node.name for node in all_recomputable_banned_nodes
        ]
        return cls(
            graph_nodes_in_order=graph_nodes_in_order,
            graph_edges=graph_edges,
            all_recomputable_banned_nodes=all_recomputable_banned_node_names,
            recorded_knapsack_input_memories=recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=recorded_knapsack_input_runtimes,
            joint_graph=joint_graph,
        )

    @property
    def recomputable_node_only_graph(self) -> nx.DiGraph:
        if self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] is None:
            self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH] = (
                self._create_recomputable_node_only_graph()
            )
        return self._lazily_initialized_graphs[self.__RECOMPUTABLE_NODE_ONLY_GRAPH]

    @property
    def recomputable_node_only_graph_with_larger_graph_context(self) -> nx.DiGraph:
        if (
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ]
            is None
        ):
            self._lazily_initialized_graphs[
                self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
            ] = self._create_recomputable_node_only_graph_with_larger_graph_context()
        return self._lazily_initialized_graphs[
            self.__RECOMPUTABLE_NODE_ONLY_GRAPH_WITH_LARGER_GRAPH_CONTEXT
        ]

    @property
    def full_joint_nx_graph(self) -> nx.DiGraph:
        if self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH] = (
                self._create_full_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__FULL_NX_JOINT_GRAPH]

    @property
    def simplified_fx_joint_graph(self) -> Graph:
        if self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] is None:
            self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH] = (
                self._recreate_psuedo_joint_graph()
            )
        return self._lazily_initialized_graphs[self.__SIMPLIFIED_FX_JOINT_GRAPH]

    def get_non_ac_peak_memory(self) -> float:
        return sum(
            self.all_node_memories[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_theoretical_max_runtime(self) -> float:
        return sum(
            self.all_node_runtimes[node_name]
            for node_name in self.all_recomputable_banned_nodes_set
        )

    def get_knapsack_memory_input(self) -> list[float]:
        return (
            self.recorded_knapsack_input_memories
            if self.recorded_knapsack_input_memories
            else [
                self.all_node_memories[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def get_knapsack_runtime_input(self) -> list[float]:
        return (
            self.recorded_knapsack_input_runtimes
            if self.recorded_knapsack_input_runtimes
            else [
                self.all_node_runtimes[node_name]
                for node_name in self.all_recomputable_banned_nodes
            ]
        )

    def _create_recomputable_node_only_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for recomputable_node in self.all_recomputable_banned_nodes:
            graph.add_node(recomputable_node)

        for a, b in self.graph_edges:
            if (
                a in self.all_recomputable_banned_nodes_set
                and b in self.all_recomputable_banned_nodes_set
            ):
                graph.add_edge(a, b)
        return graph

    def _create_recomputable_node_only_graph_with_larger_graph_context(
        self,
    ) -> nx.DiGraph:
        # Create a dictionary to store the reachable nodes for each node
        all_recomputable_banned_nodes_set = set(self.all_recomputable_banned_nodes)

        reachable_nodes = {}
        for node in all_recomputable_banned_nodes_set:
            # Use BFS to find all reachable nodes
            predecessors = dict(nx.bfs_predecessors(self.full_joint_nx_graph, node))
            reachable_recomputable_nodes = set(predecessors.keys()).intersection(
                all_recomputable_banned_nodes_set
            )
            reachable_nodes[node] = reachable_recomputable_nodes
        # Create the candidate graph
        candidate_graph = nx.DiGraph()
        candidate_graph.add_nodes_from(all_recomputable_banned_nodes_set)
        for node1 in all_recomputable_banned_nodes_set:
            for node2 in reachable_nodes[node1]:
                # Check if there is an overlapping path
                overlapping_path = False
                for intermediate_node in reachable_nodes[node1]:
                    if (
                        intermediate_node != node2
                        and node2 in reachable_nodes[intermediate_node]
                    ):
                        overlapping_path = True
                        break
                if not overlapping_path:
                    candidate_graph.add_edge(node1, node2)
        return candidate_graph

    def _create_full_joint_graph(self) -> nx.DiGraph:
        graph = nx.DiGraph()
        for node in self.graph_nodes_in_order:
            if node == "output":
                continue
            graph.add_node(node)

        for a, b in self.graph_edges:
            if a == "output" or b == "output":
                continue
            graph.add_edge(a, b)
        return graph

    def _recreate_psuedo_joint_graph(self) -> Graph:
        # Create a dictionary to store the dependencies of each node
        node_dependencies: dict[str, list[str]] = {
            node: [] for node in self.graph_nodes_in_order
        }
        for a, b in self.graph_edges:
            if a not in node_dependencies or b not in node_dependencies:
                raise ValueError(f"Edge ({a}, {b}) references a non-existent node.")
            node_dependencies[b].append(a)

        joint_graph = Graph()
        # Create nodes in the graph
        nodes: dict[str, Node] = {}
        for node_name in self.graph_nodes_in_order:
            input_nodes = [nodes[dep] for dep in node_dependencies[node_name]]
            if input_nodes:
                node = joint_graph.call_function(lambda *x: x, tuple(input_nodes))
                node.name = node_name
            else:
                node = joint_graph.placeholder(node_name)
            nodes[node_name] = node
        return joint_graph

    def _visualize_recomputable_candidate_graph_with_larger_context(
        self,
        layout_k: float = 0.5,
        layout_iterations: int = 30,
    ) -> None:
        """
        Visualize the recomputable candidate graph with larger context.
        """
        from matplotlib import cm, colors as mcolors, pyplot as plt

        pos = nx.spring_layout(
            self.recomputable_node_only_graph_with_larger_graph_context,
            k=layout_k,
            iterations=layout_iterations,
        )
        # pos = nx.spectral_layout(graph_with_indirect_edges)
        plt.figure(figsize=(20, 15))

        # Create a dictionary for node labels using the index
        labels = {
            node: self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                node
            ].get("index", node)
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        }

        # Extract memory values and normalize them
        norm = mcolors.Normalize(
            vmin=min(self.get_knapsack_memory_input()),
            vmax=max(self.get_knapsack_memory_input()),
        )
        cmap = cm.viridis  # type: ignore[attr-defined]

        # Assign colors based on memory
        node_colors = [
            cmap(
                norm(
                    float(
                        self.recomputable_node_only_graph_with_larger_graph_context.nodes[
                            node
                        ]["memory"]
                    )
                )
            )
            for node in self.recomputable_node_only_graph_with_larger_graph_context.nodes
        ]

        # Draw the graph with parsed nodes only
        nx.draw_networkx_nodes(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            node_color=node_colors,
            node_size=300,
            label="Parsed Nodes",
        )
        nx.draw_networkx_edges(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            arrows=True,
            arrowsize=10,
        )
        nx.draw_networkx_labels(
            self.recomputable_node_only_graph_with_larger_graph_context,
            pos,
            labels=labels,
            font_size=8,
            font_weight="bold",
        )

        plt.title("Memory Colour Coded Dependency Graph for Recomputable Nodes")
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Memory")
        plt.show()
