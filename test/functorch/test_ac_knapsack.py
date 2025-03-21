# Owner(s): ["module: functorch"]
from torch._functorch._activation_checkpointing.graph_info_provider import (
    GraphInfoProvider,
)
from torch._functorch._activation_checkpointing.knapsack_evaluator import (
    KnapsackEvaluator,
)
from torch.fx.graph import Graph
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGraphInfoProvider(TestCase):
    """
    Test class for GraphInfoProvider.
    The test class sets up a small graph example and tests the methods validating the graph building logic.
    """

    def setUp(self) -> None:
        super().setUp()
        self.graph_nodes_in_order = [
            "node1",
            "node2",
            "node3",
            "node4",
            "node5",
            "output",
        ]
        self.graph_edges = [
            ("node1", "node2"),
            ("node2", "node3"),
            ("node3", "node4"),
            ("node4", "node5"),
            ("node5", "output"),
            ("node1", "output"),
        ]
        self.all_recomputable_banned_nodes = ["node1", "node2", "node5"]
        self.recorded_knapsack_input_memories = [1.0, 1.0, 1.0]
        self.recorded_knapsack_input_runtimes = [1.0, 1.0, 1.0]
        self.graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )

    def test_inialize_from_graph(self):
        joint_graph = Graph()
        node1 = joint_graph.placeholder("node1")
        node2 = joint_graph.call_function(lambda x: x, (node1,))
        node2.name = "node2"
        node3 = joint_graph.call_function(lambda x: x, (node2,))
        node3.name = "node3"
        node4 = joint_graph.call_function(lambda x: x, (node3,))
        node4.name = "node4"
        node5 = joint_graph.call_function(lambda x: x, (node4,))
        node5.name = "node5"
        output = joint_graph.call_function(lambda x, y: (x, y), (node5, node1))
        output.name = "output"
        all_recomputable_banned_nodes = [node1, node2, node5]
        recorded_knapsack_input_memories = [1.0, 1.0, 1.0]
        recorded_knapsack_input_runtimes = [1.0, 1.0, 1.0]
        graph_info_provider = GraphInfoProvider.inialize_from_graph(
            joint_graph=joint_graph,
            all_recomputable_banned_nodes=all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=recorded_knapsack_input_runtimes,
        )
        self.assertEqual(
            graph_info_provider.graph_nodes_in_order,
            ["node1", "node2", "node3", "node4", "node5", "output"],
        )
        self.assertEqual(
            sorted(graph_info_provider.graph_edges),
            sorted(
                [
                    ("node1", "node2"),
                    ("node2", "node3"),
                    ("node3", "node4"),
                    ("node4", "node5"),
                    ("node5", "output"),
                    ("node1", "output"),
                ]
            ),
        )
        self.assertEqual(
            graph_info_provider.all_recomputable_banned_nodes,
            ["node1", "node2", "node5"],
        )

    def test_get_non_ac_peak_memory(self):
        self.assertEqual(
            self.graph_info_provider.get_non_ac_peak_memory(),
            sum(self.recorded_knapsack_input_memories),
        )

    def test_get_theoretical_max_runtime(self):
        self.assertEqual(
            self.graph_info_provider.get_theoretical_max_runtime(),
            sum(self.recorded_knapsack_input_runtimes),
        )

    def test_get_knapsack_memory_input(self):
        self.assertEqual(
            self.graph_info_provider.get_knapsack_memory_input(),
            self.recorded_knapsack_input_memories,
        )

    def test_get_knapsack_runtime_input(self):
        self.assertEqual(
            self.graph_info_provider.get_knapsack_runtime_input(),
            self.recorded_knapsack_input_runtimes,
        )

    def test_recomputable_node_only_graph(self):
        recomputable_node_only_graph = (
            self.graph_info_provider.recomputable_node_only_graph
        )
        expected_nodes = self.all_recomputable_banned_nodes
        expected_edges = [("node1", "node2")]
        self.assertEqual(list(recomputable_node_only_graph.nodes), expected_nodes)
        self.assertEqual(
            sorted(recomputable_node_only_graph.edges), sorted(expected_edges)
        )

    def test_recomputable_node_only_graph_with_larger_graph_context(self):
        recomputable_node_only_graph_with_larger_graph_context = (
            self.graph_info_provider.recomputable_node_only_graph_with_larger_graph_context
        )
        expected_nodes = self.all_recomputable_banned_nodes
        # node1 does not have an indirect path to node5 because of node2
        # node2 has an indirect path to node5
        expected_edges = [("node1", "node2"), ("node2", "node5")]
        self.assertEqual(
            sorted(recomputable_node_only_graph_with_larger_graph_context.nodes),
            sorted(expected_nodes),
        )
        self.assertEqual(
            sorted(recomputable_node_only_graph_with_larger_graph_context.edges),
            sorted(expected_edges),
        )

    def test_full_joint_nx_graph(self):
        graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        full_joint_nx_graph = graph_info_provider.full_joint_nx_graph
        expected_nodes = [
            node for node in self.graph_nodes_in_order if node != "output"
        ]
        expected_edges = [
            (u, v) for u, v in self.graph_edges if u != "output" and v != "output"
        ]
        self.assertEqual(list(full_joint_nx_graph.nodes), expected_nodes)
        self.assertEqual(sorted(full_joint_nx_graph.edges), sorted(expected_edges))

    def test_simplified_fx_joint_graph(self):
        graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        simplified_fx_joint_graph = graph_info_provider.simplified_fx_joint_graph
        expected_nodes = self.graph_nodes_in_order
        expected_edges = self.graph_edges
        self.assertEqual(
            [node.name for node in simplified_fx_joint_graph.nodes], expected_nodes
        )
        self.assertEqual(
            sorted(
                [
                    (node.name, user.name)
                    for node in simplified_fx_joint_graph.nodes
                    for user in node.users
                ]
            ),
            sorted(expected_edges),
        )


class TestKnapsackEvaluator(TestCase):
    """
    Test class for KnapsackEvaluator.
    The test class sets up a small graph example and tests the methods validating the knapsack evaluation logic.
    """

    def setUp(self) -> None:
        super().setUp()
        self.graph_nodes_in_order = [
            "node1",
            "node2",
            "node3",
            "node4",
            "node5",
            "output",
        ]
        self.graph_edges = [
            ("node1", "node2"),
            ("node2", "node3"),
            ("node3", "node4"),
            ("node4", "node5"),
            ("node5", "output"),
            ("node1", "output"),
        ]
        self.all_recomputable_banned_nodes = ["node1", "node2", "node5"]
        self.recorded_knapsack_input_memories = [0.1, 0.2, 0.2]
        self.recorded_knapsack_input_runtimes = [100.0, 50.0, 51.0]
        self.graph_info_provider = GraphInfoProvider(
            graph_nodes_in_order=self.graph_nodes_in_order,
            graph_edges=self.graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            recorded_knapsack_input_memories=self.recorded_knapsack_input_memories,
            recorded_knapsack_input_runtimes=self.recorded_knapsack_input_runtimes,
        )
        self.knapsack_evaluator = KnapsackEvaluator(
            graph_info_provider=self.graph_info_provider
        )
        self.knapsack_algo = lambda memory_values, runtime_values, memory_budget: {
            0.1: (101.0, [0], [1, 2]),
            0.2: (101.0, [0], [1, 2]),
            0.3: (50.0, [0, 2], [1]),
            0.4: (50.0, [0, 2], [1]),
            0.5: (0.0, [0, 1, 2], []),
        }.get(memory_budget, (0.0, [0, 1, 2], []))

    def test_evaluate_knapsack_output_not_accounting_for_backward_pass(self):
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1, 2]
        result = self.knapsack_evaluator.evaluate_knapsack_output(
            saved_nodes_idxs=saved_nodes_idxs,
            recomputable_node_idxs=recomputable_node_idxs,
        )
        self.assertEqual(result["peak_memory"], 0.1)
        self.assertEqual(result["recomputation_runtime"], 101.0)

    def test_evaluate_knapsack_output_accounting_for_backward_pass(self):
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1, 2]
        result = self.knapsack_evaluator.evaluate_knapsack_output(
            saved_nodes_idxs=saved_nodes_idxs,
            recomputable_node_idxs=recomputable_node_idxs,
            account_for_backward_pass=True,
        )
        self.assertEqual(result["peak_memory"], 0.5)
        self.assertEqual(result["recomputation_runtime"], 101.0)

    def test_evaluate_knapsack_output_with_wrong_sized_values(self):
        saved_nodes_idxs = [0]
        recomputable_node_idxs = [1]
        with self.assertRaises(AssertionError):
            self.knapsack_evaluator.evaluate_knapsack_output(
                saved_nodes_idxs=saved_nodes_idxs,
                recomputable_node_idxs=recomputable_node_idxs,
            )

    def test_evaluate_distribution_of_results_for_knapsack_algo(self):
        memory_budget_values = [0.1, 0.2, 0.3]
        results = (
            self.knapsack_evaluator.evaluate_distribution_of_results_for_knapsack_algo(
                knapsack_algo=self.knapsack_algo,
                memory_budget_values=memory_budget_values,
            )
        )
        self.assertEqual(len(results), len(memory_budget_values))
        self.assertEqual(results[0]["memory_budget"], 0.1)
        self.assertEqual(results[0]["peak_memory"], 0.1)
        self.assertEqual(results[0]["recomputation_runtime"], 101)
        self.assertEqual(results[1]["non_ac_peak_memory"], 0.5)
        self.assertEqual(results[1]["theoretical_max_runtime"], 201)
        self.assertEqual(results[2]["percentage_of_theoretical_peak_memory"], 0.3 / 0.5)
        self.assertEqual(
            results[2]["percentage_of_theoretical_peak_runtime"], 50.0 / 201
        )

    def test_get_knee_point_memory_budget(self):
        max_mem_budget = 1.0
        min_mem_budget = 0.1
        iterations = 10
        knee_point_memory_budget = self.knapsack_evaluator.get_knee_point_memory_budget(
            knapsack_algo=self.knapsack_algo,
            max_mem_budget=max_mem_budget,
            min_mem_budget=min_mem_budget,
            iterations=iterations,
        )
        self.assertEqual(knee_point_memory_budget, 0.4)

    def test_get_backward_memory_from_topologically_sorted_graph(self):
        result = self.knapsack_evaluator._get_backward_memory_from_topologically_sorted_graph(
            node_graph=self.graph_info_provider.recomputable_node_only_graph_with_larger_graph_context,
            node_memories=self.graph_info_provider.all_node_memories,
            saved_nodes_set={"node1"},
            peak_memory_after_forward_pass=0.1,
        )
        expected_result = [
            (0.1, "Initial Peak/Current Memory"),
            (0.3, "Recomputing Node: node5"),
            (0.5, "Recomputing Predecessor of node5: node2"),
            (0.3, "Dropping Node: node5"),
            (0.1, "Dropping Node(already saved): node2"),
            (0.0, "Dropping Node(already saved): node1"),
        ]
        print(result, expected_result)
        for result_item, expected_result_item in zip(result, expected_result):
            self.assertAlmostEqual(result_item[0], expected_result_item[0])
            self.assertEqual(result_item[1], expected_result_item[1])


if __name__ == "__main__":
    run_tests()
