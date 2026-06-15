# Owner(s): ["module: functorch"]
from unittest.mock import MagicMock, patch

from torch._functorch._activation_checkpointing.ac_logging_utils import (
    create_activation_checkpointing_logging_structure_payload,
    create_joint_graph_edges,
    create_joint_graph_node_information,
    create_structured_trace_for_min_cut_info,
)
from torch.fx import Graph, Node
from torch.testing._internal.common_utils import run_tests, TestCase


class TestAcLogging(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.graph: MagicMock = MagicMock(spec=Graph)
        self.node1: MagicMock = MagicMock(spec=Node)
        self.node2: MagicMock = MagicMock(spec=Node)

        self.node1.name = "node1"
        self.node1.target = "target1"
        self.node1.meta = {
            "tensor_meta": MagicMock(shape=(2, 2)),
            "stack_trace": "trace1",
        }
        self.node1.all_input_nodes = []

        self.node2.name = "node2"
        self.node2.target = "target2"
        self.node2.meta = {"tensor_meta": None, "stack_trace": "trace2"}
        self.node2.all_input_nodes = [self.node1]

        self.graph.nodes = [self.node1, self.node2]

        self.all_recomputable_banned_nodes: list[Node] = [self.node1]
        self.saved_node_idxs: list[int] = [0]
        self.recomputable_node_idxs: list[int] = []
        self.expected_runtime: int = 100
        self.memories_banned_nodes: list[int] = [50]
        self.normalized_memories_banned_nodes: list[float] = [0.10344827586206896]
        self.runtimes_banned_nodes: list[int] = [10]
        self.min_cut_saved_values: list[Node] = [self.node1]
        self.memory_budget: float = 0.5
        self.min_act_size: float = 0.001
        self.max_act_size: float = 0.01
        self.saved_values_act_size: float = 0.005
        self.more_aggressive_saved_values_mem_ratio: float = 0.9
        self.aggressive_recomputation_saved_values_mem_ratio: float = 0.8

    def test_create_joint_graph_node_information(self) -> None:
        recomputable_node_info: dict[str, int] = {"node1": 0}
        expected_output: dict[str, dict] = {
            "node1": {
                "index": 0,
                "name": "node1",
                "is_recomputable_candidate": True,
                "target": "target1",
                "shape": "(2, 2)",
                "input_arguments": [],
                "stack_trace": "trace1",
                "recomputable_candidate_info": {"recomputable_node_idx": 0},
            },
            "node2": {
                "index": 1,
                "name": "node2",
                "is_recomputable_candidate": False,
                "target": "target2",
                "shape": "[]",
                "input_arguments": ["node1"],
                "stack_trace": "trace2",
            },
        }
        result = create_joint_graph_node_information(self.graph, recomputable_node_info)
        self.assertEqual(result, expected_output)

    def test_create_joint_graph_edges(self) -> None:
        expected_edges: list[tuple[str, str]] = [("node1", "node2")]
        result = create_joint_graph_edges(self.graph)
        self.assertEqual(result, expected_edges)

    def test_create_activation_checkpointing_logging_structure_payload(self) -> None:
        input_joint_graph_node_information: dict[str, dict] = {
            "node1": {
                "index": 0,
                "name": "node1",
                "is_recomputable_candidate": True,
                "target": "target1",
                "shape": "(2, 2)",
                "input_arguments": [],
                "stack_trace": "trace1",
                "recomputable_candidate_info": {"recomputable_node_idx": 0},
            }
        }
        joint_graph_edges: list[tuple[str, str]] = [("node1", "node2")]
        expected_payload: dict[str, any] = {
            "Joint Graph Size": 2,
            "Joint Graph Edges": {"Total": 1, "Edges": joint_graph_edges},
            "Joint Graph Node Information": input_joint_graph_node_information,
            "Recomputable Banned Nodes Order": ["node1"],
            "Expected Runtime": self.expected_runtime,
            "Knapsack Saved Nodes": self.saved_node_idxs,
            "Knapsack Recomputed Nodes": self.recomputable_node_idxs,
            "Knapsack Input Memories": self.normalized_memories_banned_nodes,
            "Absolute Memories": self.memories_banned_nodes,
            "Knapsack Input Runtimes": self.runtimes_banned_nodes,
            "Min Cut Solution Saved Values": ["node1"],
            "Memory Budget": self.memory_budget,
            "Min Activation Size (GB)": self.min_act_size,
            "Max Activation Size (GB)": self.max_act_size,
            "Saved Values Activation Size (GB)": self.saved_values_act_size,
            "More Aggressive Saved Values Mem Ratio": self.more_aggressive_saved_values_mem_ratio,
            "Aggressive Recomputation Saved Values Mem Ratio": self.aggressive_recomputation_saved_values_mem_ratio,
        }
        result = create_activation_checkpointing_logging_structure_payload(
            joint_graph=self.graph,
            joint_graph_node_information=input_joint_graph_node_information,
            joint_graph_edges=joint_graph_edges,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            expected_runtime=self.expected_runtime,
            saved_node_idxs=self.saved_node_idxs,
            recomputable_node_idxs=self.recomputable_node_idxs,
            memories_banned_nodes=self.memories_banned_nodes,
            normalized_memories_banned_nodes=self.normalized_memories_banned_nodes,
            runtimes_banned_nodes=self.runtimes_banned_nodes,
            min_cut_saved_values=self.min_cut_saved_values,
            memory_budget=self.memory_budget,
            min_act_size=self.min_act_size,
            max_act_size=self.max_act_size,
            saved_values_act_size=self.saved_values_act_size,
            more_aggressive_saved_values_mem_ratio=self.more_aggressive_saved_values_mem_ratio,
            aggressive_recomputation_saved_values_mem_ratio=self.aggressive_recomputation_saved_values_mem_ratio,
        )
        self.assertEqual(result, expected_payload)

    @patch(
        "torch._functorch._activation_checkpointing.ac_logging_utils.trace_structured"
    )
    @patch("json.dumps", return_value="mocked_payload")
    def test_create_structured_trace_for_min_cut_info(
        self, mock_json_dumps: MagicMock, mock_trace_structured: MagicMock
    ) -> None:
        create_structured_trace_for_min_cut_info(
            joint_graph=self.graph,
            all_recomputable_banned_nodes=self.all_recomputable_banned_nodes,
            saved_node_idxs=self.saved_node_idxs,
            recomputable_node_idxs=self.recomputable_node_idxs,
            expected_runtime=self.expected_runtime,
            memories_banned_nodes=self.memories_banned_nodes,
            normalized_memories_banned_nodes=self.normalized_memories_banned_nodes,
            runtimes_banned_nodes=self.runtimes_banned_nodes,
            min_cut_saved_values=self.min_cut_saved_values,
            memory_budget=self.memory_budget,
            min_act_size=self.min_act_size,
            max_act_size=self.max_act_size,
            saved_values_act_size=self.saved_values_act_size,
            more_aggressive_saved_values_mem_ratio=self.more_aggressive_saved_values_mem_ratio,
            aggressive_recomputation_saved_values_mem_ratio=self.aggressive_recomputation_saved_values_mem_ratio,
        )

        self.assertEqual(mock_trace_structured.call_count, 1)

        metadata_fn_result = mock_trace_structured.call_args[1]["metadata_fn"]()
        payload_fn_result = mock_trace_structured.call_args[1]["payload_fn"]()

        self.assertEqual(
            metadata_fn_result,
            {
                "name": "min_cut_information",
                "encoding": "json",
            },
        )
        self.assertEqual(payload_fn_result, "mocked_payload")

        mock_json_dumps.assert_called_once()


if __name__ == "__main__":
    run_tests()
