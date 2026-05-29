#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Unit tests for wait_tensors proxy propagation in Dynamo tracing.

These tests verify that after calling work.wait(), the returned tensors
use the wait_tensors output proxy instead of the pre-wait collective output.

This is critical for correctness - if the return uses pre-wait tensors,
the data may not be ready yet, causing race conditions and NaN values.
"""

import logging
import os

from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

import os
import sys


# Make test/comms importable so `helpers` / `integration` resolve when this
# file is run directly (run_test.py runs `python comms/unit/<file>.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.comm_test_helpers import skip_if_torch_compile_not_supported_or_enabled

import torch
import torch._dynamo
from torch.comms import new_comm, ReduceOp


logger = logging.getLogger(__name__)


def _get_graph_output_sources(gm) -> list[str]:
    """Get the node names that are used in the graph's output."""
    output_node = None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_node = node
            break

    if output_node is None:
        return []

    def get_source_names(arg):
        if isinstance(arg, torch.fx.Node):
            return [arg.name]
        elif isinstance(arg, (list, tuple)):
            names = []
            for item in arg:
                names.extend(get_source_names(item))
            return names
        return []

    return get_source_names(output_node.args[0])


def _find_nodes_by_name_pattern(gm, pattern: str) -> list:
    """Find all nodes whose target contains the given pattern.

    Also searches inside with_effects nodes for the wrapped op.
    """
    nodes = []
    for node in gm.graph.nodes:
        if node.op == "call_function":
            target_name = str(node.target)
            if pattern in target_name:
                nodes.append(node)
            # Also check if this is a with_effects wrapping an op that matches
            elif "with_effects" in target_name and len(node.args) >= 2:
                # with_effects(token, op, *args) - check the op argument
                wrapped_op = node.args[1]
                wrapped_op_name = str(wrapped_op)
                if pattern in wrapped_op_name:
                    nodes.append(node)
    return nodes


def _check_node_in_output_path(gm, target_node) -> bool:
    """Check if target_node is in the data flow path to the output."""
    output_sources = _get_graph_output_sources(gm)

    # Build reverse dependency map: node -> nodes that use it
    users = {}
    for node in gm.graph.nodes:
        if hasattr(node, "args"):
            for arg in node.args:
                if isinstance(arg, torch.fx.Node):
                    if arg.name not in users:
                        users[arg.name] = []
                    users[arg.name].append(node.name)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, torch.fx.Node):
                            if item.name not in users:
                                users[item.name] = []
                            users[item.name].append(node.name)

    # BFS from target_node to see if we reach any output source
    visited = set()
    queue = [target_node.name]

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        if current in output_sources:
            return True

        if current in users:
            queue.extend(users[current])

    return False


@skipIfTorchDynamo("captures graphs via torch.compile; outer dynamo breaks capture")
@skip_if_torch_compile_not_supported_or_enabled()
class TestWaitProxyPropagation(TestCase):
    """Test that wait_tensors output proxies are properly propagated."""

    def setUp(self):
        from torch.comms.functional import collectives  # noqa: F401

        torch._dynamo.reset()

    def tearDown(self):
        torch._dynamo.reset()

    def _create_graph_capture_backend(self):
        """Create a backend that captures the graph for inspection."""
        captured = {"graph": None}

        def backend(gm, example_inputs):
            captured["graph"] = gm
            print("\n" + "=" * 80)
            print("CAPTURED FX GRAPH:")
            print("=" * 80)
            print(gm.print_readable(print_output=False))
            print("=" * 80 + "\n")
            return gm

        return backend, captured

    def test_wait_output_in_return_path(self):
        """Test that wait_tensors output is in the return path, not pre-wait collective output.

        This test uses a TorchComm collective with async_op=True, waits on the result,
        and returns the tensor. The FX graph should show that the returned tensor
        flows through wait_tensors, not directly from the collective.
        """
        comm = new_comm("fake", torch.device("cpu"), name="test_wait_return")
        tensor = torch.ones(4, dtype=torch.float, device="cpu")

        def my_func(comm_arg, t):
            work = comm_arg.all_reduce(t, ReduceOp.SUM, async_op=True)
            work.wait()
            return t

        backend, captured = self._create_graph_capture_backend()
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)
        compiled_func(comm, tensor)

        self.assertIsNotNone(captured["graph"], "Graph should be captured")
        gm = captured["graph"]

        wait_nodes = _find_nodes_by_name_pattern(gm, "wait_tensors")
        all_reduce_nodes = _find_nodes_by_name_pattern(gm, "all_reduce")

        print("\nDiagnostics:")
        print(f"  wait_nodes: {[n.name for n in wait_nodes]}")
        print(f"  all_reduce_nodes: {[n.name for n in all_reduce_nodes]}")
        print(f"  output_sources: {_get_graph_output_sources(gm)}")

        self.assertTrue(len(wait_nodes) > 0, "Graph should contain wait_tensors")
        self.assertTrue(len(all_reduce_nodes) > 0, "Graph should contain all_reduce")

        wait_in_path = any(_check_node_in_output_path(gm, node) for node in wait_nodes)
        self.assertTrue(
            wait_in_path,
            f"wait_tensors output should be in return path. "
            f"Wait nodes: {[n.name for n in wait_nodes]}, "
            f"Output sources: {_get_graph_output_sources(gm)}",
        )

        comm.finalize()
        comm = None
        torch._dynamo.reset()

    def test_list_return_uses_wait_output(self):
        """Test that list of tensors returned after wait uses wait outputs."""

        comm = new_comm("fake", torch.device("cpu"), name="test_wait_list")
        num_ranks = comm.get_size()
        tensor = torch.ones(4, dtype=torch.float, device="cpu") * (comm.get_rank() + 1)
        output_list = [
            torch.zeros(4, dtype=torch.float, device="cpu") for _ in range(num_ranks)
        ]

        def my_func(comm_arg, output, inp):
            work = comm_arg.all_gather(output, inp, async_op=True)
            work.wait()
            return output

        backend, captured = self._create_graph_capture_backend()
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)
        compiled_func(comm, output_list, tensor)

        self.assertIsNotNone(captured["graph"], "Graph should be captured")
        gm = captured["graph"]

        wait_nodes = _find_nodes_by_name_pattern(gm, "wait_tensors")

        print("\nDiagnostics:")
        print(f"  wait_nodes: {[n.name for n in wait_nodes]}")
        print(f"  output_sources: {_get_graph_output_sources(gm)}")

        self.assertTrue(len(wait_nodes) > 0, "Graph should contain wait_tensors")

        wait_in_path = any(_check_node_in_output_path(gm, node) for node in wait_nodes)
        self.assertTrue(
            wait_in_path,
            f"wait_tensors output should be in return path for list. "
            f"Wait nodes: {[n.name for n in wait_nodes]}, "
            f"Output sources: {_get_graph_output_sources(gm)}",
        )

        comm.finalize()
        comm = None
        torch._dynamo.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tests()
