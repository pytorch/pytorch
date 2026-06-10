#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]
# Copyright (c) Meta Platforms, Inc. and affiliates.

import itertools
import logging
import os
import unittest


os.environ["TORCHCOMMS_PATCH_FOR_COMPILE"] = "1"

from helpers.comm_test_helpers import skip_if_torch_compile_not_supported_or_enabled
from integration.helpers.TorchCommTestHelpers import skip_backend

import torch
from torch.comms.functional import is_torch_compile_supported_and_enabled


if is_torch_compile_supported_and_enabled():
    from integration.helpers.TorchCommTestHelpers import (
        get_dtype_name,
        get_op_name,
        is_full_sweep,
        TorchCommTestWrapper,
    )

    from torch.comms import ReduceOp, Timeout
else:
    from integration.helpers.TorchCommTestHelpers import is_full_sweep

    ReduceOp = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@skip_if_torch_compile_not_supported_or_enabled()
class FullgraphCompileTest(unittest.TestCase):
    """Test class for torch.compile fullgraph mode with TorchComm operations."""

    # Class variables for test parameters
    counts = [4, 1024]
    dtypes = [torch.float, torch.int] if is_full_sweep() else [torch.float]
    ops = (
        ([ReduceOp.SUM, ReduceOp.MAX] if is_full_sweep() else [ReduceOp.SUM])
        if ReduceOp is not None
        else []
    )

    def get_wrapper(self):
        return TorchCommTestWrapper()

    def setUp(self):
        """Set up test environment before each test."""
        # Clear dynamo and inductor caches to ensure fresh compilation for each test
        import torch._dynamo
        import torch._inductor.codecache
        import torch._inductor.config
        import torch._inductor.utils

        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear()
        torch._inductor.utils.clear_caches()

        # Enable inductor debug logging to see generated code
        torch._inductor.config.debug = True
        torch._inductor.config.trace.enabled = True

        self.wrapper = self.get_wrapper()
        self.torchcomm = self.wrapper.get_torchcomm()
        self.rank = self.torchcomm.get_rank()
        self.num_ranks = self.torchcomm.get_size()
        self.device = self.torchcomm.get_device()

    def _create_graph_logging_backend(self, test_name: str):
        """
        Create a custom backend that captures and logs the compiled graph.

        Args:
            test_name: Name of the test for logging purposes

        Returns:
            A backend function that logs the graph and passes to inductor
        """

        def _read_and_log_inductor_artifacts():
            """Read and log inductor output_code from the code cache."""
            import torch._inductor.codecache as codecache

            logger.info("\n%s\nINDUCTOR OUTPUT CODE\n%s", "=" * 80, "=" * 80)

            # Get the cache directory and find the most recent output_code file
            try:
                import glob
                import os

                cache_dir = codecache.cache_dir()
                if cache_dir and os.path.exists(cache_dir):
                    # Find output_code*.py files (inductor's generated wrapper code)
                    pattern = os.path.join(cache_dir, "**", "*.py")
                    py_files = glob.glob(pattern, recursive=True)
                    # Sort by modification time, most recent first
                    py_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

                    for py_file in py_files[:3]:  # Log the 3 most recent
                        try:
                            with open(py_file) as f:
                                code = f.read()
                            logger.info("\n--- %s ---", os.path.basename(py_file))
                            logger.info(code)
                        except Exception as e:
                            logger.warning("Could not read %s: %s", py_file, e)
                else:
                    logger.info("Cache dir not found: %s", cache_dir)
            except Exception as e:
                logger.warning("Could not access inductor cache: %s", e)

            logger.info("%s\n", "=" * 80)

        def graph_capture_backend(gm, example_inputs):
            """Custom backend that captures the graph before passing to inductor."""
            logger.info(
                "\n%s\nDYNAMO CAPTURED GRAPH for %s\n%s", "=" * 80, test_name, "=" * 80
            )
            logger.info("\nGraph code:")
            logger.info(gm.code)
            logger.info("\nGraph has %s nodes", len(list(gm.graph.nodes)))
            logger.info("\nGraph nodes:")

            for node in gm.graph.nodes:
                logger.info("  %s: %s -> %s", node.op, node.target, node.name)

            has_torchcomms = any(
                "torchcomms" in str(node.target) for node in gm.graph.nodes
            )
            logger.info("\nContains torchcomms ops: %s", has_torchcomms)

            # Check for custom ops that might bypass inductor
            custom_ops = [
                node
                for node in gm.graph.nodes
                if node.op == "call_function" and "torchcomms" in str(node.target)
            ]
            if custom_ops:
                logger.info("\nFound %s torchcomms custom ops:", len(custom_ops))
                for op in custom_ops:
                    logger.info("  - %s", op.target)

            logger.info("%s\n", "=" * 80)

            logger.info("Calling compile_fx (inductor default)...")
            logger.info(
                "Example inputs: %s",
                [
                    inp.shape if hasattr(inp, "shape") else type(inp)
                    for inp in example_inputs
                ],
            )

            try:
                from torch._inductor.compile_fx import compile_fx, compile_fx_inner

                # Create a wrapper to log the post-AOT/functionalized graph
                def log_and_compile(gm, example_inputs, **kwargs):
                    """Log the graph after AOT/functionalization, then call default compiler."""
                    # Log fx_graph_readable
                    logger.info(
                        "\n%s\nFX GRAPH READABLE for %s\n%s",
                        "=" * 80,
                        test_name,
                        "=" * 80,
                    )
                    try:
                        # print_readable() returns a string representation
                        readable = gm.print_readable(print_output=False)
                        logger.info(readable)
                    except Exception as e:
                        logger.info("Could not get readable graph: %s", e)
                        # Fallback to code
                        logger.info(gm.code)
                    logger.info("%s\n", "=" * 80)

                    # Call the default inductor compiler
                    result = compile_fx_inner(gm, example_inputs, **kwargs)

                    # Read and log the generated inductor output_code
                    _read_and_log_inductor_artifacts()

                    return result

                compiled = compile_fx(gm, example_inputs, inner_compile=log_and_compile)
                logger.info("Compilation complete, result type: %s", type(compiled))
                return compiled
            except Exception as e:
                logger.error("Compilation failed: %s", e)
                import traceback

                logger.error(traceback.format_exc())
                raise

        return graph_capture_backend

    def tearDown(self):
        """Clean up after each test."""
        # Reset Dynamo FIRST to release compiled graphs that hold opaque object references
        # This ensures the wrapper's __del__ can properly finalize when we release it
        torch._dynamo.reset()

        # Release references - wrapper's __del__ will call finalize()
        self.torchcomm = None
        self.wrapper = None

    def _test_fullgraph_compile_all_reduce(self, count, dtype, op, async_op):
        """Test torch.compile with fullgraph=True for all_reduce operation."""
        logger.info(
            "Testing fullgraph compile all_reduce with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(count, dtype)
        original_values = tensor.clone()

        # Define function to compile
        def my_func(t):
            res = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res.wait()
            t *= 10
            return t

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create a custom backend that logs the graph
        test_name = f"all_reduce (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(tensor)

        # Verify results
        self._verify_all_reduce_results(result, op, original_values)

    def _test_fullgraph_compile_all_reduce_multiple_calls(
        self, count, dtype, op, async_op=False
    ):
        """Test torch.compile with fullgraph=True for multiple all_reduce calls."""
        logger.info(
            "Testing fullgraph compile all_reduce with multiple calls with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(count, dtype)
        original_values = tensor.clone()

        # Define function with multiple operations
        def my_func(t):
            res1 = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res1.wait()
            t *= 2
            res2 = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res2.wait()
            t += 5
            return t

        # Reset dynamo and create graph logging backend
        import torch._dynamo

        torch._dynamo.reset()

        test_name = f"all_reduce_multiple (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function
        result = compiled_func(tensor)

        # Verify results
        self._verify_multiple_all_reduce_results(result, op, original_values)

    def _create_input_tensor(self, count, dtype):
        """Create input tensor with rank-specific values."""
        options = {"dtype": dtype, "device": self.device}
        if dtype == torch.float or dtype == torch.bfloat16:
            return torch.ones(count, **options) * float(self.rank + 1)
        elif dtype == torch.int:
            return torch.ones(count, **options) * int(self.rank + 1)
        elif dtype == torch.int8:
            return torch.ones(count, **options) * int(self.rank + 1)
        return None

    def _calculate_expected_all_reduce_result(self, op):
        """Calculate expected result for all_reduce based on operation."""
        if op == ReduceOp.SUM:
            return self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            return self.num_ranks
        else:
            raise RuntimeError("Unsupported reduce operation")

    def _calculate_expected_reduce_result(self, op):
        """Calculate expected result for reduce based on operation."""
        if op == ReduceOp.SUM:
            return self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            return self.num_ranks
        else:
            raise RuntimeError("Unsupported reduce operation")

    def synchronize_stream(self):
        """Synchronize the current stream."""
        if self.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    def _verify_all_reduce_results(self, output_tensor, op, original_values):
        """Verify the results of the all_reduce operation."""
        # Calculate expected result: all_reduce result * 10
        expected = self._calculate_expected_all_reduce_result(op) * 10

        # Compare output with expected tensor
        description = f"fullgraph compile all_reduce with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def _verify_multiple_all_reduce_results(self, output_tensor, op, original_values):
        """Verify the results of multiple all_reduce operations."""
        # Calculate expected result:
        # First all_reduce: sum/max of (rank+1)
        # Then multiply by 2: first_result * 2
        # Second all_reduce: sum/max of (first_result * 2)
        # Then add 5: second_result + 5

        first_reduce = self._calculate_expected_all_reduce_result(op)
        after_mult = first_reduce * 2

        if op == ReduceOp.SUM:
            second_reduce = after_mult * self.num_ranks
        elif op == ReduceOp.MAX:
            second_reduce = after_mult
        else:
            raise RuntimeError("Unsupported reduce operation")

        expected = second_reduce + 5

        # Compare output with expected tensor
        description = f"fullgraph compile multiple all_reduce with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def _test_fullgraph_compile_reduce(self, count, dtype, op, async_op):
        """Test torch.compile with fullgraph=True for reduce operation."""
        logger.info(
            "Testing fullgraph compile reduce with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(count, dtype)
        original_values = tensor.clone()

        # Use root rank 0 for reduce
        root = 0

        # Define function to compile
        def my_func(t):
            res = self.torchcomm.reduce(t, root, op, async_op=async_op)
            if async_op:
                res.wait()
            t *= 10
            return t

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create a custom backend that logs the graph
        test_name = f"reduce (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(tensor)

        # Verify results
        self._verify_reduce_results(result, op, original_values, root)

    def _verify_reduce_results(self, output_tensor, op, original_values, root):
        """Verify the results of the reduce operation."""
        # Only root rank should have reduced values, others keep original * 10
        if self.rank == root:
            # Root rank: reduced result * 10
            expected = self._calculate_expected_reduce_result(op) * 10
        else:
            # Non-root ranks: original value * 10
            expected = (self.rank + 1) * 10

        # Compare output with expected tensor
        description = f"fullgraph compile reduce with op {get_op_name(op)}, root={root}, rank={self.rank}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def _test_fullgraph_compile_all_reduce_premul_sum(self, async_op=False):
        """Test torch.compile with fullgraph=True for all_reduce with PREMUL_SUM operation."""
        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # PREMUL_SUM only works with float types
        dtype = torch.float
        factor = 2.0
        op = ReduceOp.PREMUL_SUM(factor)

        logger.info(
            "Testing fullgraph compile all_reduce with PREMUL_SUM(factor=%s), async_op=%s",
            factor,
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(4, dtype)

        # Define function to compile
        def my_func(t):
            res = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res.wait()
            return t

        # Create a custom backend that logs the graph
        test_name = f"all_reduce_premul_sum (factor={factor}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(tensor)

        # Calculate expected result:
        # Each rank contributes (rank + 1) * factor, then summed across all ranks
        # = factor * sum(rank + 1 for all ranks) = factor * (1 + 2 + ... + num_ranks)
        # = factor * num_ranks * (num_ranks + 1) / 2
        expected = factor * self.num_ranks * (self.num_ranks + 1) / 2

        description = f"fullgraph compile all_reduce PREMUL_SUM(factor={factor})"
        expected_tensor = torch.full_like(result.cpu(), float(expected))
        self.assertTrue(
            torch.allclose(result.cpu(), expected_tensor),
            f"Tensors not close enough for {description}. Expected {expected}, got {result[0].item()}",
        )

        logger.info("All_reduce PREMUL_SUM test passed for rank %s", self.rank)

    def _test_fullgraph_compile_broadcast(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for broadcast operation."""
        logger.info(
            "Testing fullgraph compile broadcast with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(count, dtype)
        original_values = tensor.clone()

        # Use root rank 0 for broadcast
        root = 0

        # Define function to compile
        def my_func(t):
            res = self.torchcomm.broadcast(t, root, async_op=async_op)
            if async_op:
                res.wait()
            t *= 10
            return t

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create a custom backend that logs the graph
        test_name = f"broadcast (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(tensor)

        # Verify results
        self._verify_broadcast_results(result, original_values, root)

    def _verify_broadcast_results(self, output_tensor, original_values, root):
        """Verify the results of the broadcast operation."""
        # All ranks should have the root's original value * 10
        # Root had value (root + 1) = 1 (since root=0)
        expected = (root + 1) * 10

        # Compare output with expected tensor
        description = f"fullgraph compile broadcast with root={root}, rank={self.rank}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def _test_fullgraph_compile_barrier(self, count, dtype, op, async_op):
        """Test torch.compile with fullgraph=True for barrier synchronization."""
        logger.info(
            "Testing fullgraph compile barrier with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Create input tensor with rank-specific values
        tensor = self._create_input_tensor(count, dtype)
        original_values = tensor.clone()

        # Define function with barrier - ensures all ranks sync between operations
        def my_func(t):
            # First all_reduce
            res1 = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res1.wait()
            t *= 2

            # Barrier - ensures all ranks finish first all_reduce before second
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()

            # Second all_reduce after barrier
            res2 = self.torchcomm.all_reduce(t, op, async_op=async_op)
            if async_op:
                res2.wait()
            t += 5
            return t

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create a custom backend that logs the graph
        test_name = f"barrier (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(tensor)

        # Verify results
        self._verify_barrier_results(result, op, original_values)

    def _verify_barrier_results(self, output_tensor, op, original_values):
        """Verify the results of the barrier test."""
        # Calculate expected result:
        # First all_reduce: sum/max of (rank+1)
        # Then multiply by 2: first_result * 2
        # Barrier (no effect on values)
        # Second all_reduce: sum/max of (first_result * 2)
        # Then add 5: second_result + 5

        first_reduce = self._calculate_expected_all_reduce_result(op)
        after_mult = first_reduce * 2

        if op == ReduceOp.SUM:
            second_reduce = after_mult * self.num_ranks
        elif op == ReduceOp.MAX:
            second_reduce = after_mult
        else:
            raise RuntimeError("Unsupported reduce operation")

        expected = second_reduce + 5

        # Compare output with expected tensor
        description = f"fullgraph compile barrier with op {get_op_name(op)}"

        # Create expected tensor with the same size and dtype as output
        if output_tensor.dtype == torch.float:
            expected_tensor = torch.full_like(output_tensor.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(output_tensor.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(output_tensor.cpu(), expected)
            self.assertTrue(
                torch.equal(output_tensor.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

    def _test_fullgraph_compile_all_gather(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for all_gather operation."""
        logger.info(
            "Testing fullgraph compile all_gather with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(t_list, t_input):
            res = self.torchcomm.all_gather(t_list, t_input, async_op=async_op)
            if async_op:
                res.wait()
            return t_list

        # Create tensors for compiled run
        input_tensor = self._create_input_tensor(count, dtype)
        tensor_list = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]

        # Create a custom backend that logs the graph
        test_name = f"all_gather (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution to ensure all ranks compiled
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function (this triggers compilation)
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(tensor_list, input_tensor)

        logger.info(
            "Rank %s: tensor_list[0] = %s, tensor_list[%s] = %s",
            self.rank,
            result[0][0].item(),
            self.rank,
            result[self.rank][0].item(),
        )

        # Verify results: each tensor_list[i] should contain value (i + 1)
        for rank_idx in range(self.num_ranks):
            expected_value = rank_idx + 1
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[rank_idx].cpu(), expected_tensor),
                    f"tensor_list[{rank_idx}] mismatch: expected {expected_value}, "
                    f"got {result[rank_idx][0].item()}",
                )
            else:
                expected_tensor = torch.full(
                    (count,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[rank_idx].cpu(), expected_tensor),
                    f"tensor_list[{rank_idx}] mismatch: expected {expected_value}, "
                    f"got {result[rank_idx][0].item()}",
                )

        logger.info("All_gather test passed for rank %s", self.rank)

    def _test_fullgraph_compile_window_put_get(
        self, count, dtype, signal, window_is_cpu, async_op
    ):
        """Test torch.compile with fullgraph=True for window put/get operations."""
        self.torchcomm.barrier(async_op=False)

        logger.info(
            "Testing fullgraph compile window put/get with count=%s, dtype=%s, signal=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            signal,
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create window on all ranks - need space for all ranks' tensors
        # Window is divided into sections: rank i's section is at offset i * count
        window = self.torchcomm.new_window()
        buf_device = torch.device("cpu") if window_is_cpu else self.device
        # Create buffer with correct dtype and size in elements (not bytes)
        buf_tensor = torch.empty(self.num_ranks * count, dtype=dtype, device=buf_device)
        window.tensor_register(buf_tensor)

        # Use try/finally to ensure cleanup even if test fails
        try:
            # Define function to compile - takes window as parameter to avoid closure capture
            # Each rank puts its tensor to the next rank's section in the window
            def my_func(t, win):
                barrier_res1 = self.torchcomm.barrier(async_op=async_op)
                if async_op:
                    barrier_res1.wait()

                dst_rank = (self.rank + 1) % self.num_ranks
                src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks

                # Put tensor to destination rank's section (at offset dst_rank * count)
                put_res = win.put(
                    t, dst_rank, target_disp=dst_rank * count, async_op=async_op
                )
                if async_op:
                    put_res.wait()

                # sync to notify remote rank that the put is complete
                if signal:
                    # call signal on current stream to notify remote rank that the put is complete
                    signal_res = win.signal(dst_rank, async_op=async_op)
                    if async_op:
                        signal_res.wait()
                    wait_signal_res = win.wait_signal(src_rank, async_op=async_op)
                    if async_op:
                        wait_signal_res.wait()
                    # sync to ensure that the wait_signal is complete
                    torch.cuda.current_stream().synchronize()
                else:
                    # call barrier on current stream to ensure that the put is complete
                    barrier_res2 = self.torchcomm.barrier(async_op=async_op)
                    if async_op:
                        barrier_res2.wait()

                # Get tensor from my section (at offset self.rank * count, written by previous rank)
                # get_tensor returns the full buffer, then slice to get our section
                full_tensor = win.map_remote_tensor(self.rank)
                result = full_tensor[self.rank * count : (self.rank + 1) * count]

                barrier_res3 = self.torchcomm.barrier(async_op=async_op)
                if async_op:
                    barrier_res3.wait()

                return result

            # Create input tensor
            input_tensor = self._create_input_tensor(count, dtype)

            # Create a custom backend that logs the graph
            test_name = f"window_put_get (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
            backend = self._create_graph_logging_backend(test_name)

            # Compile with custom backend
            logger.info("Rank %s: Compiling function", self.rank)
            compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

            # Execute compiled function (this triggers compilation)
            logger.info("Rank %s: Running compiled version", self.rank)
            compiled_result = compiled_func(input_tensor, window)
            logger.info(
                "Rank %s: Compiled result = %s", self.rank, compiled_result[0:4].cpu()
            )

            # Compute expected result: each rank should receive from previous rank in ring
            # Note: _create_input_tensor creates tensors with value (rank + 1)
            src_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks
            expected_result = torch.ones([count], dtype=dtype, device=self.device) * (
                src_rank + 1
            )

            # Compare result
            description = f"fullgraph compile window put/get, rank={self.rank}"

            if dtype == torch.float:
                self.assertTrue(
                    torch.allclose(compiled_result.cpu(), expected_result.cpu()),
                    f"Compiled result doesn't match expected for {description}. "
                    f"Expected {expected_result[0:4].cpu()}, got {compiled_result[0:4].cpu()}",
                )
            else:
                self.assertTrue(
                    torch.equal(compiled_result.cpu(), expected_result.cpu()),
                    f"Compiled result doesn't match expected for {description}. "
                    f"Expected {expected_result[0:4].cpu()}, got {compiled_result[0:4].cpu()}",
                )

            logger.info("Window put/get test passed for rank %s", self.rank)
        except Exception as e:
            logger.error(
                "Rank %s: Window put/get test failed with exception: %s", self.rank, e
            )
            import traceback

            logger.error("Rank %s: Traceback:\n%s", self.rank, traceback.format_exc())
            raise
        finally:
            # Always clean up and barrier, even if test fails
            del window
            self.torchcomm.barrier(async_op=False)

    def _test_fullgraph_compile_send_recv(self, count, dtype, async_op=False):
        """Test torch.compile with fullgraph=True for send/recv operations."""
        logger.info(
            "Testing fullgraph compile send/recv with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Use simple ring pattern: each rank sends to next rank and receives from previous rank
        send_rank = (self.rank + 1) % self.num_ranks
        recv_rank = (self.rank - 1 + self.num_ranks) % self.num_ranks

        # Define function to compile
        def my_func(send_tensor, recv_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()

            # Alternate send/recv order based on rank to avoid deadlock
            # Even ranks send first, then receive
            # Odd ranks receive first, then send
            if self.rank % 2 == 0:
                # Even ranks: send first, then receive
                send_work = self.torchcomm.send(
                    send_tensor, send_rank, async_op=async_op
                )
                recv_work = self.torchcomm.recv(
                    recv_tensor, recv_rank, async_op=async_op
                )
            else:
                # Odd ranks: receive first, then send
                recv_work = self.torchcomm.recv(
                    recv_tensor, recv_rank, async_op=async_op
                )
                send_work = self.torchcomm.send(
                    send_tensor, send_rank, async_op=async_op
                )

            # Wait for completion
            if async_op:
                send_work.wait()
                recv_work.wait()

            # Process received data
            recv_tensor *= 10
            return recv_tensor

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Create input tensors
        send_tensor = self._create_input_tensor(count, dtype)
        recv_tensor = torch.zeros(count, dtype=dtype, device=self.device)

        # Create a custom backend that logs the graph
        test_name = f"send_recv (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute compiled function (this triggers compilation)
        result = compiled_func(send_tensor, recv_tensor)

        # Verify results: we should receive previous rank's value multiplied by 10
        # Previous rank had value (recv_rank + 1)
        expected = (recv_rank + 1) * 10

        description = (
            f"fullgraph compile send/recv, rank={self.rank}, recv_from={recv_rank}, "
            f"async_op={async_op}"
        )

        if dtype == torch.float:
            expected_tensor = torch.full_like(result.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(result.cpu(), expected)
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

        logger.info("Send/recv test passed for rank %s", self.rank)

    def _test_fullgraph_compile_send_recv_multiple(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for multiple send/recv operations."""
        logger.info(
            "Testing fullgraph compile send/recv (multiple) with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # First exchange in ring
        send_rank1 = (self.rank + 1) % self.num_ranks
        recv_rank1 = (self.rank - 1 + self.num_ranks) % self.num_ranks

        # Second exchange in reverse ring
        send_rank2 = (self.rank - 1 + self.num_ranks) % self.num_ranks
        recv_rank2 = (self.rank + 1) % self.num_ranks

        # Define function with multiple send/recv pairs
        def my_func(send_tensor1, send_tensor2, recv_tensor1, recv_tensor2):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()

            # First exchange: forward ring
            # Alternate send/recv order based on rank to avoid deadlock
            if self.rank % 2 == 0:
                send_work1 = self.torchcomm.send(
                    send_tensor1, send_rank1, async_op=async_op
                )
                recv_work1 = self.torchcomm.recv(
                    recv_tensor1, recv_rank1, async_op=async_op
                )
            else:
                recv_work1 = self.torchcomm.recv(
                    recv_tensor1, recv_rank1, async_op=async_op
                )
                send_work1 = self.torchcomm.send(
                    send_tensor1, send_rank1, async_op=async_op
                )

            if async_op:
                send_work1.wait()
                recv_work1.wait()

            # Process received data
            recv_tensor1 *= 2

            # Second exchange: reverse ring
            # Alternate send/recv order based on rank to avoid deadlock
            if self.rank % 2 == 0:
                send_work2 = self.torchcomm.send(
                    send_tensor2, send_rank2, async_op=async_op
                )
                recv_work2 = self.torchcomm.recv(
                    recv_tensor2, recv_rank2, async_op=async_op
                )
            else:
                recv_work2 = self.torchcomm.recv(
                    recv_tensor2, recv_rank2, async_op=async_op
                )
                send_work2 = self.torchcomm.send(
                    send_tensor2, send_rank2, async_op=async_op
                )

            if async_op:
                send_work2.wait()
                recv_work2.wait()

            # Process received data
            recv_tensor2 *= 3

            # Combine results
            return recv_tensor1 + recv_tensor2

        # Reset dynamo
        import torch._dynamo

        torch._dynamo.reset()

        # Create input tensors
        send_tensor1 = self._create_input_tensor(count, dtype)
        send_tensor2 = self._create_input_tensor(count, dtype)
        recv_tensor1 = torch.zeros(count, dtype=dtype, device=self.device)
        recv_tensor2 = torch.zeros(count, dtype=dtype, device=self.device)

        # Create custom backend
        test_name = f"send_recv_multiple (count={count}, dtype={get_dtype_name(dtype)})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Execute
        result = compiled_func(send_tensor1, send_tensor2, recv_tensor1, recv_tensor2)

        # Verify: recv1 gets (recv_rank1 + 1) * 2, recv2 gets (recv_rank2 + 1) * 3
        expected = (recv_rank1 + 1) * 2 + (recv_rank2 + 1) * 3

        description = f"fullgraph compile send/recv (multiple), rank={self.rank}"

        if dtype == torch.float:
            expected_tensor = torch.full_like(result.cpu(), float(expected))
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"Tensors not close enough for {description}",
            )
        else:
            expected_tensor = torch.full_like(result.cpu(), expected)
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"Tensors not equal for {description}",
            )

        logger.info("Send/recv (multiple) test passed for rank %s", self.rank)

    def _test_fullgraph_compile_with_hints_timeout(self):
        """Test torch.compile with fullgraph=True for all_reduce with hints and timeout."""

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Use fixed test parameters - we just need to verify hints/timeout are passed correctly
        count = 4
        dtype = torch.float
        op = ReduceOp.SUM

        logger.info(
            "Testing fullgraph compile all_reduce with hints/timeout: count=%s, dtype=%s, op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
        )

        # Define test hints and timeout
        test_hints = {"hint_key": "hint_value", "another_hint": "value2"}
        test_timeout = Timeout(seconds=300)

        # Define function to compile that uses hints and timeout
        def my_func(tensor):
            self.torchcomm.barrier(async_op=False)

            # Test all_reduce with hints and timeout
            self.torchcomm.all_reduce(
                tensor,
                op,
                async_op=False,
                hints=test_hints,
                timeout=test_timeout,
            )

            # Also test barrier with hints/timeout
            self.torchcomm.barrier(
                async_op=False,
                hints=test_hints,
                timeout=test_timeout,
            )

            return tensor

        # Create input tensor
        input_tensor = self._create_input_tensor(count, dtype)

        # Create a custom backend that logs the graph
        test_name = f"hints_timeout (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution to ensure all ranks compiled
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function (this triggers compilation)
        logger.info("Rank %s: Running compiled version with hints/timeout", self.rank)
        result = compiled_func(input_tensor)

        # Verify result (same as all_reduce logic)
        expected = self._calculate_expected_all_reduce_result(op)
        description = (
            f"fullgraph compile all_reduce with hints/timeout, rank={self.rank}"
        )

        expected_tensor = torch.full_like(result.cpu(), float(expected))
        self.assertTrue(
            torch.allclose(result.cpu(), expected_tensor),
            f"Tensors not close enough for {description}",
        )

        logger.info("Hints/timeout test passed for rank %s", self.rank)

    def _test_fullgraph_compile_scatter(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for scatter operation."""
        logger.info(
            "Testing fullgraph compile scatter with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        root = 0

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor, input_tensor_list):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            scatter_res = self.torchcomm.scatter(
                output_tensor, input_tensor_list, root, async_op=async_op
            )
            if async_op:
                scatter_res.wait()
            return output_tensor

        # Create output tensor (each rank receives one tensor)
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)

        # Create input tensor list (only meaningful on root)
        if self.rank == root:
            input_tensor_list = [
                torch.ones(count, dtype=dtype, device=self.device) * (i + 1)
                for i in range(self.num_ranks)
            ]
        else:
            # Non-root ranks still need valid tensors for the op
            input_tensor_list = [
                torch.zeros(count, dtype=dtype, device=self.device)
                for _ in range(self.num_ranks)
            ]

        # Create a custom backend that logs the graph
        test_name = (
            f"scatter (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        )
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_tensor_list)

        # Verify: each rank should receive value (rank + 1) from root
        expected_value = self.rank + 1
        if dtype == torch.float:
            expected_tensor = torch.full(
                (count,), float(expected_value), dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"Scatter result mismatch: expected {expected_value}, got {result[0].item()}",
            )
        else:
            expected_tensor = torch.full(
                (count,), expected_value, dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"Scatter result mismatch: expected {expected_value}, got {result[0].item()}",
            )

        logger.info("Scatter test passed for rank %s", self.rank)

    def _test_fullgraph_compile_gather(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for gather operation."""
        logger.info(
            "Testing fullgraph compile gather with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        root = 0

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor_list, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            gather_res = self.torchcomm.gather(
                output_tensor_list, input_tensor, root, async_op=async_op
            )
            if async_op:
                gather_res.wait()
            return output_tensor_list

        # Create input tensor (each rank sends its rank + 1)
        input_tensor = self._create_input_tensor(count, dtype)

        # Create output tensor list (only meaningful on root)
        output_tensor_list = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]

        # Create a custom backend that logs the graph
        test_name = (
            f"gather (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        )
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor_list, input_tensor)

        # Verify on root: each tensor should contain value (rank + 1)
        if self.rank == root:
            for rank_idx in range(self.num_ranks):
                expected_value = rank_idx + 1
                if dtype == torch.float:
                    expected_tensor = torch.full(
                        (count,), float(expected_value), dtype=dtype, device="cpu"
                    )
                    self.assertTrue(
                        torch.allclose(result[rank_idx].cpu(), expected_tensor),
                        f"Gather result[{rank_idx}] mismatch: expected {expected_value}, "
                        f"got {result[rank_idx][0].item()}",
                    )
                else:
                    expected_tensor = torch.full(
                        (count,), expected_value, dtype=dtype, device="cpu"
                    )
                    self.assertTrue(
                        torch.equal(result[rank_idx].cpu(), expected_tensor),
                        f"Gather result[{rank_idx}] mismatch: expected {expected_value}, "
                        f"got {result[rank_idx][0].item()}",
                    )

        logger.info("Gather test passed for rank %s", self.rank)

    def _test_fullgraph_compile_all_gather_single(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for all_gather_single (all_gather_into_tensor)."""
        logger.info(
            "Testing fullgraph compile all_gather_single with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            all_gather_res = self.torchcomm.all_gather_single(
                output_tensor, input_tensor, async_op=async_op
            )
            if async_op:
                all_gather_res.wait()
            return output_tensor

        # Create input tensor (each rank has value rank + 1)
        input_tensor = self._create_input_tensor(count, dtype)

        # Create output tensor (concatenated result from all ranks)
        output_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )

        # Create a custom backend that logs the graph
        test_name = f"all_gather_single (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_tensor)

        # Verify: output should contain [1,1,..., 2,2,..., 3,3,..., ...]
        for rank_idx in range(self.num_ranks):
            start_idx = rank_idx * count
            end_idx = start_idx + count
            expected_value = rank_idx + 1
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[start_idx:end_idx].cpu(), expected_tensor),
                    f"all_gather_single result[{start_idx}:{end_idx}] mismatch",
                )
            else:
                expected_tensor = torch.full(
                    (count,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[start_idx:end_idx].cpu(), expected_tensor),
                    f"all_gather_single result[{start_idx}:{end_idx}] mismatch",
                )

        logger.info("all_gather_single test passed for rank %s", self.rank)

    def _test_fullgraph_compile_reduce_scatter_single(
        self, count, dtype, op, async_op=False
    ):
        """Test torch.compile with fullgraph=True for reduce_scatter_single."""
        logger.info(
            "Testing fullgraph compile reduce_scatter_single with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            reduce_scatter_res = self.torchcomm.reduce_scatter_single(
                output_tensor, input_tensor, op, async_op=async_op
            )
            if async_op:
                reduce_scatter_res.wait()
            return output_tensor

        # Create input tensor: each rank contributes num_ranks chunks of size count
        # Each chunk i has value (rank + 1)
        input_tensor = torch.ones(
            count * self.num_ranks, dtype=dtype, device=self.device
        ) * (self.rank + 1)

        # Create output tensor (receives one chunk after reduce-scatter)
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)

        # Create a custom backend that logs the graph
        test_name = f"reduce_scatter_single (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_tensor)

        # Calculate expected value based on op
        # Each rank contributes (rank + 1), so sum = 1 + 2 + ... + num_ranks
        if op == ReduceOp.SUM:
            expected_value = self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            expected_value = self.num_ranks
        else:
            raise ValueError(f"Unsupported op: {op}")

        if dtype == torch.float:
            expected_tensor = torch.full(
                (count,), float(expected_value), dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"reduce_scatter_single result mismatch: expected {expected_value}",
            )
        else:
            expected_tensor = torch.full(
                (count,), expected_value, dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"reduce_scatter_single result mismatch: expected {expected_value}",
            )

        logger.info("reduce_scatter_single test passed for rank %s", self.rank)

    def _test_fullgraph_compile_all_to_all_single(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for all_to_all_single."""
        logger.info(
            "Testing fullgraph compile all_to_all_single with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            all_to_all_res = self.torchcomm.all_to_all_single(
                output_tensor, input_tensor, async_op=async_op
            )
            if async_op:
                all_to_all_res.wait()
            return output_tensor

        # Create input tensor: each rank sends equal chunks to all ranks
        # Chunk i contains value (rank + 1) * 10 + i
        input_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )
        for i in range(self.num_ranks):
            start_idx = i * count
            end_idx = start_idx + count
            input_tensor[start_idx:end_idx] = (self.rank + 1) * 10 + i

        # Create output tensor
        output_tensor = torch.zeros(
            count * self.num_ranks, dtype=dtype, device=self.device
        )

        # Create a custom backend that logs the graph
        test_name = f"all_to_all_single (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_tensor)

        # Verify: chunk i should contain value from rank i = (i + 1) * 10 + self.rank
        for i in range(self.num_ranks):
            start_idx = i * count
            end_idx = start_idx + count
            expected_value = (i + 1) * 10 + self.rank
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[start_idx:end_idx].cpu(), expected_tensor),
                    f"all_to_all_single result[{start_idx}:{end_idx}] mismatch: "
                    f"expected {expected_value}",
                )
            else:
                expected_tensor = torch.full(
                    (count,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[start_idx:end_idx].cpu(), expected_tensor),
                    f"all_to_all_single result[{start_idx}:{end_idx}] mismatch: "
                    f"expected {expected_value}",
                )

        logger.info("all_to_all_single test passed for rank %s", self.rank)

    def _test_fullgraph_compile_reduce_scatter(self, count, dtype, op, async_op):
        """Test torch.compile with fullgraph=True for reduce_scatter (tensor list version)."""
        logger.info(
            "Testing fullgraph compile reduce_scatter with count=%s, dtype=%s, op=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor, input_list):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            reduce_scatter_res = self.torchcomm.reduce_scatter(
                output_tensor, input_list, op, async_op=async_op
            )
            if async_op:
                reduce_scatter_res.wait()
            return output_tensor

        # Create input list: each rank provides num_ranks tensors, each with value (rank + 1)
        input_list = [
            torch.ones(count, dtype=dtype, device=self.device) * (self.rank + 1)
            for _ in range(self.num_ranks)
        ]

        # Create output tensor (receives one reduced chunk)
        output_tensor = torch.zeros(count, dtype=dtype, device=self.device)

        # Create a custom backend that logs the graph
        test_name = f"reduce_scatter (count={count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_list)

        # Calculate expected value based on op
        # Each rank contributes (rank + 1), so sum = 1 + 2 + ... + num_ranks
        if op == ReduceOp.SUM:
            expected_value = self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            expected_value = self.num_ranks
        else:
            raise ValueError(f"Unsupported op: {op}")

        if dtype == torch.float:
            expected_tensor = torch.full(
                (count,), float(expected_value), dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"reduce_scatter result mismatch: expected {expected_value}",
            )
        else:
            expected_tensor = torch.full(
                (count,), expected_value, dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"reduce_scatter result mismatch: expected {expected_value}",
            )

        logger.info("reduce_scatter test passed for rank %s", self.rank)

    def _test_fullgraph_compile_all_to_all(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for all_to_all (tensor list version)."""
        logger.info(
            "Testing fullgraph compile all_to_all with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        def my_func(output_tensor_list, input_tensor_list):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            all_to_all_res = self.torchcomm.all_to_all(
                output_tensor_list, input_tensor_list, async_op=async_op
            )
            if async_op:
                all_to_all_res.wait()
            return output_tensor_list

        # Create input tensor list: tensor i contains value (rank + 1) * 10 + i
        input_tensor_list = [
            torch.ones(count, dtype=dtype, device=self.device)
            * ((self.rank + 1) * 10 + i)
            for i in range(self.num_ranks)
        ]

        # Create output tensor list
        output_tensor_list = [
            torch.zeros(count, dtype=dtype, device=self.device)
            for _ in range(self.num_ranks)
        ]

        # Create a custom backend that logs the graph
        test_name = f"all_to_all (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor_list, input_tensor_list)

        # Verify: result[i] should contain value from rank i = (i + 1) * 10 + self.rank
        for i in range(self.num_ranks):
            expected_value = (i + 1) * 10 + self.rank
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[i].cpu(), expected_tensor),
                    f"all_to_all result[{i}] mismatch: expected {expected_value}",
                )
            else:
                expected_tensor = torch.full(
                    (count,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[i].cpu(), expected_tensor),
                    f"all_to_all result[{i}] mismatch: expected {expected_value}",
                )

        logger.info("all_to_all test passed for rank %s", self.rank)

    def _test_fullgraph_compile_all_gather_v(self, base_count, dtype, async_op):
        """Test torch.compile with fullgraph=True for all_gather_v (variable sizes)."""
        logger.info(
            "Testing fullgraph compile all_gather_v with base_count=%s, dtype=%s, async_op=%s",
            base_count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Each rank has a different size: rank i has size base_count * (i + 1)
        my_size = base_count * (self.rank + 1)
        all_sizes = [base_count * (i + 1) for i in range(self.num_ranks)]

        # Define function to compile
        def my_func(tensor_list, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            all_gather_res = self.torchcomm.all_gather_v(
                tensor_list, input_tensor, async_op=async_op
            )
            if async_op:
                all_gather_res.wait()
            return tensor_list

        # Create input tensor (each rank has value rank + 1, but different sizes)
        input_tensor = torch.ones(my_size, dtype=dtype, device=self.device) * (
            self.rank + 1
        )

        # Create output tensor list with variable sizes
        tensor_list = [
            torch.zeros(size, dtype=dtype, device=self.device) for size in all_sizes
        ]

        # Create a custom backend that logs the graph
        test_name = f"all_gather_v (base_count={base_count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(tensor_list, input_tensor)

        # Verify: tensor_list[i] should contain value (i + 1) with size all_sizes[i]
        for rank_idx in range(self.num_ranks):
            expected_value = rank_idx + 1
            expected_size = all_sizes[rank_idx]
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (expected_size,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[rank_idx].cpu(), expected_tensor),
                    f"all_gather_v result[{rank_idx}] mismatch: expected {expected_value}",
                )
            else:
                expected_tensor = torch.full(
                    (expected_size,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[rank_idx].cpu(), expected_tensor),
                    f"all_gather_v result[{rank_idx}] mismatch: expected {expected_value}",
                )

        logger.info("all_gather_v test passed for rank %s", self.rank)

    def _test_fullgraph_compile_reduce_scatter_v(
        self, base_count, dtype, op, async_op=False
    ):
        """Test torch.compile with fullgraph=True for reduce_scatter_v (variable sizes)."""
        logger.info(
            "Testing fullgraph compile reduce_scatter_v with base_count=%s, dtype=%s, op=%s, async_op=%s",
            base_count,
            get_dtype_name(dtype),
            get_op_name(op),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Each rank receives a different size: rank i receives size base_count * (i + 1)
        my_output_size = base_count * (self.rank + 1)
        all_sizes = [base_count * (i + 1) for i in range(self.num_ranks)]

        # Define function to compile
        def my_func(output_tensor, input_list):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            reduce_scatter_res = self.torchcomm.reduce_scatter_v(
                output_tensor, input_list, op, async_op=async_op
            )
            if async_op:
                reduce_scatter_res.wait()
            return output_tensor

        # Create input list: each tensor has size all_sizes[i] and value (rank + 1)
        input_list = [
            torch.ones(size, dtype=dtype, device=self.device) * (self.rank + 1)
            for size in all_sizes
        ]

        # Create output tensor (receives chunk for this rank)
        output_tensor = torch.zeros(my_output_size, dtype=dtype, device=self.device)

        # Create a custom backend that logs the graph
        test_name = f"reduce_scatter_v (base_count={base_count}, dtype={get_dtype_name(dtype)}, op={get_op_name(op)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_list)

        # Calculate expected value based on op
        # Each rank contributes (rank + 1), so sum = 1 + 2 + ... + num_ranks
        if op == ReduceOp.SUM:
            expected_value = self.num_ranks * (self.num_ranks + 1) // 2
        elif op == ReduceOp.MAX:
            expected_value = self.num_ranks
        else:
            raise ValueError(f"Unsupported op: {op}")

        if dtype == torch.float:
            expected_tensor = torch.full(
                (my_output_size,), float(expected_value), dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.allclose(result.cpu(), expected_tensor),
                f"reduce_scatter_v result mismatch: expected {expected_value}",
            )
        else:
            expected_tensor = torch.full(
                (my_output_size,), expected_value, dtype=dtype, device="cpu"
            )
            self.assertTrue(
                torch.equal(result.cpu(), expected_tensor),
                f"reduce_scatter_v result mismatch: expected {expected_value}",
            )

        logger.info("reduce_scatter_v test passed for rank %s", self.rank)

    def _test_fullgraph_compile_all_to_all_v_single(
        self, base_count, dtype, async_op=False
    ):
        """Test torch.compile with fullgraph=True for all_to_all_v_single (variable sizes)."""
        logger.info(
            "Testing fullgraph compile all_to_all_v_single with base_count=%s, dtype=%s, async_op=%s",
            base_count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # For simplicity, use uniform split sizes (same as all_to_all_single)
        # Each rank sends/receives base_count elements to/from each other rank
        input_split_sizes = [base_count] * self.num_ranks
        output_split_sizes = [base_count] * self.num_ranks

        total_input_size = sum(input_split_sizes)
        total_output_size = sum(output_split_sizes)

        # Define function to compile
        def my_func(output_tensor, input_tensor):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()
            all_to_all_res = self.torchcomm.all_to_all_v_single(
                output_tensor,
                input_tensor,
                output_split_sizes,
                input_split_sizes,
                async_op=async_op,
            )
            if async_op:
                all_to_all_res.wait()
            return output_tensor

        # Create input tensor: chunk i contains value (rank + 1) * 10 + i
        input_tensor = torch.zeros(total_input_size, dtype=dtype, device=self.device)
        offset = 0
        for i in range(self.num_ranks):
            size = input_split_sizes[i]
            input_tensor[offset : offset + size] = (self.rank + 1) * 10 + i
            offset += size

        # Create output tensor
        output_tensor = torch.zeros(total_output_size, dtype=dtype, device=self.device)

        # Create a custom backend that logs the graph
        test_name = f"all_to_all_v_single (base_count={base_count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor, input_tensor)

        # Verify: chunk i should contain value from rank i = (i + 1) * 10 + self.rank
        offset = 0
        for i in range(self.num_ranks):
            size = output_split_sizes[i]
            expected_value = (i + 1) * 10 + self.rank
            if dtype == torch.float:
                expected_tensor = torch.full(
                    (size,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(
                        result[offset : offset + size].cpu(), expected_tensor
                    ),
                    f"all_to_all_v_single result[{offset}:{offset + size}] mismatch: "
                    f"expected {expected_value}",
                )
            else:
                expected_tensor = torch.full(
                    (size,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[offset : offset + size].cpu(), expected_tensor),
                    f"all_to_all_v_single result[{offset}:{offset + size}] mismatch: "
                    f"expected {expected_value}",
                )
            offset += size

        logger.info("all_to_all_v_single test passed for rank %s", self.rank)

    def _test_fullgraph_compile_batch_send_recv(self, count, dtype, async_op):
        """Test torch.compile with fullgraph=True for batch send/recv operations.

        Each rank sends to and receives from every other rank to test batching logic.
        """
        logger.info(
            "Testing fullgraph compile batch_send_recv with count=%s, dtype=%s, async_op=%s",
            count,
            get_dtype_name(dtype),
            async_op,
        )

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Create batch operation outside the compiled function
        batch = self.torchcomm.batch_op_create()

        # Create send tensors for each peer (one tensor per destination rank)
        # send_tensors[i] will be sent to rank i (skip self)
        # Value: (my_rank + 1) * 100 + dest_rank
        send_tensors = []
        send_peers = []
        for dest_rank in range(self.num_ranks):
            if dest_rank != self.rank:
                value = (self.rank + 1) * 100 + dest_rank
                tensor = torch.full((count,), value, dtype=dtype, device=self.device)
                send_tensors.append(tensor)
                send_peers.append(dest_rank)

        # Create recv tensors for each peer (one tensor per source rank)
        recv_tensors = []
        recv_peers = []
        for src_rank in range(self.num_ranks):
            if src_rank != self.rank:
                tensor = torch.zeros(count, dtype=dtype, device=self.device)
                recv_tensors.append(tensor)
                recv_peers.append(src_rank)

        # Define function to compile using batch send/recv
        # Note: batch is created outside and passed in (can't create opaque objects in graph)
        def my_func(send_tensors, recv_tensors, batch):
            barrier_res = self.torchcomm.barrier(async_op=async_op)
            if async_op:
                barrier_res.wait()

            # Add all sends and recvs to batch
            for tensor, peer in zip(send_tensors, send_peers):
                batch.send(tensor, peer)
            for tensor, peer in zip(recv_tensors, recv_peers):
                batch.recv(tensor, peer)

        # Create a custom backend that logs the graph
        test_name = f"batch_send_recv_all (count={count}, dtype={get_dtype_name(dtype)}, async={async_op})"
        backend = self._create_graph_logging_backend(test_name)

        # Compile and execute
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)
        compiled_func(send_tensors, recv_tensors, batch)

        # Issue the batch
        issue_res = batch.issue(async_op=async_op)
        if async_op:
            issue_res.wait()

        # so this is funky because the tensors mutated by batch.issue (i.e.,
        # inputs to open recv's) aren't inputs to the call. so it's possible you
        # access those values somewhere down here and the wait gets dropped below
        # the read. i haven't figured out a good way to solve this yet, but it should
        # be possible

        result = recv_tensors

        # Verify results: for each source rank, we should receive (src_rank + 1) * 100 + my_rank
        for i, src_rank in enumerate(recv_peers):
            expected_value = (src_rank + 1) * 100 + self.rank

            description = (
                f"fullgraph compile batch send/recv, rank={self.rank}, recv_from={src_rank}, "
                f"async_op={async_op}"
            )

            if dtype == torch.float:
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[i].cpu(), expected_tensor),
                    f"Batch send/recv result mismatch for {description}. "
                    f"Expected ~{expected_value}, got {result[i][0].item()}",
                )
            else:
                expected_tensor = torch.full(
                    (count,), expected_value, dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.equal(result[i].cpu(), expected_tensor),
                    f"Batch send/recv result mismatch for {description}. "
                    f"Expected {expected_value}, got {result[i][0].item()}",
                )

        logger.info("batch_send_recv test passed for rank %s", self.rank)

    def _test_fullgraph_compile_gather_empty_output_list(self):
        """
        Test torch.compile with gather where gather_list is semantically empty on non-dst ranks.

        For torch.compile to work uniformly across ranks, all ranks must pass the same
        input structure. The gather operation fills the list only on dst rank; non-dst
        ranks have their list ignored by the gather operation.

        This validates that:
        1. torch.compile works with gather when all ranks have uniform input structure
        2. The FX graph has proper gather->wait_tensors edges
        3. dst rank receives correct gathered data
        """
        logger.info("Testing fullgraph compile gather with uniform input structure")

        # Reset dynamo to clear any cached graphs
        import torch._dynamo

        torch._dynamo.reset()

        root = 0
        count = 4
        dtype = torch.float

        # Barrier to ensure all ranks are synced before starting
        self.torchcomm.barrier(async_op=False)

        # Define function to compile
        # All ranks pass the same structure - gather fills only on dst rank
        def my_func(output_tensor_list, input_tensor):
            gather_res = self.torchcomm.gather(
                output_tensor_list, input_tensor, root, async_op=True
            )
            if gather_res is not None:
                gather_res.wait()
            return output_tensor_list

        # Create input tensor (each rank sends its rank + 1)
        input_tensor = torch.full(
            (count,), float(self.rank + 1), dtype=dtype, device=self.device
        )

        # All ranks pass the same structure for uniform graph generation
        # Use a sentinel value (-1) to identify tensors that should NOT be modified
        # on non-dst ranks (gather ignores the list on non-dst ranks)
        sentinel_value = -1.0
        output_tensor_list = (
            [
                torch.full((count,), sentinel_value, dtype=dtype, device=self.device)
                for _ in range(self.num_ranks)
            ]
            if self.rank == root
            else [torch.empty(0, device=self.device, dtype=dtype)]
        )

        # Create a custom backend that logs the graph
        test_name = (
            f"gather_empty_output_list (count={count}, dtype={get_dtype_name(dtype)})"
        )
        backend = self._create_graph_logging_backend(test_name)

        # Compile with custom backend
        logger.info("Rank %s: Compiling function", self.rank)
        compiled_func = torch.compile(my_func, fullgraph=True, backend=backend)

        # Barrier before execution
        self.torchcomm.barrier(async_op=False)

        # Execute compiled function
        logger.info("Rank %s: Running compiled version", self.rank)
        result = compiled_func(output_tensor_list, input_tensor)

        self.torchcomm.barrier(async_op=False)

        if self.rank == root:
            self.assertEqual(len(result), self.num_ranks)
            # Root should have received all tensors with gathered data
            for rank_idx in range(self.num_ranks):
                expected_value = rank_idx + 1
                expected_tensor = torch.full(
                    (count,), float(expected_value), dtype=dtype, device="cpu"
                )
                self.assertTrue(
                    torch.allclose(result[rank_idx].cpu(), expected_tensor),
                    f"Gather result[{rank_idx}] mismatch: expected {expected_value}, "
                    f"got {result[rank_idx][0].item()}",
                )
        else:
            self.assertEqual(len(result), 1)
            self.assertTrue(
                result[0].numel() == 0,
                f"Non-dst rank {self.rank} should have an empty result, got {result}",
            )

        logger.info("Gather compile test passed for rank %s", self.rank)

    # =========================================================================
    # Test entry points (picked up by test harness)
    # =========================================================================
    def test_fullgraph_compile_all_reduce(self):
        """Test torch.compile with fullgraph=True for all_reduce operation."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_all_reduce(count, dtype, op, async_op)

    def test_fullgraph_compile_all_reduce_multiple_calls(self):
        """Test torch.compile with fullgraph=True for multiple all_reduce calls."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_all_reduce_multiple_calls(
                    count, dtype, op, async_op
                )

    @skip_backend("gloo")
    def test_fullgraph_compile_reduce(self):
        """Test torch.compile with fullgraph=True for reduce operation."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_reduce(count, dtype, op, async_op)

    def test_fullgraph_compile_all_reduce_premul_sum(self):
        """Test torch.compile with fullgraph=True for all_reduce with PREMUL_SUM operation."""
        for async_op in [False, True]:
            with self.subTest(async_op=async_op):
                self._test_fullgraph_compile_all_reduce_premul_sum(async_op)

    def test_fullgraph_compile_broadcast(self):
        """Test torch.compile with fullgraph=True for broadcast operation."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_broadcast(count, dtype, async_op)

    def test_fullgraph_compile_barrier(self):
        """Test torch.compile with fullgraph=True for barrier synchronization."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_barrier(count, dtype, op, async_op)

    def test_fullgraph_compile_all_gather(self):
        """Test torch.compile with fullgraph=True for all_gather operation."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_gather(count, dtype, async_op)

    def test_fullgraph_compile_send_recv(self):
        """Test torch.compile with fullgraph=True for send/recv operations."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_send_recv(count, dtype, async_op)

    def test_fullgraph_compile_send_recv_multiple(self):
        """Test torch.compile with fullgraph=True for multiple send/recv operations."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_send_recv_multiple(count, dtype, async_op)

    def test_fullgraph_compile_with_hints_timeout(self):
        """Test torch.compile with fullgraph=True for all_reduce with hints and timeout."""
        self._test_fullgraph_compile_with_hints_timeout()

    def test_fullgraph_compile_scatter(self):
        """Test torch.compile with fullgraph=True for scatter operation."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_scatter(count, dtype, async_op)

    def test_fullgraph_compile_gather(self):
        """Test torch.compile with fullgraph=True for gather operation."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_gather(count, dtype, async_op)

    def test_fullgraph_compile_all_gather_single(self):
        """Test torch.compile with fullgraph=True for all_gather_single."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_gather_single(count, dtype, async_op)

    def test_fullgraph_compile_reduce_scatter_single(self):
        """Test torch.compile with fullgraph=True for reduce_scatter_single."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_reduce_scatter_single(
                    count, dtype, op, async_op
                )

    def test_fullgraph_compile_all_to_all_single(self):
        """Test torch.compile with fullgraph=True for all_to_all_single."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_to_all_single(count, dtype, async_op)

    def test_fullgraph_compile_reduce_scatter(self):
        """Test torch.compile with fullgraph=True for reduce_scatter (tensor list version)."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_reduce_scatter(count, dtype, op, async_op)

    def test_fullgraph_compile_all_to_all(self):
        """Test torch.compile with fullgraph=True for all_to_all (tensor list version)."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_to_all(count, dtype, async_op)

    def test_fullgraph_compile_all_gather_v(self):
        """Test torch.compile with fullgraph=True for all_gather_v."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_gather_v(count, dtype, async_op)

    def test_fullgraph_compile_reduce_scatter_v(self):
        """Test torch.compile with fullgraph=True for reduce_scatter_v."""
        for count, dtype, op, async_op in itertools.product(
            self.counts, self.dtypes, self.ops, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, op=op, async_op=async_op):
                self._test_fullgraph_compile_reduce_scatter_v(
                    count, dtype, op, async_op
                )

    def test_fullgraph_compile_all_to_all_v_single(self):
        """Test torch.compile with fullgraph=True for all_to_all_v_single."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_all_to_all_v_single(count, dtype, async_op)

    def test_fullgraph_compile_batch_send_recv(self):
        """Test torch.compile with fullgraph=True for batch send/recv operations."""
        for count, dtype, async_op in itertools.product(
            self.counts, self.dtypes, [False, True]
        ):
            with self.subTest(count=count, dtype=dtype, async_op=async_op):
                self._test_fullgraph_compile_batch_send_recv(count, dtype, async_op)

    def test_fullgraph_compile_gather_empty_output_list(self):
        """Test gather with empty output list on non-dst ranks validates graph edges."""
        self._test_fullgraph_compile_gather_empty_output_list()


if __name__ == "__main__":
    unittest.main()
