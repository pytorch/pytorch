# mypy: allow-untyped-defs
"""
Support for multi-kernel runtime dispatch based on dynamic input sizes.

This module provides the infrastructure for generating multiple kernel variants
optimized for different input size hints, and dispatching to the best kernel
at runtime based on the actual input sizes.
"""

import functools
import logging
from typing import Any, Callable, Dict, List, Optional

from .. import config
from ..select_algorithm import ChoiceCaller
from ..virtualized import V

log = logging.getLogger(__name__)


class MultiKernelCall:
    """
    Runtime dispatcher for multi-kernel selection.

    This class generates the multi_kernel function that:
    1) Uses cache key hash to identify the input size
    2) If we already know the best kernel for this size, run it
    3) Otherwise run the first not-yet-run kernel, record time and return
    4) If this is the last kernel, find the best and cache the result
    """

    def __init__(self, kernel_choices: List[ChoiceCaller]):
        self.kernel_choices = kernel_choices
        self.kernel_name = f"multi_kernel_{id(self)}"

    def generate_dispatch_code(self) -> str:
        """Generate the multi_kernel dispatch function code."""

        # Generate the multi_kernel function that will be called at runtime
        code_lines = [
            "def multi_kernel(cache_key_args, *kernel_args):",
            "    # multi_kernel does the following:",
            "    # 1) it uses the cache key hash([s77])",
            "    # 2) given a cache key if we already know the best answer then we just run that",
            "    # 3) otherwise we run the first not yet run kernel and we record the time and return",
            "    # 4) if this is the last kernel that hasn't been run, we find the best kernel and set that to be the best answer",
            "    import time",
            "    import torch",
            "    ",
            "    # Global caches for kernel selection",
            "    if not hasattr(multi_kernel, '_kernel_cache'):",
            "        multi_kernel._kernel_cache = {}",
            "        multi_kernel._timing_cache = {}",
            "    ",
            "    cache_key = tuple(cache_key_args)",
            "    cache_key_str = str(cache_key)",
            "    ",
            "    # If we already know the best kernel, use it",
            "    if cache_key_str in multi_kernel._kernel_cache:",
            "        best_kernel_idx = multi_kernel._kernel_cache[cache_key_str]",
            "        kernel_fns = [",
        ]

        # Add lambda functions for each kernel choice
        for i, choice in enumerate(self.kernel_choices):
            code_lines.append(f"            lambda args: {choice.call_name()}(*args),")

        code_lines.extend([
            "        ]",
            "        return kernel_fns[best_kernel_idx](kernel_args)",
            "    ",
            "    # Initialize timing cache for this key if not exists",
            "    if cache_key_str not in multi_kernel._timing_cache:",
            "        multi_kernel._timing_cache[cache_key_str] = {}",
            "    ",
            "    timings = multi_kernel._timing_cache[cache_key_str]",
            "    ",
            "    # Find the first kernel we haven't benchmarked yet",
            "    next_kernel_idx = None",
            f"    for i in range({len(self.kernel_choices)}):",
            "        if i not in timings:",
            "            next_kernel_idx = i",
            "            break",
            "    ",
            "    if next_kernel_idx is not None:",
            "        # Define kernel functions",
            "        kernel_fns = [",
        ])

        # Add lambda functions for each kernel choice again
        for i, choice in enumerate(self.kernel_choices):
            code_lines.append(f"            lambda args: {choice.call_name()}(*args),")

        code_lines.extend([
            "        ]",
            "        ",
            "        kernel_fn = kernel_fns[next_kernel_idx]",
            "        ",
            "        # Warm up",
            "        for _ in range(3):",
            "            result = kernel_fn(kernel_args)",
            "        ",
            "        # Time the kernel",
            "        torch.cuda.synchronize()",
            "        start_time = time.perf_counter()",
            "        for _ in range(10):  # Run multiple times for better accuracy",
            "            result = kernel_fn(kernel_args)",
            "        torch.cuda.synchronize()",
            "        end_time = time.perf_counter()",
            "        ",
            "        avg_time = (end_time - start_time) / 10",
            "        timings[next_kernel_idx] = avg_time",
            "        ",
            "        # If this was the last kernel to benchmark, pick the best one",
            f"        if len(timings) == {len(self.kernel_choices)}:",
            "            best_kernel_idx = min(timings.keys(), key=lambda k: timings[k])",
            "            multi_kernel._kernel_cache[cache_key_str] = best_kernel_idx",
            "        ",
            "        return result",
            "    else:",
            "        # This shouldn't happen if the logic is correct",
            "        raise RuntimeError('Multi-kernel: no unbenchmarked kernels found')",
        ])

        return "\n".join(code_lines)


class RuntimeMultiKernel(ChoiceCaller):
    """
    A ChoiceCaller that wraps multiple kernel choices and generates runtime dispatch code.
    """

    def __init__(self, kernel_choices: List[ChoiceCaller], cache_key_args: List[str]):
        # Use the first choice as the template for basic properties
        first_choice = kernel_choices[0]
        super().__init__(
            name=f"multi_kernel_{len(kernel_choices)}_choices",
            input_nodes=first_choice.input_nodes,
            layout=first_choice.layout,
            description=f"Multi-kernel dispatch over {len(kernel_choices)} choices"
        )

        self.kernel_choices = kernel_choices
        self.cache_key_args = cache_key_args
        self.multi_kernel_call = MultiKernelCall(kernel_choices)

    def benchmark(self, *args, out):
        """For benchmarking, just use the first kernel choice."""
        return self.kernel_choices[0].benchmark(*args, out=out)

    def hash_key(self):
        """Generate hash key based on all constituent kernel choices."""
        choice_keys = [choice.hash_key() for choice in self.kernel_choices]
        return f"multi_kernel_{'-'.join(choice_keys)}"

    def output_node(self):
        """Generate the output node with multi-kernel dispatch."""
        from torch._inductor import ir

        # Create a custom buffer that will generate the multi-kernel call
        return ir.TensorBox.create(
            MultiKernelBuffer(
                layout=self.layout,
                inputs=self.input_nodes,
                kernel_choices=self.kernel_choices,
                cache_key_args=self.cache_key_args,
            )
        )

    def call_name(self):
        return f"multi_kernel_{id(self)}"


class MultiKernelBuffer(ir.ComputedBuffer):
    """
    A buffer that generates multi-kernel dispatch code.
    """

    def __init__(self, layout, inputs, kernel_choices, cache_key_args):
        # Create a dummy operation for the buffer
        def dummy_inner_fn(index):
            return f"multi_kernel_result[{', '.join(map(str, index))}]"

        super().__init__(
            name=None,
            layout=layout,
            data=ir.Pointwise.create(
                device=layout.device,
                dtype=layout.dtype,
                inner_fn=dummy_inner_fn,
                ranges=layout.size,
            )
        )

        self.inputs = inputs
        self.kernel_choices = kernel_choices
        self.cache_key_args = cache_key_args

    def get_read_writes(self):
        # Return read/write info from all kernel choices
        reads = set()
        writes = set()

        for choice in self.kernel_choices:
            if hasattr(choice, 'get_read_writes'):
                choice_reads, choice_writes = choice.get_read_writes()
                reads.update(choice_reads)
                writes.update(choice_writes)

        return reads, writes

    def codegen_kernel(self, wrapper):
        """Generate the multi-kernel dispatch code."""

        # Generate the multi_kernel function
        multi_kernel_code = self.generate_multi_kernel_function()

        # Add the function to the wrapper
        wrapper.writeline(multi_kernel_code)

        # Generate the call to multi_kernel
        cache_key_args = self.cache_key_args
        kernel_args = [inp.codegen_reference() for inp in self.inputs]

        call_line = f"multi_kernel([{', '.join(cache_key_args)}], {', '.join(kernel_args)})"
        wrapper.writeline(call_line)

    def generate_multi_kernel_function(self):
        """Generate the multi_kernel function code."""
        lines = [
            "def multi_kernel(cache_key_args, *kernel_args):",
            "    # multi_kernel does the following:",
            "    # 1) it uses the cache key hash([s77])",
            "    # 2) given a cache key if we already know the best answer then we just run that",
            "    # 3) otherwise we run the first not yet run kernel and we record the time and return",
            "    # 4) if this is the last kernel that hasn't been run, we find the best kernel and set that to be the best answer",
            "    import time",
            "    import torch",
            "    ",
            "    # Global caches for kernel selection",
            "    if not hasattr(multi_kernel, '_kernel_cache'):",
            "        multi_kernel._kernel_cache = {}",
            "        multi_kernel._timing_cache = {}",
            "    ",
            "    cache_key = tuple(cache_key_args)",
            "    cache_key_str = str(cache_key)",
            "    ",
            "    # If we already know the best kernel, use it",
            "    if cache_key_str in multi_kernel._kernel_cache:",
            "        best_kernel_idx = multi_kernel._kernel_cache[cache_key_str]",
            "        kernel_fns = [",
        ]

        # Add kernel function references
        for choice in self.kernel_choices:
            lines.append(f"            {choice.call_name()},")

        lines.extend([
            "        ]",
            "        return kernel_fns[best_kernel_idx](*kernel_args)",
            "    ",
            "    # Initialize timing cache for this key if not exists",
            "    if cache_key_str not in multi_kernel._timing_cache:",
            "        multi_kernel._timing_cache[cache_key_str] = {}",
            "    ",
            "    timings = multi_kernel._timing_cache[cache_key_str]",
            "    ",
            "    # Find the first kernel we haven't benchmarked yet",
            "    next_kernel_idx = None",
            f"    for i in range({len(self.kernel_choices)}):",
            "        if i not in timings:",
            "            next_kernel_idx = i",
            "            break",
            "    ",
            "    if next_kernel_idx is not None:",
            "        # Define kernel functions",
            "        kernel_fns = [",
        ])

        # Add kernel function references again
        for choice in self.kernel_choices:
            lines.append(f"            {choice.call_name()},")

        lines.extend([
            "        ]",
            "        ",
            "        kernel_fn = kernel_fns[next_kernel_idx]",
            "        ",
            "        # Warm up",
            "        for _ in range(3):",
            "            result = kernel_fn(*kernel_args)",
            "        ",
            "        # Time the kernel",
            "        torch.cuda.synchronize()",
            "        start_time = time.perf_counter()",
            "        for _ in range(10):  # Run multiple times for better accuracy",
            "            result = kernel_fn(*kernel_args)",
            "        torch.cuda.synchronize()",
            "        end_time = time.perf_counter()",
            "        ",
            "        avg_time = (end_time - start_time) / 10",
            "        timings[next_kernel_idx] = avg_time",
            "        ",
            "        # If this was the last kernel to benchmark, pick the best one",
            f"        if len(timings) == {len(self.kernel_choices)}:",
            "            best_kernel_idx = min(timings.keys(), key=lambda k: timings[k])",
            "            multi_kernel._kernel_cache[cache_key_str] = best_kernel_idx",
            "        ",
            "        return result",
            "    else:",
            "        # This shouldn't happen if the logic is correct",
            "        raise RuntimeError('Multi-kernel: no unbenchmarked kernels found')",
        ])

        return "\n".join(lines)
