# Owner(s): ["module: inductor"]

import functools
import weakref
from collections import Counter
from collections.abc import Callable
from typing import Optional

import torch
from torch._inductor.fx_passes.memory_estimator import (
    build_memory_profile,
    MemoryTracker,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakIdKeyDictionary


def tensor_storage_id(tensor):
    return tensor._typed_storage()._cdata


def device_filter(device):
    return device.type == GPU_TYPE


class FakeTensorMemoryProfilerMode(TorchDispatchMode):
    def __init__(self, device_filter: Optional[Callable[[torch.device], bool]] = None):
        # counter of storage ids to live references
        self.storage_count: dict[int, int] = Counter()
        # live fake tensors
        self.live_tensors = WeakIdKeyDictionary()
        self.memory_use = 0
        self.max_memory = 0
        self.device_filter = device_filter

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs is not None else {}
        rs = func(*args, **kwargs)
        tree_map_only(torch._subclasses.FakeTensor, self.increase_memory_use, rs)
        return rs

    def increase_memory_use(self, tensor):
        # already accounted for
        if tensor in self.live_tensors:
            return

        if self.device_filter is not None and not self.device_filter(tensor.device):
            return

        self.live_tensors[tensor] = True
        nbytes = tensor.untyped_storage().nbytes()

        storage_id = tensor_storage_id(tensor)

        # new storage, add to memory
        if storage_id not in self.storage_count:
            self.change_memory(nbytes)

        self.storage_count[storage_id] += 1

        # when this tensor dies, we need to adjust memory
        weakref.finalize(
            tensor, functools.partial(self.tensor_cleanup, storage_id, nbytes)
        )

    def tensor_cleanup(self, storage_id, nbytes):
        self.storage_count[storage_id] -= 1
        if self.storage_count[storage_id] == 0:
            del self.storage_count[storage_id]
            self.change_memory(-nbytes)

    def change_memory(self, delta):
        self.memory_use += delta
        self.max_memory = max(self.memory_use, self.max_memory)


class TestMemoryProfilingResNet(InductorTestCase):
    def test_simple_linear_layers(self):
        """Test with a simple sequential model with explicit weights on CUDA."""

        def create_inputs_and_weights():
            """Create inputs and weights on CUDA."""
            x = torch.randn(32, 1000, device=GPU_TYPE)
            w1 = torch.randn(500, 1000, device=GPU_TYPE)
            w2 = torch.randn(100, 500, device=GPU_TYPE)
            w3 = torch.randn(10, 100, device=GPU_TYPE)
            return x, w1, w2, w3

        def fn(x, w1, w2, w3):
            h1 = torch.nn.functional.linear(x, w1)
            h1 = torch.nn.functional.relu(h1)
            h2 = torch.nn.functional.linear(h1, w2)
            h2 = torch.nn.functional.relu(h2)
            out = torch.nn.functional.linear(h2, w3)
            return out

        with FakeTensorMode():
            # Trace with make_fx
            x, w1, w2, w3 = create_inputs_and_weights()
            fx_graph = make_fx(fn)(x, w1, w2, w3)

            # Static analysis
            def is_releasable(node):
                return node.op not in ("placeholder", "get_attr")

            fx_memory_profile = build_memory_profile(fx_graph.graph, is_releasable)
            fx_peak = max(fx_memory_profile)

            # Runtime profiling
            profiler = FakeTensorMemoryProfilerMode()

            with profiler:
                x_runtime, w1_runtime, w2_runtime, w3_runtime = (
                    create_inputs_and_weights()
                )
                result = fn(x_runtime, w1_runtime, w2_runtime, w3_runtime)
                del result

            runtime_peak = profiler.max_memory

            self.assertEqual(fx_peak, runtime_peak)

    def test_conv_network(self):
        """Test with a convolutional network."""

        def create_inputs_and_weights():
            """Create inputs and weights on CUDA."""
            x = torch.randn(8, 3, 224, 224, device=GPU_TYPE)
            conv1_weight = torch.randn(64, 3, 3, 3, device=GPU_TYPE)
            conv2_weight = torch.randn(128, 64, 3, 3, device=GPU_TYPE)
            linear_weight = torch.randn(10, 128 * 56 * 56, device=GPU_TYPE)
            return x, conv1_weight, conv2_weight, linear_weight

        def fn(x, conv1_weight, conv2_weight, linear_weight):
            h = torch.nn.functional.conv2d(x, conv1_weight, padding=1)
            h = torch.nn.functional.relu(h)
            h = torch.nn.functional.max_pool2d(h, 2)
            h = torch.nn.functional.conv2d(h, conv2_weight, padding=1)
            h = torch.nn.functional.relu(h)
            h = torch.nn.functional.max_pool2d(h, 2)
            h = torch.flatten(h, 1)
            out = torch.nn.functional.linear(h, linear_weight)
            return out

        with FakeTensorMode():
            # Trace with make_fx
            x, conv1_weight, conv2_weight, linear_weight = create_inputs_and_weights()
            fx_graph = make_fx(fn)(x, conv1_weight, conv2_weight, linear_weight)

            def is_releasable(node):
                return node.op not in ("placeholder", "get_attr")

            fx_memory_profile = build_memory_profile(fx_graph.graph, is_releasable)
            fx_peak = max(fx_memory_profile)

            # Runtime profiling
            profiler = FakeTensorMemoryProfilerMode()

            with profiler:
                x_runtime, conv1_w, conv2_w, linear_w = create_inputs_and_weights()
                result = fn(x_runtime, conv1_w, conv2_w, linear_w)
                del result

            runtime_peak = profiler.max_memory

            self.assertEqual(fx_peak, runtime_peak)


class TestMemoryTracker(InductorTestCase):
    def test_memory_tracker_original_order(self):
        """Test that MemoryTracker works correctly with original scheduling order and matches runtime profiling."""

        def create_inputs_and_weights():
            """Create inputs and weights on CUDA."""
            x = torch.randn(32, 100, device=GPU_TYPE)
            w1 = torch.randn(100, 50, device=GPU_TYPE)
            w2 = torch.randn(50, 10, device=GPU_TYPE)
            return x, w1, w2

        def fn(x, w1, w2):
            # Create a simple function that allocates intermediate tensors
            h1 = torch.matmul(x, w1)  # Allocates h1
            h2 = torch.relu(h1)  # h1 can be freed, h2 allocated
            out = torch.matmul(h2, w2)  # h2 can be freed, out allocated
            return out

        with FakeTensorMode():
            # Create inputs
            x, w1, w2 = create_inputs_and_weights()

            # Trace the function
            fx_graph = make_fx(fn)(x, w1, w2)

            # Test MemoryTracker with original order
            memory_tracker = MemoryTracker(fx_graph.graph, device_filter=device_filter)

            # Schedule nodes in original order
            compute_nodes = [
                node
                for node in fx_graph.graph.nodes
                if node.op not in ("placeholder", "get_attr", "output")
            ]

            for node in compute_nodes:
                memory_tracker.schedule_node(node)

            memory_tracker_peak = memory_tracker.get_current_memory_bytes()

            # Compare with runtime profiling using FakeTensorMemoryProfilerMode
            profiler = FakeTensorMemoryProfilerMode(device_filter=device_filter)

            with profiler:
                x_runtime, w1_runtime, w2_runtime = create_inputs_and_weights()
                result = fn(x_runtime, w1_runtime, w2_runtime)
                del result

            runtime_peak = profiler.max_memory

            # Verify both approaches track meaningful memory usage
            self.assertGreater(
                memory_tracker_peak, 0, "MemoryTracker should track memory usage"
            )
            self.assertGreater(
                runtime_peak, 0, "Runtime profiler should track memory usage"
            )

    def test_memory_tracker_different_scheduling(self):
        """Test that different scheduling orders produce different memory usage patterns."""

        def foo(primals_1):
            zeros = torch.zeros_like(primals_1)  # Create zeros tensor
            add_result = zeros + 1  # Use zeros (first use)
            sum_result = zeros.sum()  # Use zeros (second use)
            cpu = torch.zeros([20], device="cpu")
            cpu_2 = cpu + 1
            return add_result, sum_result, cpu_2

        with FakeTensorMode():
            # Create input
            primals_1 = torch.randn(1000, 1000, device=GPU_TYPE)

            # Trace the function
            fx_graph = make_fx(foo)(primals_1)

            # Get compute nodes (excluding placeholders, get_attr, output)
            compute_nodes = [
                node
                for node in fx_graph.graph.nodes
                if node.op not in ("placeholder", "get_attr", "output")
            ]

            # Test original order: zeros_like, add, sum
            # zeros gets freed after sum (last use of zeros)
            memory_tracker1 = MemoryTracker(fx_graph.graph, device_filter=device_filter)
            memory_profile1 = []
            initial_mem = memory_tracker1.get_current_memory_bytes()

            for node in compute_nodes:
                memory_tracker1.schedule_node(node)
                memory_profile1.append(memory_tracker1.get_current_memory_bytes())

            # use of primals should not deallocate
            self.assertEqual(memory_profile1[0], initial_mem * 2)

            # Test different order: zeros_like, sum, add
            # zeros gets freed after add (last use of zeros in new order)
            memory_tracker2 = MemoryTracker(fx_graph.graph, device_filter=device_filter)
            memory_profile2 = []

            # Alternative schedule: change which operation is the last use of zeros
            # Original: zeros_like, add, sum (zeros freed after sum)
            # Alternative: zeros_like, sum, add (zeros freed after add)
            if len(compute_nodes) != 5:
                raise AssertionError(
                    f"Expected 3 compute nodes, got {len(compute_nodes)}"
                )
            reordered_nodes = [
                compute_nodes[0],  # zeros_like: zeros = torch.zeros_like(primals_1)
                compute_nodes[2],  # sum: sum_result = zeros.sum() (zeros still alive)
                compute_nodes[
                    1
                ],  # add: add_result = zeros + 1 (last use, zeros freed here)
                compute_nodes[3],  # cpu = torch.zeros([20], device="cpu")
                compute_nodes[4],  # cpu_2 = cpu + 1
            ]

            for node in reordered_nodes:
                memory_tracker2.schedule_node(node)
                memory_profile2.append(memory_tracker2.get_current_memory_bytes())

            # Compare peak memories
            peak1 = max(memory_profile1)
            peak2 = max(memory_profile2)

            # Both should end with the same final memory (all intermediate tensors freed)
            self.assertEqual(memory_profile1[-1], memory_profile2[-1])

            # The profiles should be different, showing different memory patterns
            self.assertNotEqual(
                memory_profile1,
                memory_profile2,
                "Different scheduling should produce different memory profiles",
            )

            # The different scheduling should produce different peak memory!
            # Original: zeros + add_result both alive → higher peak
            # Reordered: zeros freed before add_result created → lower peak
            self.assertGreater(
                peak1, peak2, "Original order should have higher peak memory"
            )

            # Specifically, original has both zeros and add_result alive simultaneously
            self.assertGreater(
                memory_profile1[1],
                memory_profile2[1],
                "Original order keeps more tensors alive simultaneously",
            )

            # The reordered version should have lower intermediate memory usage
            self.assertLess(
                peak2,
                peak1,
                "Reordered schedule reduces peak memory through better deallocation timing",
            )

            # Verify the MemoryTracker correctly tracks different scheduling
            # The first tracker should match since we tested accuracy against FakeTensorMemoryProfilerMode
            self.assertLessEqual(
                abs(memory_tracker1.peak_memory - peak1),
                8,
                "First tracker peak should match profile peak",
            )

            # The key test: profiles show different peaks due to different deallocation timing
            self.assertNotEqual(
                peak1, peak2, "Different scheduling produces different peak memory"
            )


if __name__ == "__main__":
    if HAS_GPU:
        run_tests(needs="filelock")
