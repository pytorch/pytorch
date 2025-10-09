# Owner(s): ["module: inductor"]

import functools
import weakref
from collections import Counter
from typing import Callable, Optional

import torch
from torch._inductor.fx_passes.memory_estimator import build_memory_profile
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map_only
from torch.utils.weak import WeakIdKeyDictionary


def tensor_storage_id(tensor):
    return tensor._typed_storage()._cdata


def device_filter(device):
    return device.type == "cuda"


class FakeTensorMemoryProfilerMode(TorchDispatchMode):
    def __init__(self, device_filter: Optional[Callable[torch.device, bool]] = None):
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
            x = torch.randn(32, 1000, device="cuda")
            w1 = torch.randn(500, 1000, device="cuda")
            w2 = torch.randn(100, 500, device="cuda")
            w3 = torch.randn(10, 100, device="cuda")
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
            x = torch.randn(8, 3, 224, 224, device="cuda")
            conv1_weight = torch.randn(64, 3, 3, 3, device="cuda")
            conv2_weight = torch.randn(128, 64, 3, 3, device="cuda")
            linear_weight = torch.randn(10, 128 * 56 * 56, device="cuda")
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


if __name__ == "__main__":
    if IS_LINUX and HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
