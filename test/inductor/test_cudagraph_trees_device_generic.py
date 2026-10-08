# Owner(s): ["module: inductor"]
"""Device-generic coverage for cudagraph trees.

test_cudagraph_trees.py only runs on CUDA. These tests assert the parts of the
contract that must hold on every accelerator whose DeviceInterface advertises
graph support, plus the registry-driven gate itself, which needs no accelerator
at all.
"""

import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    get_interface_for_device,
    get_registered_device_interfaces,
)
from torch._dynamo.utils import counters
from torch._inductor import config as inductor_config
from torch._inductor.cudagraph_trees import get_container, reset_cudagraph_trees
from torch._inductor.cudagraph_utils import (
    _CUDAGRAPH_SUPPORTED_DEVICE_TYPES,
    check_lowering_disable_cudagraph,
    check_multiple_devices_or_any_cpu_nodes,
    format_default_skip_message,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.logging_utils import logs_to_string


class _FakeNode:
    """Minimal stand-in for an fx node in the device/node mapping."""

    def __init__(self, name):
        self.name = name
        self.users = ()
        self.meta = {}


def _mutating_graph():
    def fn(x):
        x.add_(1)
        return x

    return make_fx(fn)(torch.ones(4))


# Members of DeviceInterface.Graphs that have no usable default, so a backend
# advertising cudagraph support must provide its own.
_REQUIRED_GRAPH_MEMBERS = (
    "Graph",
    "pool_handle",
    "capture",
    "begin_allocate_current_thread_to_pool",
    "end_allocate_to_pool",
    "release_pool",
    "get_checkpoint_state",
    "set_checkpoint_pool_state",
    "check_pool_live_allocations",
    "raw_delete",
    "memory_snapshot",
    "history_recording",
    "construct_tensor",
    "has_standard_deleter",
    "free_and_remove_deleter",
)


@instantiate_parametrized_tests
class CudagraphGateTest(InductorTestCase):
    """Gate and bookkeeping tests that need no accelerator."""

    def test_supported_device_types_follow_registry(self):
        self.assertIn("cuda", _CUDAGRAPH_SUPPORTED_DEVICE_TYPES)
        self.assertIn("xpu", _CUDAGRAPH_SUPPORTED_DEVICE_TYPES)
        self.assertNotIn("cpu", _CUDAGRAPH_SUPPORTED_DEVICE_TYPES)
        self.assertEqual(
            _CUDAGRAPH_SUPPORTED_DEVICE_TYPES,
            frozenset(
                name
                for name, interface in get_registered_device_interfaces()
                if interface.Graphs.supported
            ),
        )

    def test_supported_backends_implement_the_graph_surface(self):
        for name, interface in get_registered_device_interfaces():
            if not interface.Graphs.supported:
                continue
            for member in _REQUIRED_GRAPH_MEMBERS:
                self.assertIsNot(
                    getattr(interface.Graphs, member),
                    getattr(DeviceInterface.Graphs, member),
                    f"{name} advertises cudagraph support but does not "
                    f"implement Graphs.{member}",
                )

    def test_single_supported_device_passes_gate(self):
        for device_type in _CUDAGRAPH_SUPPORTED_DEVICE_TYPES:
            self.assertIsNone(
                check_multiple_devices_or_any_cpu_nodes(
                    {torch.device(device_type, 0): None}
                )
            )

    def test_unsupported_device_is_not_reported_as_multiple_devices(self):
        reason = check_multiple_devices_or_any_cpu_nodes({torch.device("mps", 0): None})
        self.assertIsNotNone(reason)
        self.assertNotIn("multiple devices", reason)
        self.assertIn("mps", reason)

    def test_multiple_devices_still_reported(self):
        reason = check_multiple_devices_or_any_cpu_nodes(
            {torch.device("cuda", 0): None, torch.device("cuda", 1): None}
        )
        self.assertIn("multiple devices", reason)

    @inductor_config.patch("graph_partition", False)
    @parametrize(
        "devices",
        [
            (torch.device("mps", 0),),
            (torch.device("cuda", 0), torch.device("cuda", 1)),
            (torch.device("cpu"), torch.device("cuda", 0)),
        ],
    )
    def test_skip_reason_is_not_preformatted(self, devices):
        # Callers apply format_default_skip_message exactly once; a reason that
        # already carries the prefix logs "skipping cudagraphs due to skipping
        # cudagraphs due to ...". The node values must be truthy, or the cpu
        # branch's walrus falls through to the multiple-devices branch.
        mapping = {device: _FakeNode(f"n_{device.type}") for device in devices}
        reason = check_lowering_disable_cudagraph(mapping)
        self.assertIsNotNone(reason)
        self.assertEqual(format_default_skip_message(reason).count("due to"), 1)

    @inductor_config.patch("graph_partition", False)
    def test_cpu_node_reason_names_the_node(self):
        reason = check_multiple_devices_or_any_cpu_nodes(
            {torch.device("cpu"): _FakeNode("arg1_1"), torch.device("cuda", 0): None}
        )
        self.assertEqual(reason, "cpu device (arg1_1)")

    def test_dynamo_backend_mutation_reason_is_not_preformatted(self):
        # check_for_skip feeds format_default_skip_message in both of its
        # callers, so none of its branches may pre-format.
        from torch._dynamo.backends.cudagraphs import check_for_skip

        reason = check_for_skip(_mutating_graph(), 0)
        self.assertIsNotNone(reason)
        self.assertNotIn("skipping cudagraphs due to", reason)
        self.assertIn("mutated inputs", reason)

    def test_storage_from_high_data_pointer(self):
        # XPU USM addresses sit above 2**63; the binding must take them as
        # unsigned rather than rejecting them as an out-of-range int64_t.
        high_ptr = 0xFF00FFFFFFFC0000
        self.assertGreaterEqual(high_ptr, 2**63)
        storage = torch._C._construct_storage_from_data_pointer(
            high_ptr, torch.device("cpu"), 256
        )
        self.assertEqual(storage.data_ptr(), high_ptr)

    def test_containers_are_keyed_by_device_type(self):
        try:
            cuda_container = get_container(torch.device("cuda", 0))
            xpu_container = get_container(torch.device("xpu", 0))
            self.assertIsNot(cuda_container, xpu_container)
            self.assertEqual(cuda_container.device, torch.device("cuda", 0))
            self.assertEqual(xpu_container.device, torch.device("xpu", 0))
            self.assertIs(cuda_container, get_container(torch.device("cuda", 0)))
        finally:
            reset_cudagraph_trees()


class CudagraphTreesDeviceGenericTest(InductorTestCase):
    # mode="reduce-overhead" is what turns on triton.cudagraphs here; a
    # class-level config patch would hide these tests from
    # instantiate_device_type_tests, which only sees the class's own __dict__.

    def setUp(self):
        super().setUp()
        counters.clear()

    def tearDown(self):
        reset_cudagraph_trees()
        super().tearDown()

    def _tree_manager(self, device):
        return get_container(torch.device(device)).tree_manager

    def test_cudagraphs_are_recorded(self, device):
        def fn(x):
            return (x * 2).relu() + 1

        compiled = torch.compile(fn, mode="reduce-overhead")
        x = torch.randn(8, 8, device=device)
        for _ in range(3):
            compiled(x)

        manager = self._tree_manager(device)
        self.assertIsNotNone(manager)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
        self.assertGreater(manager.new_graph_id().id, 0)

    @parametrize("dynamic", [False, True])
    def test_replay_matches_eager_with_changing_inputs(self, device, dynamic):
        def fn(x, y):
            return (x @ y).sin() + x

        compiled = torch.compile(fn, mode="reduce-overhead", dynamic=dynamic)
        for i in range(5):
            x = torch.randn(16, 16, device=device)
            y = torch.randn(16, 16, device=device)
            # clone: a cudagraph output buffer is overwritten by the next replay
            self.assertEqual(compiled(x, y).clone(), fn(x, y), msg=f"iteration {i}")

        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

    def test_training_step_matches_eager(self, device):
        def make_model():
            torch.manual_seed(0)
            return torch.nn.Sequential(
                torch.nn.Linear(16, 16, device=device),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 16, device=device),
            )

        eager_model = make_model()
        compiled_model = torch.compile(make_model(), mode="reduce-overhead")

        for _ in range(3):
            x = torch.randn(4, 16, device=device)
            eager_model.zero_grad()
            eager_model(x).sum().backward()
            expected = [p.grad.clone() for p in eager_model.parameters()]

            compiled_model.zero_grad()
            compiled_model(x).sum().backward()
            self.assertEqual([p.grad for p in compiled_model.parameters()], expected)

        self.assertIsNotNone(self._tree_manager(device))

    @inductor_config.patch("graph_partition", False)
    def test_cpu_node_skip_message_is_prefixed_once(self, device):
        def fn(x, y):
            return x + 1, y + 2

        compiled = torch.compile(fn, mode="reduce-overhead")
        log_stream, ctx = logs_to_string(
            "torch._inductor.cudagraph_utils", "cudagraphs"
        )
        with ctx():
            compiled(torch.ones(4, device=device), torch.ones(4))

        logged = log_stream.getvalue()
        self.assertIn("skipping cudagraphs due to cpu device", logged)
        self.assertNotIn(
            "skipping cudagraphs due to skipping cudagraphs due to", logged
        )
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    def test_error_on_dealloc_use(self, device):
        @torch.compile(mode="reduce-overhead")
        def foo(x):
            return x * x * x

        inp = torch.rand([4], device=device)
        out = foo(inp)
        out2 = foo(inp)

        with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
            out + out

        foo(inp)

        with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
            out2 + out2

    def test_dealloc_detaches_storage_from_tensor(self, device):
        # The tensor-level invalidation must drop the TensorImpl's storage, not
        # just flag the StorageImpl, or a dead output keeps a storage alive over
        # a pool block that has been freed.
        @torch.compile(mode="reduce-overhead")
        def foo(x):
            return x * x * x

        inp = torch.rand([4], device=device)
        out = foo(inp)
        foo(inp)

        with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
            out.untyped_storage()

    # nonzero only reaches the aot graph as an unbacked-symbol node when dynamo
    # is allowed to capture data-dependent output shapes; otherwise it breaks out
    # of the graph and the backend never sees an incompatible node.
    @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
    def test_dynamo_backend_incompatible_op_message(self, device):
        def fn(x):
            return torch.nonzero(x)

        compiled = torch.compile(fn, backend="cudagraphs")
        log_stream, ctx = logs_to_string(
            "torch._inductor.cudagraph_utils", "cudagraphs"
        )
        with ctx():
            compiled(torch.ones(4, device=device))

        logged = log_stream.getvalue()
        self.assertIn("skipping cudagraphs due to incompatible op (nonzero)", logged)
        self.assertNotIn(
            "skipping cudagraphs due to skipping cudagraphs due to", logged
        )

    def test_separate_devices_get_separate_managers(self, device):
        device_type = torch.device(device).type
        if get_interface_for_device(device_type).device_count() < 2:
            self.skipTest("requires 2 devices")

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, mode="reduce-overhead")
        for index in (0, 1):
            x = torch.randn(8, device=torch.device(device_type, index))
            for _ in range(3):
                compiled(x)

        first = self._tree_manager(torch.device(device_type, 0))
        second = self._tree_manager(torch.device(device_type, 1))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)


instantiate_device_type_tests(
    CudagraphTreesDeviceGenericTest,
    globals(),
    only_for=tuple(sorted(_CUDAGRAPH_SUPPORTED_DEVICE_TYPES)),
    allow_xpu=True,
)


if __name__ == "__main__":
    run_tests(needs="filelock")
