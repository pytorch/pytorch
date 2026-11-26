# Owner(s): ["module: inductor"]
"""Device-agnostic tests for ``torch/_inductor/compile_fx.py``.

Covers ``get_device_context``: the device context helper driven by the
registered ``DeviceInterface``, which generalizes the former CUDA-only
``get_cuda_device_context``.

All tests are CPU-runnable: they use fake tensors for the per-device graph
metadata, temporarily-registered fake ``DeviceInterface``s for the hook
coverage, and mock backend availability so the accelerator-specific assertions
hold on every host.  Every assertion calls the production symbol directly; no
inlined copy of the decoupled expression is used.
"""

import contextlib
from unittest.mock import patch

import torch
from torch._dynamo.device_interface import (
    CpuInterface,
    CudaInterface,
    device_interfaces,
    DeviceInterface,
    MtiaInterface,
    register_interface_for_device,
    XpuInterface,
)
from torch._dynamo.variables.user_defined import UserDefinedClassVariable
from torch._inductor.compile_fx import get_cuda_device_context, get_device_context
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def _fake_tensor_on(device: str, mode: FakeTensorMode) -> FakeTensor:
    """Build a fake tensor tagged with an arbitrary (possibly CPU-only) device."""
    meta = mode.from_tensor(torch.empty(2, device="meta"))
    return FakeTensor(mode, meta, device=torch.device(device))


def _make_graph(*device_types: str) -> torch.fx.GraphModule:
    """Build a GraphModule whose single placeholder (and output effect) carries
    the requested device types in its ``meta["val"]``.

    ``get_all_devices`` reads placeholder nodes, so one placeholder per desired
    device is enough to exercise the device-collection logic.
    """
    graph = torch.fx.Graph()
    mode = FakeTensorMode()

    nodes = []
    for i, device_type in enumerate(device_types):
        # Placeholder names must be valid Python identifiers (device types such
        # as "cuda:0" contain a colon); use a collision-free sanitized name.
        ph = graph.placeholder(f"x_{i}")
        ph.name = f"x_{i}"
        ph.meta["val"] = _fake_tensor_on(device_type, mode)
        nodes.append(ph)

    # Output the single placeholder if there is only one; otherwise a tuple.
    if len(nodes) == 1:
        graph.output(nodes[0])
    else:
        graph.output(tuple(nodes))
    gm = torch.fx.GraphModule({}, graph)
    gm.recompile()
    return gm


class _RecordingInterface(DeviceInterface):
    """Fake interface whose ``device`` is a real-shaped guard recording entry."""

    entered: list[torch.device] = []

    @staticmethod
    def is_available() -> bool:
        return True

    class device:
        def __init__(self, dev: torch.device) -> None:
            self.dev = dev

        def __enter__(self):
            _RecordingInterface.entered.append(self.dev)
            return self

        def __exit__(self, *exc_info) -> bool | None:
            return None


@instantiate_parametrized_tests
class TestGetDeviceContext(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        super().setUp()
        _RecordingInterface.entered.clear()

    def _register(self, device: str, interface: type[DeviceInterface]) -> None:
        # The existing registry idiom (test/dynamo/test_streams.py): patch.dict
        # snapshots the whole mapping and restores it on cleanup, so preexisting
        # registrations are never popped; the in-graph-class cache is cleared
        # after the restore so it cannot keep the fake interface alive.
        self.addCleanup(UserDefinedClassVariable._in_graph_classes.cache_clear)
        patcher = patch.dict(device_interfaces)
        patcher.start()
        self.addCleanup(patcher.stop)
        register_interface_for_device(device, interface)

    def test_backwards_compatible_alias(self) -> None:
        # External callers / downstream imports alias the old symbol.
        self.assertIs(get_cuda_device_context, get_device_context)

    @parametrize(
        "device_types",
        [
            [],
            ["cpu"],
            ["meta"],
            ["cpu", "meta"],
            ["cuda:0", "cuda:1"],
            ["xpu:0", "mtia:0"],
        ],
    )
    def test_no_single_guardable_device_returns_nullcontext(
        self, device_types: list[str]
    ) -> None:
        gm = _make_graph(*device_types)
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_unregistered_device_type_is_skipped(self) -> None:
        # privateuseone is a valid device type but vanilla torch registers no
        # interface for it; it must be skipped rather than raising.  Tests that
        # inject a fake interface on an out-of-tree-shaped slot use "xpu"
        # instead: with an out-of-tree backend attached, fake privateuseone
        # tensors rebind to that backend's own device, bypassing the
        # registration here.
        gm = _make_graph("privateuseone:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_single_cuda_graph_without_cuda_returns_nullcontext(self) -> None:
        # CUDA-tagged graphs can occur on CUDA-less hosts (fake tensors, AOTI
        # cross-compilation, export); entering torch.cuda.device would fail.
        gm = _make_graph("cuda:0")
        with patch.object(torch.cuda, "is_available", return_value=False):
            ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_torch_device_interface_returns_nullcontext(self) -> None:
        # Out-of-tree backends may register interfaces whose device attr is
        # torch.device. Its context-manager protocol is a default-device mode,
        # not a device guard, so exposes_device_guard must reject it.

        class TorchDeviceInterface(DeviceInterface):
            device = torch.device

            @staticmethod
            def is_available() -> bool:
                return True

        self._register("xpu", TorchDeviceInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_base_stub_interface_returns_nullcontext(self) -> None:
        # Subclassing DeviceInterface without overriding device leaves the
        # raising base stub; it must not be entered.

        class StubInterface(DeviceInterface):
            @staticmethod
            def is_available() -> bool:
                return True

        self._register("xpu", StubInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_exposes_device_guard_matches_in_tree_interfaces(self) -> None:
        # The opt-in hook follows the same identity-sentinel contract as
        # exposes_streams: real guards (torch.<dev>.device classes) opt in,
        # leaving the base stub (cpu) out.  The context-manager requirement
        # must accept every in-tree guard, not just CUDA's.
        self.assertTrue(CudaInterface.exposes_device_guard())
        self.assertTrue(XpuInterface.exposes_device_guard())
        self.assertTrue(MtiaInterface.exposes_device_guard())
        self.assertFalse(CpuInterface.exposes_device_guard())
        self.assertFalse(DeviceInterface.exposes_device_guard())

    def test_guard_interface_is_entered(self) -> None:
        # A registered interface whose device context sets a flag proves the
        # production path both returns and lets the caller enter the guard.
        self._register("xpu", _RecordingInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIsInstance(ctx, _RecordingInterface.device)
        with ctx:
            pass
        self.assertEqual(len(_RecordingInterface.entered), 1)
        self.assertEqual(_RecordingInterface.entered[0].type, "xpu")

    @parametrize("placeholder", ["cpu", "meta"])
    def test_incidental_placeholders_do_not_suppress_guard(
        self, placeholder: str
    ) -> None:
        # CPU scalars (optimizer steps, 0-d constants) and meta placeholders
        # are incidental: the accelerator guard must not depend on them.
        self._register("xpu", _RecordingInterface)
        gm = _make_graph("xpu:0", placeholder)
        ctx = get_device_context(gm)
        self.assertIsInstance(ctx, _RecordingInterface.device)
        with ctx:
            pass
        self.assertEqual(len(_RecordingInterface.entered), 1)
        self.assertEqual(_RecordingInterface.entered[0].type, "xpu")

    def test_unavailable_interface_falls_back_to_nullcontext(self) -> None:
        # The generalized branch mirrors the cuda branch's availability
        # fallback: in-tree XPU/Mtia interfaces are registered on every host
        # and their ``device`` is a real guard, so without this gate a
        # single-device graph compiled on a host lacking the runtime would go
        # from a working no-op to a hard error when the caller enters the
        # context (mocked device tags, AOT cross-compile flows).

        class UnavailableInterface(DeviceInterface):
            @staticmethod
            def is_available() -> bool:
                return False

            class device:
                def __init__(self, dev: torch.device) -> None:
                    raise AssertionError("guard must not be constructed")

                def __enter__(self):
                    raise AssertionError("guard must not be entered")

                def __exit__(self, *exc_info) -> bool | None:
                    raise AssertionError("guard must not be exited")

        self._register("xpu", UnavailableInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_interface_without_is_available_falls_back_to_nullcontext(self) -> None:
        # A partially-implemented interface (base is_available raises
        # NotImplementedError) must be treated as unavailable rather than
        # propagating out of the compile path.

        class PartialInterface(DeviceInterface):
            class device:
                def __init__(self, dev: torch.device) -> None:
                    raise AssertionError("guard must not be constructed")

                def __enter__(self):
                    raise AssertionError("guard must not be entered")

                def __exit__(self, *exc_info) -> bool | None:
                    raise AssertionError("guard must not be exited")

        self._register("xpu", PartialInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_non_context_manager_device_returns_nullcontext(self) -> None:
        # A device override without the context-manager protocol is not
        # enterable, so it must not be reported as a guard.  This is the shape
        # of the in-tree triton extension test backend (its plain device class
        # is registered for privateuseone and compiles through compile_fx).

        class NonGuardInterface(DeviceInterface):
            class device:
                def __init__(self, dev: torch.device) -> None:
                    self.dev = dev

            @staticmethod
            def is_available() -> bool:
                return True

        self.assertFalse(NonGuardInterface.exposes_device_guard())
        self._register("xpu", NonGuardInterface)
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_multiple_indices_of_one_type_returns_nullcontext(self) -> None:
        # The single-index rule must be reached on every host: with an
        # available guard interface registered, a graph spanning two indices of
        # one type has no unambiguous current device, so the guard is returned
        # as a null context and never entered.
        self._register("xpu", _RecordingInterface)
        gm = _make_graph("xpu:0", "xpu:1")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))
        self.assertEqual(_RecordingInterface.entered, [])


class TestGetDeviceContextCuda(TestCase):
    # These tests construct torch.cuda.device to name the expected context
    # type (its __init__ only resolves the device index, so it is safe on any
    # host) and patch is_available, making the assertion meaningful on CPU CI
    # hosts rather than silently checking the nullcontext fallback.
    hw_classification = HardwareClassification.CUDA

    def test_single_cuda_graph_returns_cuda_device_context(self) -> None:
        gm = _make_graph("cuda:0")
        with patch.object(torch.cuda, "is_available", return_value=True):
            ctx = get_device_context(gm)
        self.assertIsInstance(ctx, torch.cuda.device)

    def test_mixed_device_types_preserves_cuda_precedence(self) -> None:
        # A single cuda device activates the CUDA context, even when other
        # device types are present.  If the precedence were lost, this graph
        # would return nullcontext (two device types remain after cpu/meta are
        # discarded), so the assertion pins the behavior on every host.
        gm = _make_graph("cuda:0", "xpu:0")
        with patch.object(torch.cuda, "is_available", return_value=True):
            ctx = get_device_context(gm)
        self.assertIsInstance(ctx, torch.cuda.device)


class TestGetDeviceContextXpu(TestCase):
    # Mirror of the CUDA class for the real XpuInterface path: constructs
    # torch.xpu.device (index-only __init__) with availability patched, so the
    # check holds on hosts without the XPU runtime instead of degenerating to
    # the nullcontext fallback.
    hw_classification = HardwareClassification.XPU

    def test_single_xpu_graph_returns_xpu_device_context(self) -> None:
        gm = _make_graph("xpu:0")
        with patch.object(torch.xpu, "is_available", return_value=True):
            ctx = get_device_context(gm)
        self.assertIsInstance(ctx, torch.xpu.device)

    def test_xpu_graph_with_cpu_tensors_returns_xpu_context(self) -> None:
        # An accelerator graph with incidental CPU tensors (the common
        # optimizer-state shape) must still get the accelerator guard rather
        # than have the guard depend on CPU inputs.
        gm = _make_graph("xpu:0", "cpu")
        with patch.object(torch.xpu, "is_available", return_value=True):
            ctx = get_device_context(gm)
        self.assertIsInstance(ctx, torch.xpu.device)


if __name__ == "__main__":
    run_tests()
