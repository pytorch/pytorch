# Owner(s): ["module: inductor"]
"""Device-agnostic tests for ``torch/_inductor/compile_fx.py``.

Covers ``get_device_context``: the device context helper driven by the
registered ``DeviceInterface``, which generalizes the former CUDA-only
``get_cuda_device_context``.

All tests are CPU-runnable: they use fake tensors for the per-device graph
metadata and a temporarily-registered fake ``DeviceInterface`` for the hook
coverage.  Every assertion calls the production symbol directly; no inlined copy
of the decoupled expression is used.
"""

import contextlib
import unittest.mock

import torch
from torch._dynamo.device_interface import (
    device_interfaces,
    DeviceInterface,
    register_interface_for_device,
)
from torch._inductor.compile_fx import get_cuda_device_context, get_device_context
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.testing._internal.common_utils import HardwareClassification, TestCase


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


class _FakeDeviceInterface(DeviceInterface):
    """Temporary interface whose ``device`` context manager records entry/exit."""

    device = contextlib.nullcontext

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1


# privateuseone is a constructible device type in vanilla torch (no torch_npu
# required) and has no in-tree DeviceInterface, so it is a clean slot to inject a
# temporary fake interface without disturbing any in-tree backend.
REGISTERED_FAKE_DEVICE = "privateuseone"


class TestGetDeviceContext(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def setUp(self) -> None:
        self.addCleanup(device_interfaces.pop, REGISTERED_FAKE_DEVICE, None)

    def test_backwards_compatible_alias(self) -> None:
        # External callers / downstream patches alias the old symbol.
        self.assertIs(get_cuda_device_context, get_device_context)

    def test_empty_graph_returns_nullcontext(self) -> None:
        graph = torch.fx.Graph()
        graph.output(())
        gm = torch.fx.GraphModule({}, graph)
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_cpu_graph_returns_nullcontext(self) -> None:
        gm = _make_graph("cpu")
        ctx = get_device_context(gm)
        # CPU interface does not provide a device context manager (base class).
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_single_cuda_graph_returns_cuda_device_context(self) -> None:
        gm = _make_graph("cuda:0")
        ctx = get_device_context(gm)
        if torch.cuda.is_available():
            expected = type(torch.cuda.device(torch.device("cuda:0")))
        else:
            expected = type(contextlib.nullcontext())
        self.assertIs(type(ctx), expected)

    def test_single_xpu_graph_returns_xpu_device_context(self) -> None:
        # Environment-conditional like the cuda case above: the availability
        # gate keeps hosts without the XPU runtime on the pre-decouple no-op
        # path instead of handing back a guard that hard-fails when entered.
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        if torch.xpu.is_available():
            expected = type(torch.xpu.device(torch.device("xpu:0")))
        else:
            expected = type(contextlib.nullcontext())
        self.assertIs(type(ctx), expected)

    def test_two_devices_returns_nullcontext(self) -> None:
        gm = _make_graph("cuda:0", "cuda:1")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_mixed_device_types_preserves_cuda_precedence(self) -> None:
        # A single cuda device activates the CUDA context, even when other
        # device types are present.  The generalized helper must preserve
        # this precedence exactly.
        gm = _make_graph("cuda:0", "xpu:0")
        ctx = get_device_context(gm)
        if torch.cuda.is_available():
            expected = type(torch.cuda.device(torch.device("cuda:0")))
        else:
            expected = type(contextlib.nullcontext())
        self.assertIs(type(ctx), expected)

    def test_mixed_non_cuda_device_types_returns_nullcontext(self) -> None:
        gm = _make_graph("xpu:0", "mtia:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_unregistered_device_type_is_skipped(self) -> None:
        # privateuseone is a valid device type but has no registered interface;
        # it must be skipped rather than raising.
        gm = _make_graph("privateuseone:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_meta_graph_returns_nullcontext(self) -> None:
        gm = _make_graph("meta")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_single_cuda_graph_without_cuda_returns_nullcontext(self) -> None:
        # CUDA-tagged graphs can occur on CUDA-less hosts (fake tensors, AOTI
        # cross-compilation, export); entering torch.cuda.device would fail.
        gm = _make_graph("cuda:0")
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=False):
            ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_torch_device_interface_returns_nullcontext(self) -> None:
        # Out-of-tree backends may register interfaces whose device attr is
        # torch.device. Its context-manager protocol is a default-device mode,
        # not a device guard, so it must not be returned here.

        class TorchDeviceInterface(DeviceInterface):
            device = torch.device

            @staticmethod
            def is_available() -> bool:
                return True

        register_interface_for_device(REGISTERED_FAKE_DEVICE, TorchDeviceInterface)
        gm = _make_graph(f"{REGISTERED_FAKE_DEVICE}:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_fake_interface_provides_context(self) -> None:
        register_interface_for_device(REGISTERED_FAKE_DEVICE, _FakeDeviceInterface)
        gm = _make_graph(f"{REGISTERED_FAKE_DEVICE}:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))

    def test_fake_interface_context_actually_applies(self) -> None:
        # A fake interface whose device context sets a flag proves the
        # production path enters the returned manager.  The slot is keyed on
        # "xpu" rather than "privateuseone": an attached out-of-tree backend
        # binds fake privateuseone tensors to its own device type, which
        # would bypass the interface registered here.  The original slot
        # value is restored on cleanup.
        entered = []

        class RecordingInterface(DeviceInterface):
            @staticmethod
            def is_available() -> bool:
                return True

            class device:
                def __init__(self, device: torch.device) -> None:
                    self.device = device

                def __enter__(self):
                    entered.append(self.device)
                    return self

                def __exit__(self, *exc_info) -> None:
                    return

        original = device_interfaces.get("xpu")
        device_interfaces["xpu"] = RecordingInterface
        if original is None:
            self.addCleanup(lambda: device_interfaces.pop("xpu", None))
        else:
            self.addCleanup(lambda: device_interfaces.__setitem__("xpu", original))
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        with ctx:
            pass
        self.assertEqual(len(entered), 1)
        self.assertEqual(entered[0].type, "xpu")

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
                    return self

                def __exit__(self, *exc_info) -> None:
                    return

        original = device_interfaces.get("xpu")
        device_interfaces["xpu"] = UnavailableInterface
        if original is None:
            self.addCleanup(lambda: device_interfaces.pop("xpu", None))
        else:
            self.addCleanup(lambda: device_interfaces.__setitem__("xpu", original))
        gm = _make_graph("xpu:0")
        ctx = get_device_context(gm)
        self.assertIs(type(ctx), type(contextlib.nullcontext()))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
