# Owner(s): ["module: dynamo"]
import dataclasses
import operator
import unittest
from unittest import mock

import torch
from torch._dynamo.compiled_autograd import (
    _graph_placeholders,
    _move_cpu_scalar_to_device,
    AutogradCompilerInstance,
    Op,
)
from torch.testing._internal.common_utils import run_tests, TestCase


OOT_DEVICE = "fakeoot"
requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires cuda")


@dataclasses.dataclass(frozen=True)
class _DeviceStub:
    type: str
    index: int = 0


class _MetaVal:
    def __init__(
        self, device: torch.device | _DeviceStub, shape: tuple[int, ...] = ()
    ) -> None:
        self._device = device
        self._shape = shape

    @property
    def device(self) -> torch.device | _DeviceStub:
        return self._device

    def size(self) -> torch.Size:
        return torch.Size(self._shape)


def _build_compiled_autograd_inputs_graph(
    input_vals: list[_MetaVal],
) -> torch.fx.Graph:
    graph = torch.fx.Graph()
    for name in _graph_placeholders:
        graph.create_node("placeholder", name, (), {})
    inputs = next(n for n in graph.nodes if n.target == "inputs")
    for i, val in enumerate(input_vals):
        getitem = graph.create_node("call_function", operator.getitem, (inputs, i), {})
        getitem.meta["val"] = val
    return graph


def _run_move_graph_nodes_to_cuda(
    input_vals: list[_MetaVal],
) -> tuple[list[int], torch.device | None, torch.fx.Graph]:
    graph = _build_compiled_autograd_inputs_graph(input_vals)
    inst = object.__new__(AutogradCompilerInstance)
    indices, target = inst.move_graph_nodes_to_cuda(graph)
    return indices, target, graph


class TestCompiledAutogradAcceleratorScalarMove(TestCase):
    @mock.patch.object(
        torch._C, "_get_privateuse1_backend_name", return_value=OOT_DEVICE
    )
    def test_single_oot_device_moves_scalar(self, _mock_name: mock.Mock) -> None:
        graph = _build_compiled_autograd_inputs_graph(
            [
                _MetaVal(_DeviceStub(OOT_DEVICE, 1), (2,)),
                _MetaVal(torch.device("cpu")),
            ]
        )
        inputs = next(n for n in graph.nodes if n.target == "inputs")
        getitems = list(inputs.users.keys())
        graph.create_node(
            "call_function",
            torch.ops.aten.mul.Tensor,
            (getitems[1], getitems[0]),
            {},
        )

        inst = object.__new__(AutogradCompilerInstance)
        with mock.patch(
            "torch._dynamo.compiled_autograd._move_cpu_scalar_to_device",
            side_effect=lambda val, device, **kwargs: _MetaVal(device),
        ) as move_mock:
            indices, target = inst.move_graph_nodes_to_cuda(graph)

        self.assertEqual(indices, [1])
        self.assertEqual(target, _DeviceStub(OOT_DEVICE, 1))
        move_mock.assert_called_once()
        self.assertEqual(move_mock.call_args[0][1], _DeviceStub(OOT_DEVICE, 1))

    @mock.patch.object(
        torch._C, "_get_privateuse1_backend_name", return_value=OOT_DEVICE
    )
    def test_mixed_oot_indices_skip(self, _mock_name: mock.Mock) -> None:
        indices, target, _ = _run_move_graph_nodes_to_cuda(
            [
                _MetaVal(_DeviceStub(OOT_DEVICE, 0), (2,)),
                _MetaVal(_DeviceStub(OOT_DEVICE, 1), (2,)),
                _MetaVal(torch.device("cpu")),
            ]
        )
        self.assertEqual(indices, [])
        self.assertIsNone(target)

    @mock.patch.object(
        torch._C, "_get_privateuse1_backend_name", return_value=OOT_DEVICE
    )
    def test_mixed_cuda_and_oot_skip(self, _mock_name: mock.Mock) -> None:
        indices, target, _ = _run_move_graph_nodes_to_cuda(
            [
                _MetaVal(torch.device("cuda", 0), (2,)),
                _MetaVal(_DeviceStub(OOT_DEVICE, 0), (2,)),
                _MetaVal(torch.device("cpu")),
            ]
        )
        self.assertEqual(indices, [])
        self.assertIsNone(target)

    @mock.patch.object(
        torch._C, "_get_privateuse1_backend_name", return_value=OOT_DEVICE
    )
    def test_mixed_oot_and_xpu_skip(self, _mock_name: mock.Mock) -> None:
        indices, target, _ = _run_move_graph_nodes_to_cuda(
            [
                _MetaVal(_DeviceStub(OOT_DEVICE, 0), (2,)),
                _MetaVal(_DeviceStub("xpu", 0), (2,)),
                _MetaVal(torch.device("cpu")),
            ]
        )
        self.assertEqual(indices, [])
        self.assertIsNone(target)

    def test_cpu_only_graph_skips(self) -> None:
        indices, target, _ = _run_move_graph_nodes_to_cuda(
            [_MetaVal(torch.device("cpu"), (2,)), _MetaVal(torch.device("cpu"))]
        )
        self.assertEqual(indices, [])
        self.assertIsNone(target)

    @mock.patch.object(
        torch._C, "_get_privateuse1_backend_name", return_value=OOT_DEVICE
    )
    def test_custom_function_scalar_not_moved(self, _mock_name: mock.Mock) -> None:
        accel = _MetaVal(_DeviceStub(OOT_DEVICE, 0), (2,))
        scalar = _MetaVal(torch.device("cpu"))
        graph = _build_compiled_autograd_inputs_graph([accel, scalar])
        inputs = next(n for n in graph.nodes if n.target == "inputs")
        getitems = list(inputs.users.keys())
        custom_op = Op("test_custom", lambda *args: args, is_custom_function=True)
        graph.create_node("call_function", custom_op, (getitems[1],), {})

        inst = object.__new__(AutogradCompilerInstance)
        indices, target = inst.move_graph_nodes_to_cuda(graph)
        self.assertEqual(indices, [])
        self.assertEqual(target, _DeviceStub(OOT_DEVICE, 0))

    @requires_cuda
    def test_move_cpu_scalar_to_device_preserves_cuda_index(self) -> None:
        if torch.cuda.device_count() < 2:
            self.skipTest("needs at least 2 cuda devices")
        target = torch.device("cuda", 1)
        scalar = torch.tensor(2.0, device="cpu")
        moved = _move_cpu_scalar_to_device(scalar, target)
        self.assertEqual(moved.device, target)
        runtime_moved = _move_cpu_scalar_to_device(scalar, target, for_runtime=True)
        self.assertEqual(runtime_moved.device, target)


if __name__ == "__main__":
    run_tests()
