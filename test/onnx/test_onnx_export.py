# Owner(s): ["module: onnx"]

import contextlib
import io
import itertools
import os
import sys
import unittest.mock
from typing import Callable, Iterable, Optional, Tuple, Union

import onnx
from test_pytorch_common import TestCase

import torch
from torch.onnx import OperatorExportTypes, symbolic_registry
from torch.onnx._globals import GLOBALS
from torch.onnx.symbolic_helper import _onnx_unsupported
from torch.testing._internal.common_utils import custom_op, skipIfCaffe2

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


def export_to_onnx(
    model: Union[torch.nn.Module, torch.jit.ScriptFunction],
    input: Tuple[torch.Tensor],
    custom_ops: Optional[
        Iterable[
            Union[contextlib.AbstractContextManager, contextlib.ContextDecorator],
        ]
    ] = None,
    mocks: Optional[Iterable] = None,
    operator_export_type: OperatorExportTypes = OperatorExportTypes.ONNX,
    opset_version: int = GLOBALS.export_onnx_opset_version,
) -> onnx.ModelProto:
    """Exports `model(input)` to ONNX and returns it.

    Custom operators and/or unittest patches can be used help reproducing specific behaviors.

    Args:
        model: model to export
        input: model input with same format as `torch.onnx.export(..,args,...)`
        custom_ops: list of custom operators to use during export
        mocks: list of mocks to use during export
        operator_export_type: export type as described by `torch.onnx.export(...operator_export_type,...)`
        opset_version: ONNX opset version as described by `torch.onnx.export(...opset_version,...)`
    Returns:
        A valid ONNX model (`onnx.ModelProto`)
    """
    custom_ops = custom_ops or []
    mocks = mocks or []
    with contextlib.ExitStack() as stack:
        for ctx in itertools.chain(custom_ops, mocks):
            stack.enter_context(ctx)

        f = io.BytesIO()
        torch.onnx.export(
            model,
            input,
            f,
            operator_export_type=operator_export_type,
            opset_version=opset_version,
        )

    # Validate ONNX graph before returning it
    onnx_model = onnx.load_from_string(f.getvalue())
    onnx.checker.check_model(onnx_model)
    return onnx_model


class TestONNXExport(TestCase):
    @skipIfCaffe2
    def test_clip_aten_fallback_due_exception(self):
        def bad_clamp(g, self, min, max):
            return _onnx_unsupported("Bad boy!")

        class MyClip(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-0.5, max=0.5)

        onnx_model = export_to_onnx(
            MyClip(),
            torch.randn(3, 4, requires_grad=True),
            custom_ops=[custom_op("aten::clamp", bad_clamp, 9)],
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        self.assertAtenOp(onnx_model, "clamp", "Tensor")

    @skipIfCaffe2
    def test_clip_aten_fallback_explicit_request(self):
        class MyClip(torch.nn.Module):
            def forward(self, x):
                return torch.clamp(x, min=-0.5, max=0.5)

        def break_is_registered_op_api(opname, domain, version):
            fake_missing_symbolics = ("clamp",)
            if opname in fake_missing_symbolics:
                return False
            return (
                (domain, version) in symbolic_registry._registry
                and opname in symbolic_registry._registry[(domain, version)]
            )

        # Force missing symbolic for well-known op using a mock
        onnx_model = export_to_onnx(
            MyClip(),
            torch.randn(3, 4, requires_grad=True),
            mocks=[
                unittest.mock.patch(
                    "torch.onnx.symbolic_registry.is_registered_op",
                    side_effect=break_is_registered_op_api,
                )
            ],
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
        )
        self.assertAtenOp(onnx_model, "clamp", "Tensor")

    def _helper_test_to_(self, cast_fn: Callable[[torch.Tensor], torch.Tensor]):
        """Helper to test aten::to(device) variants.

        `cast_fn` is converted into a `torch.jit.script`. It wraps `aten::to`
        during export to preventing the devices to be hard-coded.

        Needed by detectron2 after https://github.com/facebookresearch/detectron2/pull/4132/
        """
        cast_fn = torch.jit.script(cast_fn)
        onnx_model = export_to_onnx(cast_fn, torch.zeros([1, 3, 32, 32]))
        for n in onnx_model.graph.node:
            self.assertNotEqual(n.op_type, "To")
            self.assertNotEqual(n.op_type, "Cast")

    def test_to__cpu_string(self):
        def cast_cpu_string(src: torch.Tensor) -> torch.Tensor:
            return src.to("cpu")

        self._helper_test_to_(cast_cpu_string)

    def test_to__device_cpu_string(self):
        def cast_device_cpu_string(src: torch.Tensor) -> torch.Tensor:
            return src.to(device="cpu")

        self._helper_test_to_(cast_device_cpu_string)
