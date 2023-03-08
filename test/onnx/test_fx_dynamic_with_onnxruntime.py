# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy

import inspect

import io
import unittest
import warnings
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import onnx.reference
import onnx_test_common

import onnxruntime  # type: ignore[import]

import torch
import torchvision
from torch.onnx._internal import _beartype, diagnostics, fx as fx_onnx
from torch.testing._internal import common_utils
from torch.types import Number
from torch.utils import _pytree as pytree

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable]
_ONNXModelType = Union["onnx.ModelProto", bytes, str, io.BytesIO]
_InputArgsType = Union[torch.Tensor, Tuple[Any, ...]]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Union[Sequence[_NumericType], Sequence]


@_beartype.beartype
# TODO(titaiwang): bound.args makes pytorch_inputs hard to annotate
# maybe annotate it when the exporter API is launched
def _run_ort(onnx_model: _ONNXModelType, pytorch_inputs: Any) -> _OutputsType:
    session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


@_beartype.beartype
def _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
    model: _ModelType,
    input_args: _InputArgsType,
    rtol: float = 1e-3,
    atol: float = 1e-7,
    opset_version: int = 18,
    additional_test_inputs: Optional[Sequence[_InputArgsType]] = None,
    additional_test_kwargs: Optional[Sequence[_InputKwargsType]] = None,
    **input_kwargs,
):
    @_beartype.beartype
    def _try_clone_model(model: _ModelType) -> _ModelType:
        """Used for preserving original model in case forward mutates model states."""
        try:
            return copy.deepcopy(model)
        except Exception:
            warnings.warn(
                "Failed to clone model. Model state might be mutated during verification."
            )
            return model

    def compare_pytorch_onnx_with_ort(
        onnx_model: Union["onnx.ModelProto", bytes],
        model_input_args: _InputArgsType,
        model_input_kwargs,
    ):

        # Inspect the model's signature. It will be used
        # to flatten kwargs.
        if isinstance(model, torch.nn.Module):
            signature = inspect.signature(model.forward)
        else:
            signature = inspect.signature(model)

        # Bind args and kwargs to the model's signature to
        # flatten kwargs into positional args since ONNX
        # model cannot be called with kwargs.
        bound = signature.bind(*model_input_args, **model_input_kwargs)
        # Fill optional inputs.
        bound.apply_defaults()
        assert not bound.kwargs

        pt_cloned_model = _try_clone_model(model)
        ref_outputs, _ = pytree.tree_flatten(
            pt_cloned_model(*model_input_args, **model_input_kwargs)
        )
        ort_outputs = _run_ort(onnx_model, bound.args)
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(
                ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
            )

    # Feed args and kwargs into exporter.
    # Note that exporter should flatten kwargs into positional args the exported model;
    # since ONNX doesn't represent kwargs.
    onnx_model = fx_onnx.export_after_normalizing_args_and_kwargs(
        model,
        *input_args,
        opset_version=opset_version,
        use_binary_format=True,
        **input_kwargs,
    )

    compare_pytorch_onnx_with_ort(onnx_model, input_args, input_kwargs)

    # TODO(titaiwang): do not support kwargs now
    if additional_test_inputs:
        for additional_input_args in additional_test_inputs:
            compare_pytorch_onnx_with_ort(onnx_model, additional_input_args, {})


class TestFxDynamicWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        self.diag_ctx = diagnostics.engine.create_diagnostic_context(
            "test_fx_export", version=torch.__version__
        )
        self.opset_version = 18

    def tearDown(self):
        diagnostics.engine.dump(
            f"test_report_{self._testMethodName}.sarif", compress=False
        )
        super().tearDown()

    def test_shufflenet_v2_dynamic_axes(self):
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)
        # self.run_test(
        #     model,
        #     (dummy_input,),
        #     additional_test_inputs=[(dummy_input,), (test_inputs,)],
        #     input_names=["input_images"],
        #     output_names=["outputs"],
        #     dynamic_axes={
        #         "input_images": {0: "batch_size"},
        #         "output": {0: "batch_size"},
        #     },
        #     rtol=1e-3,
        #     atol=1e-5,
        # )
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
            additional_test_inputs=[(dummy_input,), (test_inputs,)],
            rtol=1e-3,
            atol=1e-5,
        )

    def test_dynamic_axes_on_add(self):
        class DynamicAdd(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.add(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        input_x = torch.randn(2, 4)
        input_y = torch.randn(2, 4)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[(input_x, input_y)]
        )

    unittest.skip(
        "The decomposed module introduces aten::expand right after inputs, "
        "which raises error that graph_building not yet supports List[fx.Node, ...]."
        "https://github.com/microsoft/onnx-script/issues/481"
    )

    def test_dynamic_axes_on_matmul(self):
        class DynamicMatMul(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.matmul(x, y)

        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicMatMul(), (x, y), additional_test_inputs=[(input_x, input_y)]
        )


if __name__ == "__main__":
    common_utils.run_tests()
