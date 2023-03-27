# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy

import inspect

import io
import unittest
import warnings
from typing import Any, Callable, Optional, Sequence, Tuple, Union

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
_OutputsType = Sequence[_NumericType]


@_beartype.beartype
def _run_ort(
    onnx_model: _ONNXModelType, pytorch_inputs: _InputArgsType
) -> _OutputsType:
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
    **input_kwargs,
):
    """Compare the results of PyTorch model with exported ONNX model

    Args:
        model (_ModelType): PyTorch model
        input_args (_InputArgsType): torch input arguments
        rtol (float, optional): relative tolerance. Defaults to 1e-3.
        atol (float, optional): absolute tolerance. Defaults to 1e-7.
        opset_version (int, optional): ONNX opset version. Defaults to 18.
        additional_test_inputs (Optional[Sequence[_InputArgsType]], optional):
            Test the models with another dataset, which is designed for dynamic axes
            testing. Defaults to None.

    """

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

    @_beartype.beartype
    def compare_pytorch_onnx_with_ort(
        onnx_model: Union["onnx.ModelProto", bytes],
        model_input_args: _InputArgsType,
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
        bound = signature.bind(*model_input_args)
        # Fill optional inputs.
        bound.apply_defaults()
        assert not bound.kwargs

        pt_cloned_model = _try_clone_model(model)
        ref_outputs, _ = pytree.tree_flatten(pt_cloned_model(*model_input_args))
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
        enable_dynamic_axes=True,  # export models with dynamic shapes
        **input_kwargs,
    )

    compare_pytorch_onnx_with_ort(onnx_model, input_args)

    # This confirms the exported mode accepts different input shapes
    # when dynamic shape is enabled.
    if additional_test_inputs:
        for additional_input_args in additional_test_inputs:
            compare_pytorch_onnx_with_ort(onnx_model, additional_input_args)


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

    @unittest.skip(
        "_aten_convolution_onnx: _add_attribute_to_torchscript_node()"
        " parameter value=[None, None] violates type hint"
        "typing.Union[float, int, str, bytes, typing.Sequence[float],"
        " typing.Sequence[int], torch.Tensor], as [None, None]:"
    )
    def test_shufflenet_v2_dynamic_axes(self):
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
            additional_test_inputs=[(dummy_input,), (test_inputs,)],
            rtol=1e-3,
            atol=1e-5,
        )

    def test_add(self):
        class DynamicAdd(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.add(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        another_x = torch.randn(3, 4)
        another_y = torch.randn(3, 4)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[(another_x, another_y)]
        )

    def test_sigmoid_add(self):
        class DynamicAdd(torch.nn.Module):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, y):
                z = torch.ops.aten.add(x, y)
                return self.sigmoid(z)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        x = x[1:, :]
        y = y[1:, :]
        input_x = torch.randn(1, 4)
        input_y = torch.randn(1, 4)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[(input_x, input_y)]
        )

    @unittest.skip("flaky test: https://github.com/microsoft/onnx-script/issues/523")
    def test_matmul(self):
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

    @unittest.skip(
        "fx.graph: doesn't handle scalar like normal tensor, so this is not yet "
        "supported! TypeError: forward() takes 1 positional argument but 2 were given"
    )
    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, x):
                return torch.scalar_tensor(x.size(0)), torch.scalar_tensor(
                    x.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            test(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @unittest.skip(
        "_aten_convolution_onnx: _add_attribute_to_torchscript_node()"
        " parameter value=[None, None] violates type hint"
        "typing.Union[float, int, str, bytes, typing.Sequence[float],"
        " typing.Sequence[int], torch.Tensor], as [None, None]:"
    )
    def test_transpose_infer_shape(self):
        class TransposeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            def forward(self, x):
                x = self.conv(x)
                return x.transpose(0, 1)

        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            TransposeModule(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @unittest.skip("torch._dynamo.exc.TorchRuntimeError")
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)]
        )
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)]
        )

    @unittest.skip(
        "AssertionError: The values for attribute 'shape' do not match:"
        " torch.Size([5, 6, 2]) != torch.Size([4, 4, 2]). Even symbolic "
        "fx.graph can't get dynamic arguments from this Module."
    )
    def test_slice(self):
        class DynamicSliceExportMod(torch.nn.Module):
            def forward(self, x):
                results = []
                for i in range(4):
                    results.append(x[: x.size(0) - i, i : x.size(2), i:3])
                return tuple(results)

        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicSliceExportMod(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @unittest.skip(
        "fx.graph: doesn't handle scalar like normal tensor, so this is not yet"
        "supported! TypeError: forward() takes 1 positional argument but 2 were given"
    )
    def test_arange(self):
        class ArangeModel(torch.nn.Module):
            def forward(self, input):
                return (
                    torch.arange(input.shape[0]),
                    torch.arange(12),
                    torch.arange(start=input.shape[0], end=input.shape[0] + 5),
                )

        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ArangeModel(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @unittest.skip(
        "fx.graph: torch._subclasses.fake_tensor.DataDependentOutputException: "
        "aten._local_scalar_dense.default"
    )
    def test_expand_as_fill_zero(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = 0
                return x

        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[(x2,)],
        )

    @unittest.skip(
        "ATenLib: INVALID_ARGUMENT : Failed to load model with error: "
        "ONNX Schema aten_copy: failed validating the check: !(it.GetName().empty())"
    )
    def test_expand_as_fill_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[(x2,)],
        )

    def test_expand_as_fill_seperate_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[(x2,)],
        )

    def test_view_dynamic_zero_dim(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        another_x = torch.empty((0,))
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ViewModel(),
            (x,),
            additional_test_inputs=[(another_x,)],
        )

    def test_flatten_dynamic_axes(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model, (x,), additional_test_inputs=[(y,)]
        )


if __name__ == "__main__":
    common_utils.run_tests()
