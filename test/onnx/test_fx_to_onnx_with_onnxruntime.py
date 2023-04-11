# Owner(s): ["module: onnx"]
from __future__ import annotations

import copy

import io
import itertools
import os
import tempfile
import unittest
import warnings

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import onnx_test_common
import onnxruntime  # type: ignore[import]
import parameterized
import pytorch_test_common
import torch
import torch.onnx
import transformers  # type: ignore[import]
from torch import nn

from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype, diagnostics, exporter, fx as fx_onnx
from torch.onnx._internal.fx.dynamo_exporter import (
    DynamoExporter,
    DynamoOptimizeExporter,
)
from torch.onnx._internal.fx.fx_symbolic_exporter import FXSymbolicTraceExporter
from torch.testing._internal import common_utils
from torch.types import Number

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, Callable]
_InputArgsType = Optional[Union[torch.Tensor, Sequence[Any], Mapping[str, Any]]]
_OutputsType = Sequence[_NumericType]
_MainExporterClass = DynamoExporter


@_beartype.beartype
def _exporter_class_name(exporter_cls: Type[exporter.Exporter]):
    return (
        "dynamo_export" if exporter_cls == _MainExporterClass else exporter_cls.__name__
    )


@_beartype.beartype
def _run_ort(
    onnx_model: Union[str, torch.onnx.ExportOutput],
    pytorch_inputs: Sequence[_InputArgsType],
) -> _OutputsType:
    if isinstance(onnx_model, torch.onnx.ExportOutput):
        buffer = io.BytesIO()
        onnx_model.save(buffer)
        ort_model = buffer.getvalue()
    else:
        ort_model = onnx_model
    session = onnxruntime.InferenceSession(
        ort_model, providers=["CPUExecutionProvider"]
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]
    if len(input_names) != len(pytorch_inputs):
        raise AssertionError(
            f"Expected {len(input_names)} inputs, got {len(pytorch_inputs)}"
        )
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


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
def _try_clone_inputs(input_args, input_kwargs):
    ref_input_args = copy.deepcopy(input_args)
    ref_input_kwargs = copy.deepcopy(input_kwargs)
    return ref_input_args, ref_input_kwargs


@_beartype.beartype
def _compare_pytorch_onnx_with_ort(
    export_output: torch.onnx.ExportOutput,
    model: _ModelType,
    input_args: Sequence[_InputArgsType],
    input_kwargs: Mapping[str, _InputArgsType],
    atol: float,
    rtol: float,
    has_mutation: bool = False,
):
    if has_mutation:
        ref_model = _try_clone_model(model)
        ref_input_args, ref_input_kwargs = _try_clone_inputs(input_args, input_kwargs)
    else:
        ref_model = model
        ref_input_args = input_args
        ref_input_kwargs = input_kwargs

    # Format original model inputs into the format expected by exported ONNX model.
    onnx_format_args = export_output.format_torch_inputs_to_onnx(
        *input_args, **input_kwargs
    )

    ref_outputs = export_output.format_torch_outputs_to_onnx(
        ref_model(*ref_input_args, **ref_input_kwargs)
    )
    ort_outputs = _run_ort(export_output, onnx_format_args)
    if len(ref_outputs) != len(ort_outputs):
        raise AssertionError(
            f"Expected {len(ref_outputs)} outputs, got {len(ort_outputs)}"
        )
    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        torch.testing.assert_close(
            ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
        )


@_beartype.beartype
def _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
    test_suite: TestFxToOnnxWithOnnxRuntime,
    model: _ModelType,
    input_args: Sequence[_InputArgsType],
    rtol: float = 1e-3,
    atol: float = 1e-7,
    opset_version: int = 18,
    has_mutation: bool = False,
    additional_test_inputs: Optional[Sequence[Sequence[_InputArgsType]]] = None,
    **input_kwargs,
):
    """Compare the results of PyTorch model with exported ONNX model

    Args:
        model (_ModelType): PyTorch model
        input_args (_InputArgsType): torch input arguments
        rtol (float, optional): relative tolerance. Defaults to 1e-3.
        atol (float, optional): absolute tolerance. Defaults to 1e-7.
        opset_version (int, optional): ONNX opset version. Defaults to 18.
        has_mutation (bool, optional): Whether the model mutates its input or state.
            `mutation` as `True` incurs extra overhead of cloning the inputs and model.
            Defaults to False.
        additional_test_inputs (Optional[Sequence[_InputArgsType]], optional):
            Test the models with another dataset, which is designed for dynamic axes
            testing. Defaults to None.

    """

    if has_mutation:
        ref_model = _try_clone_model(model)
        ref_input_args, ref_input_kwargs = _try_clone_inputs(input_args, input_kwargs)
    else:
        ref_model = model
        ref_input_args = input_args
        ref_input_kwargs = input_kwargs

    # Feed args and kwargs into exporter.
    # Note that exporter should flatten kwargs into positional args the exported model;
    # since ONNX doesn't represent kwargs.
    if test_suite.exporter_cls == _MainExporterClass:
        export_output = torch.onnx.dynamo_export(
            ref_model,
            *ref_input_args,
            **ref_input_kwargs,
            export_options=torch.onnx.ExportOptions(
                opset_version=opset_version,
                op_level_debug=test_suite.op_level_debug,
                dynamic_shapes=test_suite.dynamic_shapes,
            ),
        )
    else:
        export_output = test_suite.exporter_cls(
            torch.onnx.ExportOptions(
                opset_version=opset_version,
                op_level_debug=test_suite.op_level_debug,
                dynamic_shapes=test_suite.dynamic_shapes,
            ),
            model=ref_model,
            model_args=ref_input_args,
            model_kwargs=ref_input_kwargs,
        ).export()

    _compare_pytorch_onnx_with_ort(
        export_output,
        model,
        input_args,
        input_kwargs,
        atol,
        rtol,
        has_mutation=has_mutation,
    )
    # This confirms the exported mode accepts different input shapes
    # when dynamic shape is enabled.
    # TODO(titaiwangms): additional input kwargs.
    if additional_test_inputs and test_suite.dynamic_shapes:
        for additional_input_args in additional_test_inputs:
            _compare_pytorch_onnx_with_ort(
                export_output,
                model,
                additional_input_args,
                {},
                atol,
                rtol,
                has_mutation=has_mutation,
            )


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (True, False),
            (True, False),
            (
                DynamoExporter,
                DynamoOptimizeExporter,
            ),
        )
    )
    return {
        "attrs": ["op_level_debug", "dynamic_shapes", "exporter_cls"],
        "input_values": input_values,
    }


def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffixes = []
    for k, v in input_dicts.items():
        if k == "exporter_cls":
            suffixes.append(_exporter_class_name(v))
        else:
            suffixes.append(f"{k}_{v}")
    return f"{cls.__name__}_{'_'.join(suffixes)}"


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),
    class_name_func=_parameterize_class_name,
)
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    op_level_debug: bool
    dynamic_shapes: bool
    exporter_cls: Type[exporter.Exporter]

    def setUp(self):
        super().setUp()
        self.diag_ctx = diagnostics.engine.create_diagnostic_context(
            "test_fx_export", version=torch.__version__
        )
        self.opset_version = 18
        self.ort_version = onnxruntime.__version__

    def tearDown(self):
        # TODO(bowbao): Don't dump logs by default. Set as configurable with
        # `DiagnosticOptions`.
        diagnostics.engine.dump(
            f"test_report_{self._testMethodName}"
            f"_op_level_debug_{self.op_level_debug}"
            f"_dynamic_axes_{self.dynamic_shapes}"
            f"_{_exporter_class_name(self.exporter_cls)}"
            ".sarif",
            compress=False,
        )
        super().tearDown()

    def test_simple_function(self):
        def func(x):
            # TODO(justinchuby): Replicate torch's type casting policy
            # in the exporter for type promotion support
            y = x + 1.0
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, func, (tensor_x,))

    @pytorch_test_common.skip_fx_exporters(
        {
            DynamoExporter: (
                "AssertionError: Dynamo input/output is not consistent with traced input/output. "
                "Ref: https://github.com/pytorch/pytorch/issues/96379"
            )
        }
    )
    def test_func_with_args_and_tensor_kwargs(self):
        # Non-tensor optional kwargs are always folded into constant and
        # removed from input list in Dynamo-traced graph, if its value is not provided
        # to tracer. So for a function like
        #   def func(x, b=1.0)
        # here. E.g., if you first Dynamo-trace the model with arguments (x,),
        # and then call the traced graph with arguments (x, b=2.0), it will complain
        # somewhere that model is called with extra args because the modified
        # function is traced into
        #   def forward(self, x : torch.Tensor):
        #     add = x + 1.0;  x = None
        #     relu = add.relu()
        #     return (add, relu)
        # To summarize, in order to be traced as graph input, the value of optional kwarg
        # must be provided. Otherwise, they are treated as in-graph constants in Dynamo.
        # Tensor optional kwargs are an exception. It is always traced as input.
        # It is unclear if this behavior is intended or not. But in general it is bad
        # practice to set mutable default values.
        # `DynamoOptimizeExporter` applies a workaround by binding args and kwargs to
        # model signature and fill in the default values of unprovided optional arguments.
        def func(x, b=torch.tensor(1.0)):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        # Test without providing optional kwarg.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, func, (tensor_x,))
        # Test with only positional args.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, func, (tensor_x, torch.tensor(8.0))
        )
        # Test while specifying optional kwarg.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, func, (tensor_x,), b=torch.tensor(5.0)
        )

    # beartype.roar.BeartypeCallHintParamViolation:
    # @beartyped onnxscript.function_libs.torch_aten.graph_building.TorchScriptGraph.add_input()
    # parameter input_value=8.0 violates type hint typing.Union[torch.Tensor, NoneType],
    # as float 8.0 not <class "builtins.NoneType"> or <protocol "torch.Tensor">.
    @unittest.expectedFailure
    def test_func_with_args_and_kwargs(self):
        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, func, (tensor_x,))
        # Test with only positional args.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, func, (tensor_x, 8.0))
        # Test while specifying optional kwarg.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, func, (tensor_x,), b=5.0
        )

    @pytorch_test_common.skip_fx_exporters(
        {DynamoOptimizeExporter: "Unhandled used argument."}
    )
    def test_func_with_nested_input_structure(self):
        def func(
            x_dict: Dict[str, torch.Tensor],
            y_tuple: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            z_list: List[List[torch.Tensor]],
        ):
            if "a" in x_dict:
                x = x_dict["a"]
            elif "b" in x_dict:
                x = x_dict["b"]
            else:
                x = torch.randn(3)

            y1, (y2, y3) = y_tuple

            z = x + y1 + y2 + y3
            for z_sub_list in z_list:
                z = z + torch.stack(z_sub_list).sum()

            return z

        x_dict = {"a": torch.randn(3), "c": torch.randn(3)}
        y_tuple = (torch.randn(3), (torch.randn(3), torch.randn(3)))
        z_list = [
            [torch.randn(3), torch.randn(3)],
            [torch.randn(3), torch.randn(3), torch.randn(3)],
        ]
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, func, (x_dict, y_tuple, z_list)
        )

    @pytorch_test_common.skip_fx_exporters(
        {DynamoOptimizeExporter: "Unsupported output structure."}
    )
    def test_func_with_nested_output_structure(self):
        def func(x, y, z):
            x = x + y
            y = y + z
            z = x + y
            out1 = (x, (y, z))
            out2 = [[x, y], [y, z]]
            out3 = {"z": z, "x": x}
            return out1, out2, out3

        x = torch.randn(3)
        y = torch.randn(3)
        z = torch.randn(3)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, func, (x, y, z))

    @pytorch_test_common.skip_min_ort_version(reason="SegFault", version="1.15")
    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 2, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc2(tensor_x)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, MNISTModel(), (tensor_x,)
        )

    # test single op with no kwargs
    def test_sigmoid(self):
        x = torch.randn(1, 4, 2, 3)

        class SigmoidModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, SigmoidModel(), (x,))

    @pytorch_test_common.skip_if_no_torchvision
    @pytorch_test_common.skip_dynamic_fx_test(
        "_aten_convolution_onnx: _add_attribute_to_torchscript_node()"
        " parameter value=[None, None] violates type hint"
        "typing.Union[float, int, str, bytes, typing.Sequence[float],"
        " typing.Sequence[int], torch.Tensor], as [None, None]:"
    )
    def test_shufflenet_v2_dynamic_axes(self):
        import torchvision

        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self,
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
            self, DynamicAdd(), (x, y), additional_test_inputs=[(another_x, another_y)]
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
            self, DynamicAdd(), (x, y), additional_test_inputs=[(input_x, input_y)]
        )

    @pytorch_test_common.skip_fx_exporters(
        {
            DynamoExporter: (
                "[ONNXRuntimeError] : 2 : INVALID_ARGUMENT : Non-zero status code returned "
                "while running Expand node. Name:'_0x55b501ebaf10_n2' "
                "Status Message: invalid expand shape"
            ),
        }
    )
    @pytorch_test_common.skip_min_ort_version(reason="SegFault", version="1.15")
    def test_matmul(self):
        class DynamicMatMul(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.matmul(x, y)

        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, DynamicMatMul(), (x, y), additional_test_inputs=[(input_x, input_y)]
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            test(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            TransposeModule(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @pytorch_test_common.skip_dynamic_fx_test("torch._dynamo.exc.TorchRuntimeError")
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)]
        )
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)]
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            DynamicSliceExportMod(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @pytorch_test_common.skip_min_ort_version(reason="SegFault", version="1.15")
    def test_mutation(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x):
                x.view(3, 2, -1).add_(2.0)
                return x

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, MutationModel(), (torch.randn(12),), has_mutation=True
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            ArangeModel(),
            (x,),
            additional_test_inputs=[(y,)],
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            Model(),
            (x,),
            additional_test_inputs=[(x2,)],
        )

    @pytorch_test_common.skip_dynamic_fx_test(
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
            self,
            Model(),
            (x,),
            additional_test_inputs=[(x2,)],
        )

    @pytorch_test_common.skip_fx_exporters(
        {DynamoExporter: "Unknown call_function target: aten.lift_fresh_copy.default"}
    )
    def test_expand_as_fill_seperate_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self,
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
            self,
            ViewModel(),
            (x,),
            additional_test_inputs=[(another_x,)],
        )

    @pytorch_test_common.skip_fx_exporters(
        {DynamoExporter: "Shapes are assumed static by default by 'dynamo.export'."}
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
            self, model, (x,), additional_test_inputs=[(y,)]
        )

    def test_none_input(self):
        class NoneInputModel(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: Optional[torch.Tensor], z: torch.Tensor
            ):
                if y is None:
                    return x + z
                return x + y + z

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            self, NoneInputModel(), (torch.randn(1, 2), None, torch.randn(1, 2))
        )

    @pytorch_test_common.skip_min_ort_version(reason="SegFault", version="1.15")
    def test_gpt2_tiny(self):
        model_name = "sshleifer/tiny-gpt2"
        # Download pytorch model
        model = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Transform input tokens
        inputs = tokenizer("Hello world!", return_tensors="pt")

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(self, model, [], **inputs)

    @_beartype.beartype
    def _test_large_scale_exporter(
        self,
        model_name: str,
        create_model: Callable,
        create_args: Callable,
        create_pytorch_only_kwargs: Callable,
    ):
        """Test helper for large-scale exporter.

        Arguments:
            model_name: Name of the model. It used to name temporary files.
            create_model: A function that creates a model. It should always create the same model.
            create_args: A function that creates random input arguments for the model.
            create_pytorch_only_kwargs: A function that creates kwargs for calling PyTorch model with real tensors.

        This test contains several steps.

        1. Create a toy model.
        2. Save the toy's state (parameters) to a file. This is for simulating a checkpoint file.
        3. Load it back and export it to ONNX with large-scale exporter.
            All operations (including model loading) are done under
            FakeTensorMode so no real tensor is created and no real
            computation happens.
        4. The ONNX model generated in step 3 doesn't contain parameters,
            and this step adds them as external data and save a new ONNX model.
        5. Run PyTorch and ONNX models and compare their results.
        """

        # Create the toy model.
        model = create_model()

        with tempfile.NamedTemporaryFile(
            prefix=model_name, suffix=".pt"
        ) as tmp_file, tempfile.TemporaryDirectory(
            suffix="large_scale_export"
        ) as tmp_folder:
            # Dump state_dict to a file to simulate how HuggingFace model is initialized.
            # The file will be loaded via .load_state_dict(...)
            torch.save(model.state_dict(), tmp_file.name)

            ftm = fake_tensor.FakeTensorMode(
                allow_non_fake_inputs=True, allow_fallback_kernels=False
            )
            ctx = fx_onnx.FxToOnnxContext()
            # NOTE: FakeTensorMode disallows symbolic shape of fx graph
            # The following coed block does several things.
            #  1. Create a model whose parameters and buffers are all FakeTensor's.
            #  2. Convert nn.Module into ONNX model without initializers.
            #  3. Record the file paths to find real initializers.
            with ctx, ftm:
                # Toy model with parameters and buffers as FakeTensor's.
                fake_model = create_model()
                fake_model.load_state_dict(torch.load(tmp_file.name))
                # Toy inputs as FakeTensor's.
                fake_args = create_args()
                # Export ONNX model without initializers while ctx.paths records
                # all files that contains real initializers.

                export_output = FXSymbolicTraceExporter(
                    options=torch.onnx.ExportOptions(
                        opset_version=self.opset_version,
                        dynamic_shapes=self.dynamic_shapes,
                        op_level_debug=self.op_level_debug,
                    ),
                    model=fake_model,
                    model_args=fake_args,
                    model_kwargs={},
                ).export()

                onnx_model = export_output.model_proto

            # Tasks done by the following block.
            #  1. Iterate through all tensors stored in ctx.paths (the file content is loaded torch.load)
            #  2. If a tensor's name matches a "onnx_model"'s input name, an initializer is created and saved to
            #     a seperated folder.
            #  3. A new ONNX model is saved into file with the initializers saved in the previous step.
            #  4. ORT executes the new ONNX model and compares the results with the original GPT model.

            # Model saved to tmp_folder/onnx_model_location
            # Initializers are saved to tmp_folder/onnx_initializer_location/*.onnx
            onnx_model_location = model_name + "_external_data.onnx"
            onnx_initializer_location = model_name + "_initializers"
            fx_onnx.save_model_with_external_data(
                tmp_folder,
                onnx_model_location,
                onnx_initializer_location,
                tuple(ctx.paths),
                onnx_model,
            )

            # Generate random inputs.
            args = create_args()
            kwargs = create_pytorch_only_kwargs()
            # Original outputs.
            ref_outputs = export_output.format_torch_outputs_to_onnx(
                model(*args, **kwargs)
            )
            # ORT outputs.
            args_not_none = export_output.format_torch_inputs_to_onnx(*args)
            ort_outputs = _run_ort(
                os.path.join(tmp_folder, onnx_model_location),
                args_not_none,
            )

            assert len(ref_outputs) == len(ort_outputs)

            for ref_output, ort_output in zip(ref_outputs, ort_outputs):
                torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    @pytorch_test_common.skip_dynamic_fx_test(
        "FakeTensor exporting is not supported by dynamic axes."
    )
    def test_large_scale_exporter_with_toy_mlp(self):
        class MLPModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc0 = nn.Linear(8, 8, bias=True)
                self.fc1 = nn.Linear(8, 4, bias=True)
                self.fc2 = nn.Linear(4, 2, bias=True)
                self.fc3 = nn.Linear(2, 2, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.fc0(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                output = self.fc3(tensor_x)
                return output

        def create_model() -> nn.Module:
            return MLPModel()

        def create_args():
            return (torch.rand((97, 8), dtype=torch.float32),)

        def create_pytorch_only_extra_kwargs():
            return {}

        self._test_large_scale_exporter(
            "toy_mlp1",
            create_model,
            create_args,
            create_pytorch_only_extra_kwargs,
        )

    @pytorch_test_common.skip_dynamic_fx_test(
        "FakeTensor exporting is not supported by dynamic axes."
    )
    def test_large_scale_exporter_with_tiny_gpt2(self):
        model_name = "sshleifer/tiny-gpt2"

        def create_model() -> nn.Module:
            return transformers.AutoModel.from_pretrained(model_name)

        def create_args():
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            kwargs = tokenizer("Hello world!", return_tensors="pt")
            input_ids = kwargs["input_ids"]
            attention_mask = kwargs["attention_mask"]
            return input_ids, None, attention_mask

        def create_pytorch_only_extra_kwargs():
            return {"return_dict": False}

        self._test_large_scale_exporter(
            "tiny_gpt2",
            create_model,
            create_args,
            create_pytorch_only_extra_kwargs,
        )


if __name__ == "__main__":
    common_utils.run_tests()
