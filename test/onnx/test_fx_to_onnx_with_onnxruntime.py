# Owner(s): ["module: onnx"]
from __future__ import annotations

import itertools
import os
import tempfile
import unittest

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type

import onnx_test_common
import onnxruntime  # type: ignore[import]
import parameterized
import pytorch_test_common
import torch
import torch.onnx
import transformers  # type: ignore[import]
from torch import nn

from torch._subclasses import fake_tensor
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
    fx_symbolic_graph_extractor,
    patcher,
    serialization as fx_serialization,
)
from torch.testing._internal import common_utils

try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skip_if_no_torchvision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")

ONNX_OPSET_VERSION_TO_TEST = 18


def _parameterized_class_attrs_and_values():
    input_values = []
    input_values.extend(
        itertools.product(
            (True, False),
            (True, False),
        )
    )
    return {
        "attrs": ["op_level_debug", "dynamic_shapes"],
        "input_values": input_values,
    }


def _parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffixes = []
    for k, v in input_dicts.items():
        suffixes.append(f"{k}_{v}")
    return f"{cls.__name__}_{'_'.join(suffixes)}"


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values(),
    class_name_func=_parameterize_class_name,
)
class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    op_level_debug: bool
    dynamic_shapes: bool

    def setUp(self):
        super().setUp()
        self.opset_version = ONNX_OPSET_VERSION_TO_TEST
        self.ort_version = onnxruntime.__version__

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_simple_function(self):
        def func(x):
            # TODO(justinchuby): Replicate torch's type casting policy
            # in the exporter for type promotion support
            y = x + 1.0
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))

    @pytorch_test_common.xfail(
        "AssertionError: Dynamo input/output is not consistent with traced input/output. "
        "Ref: https://github.com/pytorch/pytorch/issues/96379"
    )
    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
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

        tensor_x = torch.randn(1, 2, 3, dtype=torch.float32)

        # Test without providing optional kwarg.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))
        # Test with only positional args.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x, torch.tensor(8.0))
        )
        # Test while specifying optional kwarg.
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x,), {"b": torch.tensor(5.0)}
        )

    @pytorch_test_common.xfail(
        "https://github.com/pytorch/pytorch/issues/99534"
        "Non-tensor input is not traceable in dynamo."
    )
    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_xfail_func_with_non_tensor_args(self):
        def func(x, b=1.0):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        export_output = torch.onnx.dynamo_export(
            func,
            tensor_x,
            8.0,
            export_options=torch.onnx.ExportOptions(
                opset_version=self.opset_version,
                op_level_debug=self.op_level_debug,
                dynamic_shapes=self.dynamic_shapes,
            ),
        )
        onnx_format_args = export_output.adapt_torch_inputs_to_onnx(tensor_x, 8.0)
        ref_outputs = export_output.adapt_torch_outputs_to_onnx(func(tensor_x, 8.0))
        ort_outputs = onnx_test_common.run_ort(export_output, onnx_format_args)
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

        # test on different non-tensor input - xfail
        onnx_format_args = export_output.adapt_torch_inputs_to_onnx(tensor_x, 9.0)
        ref_outputs = export_output.adapt_torch_outputs_to_onnx(func(tensor_x, 9.0))
        _ = onnx_test_common.run_ort(export_output, onnx_format_args)
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (x_dict, y_tuple, z_list)
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (x, y, z))

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_mnist(self):
        class MNISTModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=True)
                self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=True)
                self.fc1 = nn.Linear(9216, 128, bias=True)
                self.fc2 = nn.Linear(128, 10, bias=True)

            def forward(self, tensor_x: torch.Tensor):
                tensor_x = self.conv1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.conv2(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = torch.max_pool2d(tensor_x, 2)
                tensor_x = torch.flatten(tensor_x, 1)
                tensor_x = self.fc1(tensor_x)
                tensor_x = torch.sigmoid(tensor_x)
                tensor_x = self.fc2(tensor_x)
                output = torch.log_softmax(tensor_x, dim=1)
                return output

        tensor_x = torch.rand((64, 1, 28, 28), dtype=torch.float32)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MNISTModel(), (tensor_x,)
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_log_sigmoid(self):
        # This produces op as `torch.ops.aten.log_sigmoid_forward`, instead of the more
        # conventional `torch.ops.aten.log_sigmoid`.
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = torch.nn.LogSigmoid()

            def forward(self, x):
                return self.m(x)

        input = torch.randn(2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(Model(), (input,))

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    @skip_if_no_torchvision
    def test_resnet18(self):
        # TODO(bowbao): Note [training vs eval in dynamo_export]
        # So we are effectively exporting all models in traning mode by
        # default. But for the sake of this export we are only interested in eval mode.
        # The question is, should we call `model.eval()` in `dynamo_export`?
        # This particular test fails 'functionalization' in training mode.
        # So we are explicitly calling `model.eval()` for any model that contains
        # batch norm.
        # Ref: https://github.com/pytorch/pytorch/issues/99662#issuecomment-1528178221
        model = torchvision.models.resnet18(pretrained=False).eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
        )

    @pytorch_test_common.xfail(
        "RuntimeError: Unknown call_function target: aten.mean.dim"
    )
    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    @skip_if_no_torchvision
    def test_shufflenet_v2(self):
        # TODO(bowbao): see Note [training vs eval in dynamo_export]
        model = torchvision.models.shufflenet_v2_x0_5(pretrained=False).eval()
        dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
        test_inputs = torch.randn(3, 3, 224, 224, requires_grad=True)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model,
            (dummy_input,),
            additional_test_inputs=[((test_inputs,),)],
            rtol=1e-3,
            atol=1e-5,
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_add(self):
        class DynamicAdd(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.add(x, y)

        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        another_x = torch.randn(3, 4)
        another_y = torch.randn(3, 4)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(),
            (x, y),
            additional_test_inputs=[((another_x, another_y),)],
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
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

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicAdd(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_matmul(self):
        class DynamicMatMul(torch.nn.Module):
            def forward(self, x, y):
                return torch.ops.aten.matmul(x, y)

        x = torch.randn(2, 3, 6)
        y = torch.randn(2, 6, 4)
        input_x = torch.randn(2, 3, 4)
        input_y = torch.randn(2, 4, 4)

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicMatMul(), (x, y), additional_test_inputs=[((input_x, input_y),)]
        )

    @pytorch_test_common.skip_dynamic_fx_test(
        "fx graph does not capture symbolic value for aten::scalar_tensor."
    )
    def test_scalar_tensor(self):
        class test(torch.nn.Module):
            def forward(self, x):
                return torch.scalar_tensor(x.size(0)), torch.scalar_tensor(
                    x.size(1), dtype=torch.int64
                )

        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            test(),
            (x,),
            additional_test_inputs=[((y,),)],
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            TransposeModule(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    @pytorch_test_common.xfail("torch._dynamo.exc.TorchRuntimeError")
    def test_squeeze_runtime_dim(self):
        class Squeeze(torch.nn.Module):
            def forward(self, d1, d2):
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)

        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d1, d4), additional_test_inputs=[((d3, d4),)]
        )
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Squeeze(), (d3, d4), additional_test_inputs=[((d1, d3),)]
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            DynamicSliceExportMod(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    # TODO(titaiwang): This is also detected flaky in static shape:
    # https://github.com/pytorch/pytorch/issues/98622
    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=False,
    )
    def test_mutation(self):
        class MutationModel(torch.nn.Module):
            def forward(self, x):
                x.view(3, 2, -1).add_(2.0)
                return x

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            MutationModel(), (torch.randn(12),), has_mutation=True
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=False,
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ArangeModel(),
            (x,),
            additional_test_inputs=[((y,),)],
        )

    @pytorch_test_common.xfail(
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
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.xfail(
        "RuntimeError: Unknown call_function target: aten.lift_fresh_copy.default"
    )
    def test_expand_as_fill_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                x[:, x.size(0) :] = torch.tensor([1, 2, 3])
                return x

        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.xfail(
        "Unknown call_function target: aten.lift_fresh_copy.default"
    )
    def test_expand_as_fill_seperate_tensor(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)

        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            Model(),
            (x,),
            additional_test_inputs=[((x2,),)],
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_view_dynamic_zero_dim(self):
        class ViewModel(torch.nn.Module):
            def forward(self, input):
                input = input.view(-1, 2)
                return input.view(1, -1)

        x = torch.ones(2)
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            ViewModel(),
            (x,),
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_flatten_dynamic_axes(self):
        class MyModule(torch.nn.Module):
            def forward(self, x):
                return torch.flatten(x, start_dim=2, end_dim=3)

        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model, (x,), additional_test_inputs=[((y,),)]
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_none_input(self):
        class NoneInputModel(torch.nn.Module):
            def forward(
                self, x: torch.Tensor, y: Optional[torch.Tensor], z: torch.Tensor
            ):
                if y is None:
                    return x + z
                return x + y + z

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            NoneInputModel(), (torch.randn(1, 2), None, torch.randn(1, 2))
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_gpt2_tiny(self):
        model_name = "sshleifer/tiny-gpt2"
        # Download pytorch model
        model = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Transform input tokens
        inputs = tokenizer("Hello world!", return_tensors="pt")
        another_inputs = tokenizer("Another Hello world!", return_tensors="pt")

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            model, [], inputs, additional_test_inputs=[((), another_inputs)]
        )

    @pytorch_test_common.skip_min_ort_version(
        reason="ORT doesn't support dynamic fx exporter yet making SegFault flaky test",
        version="1.15",
        dynamic_only=True,
    )
    def test_prims_device_put(self):
        class CustomModule(nn.Module):
            def forward(self, x):
                # Assuming x is a tensor on the CPU, move it to the desired device using device_put()
                x = torch.ops.prims.device_put(x, "cpu")
                return x

        self.run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            CustomModule(), (torch.randn(1, 2, 3),)
        )

    @_beartype.beartype
    def _test_fx_symbolic_tracer_large_scale_exporter(
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
            ctx = patcher.ONNXTorchPatcher()
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

                options = torch.onnx.ExportOptions(
                    opset_version=self.opset_version,
                    dynamic_shapes=self.dynamic_shapes,
                    op_level_debug=self.op_level_debug,
                )
                export_options = torch.onnx._internal.exporter.ResolvedExportOptions(
                    options
                )
                export_options.fx_tracer = (
                    fx_symbolic_graph_extractor.FXSymbolicTracer()
                )
                export_output = torch.onnx.dynamo_export(
                    fake_model,
                    *fake_args,
                    export_options=export_options,
                )

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
            # TODO: We are using the internal `save_model_with_external_data` instead of public
            # `ExportOutput.save` because we need to rename ONNX initializers before saving.
            # This is only needed/allowed because we are using `fx_tracer=FXSymbolicTracer`,
            # which is not an official FX tracer.
            fx_serialization.save_model_with_external_data(
                tmp_folder,
                onnx_model_location,
                onnx_initializer_location,
                tuple(ctx.paths),
                onnx_model,
                rename_initializer=True,
            )
            # Generate random inputs.
            args = create_args()
            kwargs = create_pytorch_only_kwargs()
            # Original outputs.
            ref_outputs = export_output.adapt_torch_outputs_to_onnx(
                model(*args, **kwargs)
            )
            # ORT outputs.
            args_not_none = export_output.adapt_torch_inputs_to_onnx(*args)

            # Drop Parameters and buffers added by fx_serialization.save_model_with_external_data
            args_not_none = args_not_none[: len(args) - len(kwargs)]

            ort_outputs = onnx_test_common.run_ort(
                os.path.join(tmp_folder, onnx_model_location),
                args_not_none,
            )

            assert len(ref_outputs) == len(ort_outputs)

            for ref_output, ort_output in zip(ref_outputs, ort_outputs):
                torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    @pytorch_test_common.skip_dynamic_fx_test(
        "FakeTensor exporting is not supported by dynamic axes."
    )
    def test_fx_symbolic_tracer_large_scale_exporter_with_toy_mlp(self):
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

        self._test_fx_symbolic_tracer_large_scale_exporter(
            "toy_mlp1",
            create_model,
            create_args,
            create_pytorch_only_extra_kwargs,
        )

    @pytorch_test_common.skip_dynamic_fx_test(
        "FakeTensor exporting is not supported by dynamic axes."
    )
    def test_fx_symbolic_tracer_large_scale_exporter_with_tiny_gpt2(self):
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

        self._test_fx_symbolic_tracer_large_scale_exporter(
            "tiny_gpt2",
            create_model,
            create_args,
            create_pytorch_only_extra_kwargs,
        )


def _parameterized_class_attrs_and_values_with_fake_options():
    input_values = []
    input_values.extend(
        itertools.product((True, False), (True, False), (True, False), (True, False))
    )
    return {
        "attrs": [
            "op_level_debug",
            "dynamic_shapes",
            "load_checkpoint_during_init",
            "export_within_fake_mode",
        ],
        "input_values": input_values,
    }


@parameterized.parameterized_class(
    **_parameterized_class_attrs_and_values_with_fake_options(),
    class_name_func=_parameterize_class_name,
)
class TestFxToOnnxFakeTensorWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    """ONNX export test for specific Fake Tensor scenarios

    TODO: Should we merge this with  `TestFxToOnnxWithOnnxRuntime`? Considerably increases export time
    """

    op_level_debug: bool
    dynamic_shapes: bool
    load_checkpoint_during_init: bool
    export_within_fake_mode: bool

    def setUp(self):
        super().setUp()
        self.opset_version = ONNX_OPSET_VERSION_TO_TEST
        self.ort_version = onnxruntime.__version__

    @_beartype.beartype
    def _test_fake_tensor_mode_exporter(
        self,
        model_name: str,
        create_model: Callable,
        create_args: Callable,
        create_kwargs: Callable,
        load_checkpoint_during_init: bool,
        export_within_fake_mode: bool,
    ):
        """Test helper for FakeTensorMode-enabled exporter.

        Arguments:
            model_name: Name of the model. It used to name temporary files.
            create_model: A function that creates a model.
            create_args: A function that creates positional inputs for the model.
            create_kwargs: A function that creates keyword inputs for ther model.
            load_checkpoint_during_init: Whether to load a checkpoint during model initialization.
                (after or during model creation, but before exporting starts)
            export_within_fake_mode: Whether to call torch.onnx._dynamo_export within torch._subclasses.FakeTensorMode

        This test contains several steps.

        1. Create a toy model.
        2. Save the toy's state (parameters) to a file. This is for simulating a checkpoint file.
        3. Load it back and export it to ONNX with Fake Mode enabled.
            Because all operations (including model and input loading) are done under
            FakeTensorMode, no real tensor are created and no real computation happens.
        4. The ONNX model generated in step 3 doesn't contain parameters,
            and this step adds them as external data on an ONNX model.
        5. Run PyTorch and ONNX models and compare their results.
        """

        # Create the toy model with real weight.
        real_model = create_model()

        with tempfile.NamedTemporaryFile(
            prefix=model_name, suffix=".pt"
        ) as tmp_checkpoint_file:
            # Dump state_dict to a file to simulate how HuggingFace model is initialized.
            # The file will be loaded via .load_state_dict(...)
            state_dict = real_model.state_dict()
            torch.save(state_dict, tmp_checkpoint_file.name)

            with torch.onnx.enable_fake_mode() as fake_context:
                fake_args = create_args()
                fake_kwargs = create_kwargs()
                fake_model = create_model()
                if load_checkpoint_during_init:
                    fake_model.load_state_dict(torch.load(tmp_checkpoint_file.name))

                # Export the model with fake inputs and parameters
                export_options = torch.onnx.ExportOptions(
                    opset_version=self.opset_version,
                    dynamic_shapes=self.dynamic_shapes,
                    op_level_debug=self.op_level_debug,
                    fake_context=fake_context,
                )

                if export_within_fake_mode:
                    export_output = torch.onnx.dynamo_export(
                        fake_model,
                        *fake_args,
                        **fake_kwargs,
                        export_options=export_options,
                    )

            if not export_within_fake_mode:
                export_output = torch.onnx.dynamo_export(
                    fake_model, *fake_args, **fake_kwargs, export_options=export_options
                )

            with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp_onnx_file:
                export_output.save(
                    tmp_onnx_file.name, model_state_dict=tmp_checkpoint_file.name
                )

                # Generate random inputs.
                args = create_args()
                kwargs = create_kwargs()
                # Original outputs.
                ref_outputs = export_output.adapt_torch_outputs_to_onnx(
                    real_model(*args, **kwargs)
                )
                # ORT outputs.
                args_not_none = export_output.adapt_torch_inputs_to_onnx(
                    *args, **kwargs
                )

                ort_outputs = onnx_test_common.run_ort(
                    tmp_onnx_file.name,
                    args_not_none,
                )

                assert len(ref_outputs) == len(ort_outputs)

                for ref_output, ort_output in zip(ref_outputs, ort_outputs):
                    torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    def test_fake_tensor_mode_simple(self):
        def create_model() -> nn.Module:
            class Model(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2)

                def forward(self, x):
                    out = self.linear(x)
                    return out

            return Model()

        def create_args():
            return (torch.rand(5, 2, 2),)

        def create_kwargs():
            return {}

        self._test_fake_tensor_mode_exporter(
            "simple",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
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

        def create_kwargs():
            return {"return_dict": False}

        self._test_fake_tensor_mode_exporter(
            "tiny_gpt2",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
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

        def create_kwargs():
            return {}

        self._test_fake_tensor_mode_exporter(
            "toy_mlp1",
            create_model,
            create_args,
            create_kwargs,
            load_checkpoint_during_init=self.load_checkpoint_during_init,
            export_within_fake_mode=self.export_within_fake_mode,
        )


if __name__ == "__main__":
    common_utils.run_tests()
