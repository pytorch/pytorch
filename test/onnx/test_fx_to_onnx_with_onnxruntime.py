# Owner(s): ["module: onnx"]
from __future__ import annotations

import inspect

import io
import os
import tempfile
import unittest

from typing import Any, Callable, Sequence, Tuple, Union

import onnx.reference
import onnx_test_common

import onnxruntime  # type: ignore[import]

import torch
import transformers  # type: ignore[import]
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.onnx._internal import diagnostics, fx as fx_onnx
from torch.testing._internal import common_utils
from torch.utils import _pytree as pytree


def _run_onnx_reference_runtime(
    onnx_model: Union[str, io.BytesIO],
    pytorch_inputs: Tuple[Any, ...],
    verbose: int = 10,
) -> Sequence[Any]:
    session = onnx.reference.ReferenceEvaluator(onnx_model, verbose=verbose)
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(session.input_names, pytorch_inputs)}
    )


def _run_ort(
    onnx_model: Union[str, io.BytesIO], pytorch_inputs: Tuple[Any, ...]
) -> Sequence[Any]:
    session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


def _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
    model: Union[torch.nn.Module, Callable],
    input_args,
    rtol: float = 1e-3,
    atol: float = 1e-7,
    opset_version: int = 17,
    **input_kwargs,
):
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

    # Inspect the model's signature. It will be used
    # to flatten kwargs.
    if isinstance(model, torch.nn.Module):
        signature = inspect.signature(model.forward)
    else:
        signature = inspect.signature(model)

    # Bind args and kwargs to the model's signature to
    # flatten kwargs into positional args since ONNX
    # model cannot be called with kwargs.
    bound = signature.bind(*input_args, **input_kwargs)
    # Fill optional inputs.
    bound.apply_defaults()
    assert not bound.kwargs

    ref_outputs, _ = pytree.tree_flatten(model(*input_args, **input_kwargs))
    ort_outputs = _run_ort(onnx_model, bound.args)
    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        torch.testing.assert_close(
            ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
        )


class TestFxToOnnxWithOnnxRuntime(onnx_test_common._TestONNXRuntime):
    def setUp(self):
        super().setUp()
        self.diag_ctx = diagnostics.engine.create_diagnostic_context(
            "test_fx_export", version=torch.__version__
        )
        self.opset_version = 17

    def tearDown(self):
        diagnostics.engine.dump(
            f"test_report_{self._testMethodName}.sarif", compress=False
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

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))

    def test_func_with_args_and_kwargs(self):
        # Non-tensor optional kwargs are always folded into constant and
        # removed from input list in Dynamo-traced graph, so we can't
        # define a function like
        #   def func(x, b=1.0)
        # here. E.g., if you change the `b` to 1.0 below, it will complain
        # somewhere that model is called with extra args because the modified
        # function is traced into
        #   def forward(self, x : torch.Tensor):
        #     add = x + 1.0;  x = None
        #     relu = add.relu()
        #     return (add, relu)
        # To summarize, optional kwargs must be tensors; otherwise, they are
        # treated as in-graph constants in Dynamo.
        def func(x, b=torch.tensor(1.0)):
            y = x + b
            z = y.relu()
            return (y, z)

        tensor_x = torch.randn(1, 1, 2, dtype=torch.float32)

        # Test without providing optional kwarg.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(func, (tensor_x,))
        # Test with only positional args.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x, torch.tensor(8.0))
        )
        # Test while specifying optional kwarg.
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
            func, (tensor_x,), b=torch.tensor(5.0)
        )

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
        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(MNISTModel(), (tensor_x,))

    # test single op with no kwargs
    def test_sigmoid(self):
        x = torch.randn(1, 4, 2, 3)

        class SigmoidModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                return self.sigmoid(x)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(SigmoidModel(), (x,))

    # test single op with no kwargs
    def test_sigmoid_add(self):
        self.opset_version = 17
        # TODO(titaiwang): change to randn once it's ready
        x = torch.tensor([1.0, 2.0], dtype=torch.float)

        class SigmoidAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                x = torch.ops.aten.add(x, 1.0, alpha=2.0)
                return self.sigmoid(x)

        _run_test_with_fx_to_onnx_exporter_and_onnx_runtime(SigmoidAddModel(), (x,))

    def test_gpt2_tiny(self):
        model_name = "sshleifer/tiny-gpt2"
        # Download pytorch model
        model = transformers.AutoModel.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        # Transform input tokens
        inputs = tokenizer("Hello world!", return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        onnx_model = fx_onnx.export_after_normalizing_args_and_kwargs(
            model, use_binary_format=True, opset_version=self.opset_version, **inputs
        )

        ref_outputs, _ = pytree.tree_flatten(model(**inputs, return_dict=False))
        ort_outputs = _run_ort(onnx_model, (input_ids, attention_mask))
        assert len(ref_outputs) == len(ort_outputs)
        assert len(ref_outputs) == 5
        for ref_output, ort_output in zip(ref_outputs, ort_outputs):
            torch.testing.assert_close(ref_output, torch.tensor(ort_output))

    def _test_large_scale_exporter(
        self,
        model_name,
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

            ftm = FakeTensorMode(
                allow_non_fake_inputs=True, allow_fallback_kernels=False
            )
            ctx = fx_onnx.FxToOnnxContext()

            # The following coed block does several things.
            #  1. Create a model whose parameters and buffers are all FakeTensor's.
            #  2. Convert nn.Module into ONNX model without initializers.
            #  3. Record the file paths to find real initializers.
            with ftm, ctx:
                # Toy model with parameters and buffers as FakeTensor's.
                fake_model = create_model()
                fake_model.load_state_dict(torch.load(tmp_file.name))
                # Toy inputs as FakeTensor's.
                fake_args = create_args()
                # Export ONNX model without initializers while ctx.paths records
                # all files that contains real initializers.
                (onnx_model, _, _, _) = fx_onnx.export_without_parameters_and_buffers(
                    fake_model,
                    *fake_args,
                    use_binary_format=False,
                    opset_version=self.opset_version,
                )

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
            ref_outputs, _ = pytree.tree_flatten(model(*args, **kwargs))
            # ORT outputs.
            ort_outputs = _run_ort(
                os.path.join(tmp_folder, onnx_model_location),
                (arg for arg in args if arg is not None),
            )

            assert len(ref_outputs) == len(ort_outputs)

            for ref_output, ort_output in zip(ref_outputs, ort_outputs):
                torch.testing.assert_close(ref_output, torch.tensor(ort_output))

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

        def create_model():
            return MLPModel()

        def create_args():
            return (torch.rand((97, 8), dtype=torch.float32),)

        def create_pytorch_only_extra_kwargs():
            return {}

        self._test_large_scale_exporter(
            "toy_mlp1", create_model, create_args, create_pytorch_only_extra_kwargs
        )

    @unittest.skip("To pass this test, if-else conditions in GPT2 should be removed.")
    def test_large_scale_exporter_with_tiny_gpt2(self):
        model_name = "sshleifer/tiny-gpt2"

        def create_model():
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
            "tiny_gpt2", create_model, create_args, create_pytorch_only_extra_kwargs
        )


if __name__ == "__main__":
    common_utils.run_tests()
