# Owner(s): ["module: onnx"]

from __future__ import annotations

import dataclasses

import inspect
import io
import os
from typing import Any, Callable, Mapping, Sequence, Tuple, Type, Union

import onnxruntime
import pytorch_test_common

import torch
from torch.onnx import _constants, verification
from torch.onnx._internal import fx as fx_onnx
from torch.utils import _pytree as pytree

onnx_model_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.pardir,
    "repos",
    "onnx",
    "onnx",
    "backend",
    "test",
    "data",
)


pytorch_converted_dir = os.path.join(onnx_model_dir, "pytorch-converted")


pytorch_operator_dir = os.path.join(onnx_model_dir, "pytorch-operator")


def run_model_test(test_suite: _TestONNXRuntime, *args, **kwargs):
    options = verification.VerificationOptions()

    kwargs["opset_version"] = test_suite.opset_version
    kwargs["keep_initializers_as_inputs"] = test_suite.keep_initializers_as_inputs
    if hasattr(test_suite, "check_shape"):
        options.check_shape = test_suite.check_shape
    if hasattr(test_suite, "check_dtype"):
        options.check_dtype = test_suite.check_dtype

    names = {f.name for f in dataclasses.fields(options)}
    keywords_to_pop = []
    for k, v in kwargs.items():
        if k in names:
            setattr(options, k, v)
            keywords_to_pop.append(k)
    for k in keywords_to_pop:
        kwargs.pop(k)

    return verification.verify(*args, options=options, **kwargs)


def run_ort(
    onnx_model: Union[str, io.BytesIO], pytorch_inputs: Tuple[Any, ...]
) -> Sequence[Any]:
    session = onnxruntime.InferenceSession(
        onnx_model, providers=["CPUExecutionProvider"]
    )
    input_names = [ort_input.name for ort_input in session.get_inputs()]
    return session.run(
        None, {k: v.cpu().numpy() for k, v in zip(input_names, pytorch_inputs)}
    )


def run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
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
    ort_outputs = run_ort(onnx_model, bound.args)
    for ref_output, ort_output in zip(ref_outputs, ort_outputs):
        torch.testing.assert_close(
            ref_output, torch.tensor(ort_output), rtol=rtol, atol=atol
        )


def parameterize_class_name(cls: Type, idx: int, input_dicts: Mapping[Any, Any]):
    """Combine class name with the parameterized arguments.

    This function is passed to `parameterized.parameterized_class` as the
    `class_name_func` argument.
    """
    suffix = "_".join(f"{k}_{v}" for k, v in input_dicts.items())
    return f"{cls.__name__}_{suffix}"


class _TestONNXRuntime(pytorch_test_common.ExportTestCase):
    opset_version = _constants.ONNX_DEFAULT_OPSET
    keep_initializers_as_inputs = True  # For IR version 3 type export.
    is_script = False
    is_fx = False
    check_shape = True
    check_dtype = True

    def setUp(self):
        super().setUp()
        onnxruntime.set_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        os.environ["ALLOW_RELEASED_ONNX_OPSET_ONLY"] = "0"

    # The exported ONNX model may have less inputs than the pytorch model because of const folding.
    # This mostly happens in unit test, where we widely use torch.size or torch.shape.
    # So the output is only dependent on the input shape, not value.
    # remained_onnx_input_idx is used to indicate which pytorch model input idx is remained in ONNX model.
    def run_test(
        self,
        model,
        input_args,
        input_kwargs=None,
        rtol=1e-3,
        atol=1e-7,
        do_constant_folding=True,
        dynamic_axes=None,
        additional_test_inputs=None,
        input_names=None,
        output_names=None,
        fixed_batch_size=False,
        training=torch.onnx.TrainingMode.EVAL,
        remained_onnx_input_idx=None,
        verbose=False,
    ):
        def _run_test(m, remained_onnx_input_idx, flatten=True, ignore_none=True):
            return run_model_test(
                self,
                m,
                input_args=input_args,
                input_kwargs=input_kwargs,
                rtol=rtol,
                atol=atol,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                additional_test_inputs=additional_test_inputs,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                remained_onnx_input_idx=remained_onnx_input_idx,
                flatten=flatten,
                ignore_none=ignore_none,
                verbose=verbose,
            )

        if isinstance(remained_onnx_input_idx, dict):
            scripting_remained_onnx_input_idx = remained_onnx_input_idx["scripting"]
            tracing_remained_onnx_input_idx = remained_onnx_input_idx["tracing"]
        else:
            scripting_remained_onnx_input_idx = remained_onnx_input_idx
            tracing_remained_onnx_input_idx = remained_onnx_input_idx

        is_model_script = isinstance(
            model, (torch.jit.ScriptModule, torch.jit.ScriptFunction)
        )

        if self.is_script:
            # Run test for export route of `torch.nn.Module` -> `torch.jit.ScriptModule` -> ONNX.
            script_model = model if is_model_script else torch.jit.script(model)
            _run_test(
                script_model,
                scripting_remained_onnx_input_idx,
                flatten=False,
                ignore_none=False,
            )
        elif self.is_fx:
            # Run test with FX ONNX Exporter
            input_kwargs = input_kwargs or {}
            if isinstance(input_args, torch.Tensor):
                input_args = (input_args,)
            if training == torch.onnx.TrainingMode.EVAL:
                model.eval()
            run_test_with_fx_to_onnx_exporter_and_onnx_runtime(
                model,
                input_args,
                rtol,
                atol,
                opset_version=self.opset_version,
                **input_kwargs,
            )
        elif not is_model_script:
            # Run test on export route of `torch.nn.Module` -> `ONNX`.
            _run_test(model, tracing_remained_onnx_input_idx)
