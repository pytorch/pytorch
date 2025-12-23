# mypy: allow-untyped-defs
"""The ONNX verification module provides a set of tools to verify the correctness of ONNX models."""

from __future__ import annotations


__all__ = [
    "OnnxBackend",
    "VerificationOptions",
    "verify",
]

import contextlib
import copy
import dataclasses
import enum
import io
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

import torch
import torch._C._onnx as _C_onnx
from torch.onnx._internal.torchscript_exporter import utils
from torch.types import Number


# Everything below are deprecated ##############################################

_ORT_PROVIDERS = ("CPUExecutionProvider",)

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, torch.jit.ScriptModule]
_InputArgsType = Union[torch.Tensor, tuple[Any, ...]]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Union[Sequence[_NumericType], Sequence]


class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.
    """

    REFERENCE = "ONNXReferenceEvaluator"
    ONNX_RUNTIME_CPU = "CPUExecutionProvider"
    ONNX_RUNTIME_CUDA = "CUDAExecutionProvider"


@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """

    flatten: bool = True
    ignore_none: bool = True
    check_shape: bool = True
    check_dtype: bool = True
    backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    rtol: float = 1e-3
    atol: float = 1e-7
    remained_onnx_input_idx: Sequence[int] | None = None
    acceptable_error_percentage: float | None = None


def _flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(_flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened


def _to_numpy(elem) -> list | npt.NDArray:
    if isinstance(elem, torch.Tensor):
        if elem.requires_grad:
            return elem.detach().cpu().numpy()
        else:
            return elem.cpu().numpy()
    elif isinstance(elem, (list, tuple)):
        return [_to_numpy(inp) for inp in elem]
    elif isinstance(elem, (bool, int, float)):
        return np.array(elem)
    elif isinstance(elem, dict):
        flattened = []
        for k in elem:
            flattened.extend([_to_numpy(k), _to_numpy(elem[k])])
        return flattened
    return elem


def _inline_flatten_list(inputs, res_list) -> list:
    for i in inputs:
        res_list.append(i) if not isinstance(
            i, (list, tuple)
        ) else _inline_flatten_list(i, res_list)
    return res_list


def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    value_unpacked = []
    for value in values:
        value_unpacked.extend(
            utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted)
        )
    return [_to_numpy(v) for v in value_unpacked]


def _run_onnx(onnx_session, inputs) -> _OutputsType:
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    if hasattr(onnx_session, "get_inputs"):
        # onnxruntime.InferenceSession
        input_names = [i.name for i in onnx_session.get_inputs()]
    elif hasattr(onnx_session, "input_names"):
        # onnx.reference.ReferenceEvaluator
        input_names = onnx_session.input_names
    else:
        raise ValueError(f"Unknown ONNX backend type: {type(onnx_session)}.")

    for i, input in enumerate(inputs):
        if i == len(input_names) or input_names[i] in ort_inputs:
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}. "
                f"input names: {input_names}."
            )
        ort_inputs[input_names[i]] = input
    onnx_outs = onnx_session.run(None, ort_inputs)
    return onnx_outs


def _ort_session(
    model: str | io.BytesIO, ort_providers: Sequence[str] = _ORT_PROVIDERS
):
    try:
        import onnxruntime  # type: ignore[import]
    except ImportError as e:
        raise ImportError("onnxruntime is required for export verification.") from e

    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS

    session_options = onnxruntime.SessionOptions()
    # suppress ort warnings.
    # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
    session_options.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(
        model if isinstance(model, str) else model.getvalue(),
        session_options,
        providers=ort_providers,
    )
    return ort_session


def _onnx_backend_session(model: str | io.BytesIO, backend: OnnxBackend):
    if backend == OnnxBackend.REFERENCE:
        raise NotImplementedError
    elif backend in {OnnxBackend.ONNX_RUNTIME_CPU, OnnxBackend.ONNX_RUNTIME_CUDA}:
        onnx_session = _ort_session(model, (backend.value,))
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return onnx_session


def _compare_onnx_pytorch_outputs_in_np(
    onnx_outs: _OutputsType,
    pt_outs: _OutputsType,
    options: VerificationOptions,
) -> None:
    assert len(onnx_outs) == len(pt_outs), (
        f"Number of outputs differ ONNX runtime: ({len(onnx_outs)}) PyTorch: ({len(pt_outs)})"
    )
    acceptable_error_percentage = options.acceptable_error_percentage
    if acceptable_error_percentage and (
        acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0
    ):
        raise ValueError(
            "If set, acceptable_error_percentage should be between 0.0 and 1.0"
        )

    for ort_out, pt_out in zip(onnx_outs, pt_outs):
        try:
            # TODO: Remove `check_shape` option once every shape inconsistent issue is addressed.
            if not options.check_shape:
                # Allow different but broadcastable output shapes.
                ort_out, pt_out = np.broadcast_arrays(ort_out, pt_out)
            torch.testing.assert_close(
                ort_out,
                pt_out,
                rtol=options.rtol,
                atol=options.atol,
                check_dtype=options.check_dtype,
                equal_nan=True,
            )
        except AssertionError as e:
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(
                    np.isclose(ort_out, pt_out, rtol=options.rtol, atol=options.atol)
                ) / np.prod(ort_out.shape)  # pyrefly: ignore [missing-attribute]
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(
                        f"Suppressed AssertionError:\n{e}.\n"
                        f"Error percentage {error_percentage} "
                        f"within acceptable range {acceptable_error_percentage}.",
                        stacklevel=2,
                    )
                    continue
            # pyrefly: ignore [missing-attribute]
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn("ONNX output is quantized", stacklevel=2)
            # pyrefly: ignore [missing-attribute]
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn("PyTorch output is quantized", stacklevel=2)
            raise


def _compare_onnx_pytorch_outputs(
    onnx_outs: _OutputsType,
    pt_outs: Any,
    options: VerificationOptions,
) -> None:
    """
    Compare ONNX and PyTorch outputs.

    Args:
        onnx_outs: outputs from ONNX backend.
        pt_outs: outputs from PyTorch.
        options: options for verification.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if options.ignore_none:
        # torch.jit._flatten filters None type
        pt_outs, _ = torch.jit._flatten(pt_outs)
    else:
        pt_outs = _inline_flatten_list([pt_outs], [])
    pt_outs_np = _unpack_to_numpy(pt_outs, cast_onnx_accepted=False)
    onnx_outs = _inline_flatten_list(onnx_outs, [])
    _compare_onnx_pytorch_outputs_in_np(onnx_outs, pt_outs_np, options)


def _prepare_input_for_pytorch(args, kwargs):
    """Prepare input for PyTorch model execution.

    Any future changes/formatting to the input before dispatching to the PyTorch
    model should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
    """
    if isinstance(args, (torch.Tensor, dict)):
        args = (args,)
    # In-place operators will update input tensor data as well.
    # Thus inputs are replicated before every forward call.
    args = copy.deepcopy(args)
    if kwargs:
        kwargs = copy.deepcopy(kwargs)
    else:
        kwargs = {}
    return args, kwargs


def _prepare_input_for_export(args, kwargs):
    """Prepare input for ONNX model export.

    Any future changes/formatting to the input before dispatching to the
    :func:`torch.onnx.export` api should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        onnx_inputs: positional arguments for ONNX model export, as `args` in
            :func:`torch.onnx.export`.
    """
    args, kwargs = _prepare_input_for_pytorch(args, kwargs)
    if not kwargs and len(args) > 0 and isinstance(args[-1], dict):
        onnx_inputs = args + ({},)
    elif kwargs:
        onnx_inputs = args + (kwargs,)
    else:
        onnx_inputs = args
    return onnx_inputs


def _prepare_input_for_onnx(
    args, kwargs, remained_onnx_input_idx: Sequence[int] | None, flatten: bool
):
    """Prepare input for ONNX model execution in ONNX backend.

    Any future changes/formatting to the input before dispatching to the ONNX backend
    run should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.
        remained_onnx_input_idx: indices of inputs to be used for ONNX model execution.
        flatten: whether to flatten the input before dispatching to the ONNX model execution.

    Returns:
        onnx_inputs: positional arguments for ONNX model execution in ONNX backend.
    """
    onnx_inputs = _prepare_input_for_export(args, kwargs)
    if flatten:
        onnx_inputs, _ = torch.jit._flatten(onnx_inputs)
    elif onnx_inputs and onnx_inputs[-1] == {}:
        # Handle empty kwargs (normally removed by flatten).
        onnx_inputs = onnx_inputs[:-1]
    if remained_onnx_input_idx is not None:
        return [onnx_inputs[i] for i in remained_onnx_input_idx]
    else:
        return onnx_inputs


def _try_clone_model(model):
    """Used for preserving original model in case forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification.",
            stacklevel=2,
        )
        return model


def _compare_onnx_pytorch_model(
    pt_model: _ModelType,
    onnx_model_f: str | io.BytesIO,
    input_args: _InputArgsType,
    input_kwargs: _InputKwargsType | None,
    additional_test_inputs: Sequence[_InputArgsType] | None,
    options: VerificationOptions,
) -> None:
    """Compare outputs from ONNX model runs with outputs from PyTorch model runs.

    Args:
        pt_model: PyTorch model.
        onnx_model_f: ONNX model file path or file-like object.
        input_args: positional arguments for PyTorch model forward method.
        input_kwargs: keyword arguments for PyTorch model forward method.
        additional_test_inputs: additional positional arguments for PyTorch model
            forward method.
        options: options for verification.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
    """
    onnx_session = _onnx_backend_session(onnx_model_f, options.backend)

    def compare_onnx_pytorch_model_with_input(input_args, input_kwargs) -> None:
        pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        pt_model_copy = _try_clone_model(pt_model)
        pt_outs = pt_model_copy(*pt_args, **pt_kwargs)

        onnx_inputs = _prepare_input_for_onnx(
            input_args, input_kwargs, options.remained_onnx_input_idx, options.flatten
        )

        onnx_outs = _run_onnx(onnx_session, onnx_inputs)

        _compare_onnx_pytorch_outputs(
            onnx_outs=onnx_outs,
            pt_outs=pt_outs,
            options=options,
        )

    compare_onnx_pytorch_model_with_input(input_args, input_kwargs)

    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_onnx_pytorch_model_with_input(test_input_args, {})


def verify(
    model: _ModelType,
    input_args: _InputArgsType,
    input_kwargs: _InputKwargsType | None = None,
    do_constant_folding: bool = True,
    dynamic_axes: Mapping[str, Mapping[int, str] | Mapping[str, Sequence[int]]]
    | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    training: _C_onnx.TrainingMode = _C_onnx.TrainingMode.EVAL,
    opset_version: int | None = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    fixed_batch_size: bool = False,
    use_external_data: bool = False,
    additional_test_inputs: Sequence[_InputArgsType] | None = None,
    options: VerificationOptions | None = None,
) -> None:
    """Verify model export to ONNX against original PyTorch model.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Args:
        model: See :func:`torch.onnx.export`.
        input_args: See :func:`torch.onnx.export`.
        input_kwargs: See :func:`torch.onnx.export`.
        do_constant_folding: See :func:`torch.onnx.export`.
        dynamic_axes: See :func:`torch.onnx.export`.
        input_names: See :func:`torch.onnx.export`.
        output_names: See :func:`torch.onnx.export`.
        training: See :func:`torch.onnx.export`.
        opset_version: See :func:`torch.onnx.export`.
        keep_initializers_as_inputs: See :func:`torch.onnx.export`.
        verbose: See :func:`torch.onnx.export`.
        fixed_batch_size: Legacy argument, used only by rnn test cases.
        use_external_data: Explicitly specify whether to export the model with external data.
        additional_test_inputs: List of tuples. Each tuple is a group of
            input arguments to test. Currently only ``*args`` are supported.
        options: A VerificationOptions object that controls the verification behavior.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if options is None:
        options = VerificationOptions()

    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: str | io.BytesIO = io.BytesIO()
        if use_external_data:
            tmpdir_path = stack.enter_context(tempfile.TemporaryDirectory())
            model_f = os.path.join(tmpdir_path, "model.onnx")

        inputs_for_export = _prepare_input_for_export(input_args, input_kwargs)

        # TODO(#77679): remove this and treat mutating model separately.
        model_copy = _try_clone_model(model)
        utils._export(
            model,
            inputs_for_export,
            model_f,
            opset_version=opset_version,
            do_constant_folding=do_constant_folding,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            dynamic_axes=dynamic_axes,
            input_names=input_names,
            output_names=output_names,
            fixed_batch_size=fixed_batch_size,
            training=training,
            verbose=verbose,
        )

        _compare_onnx_pytorch_model(
            pt_model=model_copy,
            onnx_model_f=model_f,
            input_args=input_args,
            input_kwargs=input_kwargs,
            additional_test_inputs=additional_test_inputs,
            options=options,
        )
