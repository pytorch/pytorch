"""Functions to verify exported ONNX model is functionally equivalent to original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""

import contextlib
import copy
import io
import os
import tempfile
import warnings
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import torch
from torch import Tensor
from torch.onnx.utils import unpack_quantized_tensor

_ORT_PROVIDERS = ("CPUExecutionProvider",)


def _flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(_flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened


def _to_numpy(elem):
    if isinstance(elem, Tensor):
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
            flattened += [_to_numpy(k)] + [_to_numpy(elem[k])]
        return flattened
    return elem


def _inline_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(
            i, (list, tuple)
        ) else _inline_flatten_list(i, res_list)
    return res_list


def _unpack_to_numpy(values):
    value_unpacked = []
    for value in values:
        value_unpacked.extend(unpack_quantized_tensor(value))
    return [_to_numpy(v) for v in value_unpacked]


def _run_ort(ort_session, inputs):
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    ort_session_inputs = ort_session.get_inputs()
    for i, input in enumerate(inputs):
        if i == len(ort_session_inputs) or ort_session_inputs[i].name in ort_inputs:
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}"
            )
        ort_inputs[ort_session_inputs[i].name] = input
    ort_outs = ort_session.run(None, ort_inputs)
    return _inline_flatten_list(ort_outs, [])


def _ort_session(
    model: Union[str, io.BytesIO], ort_providers: Sequence[str] = _ORT_PROVIDERS
):
    try:
        import onnxruntime  # type: ignore[import]
    except ImportError:
        raise ImportError("onnxruntime is required for export verification.")

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


def _compare_ort_pytorch_outputs(ort_outs, pt_outs, rtol, atol):
    pt_outs, _ = torch.jit._flatten(pt_outs)
    pt_outs = _unpack_to_numpy(pt_outs)

    assert len(pt_outs) == len(ort_outs), "number of outputs differ"

    for ort_out, pt_out in zip(ort_outs, pt_outs):
        np.testing.assert_allclose(ort_out, pt_out, rtol=rtol, atol=atol)


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
    if isinstance(args, (Tensor, dict)):
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
    if not kwargs and isinstance(args[-1], dict):
        onnx_inputs = args + ({},)
    elif kwargs:
        onnx_inputs = args + (kwargs,)
    else:
        onnx_inputs = args
    return onnx_inputs


def _prepare_input_for_ort(args, kwargs, remained_onnx_input_idx, flatten):
    """Prepare input for ONNX model execution in ONNX Runtime.

    Any future changes/formatting to the input before dispatching to the ONNX Runtime
    InferenceSession run should be made in this function.

    Args:
        args: positional arguments for PyTorch model forward method.
        kwargs: keyword arguments for PyTorch model forward method.

    Returns:
        onnx_inputs: positional arguments for ONNX model execution in ONNX Runtime.
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
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


def _compare_ort_pytorch_model(
    model,
    ort_session,
    input_args,
    input_kwargs,
    additional_test_inputs,
    remained_onnx_input_idx,
    flatten,
    rtol,
    atol,
):
    """Compare outputs from ONNX model runs with outputs from PyTorch model runs.

    ONNX Runtime is used for model execution backend for ONNX model.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
    """

    def compare_ort_pytorch_model_with_input(input_args, input_kwargs):
        pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        model_copy = _try_clone_model(model)
        pt_outs = model_copy(*pt_args, **pt_kwargs)

        ort_inputs = _prepare_input_for_ort(
            input_args, input_kwargs, remained_onnx_input_idx, flatten
        )
        ort_outs = _run_ort(ort_session, ort_inputs)

        _compare_ort_pytorch_outputs(ort_outs, pt_outs, rtol, atol)

    compare_ort_pytorch_model_with_input(input_args, input_kwargs)

    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_ort_pytorch_model_with_input(test_input_args, {})


def verify(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    input_args: Tuple[Any, ...],
    input_kwargs: Optional[Mapping[str, Any]] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Mapping[str, Union[Mapping[int, str], Mapping[str, Sequence[int]]]]
    ] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    training: Optional[bool] = None,
    opset_version: Optional[int] = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    fixed_batch_size: bool = False,
    use_external_data: bool = False,
    additional_test_inputs: Optional[Sequence[Tuple[Any, ...]]] = None,
    remained_onnx_input_idx: Optional[Sequence[int]] = None,
    flatten: bool = True,
    ort_providers: Sequence[str] = _ORT_PROVIDERS,
    rtol: float = 0.001,
    atol: float = 1e-7,
    **kwargs,
):
    """Verify model export to ONNX with ONNX Runtime.

    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): See :func:`torch.onnx.export`.
        input_args (tuple): See :func:`torch.onnx.export`.
        input_kwargs (dict): See :func:`torch.onnx.export`.
        do_constant_folding (bool, optional): See :func:`torch.onnx.export`.
        dynamic_axes (dict, optional): See :func:`torch.onnx.export`.
        input_names (list, optional): See :func:`torch.onnx.export`.
        output_names (list, optional): See :func:`torch.onnx.export`.
        training (bool, optional): See :func:`torch.onnx.export`.
        opset_version (int, optional): See :func:`torch.onnx.export`.
        keep_initializers_as_inputs (bool, optional): See :func:`torch.onnx.export`.
        verbose (bool, optional): See :func:`torch.onnx.export`.
        fixed_batch_size (bool, optional): Legacy argument, used only by rnn test cases.
        use_external_data (bool, optional): Explicitly specify whether to export the
            model with external data.
        additional_test_inputs (list, optional): List of tuples. Each tuple is a set of
            input arguments to test. Currently only *args are supported.
        remained_onnx_input_idx (list, optional): If provided, only the specified inputs
            will be passed to the ONNX model. Supply a list when there are unused inputs
            in the model. Since unused inputs will be removed in the exported ONNX
            model, supplying all inputs will cause an error on unexpected inputs.
            This parameter tells the verifier which inputs to pass into the ONNX model.
        flatten (bool, optional): Default True. If True, unpack nested list/tuple/dict
            inputs into a flattened list of Tensors for ONNX. Set this to False if nested
            structures are to be preserved for ONNX, which is usually the case with
            exporting ScriptModules.
        ort_providers (sequence, optional): ONNX Runtime providers to use.
        rtol (float, optional): relative tolerance in comparison between ONNX and PyTorch outputs.
        atol (float, optional): absolute tolerance in comparison between ONNX and PyTorch outputs.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
    """
    if training is not None and training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training is None or training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: Union[str, io.BytesIO] = io.BytesIO()
        if use_external_data:
            tmpdirname = stack.enter_context(tempfile.TemporaryDirectory())
            model_f = os.path.join(tmpdirname, "model.onnx")

        inputs_for_export = _prepare_input_for_export(input_args, input_kwargs)

        # TODO: remove this and treat mutating model separately. See #77679
        model_copy = _try_clone_model(model)
        torch.onnx._export(
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

        ort_session = _ort_session(model_f, ort_providers)

        _compare_ort_pytorch_model(
            model_copy,
            ort_session,
            input_args,
            input_kwargs,
            additional_test_inputs,
            remained_onnx_input_idx,
            flatten,
            rtol,
            atol,
        )
