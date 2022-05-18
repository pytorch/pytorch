"""Functions to verify exported ONNX model against original PyTorch model.

ONNXRuntime is required, and is used as the ONNX backend for export verification.
The goal is to verify that the exported ONNX model is functionally equivalent to the
original PyTorch model.
"""

import contextlib
import copy
import io
import os
import tempfile
from typing import Optional, Tuple, Union

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


def _create_ort_session(
    model: Union[str, io.BytesIO], ort_providers: Optional[Tuple[str, ...]] = None
):
    try:
        import onnxruntime  # type: ignore[import]
    except ImportError:
        raise RuntimeError("ONNXRuntime is required for export verification.")

    if ort_providers is None:
        ort_providers = _ORT_PROVIDERS

    so = onnxruntime.SessionOptions()
    # suppress ort warnings.
    # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
    so.log_severity_level = 3
    ort_session = onnxruntime.InferenceSession(
        model if isinstance(model, str) else model.getvalue(),
        so,
        providers=ort_providers,
    )
    return ort_session


def _compare_ort_pytorch_outputs(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = _unpack_to_numpy(output)

    assert len(outputs) == len(ort_outs), "number of outputs differ"

    for out, ort_out in zip(outputs, ort_outs):
        np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol)


def _format_input_for_pytorch(args, kwargs):
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


def _format_input_for_export(args, kwargs):
    args, kwargs = _format_input_for_pytorch(args, kwargs)
    if not kwargs and isinstance(args[-1], dict):
        onnx_inputs = args + ({},)
    elif kwargs:
        onnx_inputs = args + (kwargs,)
    else:
        onnx_inputs = args
    return onnx_inputs


def _format_input_for_ort(args, kwargs, remained_onnx_input_idx, flatten):
    onnx_inputs = _format_input_for_export(args, kwargs)
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
    """Used for preserving original model incase forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
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
    def compare_ort_pytorch_model_with_input(input_args, input_kwargs):
        pt_args, pt_kwargs = _format_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        model_copy = _try_clone_model(model)
        pt_outs = model_copy(*pt_args, **pt_kwargs)

        ort_inputs = _format_input_for_ort(
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
    input_args,
    input_kwargs=None,
    do_constant_folding=True,
    dynamic_axes=None,
    input_names=None,
    output_names=None,
    training=None,
    opset_version=None,
    keep_initializers_as_inputs=True,
    verbose=False,
    fixed_batch_size=False,
    use_external_data=False,
    additional_test_inputs=None,
    remained_onnx_input_idx=None,
    flatten=True,
    ort_providers=_ORT_PROVIDERS,
    rtol=0.001,
    atol=1e-7,
):
    """Verify model export to ONNX with ONNXRuntime.

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
        fixed_batch_size (bool, optional):
        use_external_data (bool, optional):
        additional_test_inputs (list, optional):
        remained_onnx_input_idx (list, optional):
        flatten (bool, optional):
        ort_providers (list, optional):
        rtol (float, optional):
        atol (float, optional):


        dict_check (bool, optional): If True, expect last input, if dictionary,
            to be keyword arguments to model forward. Otherwise, last input, if
            dictionary, is treated as the last positional argument. An empty dictionary
            will appended to the end of the input list, informing the export api that
            keyword arguments to model forward is empty.
        ort_providers (tuple, optional): A tuple of ONNX Runtime providers to use.
            Default is _ORT_PROVIDERS in torch.onnx.verification.
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

        inputs_for_export = _format_input_for_export(input_args, input_kwargs)

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

        ort_session = _create_ort_session(model_f, ort_providers)

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
