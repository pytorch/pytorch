"""Functions to verify exported ONNX model against original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""

import contextlib
import copy
import io
import os
import tempfile
from typing import Optional, Union

import numpy as np

import torch
from torch import Tensor
from torch.onnx.utils import unpack_quantized_tensor

_ORT_PROVIDERS = ["CPUExecutionProvider"]


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
    elif isinstance(elem, bool):
        return np.array(elem, dtype=bool)
    elif isinstance(elem, int):
        return np.array(elem, dtype=int)
    elif isinstance(elem, float):
        return np.array(elem, dtype=float)
    elif isinstance(elem, dict):
        flattened = []
        for k in elem:
            flattened += [_to_numpy(k)] + [_to_numpy(elem[k])]
        return flattened
    return elem


def _convert_to_onnx(
    model,
    model_f: Optional[Union[str, io.BytesIO]] = None,
    input=None,
    opset_version=None,
    do_constant_folding=True,
    keep_initializers_as_inputs=True,
    dynamic_axes=None,
    input_names=None,
    output_names=None,
    fixed_batch_size=False,
    training=None,
    verbose=False,
    ort_providers=_ORT_PROVIDERS,
    ort_optim_on=True,
):
    if model_f is None:
        model_f = io.BytesIO()
    input_copy = copy.deepcopy(input)

    torch.onnx._export(
        model,
        input_copy,
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

    try:
        import onnxruntime  # type: ignore[import]
    except ImportError:
        raise RuntimeError("ONNXRuntime is required for export verification.")

    # compute onnxruntime output prediction
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if ort_optim_on
        else onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    # suppress ort warnings.
    # 0:Verbose, 1:Info, 2:Warning. 3:Error, 4:Fatal. Default is 2.
    so.log_severity_level = 3
    ort_sess = onnxruntime.InferenceSession(
        model_f if isinstance(model_f, str) else model_f.getvalue(),
        so,
        providers=ort_providers,
    )
    return ort_sess


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


def _run_ort(ort_sess, inputs):
    kw_inputs = {}
    if inputs and isinstance(inputs[-1], dict):
        kw_inputs = inputs[-1]
        inputs = inputs[:-1]
    inputs = _unpack_to_numpy(_flatten_tuples(inputs))
    ort_inputs = {}
    for input_name, input in kw_inputs.items():
        ort_inputs[input_name] = _to_numpy(input)
    inputs = _to_numpy(inputs)
    ort_sess_inputs = ort_sess.get_inputs()
    for i, input in enumerate(inputs):
        if i == len(ort_sess_inputs) or ort_sess_inputs[i].name in ort_inputs:
            raise ValueError(
                f"got too many positional inputs. inputs: {inputs}. kw_inputs: {kw_inputs}"
            )
        ort_inputs[ort_sess_inputs[i].name] = input
    ort_outs = ort_sess.run(None, ort_inputs)
    return _inline_flatten_list(ort_outs, [])


def _ort_compare_with_pytorch(ort_outs, output, rtol, atol):
    output, _ = torch.jit._flatten(output)
    outputs = _unpack_to_numpy(output)

    # compare onnxruntime and PyTorch results
    assert len(outputs) == len(ort_outs), "number of outputs differ"

    # compare onnxruntime and PyTorch results
    [
        np.testing.assert_allclose(out, ort_out, rtol=rtol, atol=atol)
        for out, ort_out in zip(outputs, ort_outs)
    ]


def verify(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    input=None,
    batch_size=2,
    rtol=0.001,
    atol=1e-7,
    do_constant_folding=True,
    dynamic_axes=None,
    test_with_inputs=None,
    input_names=None,
    output_names=None,
    fixed_batch_size=False,
    dict_check=True,
    training=None,
    remained_onnx_input_idx=None,
    flatten=True,
    verbose=False,
    ort_providers=_ORT_PROVIDERS,
    ort_optim_on=True,
    use_external_data=False,
    opset_version: Optional[int] = None,
    keep_initializers_as_inputs: bool = True,
):
    """Verify model export to ONNX with ONNXRuntime."""
    if training is not None and training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training is None or training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    if input is None:
        input = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    with torch.no_grad():
        if isinstance(input, (Tensor, dict)):
            input = (input,)
        # In-place operators will update input tensor data as well.
        # Thus inputs are replicated before every forward call.
        input_args = copy.deepcopy(input)
        input_kwargs = {}
        if dict_check and isinstance(input_args[-1], dict):
            input_kwargs = input_args[-1]
            input_args = input_args[:-1]
        try:
            model_copy = copy.deepcopy(model)
            output = model_copy(*input_args, **input_kwargs)
        except Exception:
            output = model(*input_args, **input_kwargs)
        if isinstance(output, Tensor):
            output = (output,)

        if not dict_check and isinstance(input[-1], dict):
            input = input + ({},)

        with contextlib.ExitStack() as stack:
            model_f: Union[str, io.BytesIO] = io.BytesIO()
            if use_external_data:
                tmpdirname = stack.enter_context(tempfile.TemporaryDirectory())
                model_f = os.path.join(tmpdirname, "model.onnx")

            ort_sess = _convert_to_onnx(
                model,
                model_f,
                input=input,
                opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                dynamic_axes=dynamic_axes,
                input_names=input_names,
                output_names=output_names,
                fixed_batch_size=fixed_batch_size,
                training=training,
                verbose=verbose,
                ort_providers=ort_providers,
                ort_optim_on=ort_optim_on,
            )
            # compute onnxruntime output prediction
            if remained_onnx_input_idx is not None:
                input_onnx = []
                for idx in remained_onnx_input_idx:
                    input_onnx.append(input[idx])
                input = input_onnx

            input_copy = copy.deepcopy(input)
            if flatten:
                input_copy, _ = torch.jit._flatten(input_copy)
            elif input_copy and input_copy[-1] == {}:
                # Handle empty kwargs (normally removed by flatten).
                input_copy = input_copy[:-1]
            ort_outs = _run_ort(ort_sess, input_copy)
            _ort_compare_with_pytorch(ort_outs, output, rtol, atol)

            # if additional test inputs are provided run the onnx
            # model with these inputs and check the outputs
            if test_with_inputs is not None:
                for test_input in test_with_inputs:
                    if isinstance(test_input, Tensor):
                        test_input = (test_input,)
                    test_input_copy = copy.deepcopy(test_input)
                    output = model(*test_input_copy)
                    if isinstance(output, Tensor):
                        output = (output,)
                    if remained_onnx_input_idx is not None:
                        test_input_onnx = []
                        for idx in remained_onnx_input_idx:
                            test_input_onnx.append(test_input[idx])
                        test_input = test_input_onnx
                    if flatten:
                        test_input, _ = torch.jit._flatten(test_input)
                    ort_outs = _run_ort(ort_sess, test_input)
                    _ort_compare_with_pytorch(ort_outs, output, rtol, atol)
