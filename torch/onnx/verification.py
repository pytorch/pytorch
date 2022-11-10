"""Functions to verify exported ONNX model is functionally equivalent to original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import difflib
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype
from torch.types import Number

_ORT_PROVIDERS = ("CPUExecutionProvider",)

_NumericType = Union[Number, torch.Tensor, np.ndarray]


@_beartype.beartype
def _flatten_tuples(elem):
    flattened = []
    for t in elem:
        if isinstance(t, tuple):
            flattened.extend(_flatten_tuples(t))
        else:
            flattened.append(t)
    return flattened


# TODO(justinchuby): Add type checking by narrowing down the return type when input is None
def _to_numpy(elem) -> Union[list, np.ndarray]:
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


@_beartype.beartype
def _inline_flatten_list(inputs, res_list) -> list:
    for i in inputs:
        res_list.append(i) if not isinstance(
            i, (list, tuple)
        ) else _inline_flatten_list(i, res_list)
    return res_list


@_beartype.beartype
def _unpack_to_numpy(values, cast_onnx_accepted=True) -> list:
    value_unpacked = []
    for value in values:
        value_unpacked.extend(
            utils.unpack_quantized_tensor(value, cast_onnx_accepted=cast_onnx_accepted)
        )
    return [_to_numpy(v) for v in value_unpacked]


@_beartype.beartype
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
    return ort_outs


@_beartype.beartype
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


@_beartype.beartype
def _compare_ort_pytorch_outputs(
    ort_outs: Union[Sequence[_NumericType], Sequence],
    pt_outs: Optional[Union[_NumericType, Sequence[_NumericType], Sequence, Dict]],
    rtol: float,
    atol: float,
    check_shape: bool,
    check_dtype: bool,
    ignore_none: bool,
    acceptable_error_percentage: Optional[float],
):
    """
    Compare ONNX Runtime and PyTorch outputs.

    Args:
        ort_outs: outputs from ONNX Runtime.
        pt_outs: outputs from PyTorch.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        ignore_none: Whether to ignore None type in
            torch output, which is usually the case with tracing. Set this to False, if
            torch output should keep None type, which is usually the case with exporting
            ScriptModules.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if ignore_none:
        # torch.jit._flatten filters None type
        pt_outs, _ = torch.jit._flatten(pt_outs)
    else:
        pt_outs = _inline_flatten_list([pt_outs], [])
    pt_outs_np = _unpack_to_numpy(pt_outs, cast_onnx_accepted=False)
    ort_outs = _inline_flatten_list(ort_outs, [])
    assert len(ort_outs) == len(
        pt_outs_np
    ), f"Number of outputs differ ONNX runtime: ({len(ort_outs)}) PyTorch: ({len(pt_outs_np)})"
    if acceptable_error_percentage and (
        acceptable_error_percentage > 1.0 or acceptable_error_percentage < 0.0
    ):
        raise ValueError(
            "If set, acceptable_error_percentage should be between 0.0 and 1.0"
        )

    for ort_out, pt_out in zip(ort_outs, pt_outs_np):
        try:
            # TODO: Remove `check_shape` option once every shape inconsistent issue is addressed.
            if not check_shape:
                # Allow different but broadcastable output shapes.
                ort_out, pt_out = np.broadcast_arrays(ort_out, pt_out)
            torch.testing.assert_close(
                ort_out,
                pt_out,
                rtol=rtol,
                atol=atol,
                check_dtype=check_dtype,
                equal_nan=True,
            )
        except AssertionError as e:
            if acceptable_error_percentage:
                error_percentage = 1 - np.sum(
                    np.isclose(ort_out, pt_out, rtol=rtol, atol=atol)
                ) / np.prod(ort_out.shape)
                if error_percentage <= acceptable_error_percentage:
                    warnings.warn(
                        f"Suppressed AssertionError:\n{e}.\n"
                        f"Error percentage {error_percentage} "
                        f"within acceptable range {acceptable_error_percentage}."
                    )
                    continue
            if ort_out.dtype == np.uint8 or ort_out.dtype == np.int8:
                warnings.warn("ONNX output is quantized")
            if pt_out.dtype == np.uint8 or pt_out.dtype == np.int8:
                warnings.warn("PyTorch output is quantized")
            raise


@_beartype.beartype
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


@_beartype.beartype
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


@_beartype.beartype
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


@_beartype.beartype
def _try_clone_model(model):
    """Used for preserving original model in case forward mutates model states."""
    try:
        return copy.deepcopy(model)
    except Exception:
        warnings.warn(
            "Failed to clone model. Model state might be mutated during verification."
        )
        return model


@_beartype.beartype
def _compare_ort_pytorch_model(
    model,
    ort_session,
    input_args,
    input_kwargs,
    additional_test_inputs,
    remained_onnx_input_idx,
    flatten,
    ignore_none,
    rtol,
    atol,
    check_shape,
    check_dtype,
    acceptable_error_percentage: Optional[float],
):
    """Compare outputs from ONNX model runs with outputs from PyTorch model runs.

    ONNX Runtime is used for model execution backend for ONNX model.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
    """

    @_beartype.beartype
    def compare_ort_pytorch_model_with_input(input_args, input_kwargs):
        pt_args, pt_kwargs = _prepare_input_for_pytorch(input_args, input_kwargs)
        # TODO: remove this and treat mutating model separately. See #77679
        model_copy = _try_clone_model(model)
        pt_outs = model_copy(*pt_args, **pt_kwargs)

        ort_inputs = _prepare_input_for_ort(
            input_args, input_kwargs, remained_onnx_input_idx, flatten
        )
        ort_outs = _run_ort(ort_session, ort_inputs)

        _compare_ort_pytorch_outputs(
            ort_outs=ort_outs,
            pt_outs=pt_outs,
            rtol=rtol,
            atol=atol,
            check_shape=check_shape,
            check_dtype=check_dtype,
            ignore_none=ignore_none,
            acceptable_error_percentage=acceptable_error_percentage,
        )

    compare_ort_pytorch_model_with_input(input_args, input_kwargs)

    if additional_test_inputs:
        for test_input_args in additional_test_inputs:
            compare_ort_pytorch_model_with_input(test_input_args, {})


class _GraphDiff:
    """A class to represent the difference between two graphs."""

    @_beartype.beartype
    def __init__(self, graph_a: _C.Graph, graph_b: _C.Graph):
        """Construct a _GraphDiff object.

        Args:
            graph_a (_C.Graph): First graph to compare.
            graph_b (_C.Graph): Second graph to compare.
        """
        self.graph_a = graph_a
        self.graph_b = graph_b

    @_beartype.beartype
    def __str__(self):
        """See function :func:`diff_report`."""
        return self.diff_report()

    @_beartype.beartype
    def _indent(self, lines: str) -> str:
        return "\n".join(["\t" + line for line in lines.splitlines()])

    @_beartype.beartype
    def diff_report(self) -> str:
        """Return a string representation of the graph difference.

        The report shows the first pair of nodes that diverges. It also shows the source
        location of the pair of nodes.

        Returns:
            graph_diff_report (str): A string representation of the graph difference.
        """
        graph_a = self.graph_a
        graph_b = self.graph_b

        graph_a_str = str(graph_a)
        graph_b_str = str(graph_b)

        if graph_a_str == graph_b_str:
            return ""

        graph_diff = difflib.ndiff(
            graph_a_str.splitlines(True), graph_b_str.splitlines(True)
        )
        graph_diff_report = ["Graph diff:", self._indent("".join(graph_diff))]

        for node_a, node_b in itertools.zip_longest(graph_a.nodes(), graph_b.nodes()):
            if str(node_a) != str(node_b):
                graph_diff_report.append("First diverging operator:")
                node_diff = difflib.ndiff(
                    str(node_a).splitlines(True), str(node_b).splitlines(True)
                )
                source_printout = ["node diff:", self._indent("".join(node_diff))]

                stack_a = node_a.sourceRange() if node_a else None
                if stack_a:
                    source_printout.extend(
                        ["Former source location:", self._indent(str(stack_a))]
                    )
                stack_b = node_b.sourceRange() if node_b else None
                if stack_b:
                    source_printout.extend(
                        ["Latter source location:", self._indent(str(stack_b))]
                    )

                graph_diff_report.extend(source_printout)

                break

        return "\n".join(graph_diff_report)


@_beartype.beartype
def _check_graph_diff(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    export_options: _experimental.ExportOptions,
    model_to_graph_func: Callable[
        [
            torch.nn.Module,
            Tuple[Any, ...],
            Mapping[str, Any],
            _experimental.ExportOptions,
        ],
        _C.Graph,
    ],
) -> str:
    """Check if graph produced by `model_to_graph_func` is the same across `test_input_groups`.

    Args:
        model: See :func:`check_export_model_diff`.
        test_input_groups: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.
        model_to_graph_func: A function to convert a PyTorch model to a JIT IR graph.

    Returns:
        graph_diff_report (str): A string representation of the graph difference.
    """
    if len(test_input_groups) < 2:
        raise ValueError("Need at least two groups of test inputs to compare.")

    ref_jit_graph = None
    for args, kwargs in test_input_groups:
        jit_graph = model_to_graph_func(model, args, kwargs, export_options)
        if ref_jit_graph is None:
            ref_jit_graph = jit_graph
            continue

        graph_diff_report = _GraphDiff(ref_jit_graph, jit_graph).diff_report()
        if graph_diff_report:
            return graph_diff_report
    return ""


@_beartype.beartype
def _traced_graph_from_model(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    export_options: _experimental.ExportOptions,
) -> _C.Graph:
    """As part of the ONNX export steps, create a traced JIT graph from a PyTorch model.

    Args:
        model: See :func:`check_export_model_diff`.
        args: See :func:`check_export_model_diff`.
        kwargs: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.

    Returns:
        jit_graph (_C.Graph): A traced JIT graph.
    """
    training = export_options.training
    verbose = export_options.verbose

    with utils.exporter_context(model, training, verbose):
        export_inputs = _prepare_input_for_export(args, kwargs)
        model = utils._pre_trace_quant_model(model, export_inputs)
        jit_graph, _, _, _ = utils._create_jit_graph(model, export_inputs)
        return jit_graph


@_beartype.beartype
def _onnx_graph_from_model(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    export_options: _experimental.ExportOptions,
) -> _C.Graph:
    """As part of the ONNX export steps, export an ONNX JIT graph from a PyTorch model.

    Args:
        model: See :func:`check_export_model_diff`.
        args: See :func:`check_export_model_diff`.
        kwargs: See :func:`check_export_model_diff`.
        export_options: See :func:`check_export_model_diff`.

    Returns:
        onnx_graph (_C.Graph): An ONNX JIT graph.
    """
    # TODO: refactor utils.py to remove duplicated code of context setup. See #78834
    opset_version = export_options.opset_version
    operator_export_type = export_options.operator_export_type
    export_modules_as_functions = export_options.export_modules_as_functions
    training = export_options.training
    verbose = export_options.verbose
    dynamic_axes = export_options.dynamic_axes
    input_names = export_options.input_names
    output_names = export_options.output_names

    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET

    utils._setup_trace_module_map(model, export_modules_as_functions)

    if not operator_export_type:
        if _C_onnx._CAFFE2_ATEN_FALLBACK:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = _C_onnx.OperatorExportTypes.ONNX

    GLOBALS.export_onnx_opset_version = opset_version
    GLOBALS.operator_export_type = operator_export_type

    with utils.exporter_context(model, training, verbose):
        do_constant_folding = utils._decide_constant_folding(
            export_options.do_constant_folding, operator_export_type, training
        )

        if dynamic_axes is None:
            dynamic_axes = {}
        utils._validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

        export_inputs = _prepare_input_for_export(args, kwargs)
        export_inputs = utils._decide_input_format(model, export_inputs)
        onnx_graph, _, _ = utils._model_to_graph(
            model,
            export_inputs,
            verbose,
            input_names,
            output_names,
            operator_export_type,
            do_constant_folding,
            training=training,
            dynamic_axes=dynamic_axes,
        )

        return onnx_graph


@_beartype.beartype
def check_export_model_diff(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    test_input_groups: Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]],
    export_options: Optional[_experimental.ExportOptions] = None,
) -> str:
    """Verify exported model discrepancy between different groups of inputs.

    A graph is exported for each group of inputs. The exported graphs are then compared
    to each other, and discrepancies of first pair of nodes are reported. This function
    first checks the jit graph. If no discrepancies were found, it then checks the onnx
    graph.

    Unless otherwise specified, the jit/ONNX graph is expected to be the same, regardless
    of the inputs used for exporting. A discrepancy implies the graph exported is
    not accurate when run on other groups of inputs, which will typically results in
    runtime errors or mismatching output.

    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): The model to be exported.
        test_input_groups (Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]]): A sequence
            of input groups to be used to export the model. Each input group is a pair of
            (args, kwargs).
        export_options (_experimental.ExportOptions, optional): An _experimental.ExportOptions
            object that controls the export behavior.

    Returns:
        str: A string containing the diff of the exported models.
    """
    export_options = (
        _experimental.ExportOptions() if export_options is None else export_options
    )

    jit_diff_report = _check_graph_diff(
        model, test_input_groups, export_options, _traced_graph_from_model
    )
    if jit_diff_report:
        return jit_diff_report

    return _check_graph_diff(
        model, test_input_groups, export_options, _onnx_graph_from_model
    )


@_beartype.beartype
def verify(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    input_args: Union[torch.Tensor, Tuple[Any, ...]],
    input_kwargs: Optional[Mapping[str, Any]] = None,
    do_constant_folding: bool = True,
    dynamic_axes: Optional[
        Mapping[str, Union[Mapping[int, str], Mapping[str, Sequence[int]]]]
    ] = None,
    input_names: Optional[Sequence[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    training: torch.onnx.TrainingMode = torch.onnx.TrainingMode.EVAL,
    opset_version: Optional[int] = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    fixed_batch_size: bool = False,
    use_external_data: bool = False,
    additional_test_inputs: Optional[
        Sequence[Union[torch.Tensor, Tuple[Any, ...]]]
    ] = None,
    remained_onnx_input_idx: Optional[Sequence[int]] = None,
    flatten: bool = True,
    ignore_none: bool = True,
    check_shape: bool = True,
    check_dtype: bool = True,
    ort_providers: Sequence[str] = _ORT_PROVIDERS,
    rtol: float = 0.001,
    atol: float = 1e-7,
    acceptable_error_percentage: Optional[float] = None,
    **_,
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
        training (torch.onnx.TrainingMode): See :func:`torch.onnx.export`.
        opset_version (int, optional): See :func:`torch.onnx.export`.
        keep_initializers_as_inputs (bool, optional): See :func:`torch.onnx.export`.
        verbose (bool, optional): See :func:`torch.onnx.export`.
        fixed_batch_size (bool, optional): Legacy argument, used only by rnn test cases.
        use_external_data (bool, optional): Explicitly specify whether to export the
            model with external data.
        additional_test_inputs (list, optional): List of tuples. Each tuple is a group of
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
        ignore_none (bool, optional): Whether to ignore None type in
            torch output, which is usually the case with tracing. Set this to False, if
            torch output should keep None type, which is usually the case with exporting
            ScriptModules. Default to True.
        check_shape (bool, optional): Whether to check the shapes between
            PyTorch and ONNX Runtime outputs are exactly the same. Set this to False to allow
            output shape broadcasting. Default to True.
        check_dtype (bool, optional): Whether to check the dtypes between
            PyTorch and ONNX Runtime outputs are consistent. Default to True.
        ort_providers (sequence, optional): ONNX Runtime providers to use.
        rtol (float, optional): relative tolerance in comparison between ONNX and PyTorch outputs.
        atol (float, optional): absolute tolerance in comparison between ONNX and PyTorch outputs.
        acceptable_error_percentage (float, optional): acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad(), contextlib.ExitStack() as stack:
        model_f: Union[str, io.BytesIO] = io.BytesIO()
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

        ort_session = _ort_session(model_f, ort_providers)

        _compare_ort_pytorch_model(
            model=model_copy,
            ort_session=ort_session,
            input_args=input_args,
            input_kwargs=input_kwargs,
            additional_test_inputs=additional_test_inputs,
            remained_onnx_input_idx=remained_onnx_input_idx,
            flatten=flatten,
            ignore_none=ignore_none,
            rtol=rtol,
            atol=atol,
            check_shape=check_shape,
            check_dtype=check_dtype,
            acceptable_error_percentage=acceptable_error_percentage,
        )


@_beartype.beartype
def _remove_none_inputs(graph: torch.Graph, input_args):
    """Remove None inputs from the graph.

    Args:
        graph (torch.Graph): The graph to remove None inputs.
        input_args (tuple): The input arguments to the model.
    """
    idx_to_erase = []
    for i, arg in enumerate(input_args):
        if arg is None:
            idx_to_erase.append(i)
    for i in reversed(idx_to_erase):
        graph.eraseInput(i)
        input_args.erase(i)

    return graph, input_args


@_beartype.beartype
def verify_aten_graph(
    graph: torch.Graph,
    input_args: Tuple[Any, ...],
    export_options: _experimental.ExportOptions,
    params_dict: Optional[Dict[str, Any]] = None,
    flatten: bool = True,
    ignore_none: bool = True,
    check_shape: bool = True,
    check_dtype: bool = True,
    ort_providers: Sequence[str] = _ORT_PROVIDERS,
    rtol: float = 0.001,
    atol: float = 1e-7,
    acceptable_error_percentage: Optional[float] = None,
    **_,
) -> Tuple[bool, Union[_NumericType, Sequence[_NumericType]]]:
    operator_export_type = export_options.operator_export_type
    dynamic_axes = {}
    input_names = []
    training = export_options.training
    do_constant_folding = export_options.do_constant_folding
    opset_version = export_options.opset_version
    if params_dict is None:
        params_dict = {}

    graph_inputs = [v for v in graph.inputs()]
    jit_inputs = tuple([arg for arg in input_args if arg is not None])
    weights = [params_dict[v.debugName()] for v in graph_inputs[len(jit_inputs) :]]
    assert all([w is not None for w in weights])

    jit_input_and_parameters = jit_inputs + tuple(weights)
    jit_outs = torch._C._jit_interpret_graph(graph, jit_input_and_parameters)
    jit_graph = graph
    graph = utils._optimize_graph(
        graph,
        operator_export_type,
        params_dict=params_dict,
        dynamic_axes=dynamic_axes,
        input_names=input_names,
    )

    if training is None or training == _C_onnx.TrainingMode.EVAL:
        params_dict = torch._C._jit_pass_onnx_eval_peephole(graph, params_dict)

    if (
        do_constant_folding
        and GLOBALS.export_onnx_opset_version
        >= _constants.ONNX_CONSTANT_FOLDING_MIN_OPSET
    ):
        params_dict = _C._jit_pass_onnx_constant_fold(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )
        _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    if GLOBALS.onnx_shape_inference:
        _C._jit_pass_onnx_graph_shape_type_inference(
            graph, params_dict, GLOBALS.export_onnx_opset_version
        )

    params_dict = _C._jit_pass_onnx_eliminate_unused_items(graph, params_dict)

    # For ONNX opset < 9, constants only have three data types: float16, float, double.
    # In this pass transform constants of other data types to float/double + cast operator.
    if GLOBALS.export_onnx_opset_version < 9:
        _C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    params_dict = _C._jit_pass_filter_non_tensor_arguments(params_dict)
    _C._jit_decay_packed_param_input_types(graph)

    _C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    model_f: Union[str, io.BytesIO] = io.BytesIO()
    proto, _, _, _ = graph._export_onnx(
        params_dict,
        opset_version,
        dynamic_axes,
        False,
        operator_export_type,
        True,
        True,
        {},
        False,
        "",
        {},
    )

    # NOTE: Input might be dce'ed, so we need to remove those from the input args.
    new_input_names = set(v.debugName() for v in graph.inputs())
    new_input_args = []
    for v, arg in zip(jit_graph.inputs(), input_args):
        if v.debugName() in new_input_names:
            new_input_args.append(arg)
    input_args = tuple(new_input_args)

    with torch.serialization._open_file_like(model_f, "wb") as opened_file:
        opened_file.write(proto)
    ort_session = _ort_session(model_f, ort_providers)
    ort_inputs = _prepare_input_for_ort(
        input_args, {}, flatten=flatten, remained_onnx_input_idx=None
    )
    ort_outs = _run_ort(ort_session, ort_inputs)

    has_mismatch = False
    try:
        _compare_ort_pytorch_outputs(
            ort_outs=ort_outs,
            pt_outs=jit_outs,
            rtol=rtol,
            atol=atol,
            check_shape=check_shape,
            check_dtype=check_dtype,
            ignore_none=ignore_none,
            acceptable_error_percentage=acceptable_error_percentage,
        )
    except AssertionError as e:
        print("Has mismatch: ", e)
        has_mismatch = True

    return has_mismatch, jit_outs


@dataclasses.dataclass
class GraphInfo:
    graph: torch.Graph
    input_args: Tuple[Any, ...]
    params_dict: Dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(
        default_factory=_experimental.ExportOptions
    )
    has_mismatch: Optional[bool] = None
    pt_outs: Optional[Union[_NumericType, Sequence[_NumericType]]] = None
    upper_graph_info: Optional[GraphInfo] = None
    lower_graph_info: Optional[GraphInfo] = None
    tree_tag: str = dataclasses.field(default="")
    _excluded_node_kinds: FrozenSet[str] = dataclasses.field(
        default_factory=lambda: frozenset({"prim::Constant", "prim::ListConstruct"}),
    )

    def clear(self):
        self.has_mismatch = None
        self.pt_outs = None
        self.upper_graph_info = None
        self.lower_graph_info = None

    def essential_node_count(self) -> int:
        return sum(
            1 for n in self.graph.nodes() if n.kind() not in self._excluded_node_kinds
        )

    def mismatch_partition(self) -> Optional[GraphInfo]:
        if not self.has_mismatch:
            return None
        if self.upper_graph_info is not None and self.upper_graph_info.has_mismatch:
            return self.upper_graph_info.mismatch_partition()
        if self.lower_graph_info is not None and self.lower_graph_info.has_mismatch:
            return self.lower_graph_info.mismatch_partition()
        if self.upper_graph_info is None and self.lower_graph_info is None:
            return self
        raise RuntimeError("Mismatch detected, but cannot locate the mismatching op.")

    def _graph_partition_pivot(self) -> int:
        """Find the pivot to partition the graph.

        The pivot is the node that splits the graph into two parts. Each part should
        have the same amount of nodes, excluding primitive ops, such as `prim::Constant`.
        If the graph has an odd number of nodes, the upper part will have one more node.
        If the graph does not have any node that can be partitioned, return -1.

        Returns:
            The index of the pivot node.
        """
        non_const_indices = [
            i
            for i, n in enumerate(self.graph.nodes())
            if n.kind() not in self._excluded_node_kinds
        ]
        half_idx = len(non_const_indices) // 2 - 1
        if half_idx >= 0 and len(non_const_indices) > half_idx:
            return non_const_indices[half_idx] + 1
        return -1

    def _partition_upper_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())

        def _process_node_output_for_upper(
            new_outputs: List[torch.Value], output: torch.Value
        ) -> torch.Value:
            new_outputs.append(output)
            return output

        new_outputs = []
        process_value_for_upper = functools.partial(
            _process_node_output_for_upper, new_outputs
        )
        upper_nodes, lower_nodes, _ = self._partition_nodes(
            graph, pivot, process_value_for_upper
        )

        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)

        for node in reversed(lower_nodes):
            node.destroy()

        for i, input in reversed(list(enumerate(list(graph.inputs())))):
            if not _has_uses_by(input, upper_nodes):
                graph.eraseInput(i)

        return graph

    def _partition_lower_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()
        original_outputs = list(graph.outputs())
        original_inputs = list(graph.inputs())

        new_outputs = []

        def _process_node_output_for_lower(
            graph: torch.Graph, output: torch.Value
        ) -> torch.Value:
            new_input = graph.addInput()
            output.replaceAllUsesWith(new_input)
            new_input.copyMetadata(output)
            return new_input

        process_value_for_lower = functools.partial(
            _process_node_output_for_lower, graph
        )

        upper_nodes, lower_nodes, keep_nodes = self._partition_nodes(
            graph, pivot, process_value_for_lower
        )

        for output in original_outputs:
            if _produced_by(output, lower_nodes):
                new_outputs.append(output)
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)

        for input in original_inputs:
            if _has_uses_by(input, lower_nodes):
                new_input = graph.addInput()
                input.replaceAllUsesWith(new_input)
                new_input.copyMetadata(input)

        for node in reversed(upper_nodes):
            if not node in keep_nodes:
                node.destroy()

        for _ in original_inputs:
            graph.eraseInput(0)

        return graph

    def _partition_node(
        self,
        node: torch.Node,
        upper_nodes_set: Set[torch.Node],
        lower_nodes_set: Set[torch.Node],
        original_graph_outputs: Set[torch.Value],
        nodes_to_duplicate: Set[torch.Node],
        covered_bridge_values: Set[torch.Value],
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ):
        if node in lower_nodes_set:
            return

        if (
            _node_has_uses_by(node, lower_nodes_set)
            and node.kind() in self._excluded_node_kinds
        ):
            lower_nodes_set.add(node)
            nodes_to_duplicate.add(node)
            for input in node.inputs():
                if input in covered_bridge_values:
                    continue
                self._partition_node(
                    input.node(),
                    upper_nodes_set,
                    lower_nodes_set,
                    original_graph_outputs,
                    nodes_to_duplicate,
                    covered_bridge_values,
                    process_bridge_value,
                )
        else:
            for output in node.outputs():
                if output in covered_bridge_values:
                    continue
                if (
                    _has_uses_by(output, lower_nodes_set)
                    or output in original_graph_outputs
                ):
                    covered_bridge_values.add(process_bridge_value(output))

    def _partition_nodes(
        self,
        graph: torch.Graph,
        pivot: int,
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ) -> Tuple[List[torch.Node], List[torch.Node], Set[torch.Node]]:
        nodes = list(graph.nodes())
        upper_nodes = nodes[:pivot]
        lower_nodes = nodes[pivot:]
        upper_nodes_set = set(upper_nodes)
        lower_nodes_set = set(lower_nodes)
        original_graph_outputs = set(graph.outputs())
        nodes_to_duplicate = set()
        covered_bridge_values = set()
        for node in upper_nodes:
            self._partition_node(
                node,
                upper_nodes_set,
                lower_nodes_set,
                original_graph_outputs,
                nodes_to_duplicate,
                covered_bridge_values,
                process_bridge_value,
            )
        return upper_nodes, lower_nodes, nodes_to_duplicate

    def _bridge_kwargs(self):
        pt_outs = self.pt_outs
        if pt_outs is None:
            raise RuntimeError("pt_outs is not set")
        if not isinstance(pt_outs, (list, tuple)):
            pt_outs = [pt_outs]
        graph_outputs = list(self.graph.outputs())
        # TODO: Handle diff caused by prim::TupleConstruct
        assert len(graph_outputs) == len(
            pt_outs
        ), f"{len(graph_outputs)} vs {len(pt_outs)}"
        return {v.debugName(): o for v, o in zip(graph_outputs, pt_outs)}

    def _args_and_params_for_partition_graph(
        self,
        graph: torch.Graph,
        bridge_kwargs: Mapping[str, Union[_NumericType, Sequence[_NumericType]]],
        full_kwargs: Mapping[str, torch.Tensor],
        full_params: Mapping[str, torch.Tensor],
    ):
        input_names = [input.debugName() for input in graph.inputs()]
        args = tuple(bridge_kwargs[k] for k in input_names if k in bridge_kwargs)
        args += tuple(full_kwargs[k] for k in input_names if k in full_kwargs)
        params = {k: full_params[k] for k in input_names if k in full_params}
        assert len(args) + len(params) == len(input_names)
        return args, params

    def find_first_mismatch(
        self,
        flatten: bool = True,
        ignore_none: bool = True,
        check_shape: bool = True,
        check_dtype: bool = True,
        ort_providers: Sequence[str] = _ORT_PROVIDERS,
        rtol: float = 0.001,
        atol: float = 1e-7,
        acceptable_error_percentage: Optional[float] = None,
    ):
        verification_kwargs = {
            "flatten": flatten,
            "ignore_none": ignore_none,
            "check_shape": check_shape,
            "check_dtype": check_dtype,
            "ort_providers": ort_providers,
            "rtol": rtol,
            "atol": atol,
            "acceptable_error_percentage": acceptable_error_percentage,
        }
        if self.export_options.verbose:
            print(self.graph)

        if len(list(self.graph.outputs())) == 0:
            return

        assert len(self.input_args) + len(self.params_dict) == len(
            list(self.graph.inputs())
        ), (
            f"Number of graph inputs({len(list(self.graph.inputs()))}) does not match "
            f"the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)})."
        )

        # make copy because altered by export process.
        jit_graph = self.graph.copy()
        self.has_mismatch, self.pt_outs = verify_aten_graph(
            jit_graph,
            input_args=self.input_args,
            params_dict=self.params_dict,
            export_options=self.export_options,
            **verification_kwargs,
        )

        if not self.has_mismatch:
            print(f"No mismatch found in graph {self.tree_tag}".center(80, "-"))
            return
        print(f"Has mismatch in graph {self.tree_tag}".center(80, "-"))

        if self.essential_node_count() <= 1:
            # No nodes to partition.
            print(self.graph)
            print(f"Found mismatch in graph {self.tree_tag}".center(80, "-"))
            return

        full_kwargs = {
            k.debugName(): v for k, v in zip(self.graph.inputs(), self.input_args)
        }
        full_params = self.params_dict

        upper_graph = self._partition_upper_graph()
        upper_args, upper_params = self._args_and_params_for_partition_graph(
            upper_graph, {}, full_kwargs, full_params
        )
        self.upper_graph_info = GraphInfo(
            upper_graph,
            upper_args,
            upper_params,
            self.export_options,
            tree_tag=self.tree_tag + "0",
        )

        print(f"Check upper graph {self.upper_graph_info.tree_tag}".center(80, "-"))
        self.upper_graph_info.find_first_mismatch(**verification_kwargs)

        if self.upper_graph_info.has_mismatch:
            return

        bridge_kwargs = self.upper_graph_info._bridge_kwargs()
        lower_graph = self._partition_lower_graph()
        lower_args, lower_params = self._args_and_params_for_partition_graph(
            lower_graph, bridge_kwargs, full_kwargs, full_params
        )
        self.lower_graph_info = GraphInfo(
            lower_graph,
            lower_args,
            lower_params,
            self.export_options,
            tree_tag=self.tree_tag + "1",
        )

        print(f"Check lower graph {self.lower_graph_info.tree_tag}".center(80, "-"))
        self.lower_graph_info.find_first_mismatch(**verification_kwargs)

        assert self.lower_graph_info.has_mismatch, (
            "Mismatch found in graph, but not found in neither partition."
            "Please try tightening the tolerance."
        )


def _has_uses_by(value: torch.Value, nodes: Collection[torch.Node]):
    if any(use.user in nodes for use in value.uses()):
        return True
    return False


def _node_has_uses_by(node: torch.Node, nodes: Collection[torch.Node]):
    for output in node.outputs():
        if _has_uses_by(output, nodes):
            return True
    return False


def _produced_by(value: torch.Value, nodes: Collection[torch.Node]):
    return value.node() in nodes


# TODO:
# * Find all op level mismatches.
# * kwargs, default values.
# * unify argument preprocessing (flatten, ignore_none, etc.)
# * Clean ups for util.py.
@_beartype.beartype
def verify_ops(
    model: Union[torch.nn.Module, torch.jit.ScriptModule],
    input_args: Tuple[Any, ...],
    do_constant_folding: bool = True,
    training: torch.onnx.TrainingMode = torch.onnx.TrainingMode.EVAL,
    opset_version: Optional[int] = None,
    keep_initializers_as_inputs: bool = True,
    verbose: bool = False,
    flatten: bool = True,
    ignore_none: bool = True,
    check_shape: bool = True,
    check_dtype: bool = True,
    ort_providers: Sequence[str] = _ORT_PROVIDERS,
    rtol: float = 0.001,
    atol: float = 1e-7,
    acceptable_error_percentage: Optional[float] = None,
    **_,
) -> Optional[GraphInfo]:
    if opset_version is None:
        opset_version = _constants.ONNX_DEFAULT_OPSET
    """From aten graph, do binary search on graph partition to find operator export discrepancy."""
    # TODO: Copied from utils.py `export` until `_optimize_graph`.
    if training == torch.onnx.TrainingMode.TRAINING:
        model.train()
    elif training == torch.onnx.TrainingMode.EVAL:
        model.eval()
    with torch.no_grad():
        inputs_for_export = _prepare_input_for_export(input_args, {})
        args = utils._decide_input_format(model, inputs_for_export)

        model = utils._pre_trace_quant_model(model, args)
        graph, params, torch_out, module = utils._create_jit_graph(model, args)
        params_dict = utils._get_named_param_dict(graph, params)

        utils._apply_friendly_debug_names(graph, params_dict)
        jit_graph = graph.copy()

        graph_info = GraphInfo(
            jit_graph,
            input_args,
            params_dict,
            _experimental.ExportOptions(
                do_constant_folding=do_constant_folding,
                training=training,
                opset_version=opset_version,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                verbose=verbose,
            ),
        )
        graph_info.find_first_mismatch(
            flatten=flatten,
            ignore_none=ignore_none,
            check_shape=check_shape,
            check_dtype=check_dtype,
            ort_providers=ort_providers,
            rtol=rtol,
            atol=atol,
            acceptable_error_percentage=acceptable_error_percentage,
        )

        return graph_info.mismatch_partition()
