from __future__ import absolute_import, division, print_function, unicode_literals

r"""
The torch.onnx module contains functions to export models into the ONNX
IR format.  These models can be loaded with the ONNX library and then
converted to models which run on other deep learning frameworks.
"""

import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
from torch._six import container_abcs
import contextlib
import numbers
import warnings
from torch._six import string_classes
from torch.jit import _unique_state_dict
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
from torch._C import ListType, _propagate_and_assign_input_shapes, _assign_output_shapes


# the flag to tell the user whether it's in the middle of ONNX export or not
__IN_ONNX_EXPORT = False


def is_in_onnx_export():
    global __IN_ONNX_EXPORT
    return __IN_ONNX_EXPORT


@contextlib.contextmanager
def set_training(model, mode):
    if mode is None:
        yield
        return
    old_mode = model.training
    if old_mode != mode:
        model.train(mode)
    try:
        yield
    finally:
        if old_mode != mode:
            model.train(old_mode)


def export(model, args, f, export_params=True, verbose=False, training=False,
           input_names=None, output_names=None, aten=False, export_raw_ir=False,
           operator_export_type=None, opset_version=None, _retain_param_name=True,
           do_constant_folding=False, example_outputs=None, strip_doc_string=True,
           dynamic_axes=None, keep_initializers_as_inputs=None):
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
            operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
        else:
            operator_export_type = OperatorExportTypes.ONNX
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
            operator_export_type=operator_export_type, opset_version=opset_version,
            _retain_param_name=_retain_param_name, do_constant_folding=do_constant_folding,
            example_outputs=example_outputs, strip_doc_string=strip_doc_string,
            dynamic_axes=dynamic_axes, keep_initializers_as_inputs=keep_initializers_as_inputs)


# ONNX can't handle constants that are lists of tensors, which can
# get generated in constant prop. So we split them back into prim::ListConstructs
def _split_tensor_list_constants(g, block):
    for node in block.nodes():
        for subblock in node.blocks():
            _split_tensor_list_constants(g, subblock)
        if node.kind() == "prim::Constant":
            output_type = node.output().type()
            if output_type.isSubtypeOf(ListType.ofTensors()):
                inputs = [g.create("prim::Constant").t_('value', t)
                           .insertBefore(node).output()
                          for t in node['value']]
                lc = (g.create("prim::ListConstruct", inputs)
                      .insertBefore(node)
                      .output()
                      .setType(ListType.ofTensors()))
                node.output().replaceAllUsesWith(lc)


def _optimize_graph(graph, operator_export_type, _disable_torch_constant_prop=False, fixed_batch_size=False):
    # Inline everyting
    torch._C._jit_pass_inline(graph)

    # Remove fork/wait nodes
    torch._C._jit_pass_inline_fork_wait(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_remove_inplace_ops(graph)
    # we record now record some ops like ones/zeros
    # into a trace where we previously recorded constants
    # use constant prop to maintain our current level of onnx support
    # without implementing symbolics for all of them
    if _disable_torch_constant_prop is False:
        torch._C._jit_pass_constant_propagation(graph)
    _split_tensor_list_constants(graph, graph)
    # run dce to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_canonicalize_ops(graph)
    torch._C._jit_pass_lint(graph)

    torch._C._jit_pass_peephole(graph, True)
    torch._C._jit_pass_lint(graph)

    if operator_export_type != OperatorExportTypes.RAW:
        # onnx does not support tuples, so try to remove them
        torch._C._jit_pass_lower_all_tuples(graph)
        torch._C._jit_pass_peephole(graph, True)
        torch._C._jit_pass_lint(graph)

        # onnx only supports tensors, but 1 / 2 = 0.5 and tensor(1) / tensor(2) = 0
        torch._C._jit_pass_prepare_division_for_onnx(graph)

        torch._C._jit_pass_onnx_remove_print(graph)

        torch._C._jit_pass_onnx_preprocess_caffe2(graph)

        # onnx only supports tensors, so we turn all out number types into tensors
        torch._C._jit_pass_erase_number_types(graph)

        graph = torch._C._jit_pass_onnx(graph, operator_export_type)
        torch._C._jit_pass_lint(graph)

        torch._C._jit_pass_onnx_scalar_type_analysis(graph)
        torch._C._jit_pass_lint(graph)

        from torch.onnx.symbolic_helper import _export_onnx_opset_version
        torch._C._jit_pass_onnx_peephole(graph, _export_onnx_opset_version, fixed_batch_size)
        torch._C._jit_pass_lint(graph)

    # graph is not a valid jit graph anymore because types have been replaced
    # (e.g. int with Tensor), so it now contains operators that don't actually
    # exist. We can't run normal dead code elimination because it'd fail trying
    # to look up if an operator has side effects, but we can run a dead code
    # elimination variant that doesn't need to look up if an op has side effects.
    torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_fixup_onnx_loops(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph


# We accept dictionnaries and strings as ONNX inputs,
# but they should be only for configuration use.
# we detect here if these inputs are modified, and if so
# we warn the user that the changes won't take effect in the
# traced ONNX graph
def warn_on_static_input_change(input_states):
    for input, traced_input in zip(input_states[0], input_states[1]):
        if isinstance(input, dict):
            if list(input.keys()) != list(traced_input.keys()):
                warning = "We detected that you are modifying a dictionnary that is an input to your " \
                          "model. " \
                          "Note that dictionaries are allowed as inputs in ONNX but they should be " \
                          "handled with care. " \
                          "Usages of dictionaries is not recommended, and should not be used except " \
                          "for configuration use. " \
                          "Also note that the order and values of the keys must remain the same. "
                warnings.warn(warning)
        elif isinstance(input, str):
            if input != traced_input:
                warning = "The model seems to have string inputs/outputs. " \
                          "Note that strings will not appear as inputs/outputs of the ONNX graph. "
                warnings.warn(warning)

def _decide_keep_init_as_input(keep_initializers_as_inputs, operator_export_type,
                               opset_version):
    # This method encapsulates the logic to decide whether the initializers in the graph
    # should be listed as ONNX graph inputs (i.e., whether to choose ONNX IR v3 or v4).
    # If keep_initializers_as_inputs is not specified (None), then we decide whether to keep
    # intializers as graph inputs (val_keep_init_as_ip) based on export type. If export type
    # is ONNX, then do not keep initializers as input (val_keep_init_as_ip=False). For all other 
    # export types keep initializers as input (val_keep_init_as_ip=True).
    # If keep_initializers_as_inputs is specified, then respect it. Unless opset version <= 8,
    # in which case it must be ignored because for opset version <= 8, all initializers MUST be
    # part of graph input (only ONNX IR v3 is allowed), i.e. val_keep_init_as_ip=True.

    # Special handling is needed for opset version 8 or lower, because irrespective
    # of user input for keep_initializers_as_inputs, the graph must follow ONNX IR v3
    # semantics, i.e. all intializers must be listed as ONNX graph input.
    if opset_version < 9:
        if keep_initializers_as_inputs is False:
            warnings.warn("Setting 'keep_initializers_as_inputs=False' for opset version"
                          "8 or lower would lead to an invalid ONNX graph. Therefore, "
                          "'keep_initializers_as_inputs=False' is ignored during export."
                          "Exported model will have initialiers as graph inputs (compliant "
                          " to ONNX IR v3).")
        return True  # i.e. True == initializers are part of graph input (ONNX IR v3)
    val_keep_init_as_ip = True if keep_initializers_as_inputs is None else keep_initializers_as_inputs
    if keep_initializers_as_inputs is None and operator_export_type is OperatorExportTypes.ONNX:
        val_keep_init_as_ip = False
    return val_keep_init_as_ip

def _trace(func, args, operator_export_type, return_outs=False):
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(func, args, _force_outplace=True, _return_inputs_states=True)
    warn_on_static_input_change(inputs_states)

    trace_graph = _optimize_graph(trace_graph, operator_export_type)
    if return_outs:
        return trace_graph, torch_out
    return trace_graph


def _trace_and_get_graph_from_model(model, args, training):

    # A basic sanity check: make sure the state_dict keys are the same
    # before and after running the model.  Fail fast!
    orig_state_dict_keys = _unique_state_dict(model).keys()

    # By default, training=False, which is good because running a model in
    # training mode could result in internal buffers getting updated, dropout
    # getting applied, etc.  If you really know what you're doing, you
    # can turn training=True (or None, to preserve whatever the original
    # training mode was.)
    with set_training(model, training):
        trace_graph, torch_out, inputs_states = torch.jit._get_trace_graph(model, args, _force_outplace=True, _return_inputs_states=True)
        warn_on_static_input_change(inputs_states)

    if orig_state_dict_keys != _unique_state_dict(model).keys():
        raise RuntimeError("state_dict changed after running the tracer; "
                           "something weird is happening in your model!")

    return trace_graph, torch_out


def _model_to_graph(model, args, verbose=False, training=False,
                    input_names=None, output_names=None,
                    operator_export_type=OperatorExportTypes.ONNX,
                    example_outputs=None, propagate=False,
                    _retain_param_name=False, do_constant_folding=False,
                    _disable_torch_constant_prop=False, fixed_batch_size=False):
    from torch.onnx.symbolic_helper import _export_onnx_opset_version
    # Special case for common case of passing a single Tensor
    if isinstance(args, torch.Tensor):
        args = (args, )

    if isinstance(example_outputs, torch.Tensor):
        example_outputs = [example_outputs]

    torch_out = None

    if isinstance(model, torch.jit.ScriptModule):
        assert example_outputs is not None, "example_outputs must be provided when exporting a ScriptModule"
        try:
            method_graph, params = torch._C._jit_pass_lower_graph(model.forward.graph, model._c)
            in_vars, in_desc = torch.jit._flatten(tuple(args) + tuple(params))
            graph = _propagate_and_assign_input_shapes(
                method_graph, tuple(in_vars), False, propagate)
        except AttributeError:
            raise RuntimeError('\'forward\' method must be a script method')
    elif isinstance(model, torch.jit.ScriptFunction):
        assert example_outputs is not None, "example_outputs must be provided when exporting a TorchScript ScriptFunction"
        method = model
        params = ()
        in_vars, in_desc = torch.jit._flatten(tuple(args))
        graph = _propagate_and_assign_input_shapes(
            model.graph, tuple(in_vars), False, propagate)
    else:
        graph, torch_out = _trace_and_get_graph_from_model(model, args, training)
        state_dict = _unique_state_dict(model)
        params = list(state_dict.values())
        if _retain_param_name:
            graph_inputs = list(graph.inputs())
            user_input_num = len(graph_inputs) - len(state_dict)
            param_names = list(state_dict.keys())
            for i, inp in enumerate(graph_inputs):
                if i >= user_input_num:
                    inp.setDebugName(param_names[i - user_input_num])

    graph = _optimize_graph(graph, operator_export_type,
                            _disable_torch_constant_prop=_disable_torch_constant_prop,
                            fixed_batch_size=fixed_batch_size)

    if isinstance(model, torch.jit.ScriptModule) or isinstance(model, torch.jit.ScriptFunction):
        out_vars, _ = torch.jit._flatten(tuple(example_outputs))
        graph = _assign_output_shapes(graph, out_vars)

    # NB: ONNX requires complete information about output types, which might be
    # erased by some optimizations, so we need to set it explicitly again.
    if torch_out is not None:
        output_tensors, _ = torch._C._jit_flatten(torch_out)
        for output, tensor in zip(graph.outputs(), output_tensors):
            output.inferTypeFrom(tensor)

    _set_input_and_output_names(graph, input_names, output_names)

    # make sure that the param dict and the graph match each other
    flatten_args, _ = torch._C._jit_flatten(args)
    assert len(params) + len(flatten_args) == sum(1 for _ in graph.inputs())

    input_and_param_names = [val.debugName() for val in graph.inputs()]
    param_names = input_and_param_names[len(input_and_param_names) - len(params):]
    params_dict = dict(zip(param_names, params))

    if do_constant_folding and _export_onnx_opset_version in [9, 10, 11]:
        params_dict = torch._C._jit_pass_onnx_constant_fold(graph, params_dict,
                                                            _export_onnx_opset_version)
        torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects(graph)

    # For ONNX opset < 9, constants only have three data types: float16, float, double.
    # In this pass transform constants of other data types to float/double + cast operator.
    if _export_onnx_opset_version < 9:
        torch._C._jit_pass_onnx_cast_all_constant_to_floating(graph)

    if verbose:
        print(graph)

    return graph, params_dict, torch_out


def export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                            input_names=None, output_names=None, aten=False, export_raw_ir=False,
                            operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE,
                            example_outputs=None, propagate=False, google_printer=False,
                            opset_version=None, _retain_param_name=True,
                            keep_initializers_as_inputs=None):
    if aten or export_raw_ir:
        assert operator_export_type is None
        assert aten ^ export_raw_ir
        operator_export_type = OperatorExportTypes.ATEN if aten else OperatorExportTypes.RAW
    elif operator_export_type is None:
        operator_export_type = OperatorExportTypes.ONNX
    return _export_to_pretty_string(model, args, f, export_params, verbose, training,
                                    input_names, output_names, operator_export_type,
                                    export_type, example_outputs, propagate, google_printer,
                                    opset_version, _retain_param_name,
                                    keep_initializers_as_inputs=keep_initializers_as_inputs)


def _export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False,
                             input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX,
                             export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False,
                             google_printer=False, opset_version=None, _retain_param_name=False,
                             do_constant_folding=False, keep_initializers_as_inputs=None, fixed_batch_size=False):
    from torch.onnx.symbolic_helper import _default_onnx_opset_version, _set_opset_version
    from torch.onnx.symbolic_helper import _set_operator_export_type
    if opset_version is None:
        opset_version = _default_onnx_opset_version
    _set_opset_version(opset_version)
    _set_operator_export_type(operator_export_type)
    val_keep_init_as_ip = _decide_keep_init_as_input(keep_initializers_as_inputs,
                                                     operator_export_type,
                                                     opset_version)
    graph, params_dict, torch_out = _model_to_graph(model, args, verbose,
                                                    training, input_names,
                                                    output_names, operator_export_type,
                                                    example_outputs, propagate, _retain_param_name,
                                                    do_constant_folding, fixed_batch_size=fixed_batch_size)

    return graph._pretty_print_onnx(params_dict, opset_version, False,
                                    operator_export_type, google_printer,
                                    val_keep_init_as_ip)


# NOTE: the output `torch_out` will contain the output tensors resulting from
# the trace of a Module. In the case that a torch.nn.ScriptModule is passed in,
# this output will be None, since we are not doing any tracing but rather
# directly extracting the graph.
def _export(model, args, f, export_params=True, verbose=False, training=False,
            input_names=None, output_names=None, operator_export_type=None,
            export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False,
            opset_version=None, _retain_param_name=False, do_constant_folding=False,
            strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, fixed_batch_size=False):
    if isinstance(model, torch.nn.DataParallel):
        raise ValueError('torch.nn.DataParallel is not supported by ONNX '
                         'exporter, please use \'attribute\' module to '
                         'unwrap model from torch.nn.DataParallel. Try '
                         'torch.onnx.export(model.module, ...)')
    global __IN_ONNX_EXPORT
    assert __IN_ONNX_EXPORT is False
    __IN_ONNX_EXPORT = True
    try:
        from torch.onnx.symbolic_helper import _default_onnx_opset_version, _set_opset_version
        from torch.onnx.symbolic_helper import _set_operator_export_type
        if opset_version is None:
            opset_version = _default_onnx_opset_version
        if not operator_export_type:
            if torch.onnx.PYTORCH_ONNX_CAFFE2_BUNDLE:
                operator_export_type = OperatorExportTypes.ONNX_ATEN_FALLBACK
            else:
                operator_export_type = OperatorExportTypes.ONNX
        _set_opset_version(opset_version)
        _set_operator_export_type(operator_export_type)
        val_keep_init_as_ip = _decide_keep_init_as_input(keep_initializers_as_inputs,
                                                         operator_export_type,
                                                         opset_version)
        graph, params_dict, torch_out = _model_to_graph(model, args, verbose,
                                                        training, input_names,
                                                        output_names, operator_export_type,
                                                        example_outputs, propagate,
                                                        _retain_param_name, do_constant_folding,
                                                        fixed_batch_size=fixed_batch_size)

        # TODO: Don't allocate a in-memory string for the protobuf
        defer_weight_export = export_type is not ExportTypes.PROTOBUF_FILE
        if dynamic_axes is None:
            dynamic_axes = {}

        _validate_dynamic_axes(dynamic_axes, model, input_names, output_names)

        if export_params:
            proto, export_map = graph._export_onnx(
                params_dict, opset_version, dynamic_axes, defer_weight_export,
                operator_export_type, strip_doc_string, val_keep_init_as_ip)
        else:
            proto, export_map = graph._export_onnx(
                {}, opset_version, dynamic_axes, False, operator_export_type,
                strip_doc_string, val_keep_init_as_ip)

        if export_type == ExportTypes.PROTOBUF_FILE:
            assert(len(export_map) == 0)
            with torch.serialization._open_file_like(f, 'wb') as opened_file:
                opened_file.write(proto)
        elif export_type in [ExportTypes.ZIP_ARCHIVE, ExportTypes.COMPRESSED_ZIP_ARCHIVE]:
            import zipfile
            compression = zipfile.ZIP_DEFLATED \
                if export_type == ExportTypes.COMPRESSED_ZIP_ARCHIVE \
                else zipfile.ZIP_STORED
            with zipfile.ZipFile(f, 'w', compression=compression) as z:
                z.writestr(ONNX_ARCHIVE_MODEL_PROTO_NAME, proto)
                for k, v in export_map.items():
                    z.writestr(k, v)
        elif export_type == ExportTypes.DIRECTORY:
            import os
            if os.path.exists(f):
                assert(os.path.isdir(f))
            else:
                os.makedirs(f)

            model_proto_file = os.path.join(f, ONNX_ARCHIVE_MODEL_PROTO_NAME)
            with torch.serialization._open_file_like(model_proto_file, 'wb') as opened_file:
                opened_file.write(proto)

            for k, v in export_map.items():
                weight_proto_file = os.path.join(f, k)
                with torch.serialization._open_file_like(weight_proto_file, 'wb') as opened_file:
                    opened_file.write(v)
        else:
            raise RuntimeError('Unknown export type')
    finally:
        assert __IN_ONNX_EXPORT
        __IN_ONNX_EXPORT = False
    return torch_out


def _set_input_and_output_names(graph, input_names, output_names):
    def set_names(node_list, name_list, descriptor):
        if name_list is None:
            return
        if len(name_list) > len(node_list):
            raise RuntimeError(
                "number of %s names provided (%d) exceeded number of %ss (%d)"
                % (descriptor, len(name_list), descriptor, len(node_list)))
        for name, node in zip(name_list, node_list):
            if node.debugName() != name:
                node.setDebugName(name)
    set_names(list(graph.inputs()), input_names, 'input')
    set_names(list(graph.outputs()), output_names, 'output')

attr_pattern = re.compile("^(.+)_([ifstgz])$")


def _run_symbolic_method(op_name, symbolic_fn, args):
    r"""
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    try:
        return symbolic_fn(*args)
    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch
        # to symbolic_fn.  Otherwise, the backtrace will have the clues
        # you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


def _is_onnx_list(value):
    if not isinstance(value, string_classes) and \
            not isinstance(value, torch.Tensor) and \
            isinstance(value, container_abcs.Iterable):
        return True
    return False


def _add_attribute(node, key, value, aten):
    r""" initializes the right attribute based on type of value """
    m = attr_pattern.match(key)
    if m is None:
        raise IndexError((
            "Invalid attribute specifier '{}' names " +
            " must be suffixed with type, e.g. 'dim_i' or 'dims_i'").format(key))
    name, kind = m.group(1), m.group(2)
    if _is_onnx_list(value):
        kind += "s"
    if aten:
        if isinstance(value, torch.Tensor):
            # Caffe2 proto does not support tensor attribute.
            if value.numel() > 1:
                raise ValueError("Should not pass tensor attribute")
            value = _scalar(value)
            if isinstance(value, float):
                kind = "f"
            else:
                kind = "i"
    return getattr(node, kind + "_")(name, value)


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]


def _newNode(g, opname, outputs, *args, **kwargs):
    if "::" in opname:
        aten = False
        ns_opname = opname
    else:
        aten = kwargs.pop("aten", False)
        ns = "aten" if aten else "onnx"
        ns_opname = ns + "::" + opname
    n = g.create(ns_opname, args, outputs)
    for k, v in sorted(kwargs.items()):
        # TODO: enable inplace in aten exporting mode.
        if k == "inplace":
            continue
        _add_attribute(n, k, v, aten=aten)
    return n


def _graph_op(g, opname, *raw_args, **kwargs):
    r"""
    Create an ONNX operator 'opname', taking 'args' as inputs and attributes
    'kwargs'; returning the node representing the single output of this operator
    (see the `outputs` keyword argument for multi-return nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
    outputs = kwargs.pop('outputs', 1)

    # Filter out None attributes, this can be convenient client side because
    # now they can pass through None attributes, and have them not show up
    kwargs = dict((k, v) for k, v in kwargs.items() if v is not None)

    def const_if_tensor(arg):
        if arg is None:
            return arg
        elif isinstance(arg, torch._C.Value):
            return arg
        else:
            return g.op("Constant", value_z=arg)

    args = list(const_if_tensor(arg) for arg in raw_args)
    n = g.insertNode(_newNode(g, opname, outputs, *args, **kwargs))
    if outputs == 1:
        return n.output()
    return tuple(o for o in n.outputs())


# Note [Export inplace]
# ~~~~~~~~~~~~~~~~~~~~~
# In abstract, it would be better for us to export inplace annotations,
# than to not export them, since it is useful information that can
# help the target of an ONNX export export more efficiently.  However,
# ONNX doesn't currently formalize inplace. Fortunately, it's sound to drop
# inplace annotations, but we are losing information this way.


def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    # NB: Returning None means the node gets cloned as is into
    # the new graph
    try:
        from torch.onnx.symbolic_helper import _export_onnx_opset_version as opset_version
        import torch.onnx.symbolic_registry as sym_registry

        if operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK:
            sym_registry.register_quantized_ops('caffe2', opset_version)
        sym_registry.register_version('', opset_version)

        # See Note [Export inplace]
        # TODO: I think this is not necessary anymore
        if n.kind().endswith('_'):
            ns_op_name = n.kind()[:-1]
        else:
            ns_op_name = n.kind()
        ns, op_name = ns_op_name.split("::")
        if ns == "onnx":
            # Use the original node directly
            return None

        elif ns == "aten":
            is_exportable_aten_op = sym_registry.is_registered_op(op_name, '', opset_version)
            is_onnx_aten_export = operator_export_type == OperatorExportTypes.ONNX_ATEN
            is_aten_fallback_export = operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK
            if is_onnx_aten_export or (not is_exportable_aten_op and is_aten_fallback_export):
                # Direct ATen export requested
                attrs = {k + "_" + n.kindOf(k)[0]: n[k] for k in n.attributeNames()}
                outputs = n.outputsSize()
                attrs["outputs"] = outputs
                return _graph_at(g, op_name, *inputs, aten=True, **attrs)

            else:
                # Export it regularly
                attrs = {k: n[k] for k in n.attributeNames()}
                if not is_exportable_aten_op:
                    warnings.warn("ONNX export failed on ATen operator {} because "
                                  "torch.onnx.symbolic_opset{}.{} does not exist"
                                  .format(op_name, opset_version, op_name))
                op_fn = sym_registry.get_registered_op(op_name, '', opset_version)
                return op_fn(g, *inputs, **attrs)

        elif ns == "prim":
            if op_name == "Constant" and not n.mustBeNone():
                if n.kindOf("value") == "t":
                    return g.op("Constant", value_t=n["value"])
                if n.kindOf("value") == "s":
                    return g.op("Constant", value_s=n["value"])
                elif n.kindOf("value") == "is":
                    value = torch.stack([torch.tensor(v) for v in n["value"]]) if n["value"] else []
                    return g.op("Constant", value_t=value)
                elif n.output().type().kind() == "DeviceObjType":
                    return None
                else:
                    raise RuntimeError("Unsupported prim::Constant kind: `{}`. Send a bug report.".format(
                        n.kindOf("value")))
            elif n.mustBeNone() or op_name == "ListConstruct" or op_name == "ListUnpack":
                # None is not an ONNX operator; keep it as None
                # let the exporter handle finally eliminating these

                # For ListConstruct/ListUnpack, it will be erased in the ONNX peephole pass
                return None
            elif op_name == 'Loop' or op_name == 'If':
                new_op_outputs = g.op(op_name, *inputs, outputs=n.outputsSize())
                new_node = new_op_outputs[0].node() if n.outputsSize() > 1 else new_op_outputs.node()
                for b in n.blocks():
                    new_block = new_node.addBlock()
                    torch._C._jit_pass_onnx_block(b, new_block, operator_export_type, env)
                return new_op_outputs
            else:
                # TODO: we sould lift prim's symbolic out
                symbolic_name = 'prim_' + op_name
                is_exportable = sym_registry.is_registered_op(symbolic_name, '', opset_version)
                if not is_exportable:
                    warnings.warn("ONNX export failed on primitive operator {}; please report a bug".format(op_name))
                symbolic_fn = sym_registry.get_registered_op(symbolic_name, '', opset_version)
                attrs = {k: n[k] for k in n.attributeNames()}
                return symbolic_fn(g, *inputs, **attrs)

        elif ns == "quantized":
            domain = ''
            if operator_export_type == OperatorExportTypes.ONNX_ATEN_FALLBACK:
                domain = 'caffe2'
            attrs = {k: n[k] for k in n.attributeNames()}

            if not sym_registry.is_registered_op(op_name, domain, opset_version):
                warnings.warn("ONNX export failed on quantized operator {}::{} because "
                              "torch.onnx.symbolic_opset{}.{} does not exist. "
                              .format(ns, op_name, opset_version, op_name))
            op_fn = sym_registry.get_registered_op(op_name, domain, opset_version)
            return op_fn(g, *inputs, **attrs)

        # custom ops
        elif sym_registry.is_registered_version(ns, opset_version):
            if not sym_registry.is_registered_op(op_name, ns, opset_version):
                warnings.warn("ONNX export failed on custom operator {}::{} because "
                              "torch.onnx.symbolic_opset{}.{} does not exist. "
                              "Have you registered your symbolic function with "
                              "torch.onnx.register_custom_op_symbolic(symbolic_name, symbolic_fn)?"
                              .format(ns, op_name, opset_version, op_name))
            symbolic_fn = sym_registry.get_registered_op(op_name, ns, opset_version)
            attrs = {k: n[k] for k in n.attributeNames()}
            return symbolic_fn(g, *inputs, **attrs)

        else:
            warnings.warn("ONNX export failed on an operator with unrecognized namespace {}::{}; "
                          "If you are trying to export a custom operator, make sure you registered "
                          "it with the right domain and version."
                          "Otherwise please report a bug".format(ns, op_name))
            return None

    except TypeError as e:
        # Handle the specific case where we didn't successfully dispatch.
        # Otherwise, the backtrace will have the clues you need.
        e.args = ("{} (occurred when translating {})".format(e.args[0], op_name), )
        raise


# Generate an ONNX ATen op node.
def _graph_at(g, opname, *args, **kwargs):
    return g.op("ATen", *args, operator_s=opname, **kwargs)


# This helper function can create either constant tensor or constant scalar.
# If dims is None or 0 or [0], generate a 0-d tensor (scalar).
#
# TODO: We might not need this anymore, since most scalars now show up
# as tensors
def _graph_constant(g, value, dims, type, *args, **kwargs):
    assert isinstance(value, numbers.Number)
    assert type is not None
    isscalar = False
    if dims is None or dims == 0 or set(dims) == set([0]):
        dims = [1]
        isscalar = True
    type = type.lower()
    if type == "char":
        tensor = torch.CharTensor(*dims)
    elif type == "short":
        tensor = torch.ShortTensor(*dims)
    elif type == "int":
        tensor = torch.IntTensor(*dims)
    elif type == "long":
        tensor = torch.LongTensor(*dims)
    elif type == "half":
        tensor = torch.HalfTensor(*dims)
    elif type == "float":
        tensor = torch.FloatTensor(*dims)
    elif type == "double":
        tensor = torch.DoubleTensor(*dims)
    else:
        raise ValueError("Unknown type, type should be one of the following strings: "
                         "char, short, int, long, half, float, double")
    tensor.fill_(value)
    if isscalar:
        return g.op("Constant", *args, value_z=tensor, **kwargs)
    return g.op("Constant", *args, value_t=tensor, **kwargs)


def _node_getitem(self, k):
    r"""
    Accessor for attributes of a node which is polymorphic over
    return type.

    NB: This is monkey-patched onto Node.
    """
    sel = self.kindOf(k)
    return getattr(self, sel)(k)


def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    if not bool(re.match(r"^[a-zA-Z0-9-_]*::[a-zA-Z]+[a-zA-Z0-9-_]*$", symbolic_name)):
        raise RuntimeError("Failed to register operator {}. \
                           The symbolic name must match the format Domain::Name, \
                           and sould start with a letter and contain only \
                           alphanumerical characters"
                           .format(symbolic_name))
    ns, op_name = symbolic_name.split('::')
    unaccepted_domain_names = ["onnx", "aten", "prim"]
    if ns in unaccepted_domain_names:
        raise RuntimeError("Failed to register operator {}. The domain {} is already a used domain."
                           .format(symbolic_name, ns))
    import torch.onnx.symbolic_registry as sym_registry
    sym_registry.register_op(op_name, symbolic_fn, ns, opset_version)

# This helper function ensures dynamic axes argument is following the expected format
def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names):
    if len(dynamic_axes) == 0:
        return

    if(hasattr(model, 'graph')):
        # Extracting set of valid input/output names that shall be used for dynamic_axes
        if (input_names is None) or len(input_names) == 0:
            input_names = [x.debugName() for x in model.graph.inputs()]
        if (output_names is None) or len(output_names) == 0:
            output_names = [y.debugName() for y in model.graph.outputs()]

    valid_names = set((input_names or []) + (output_names or []))

    # If dynamic axes are provided as a list rather than dictionary, they should
    # first get converted to a dictionary in expected format. If desired axes names
    # are not provided for dynamic axes, automatic names shall be generated for
    # provided dynamic axes of specified input/output
    for key, value in dynamic_axes.items():
        if key not in valid_names:
            warnings.warn("Provided key {} for dynamic axes is not a valid input/output name".format(key))
        if isinstance(value, list):
            warnings.warn('No names were found for specified dynamic axes of provided input.'
                          'Automatically generated names will be applied to each dynamic axes of input {}'.format(key))

            value_dict = {}
            for i, x in enumerate(value):
                if not isinstance(x, int):
                    raise ValueError("The type of axis index is expected to be an integer")
                if x in value_dict:
                    warnings.warn('Duplicate dynamic axis index {} was provided for input {}.'
                                  .format(x, key))
                else:
                    value_dict[x] = str(key) + '_dynamic_axes_' + str(i + 1)
            dynamic_axes[key] = value_dict

torch._C.Graph.op = _graph_op
torch._C.Graph.at = _graph_at
torch._C.Graph.constant = _graph_constant
torch._C.Node.__getitem__ = _node_getitem
